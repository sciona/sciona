#!/usr/bin/env python3
"""Validate serialized Feng KNN signal-to-score execution with synthetic clips."""
import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.signal import butter, sosfilt
from sciona.feng_expanded_execution import build_feng_expanded_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from validate_feng_filter import load_reference as filter_reference, SOURCE_SHA as FILTER_SHA
from validate_feng_expanded_features import load_reference as fft_reference, SOURCE_SHA as FFT_SHA
from validate_feng_expanded_knn import load_reference as knn_reference, SOURCE_SHA as KNN_SHA
from validate_feng_glm import load_reference as glm_reference, SOURCE_SHA as GLM_SHA


async def validate(directory):
    filtering, fft, knn = filter_reference(directory), fft_reference(directory), knn_reference(directory)
    glm = glm_reference(directory)
    graph = build_feng_expanded_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': 'synthetic'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'synthetic'} for e in edges]}, version_id='synthetic', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    cases = []
    maximum = 0.
    for seed in [9321, 9322, 9323]:
        rng = np.random.default_rng(seed)
        labels = np.array([0, 1, 0, 1, 0, 1])
        def segments(classes):
            result = []
            for label in classes:
                # Full source duration; deliberately non-400-Hz source sample
                # count exercises the complete resampling path.
                t = np.arange(180000)/300.
                frequency = 7. if label == 0 else 43.
                x = np.stack([np.sin(2*np.pi*frequency*t+p) for p in np.linspace(.1, 2.5, 16)], axis=1)
                noise = rng.normal(size=x.shape)
                sos = butter(4, 20 if label == 0 else 35, btype='lowpass' if label == 0 else 'highpass', fs=300, output='sos')
                result.append(.2*x + sosfilt(sos, noise, axis=0) + .01*rng.normal(size=x.shape))
            return result
        training, prediction = segments(labels), segments([0, 1, 0, 1])
        def source_features(partition):
            result = []
            for segment in partition:
                x = filtering(segment.astype(np.float32), 400, 600, .1, 180)
                result.append(np.c_[fft['compute_fft_more'](x, 600, 400, 6, 50, 50, 'meanlog_std'),
                                    fft['compute_timedomaincoef'](x, 600, 400, 50, 50),
                                    fft['compute_frequencydomaincoef'](x, 600, 400, 6, 50, 50)])
            return np.stack(result)
        train_features, pred_features = source_features(training), source_features(prediction)
        knn['load_train_data_knn'] = lambda *_: {'x': train_features.astype(np.float32), 'y': labels.copy()}
        knn['load_test_data'] = lambda *_: {'x': pred_features.astype(np.float32), 'id': list(range(len(prediction)))}
        expected_knn = np.array([v for _, v in knn['predict'](None, knn['train'](None, None), None)])
        glm['load_train_data_lasso'] = lambda *_: {'x': train_features.astype(np.float32), 'y': labels.copy()}
        glm['load_test_data'] = lambda *_: {'x': pred_features.astype(np.float32), 'id': list(range(len(prediction)))}
        model, scaler = glm['train'](None, None)
        expected_glm = np.array([v for _, v in glm['predict'](None, model, scaler, None)])
        expected_outputs = {'knn': expected_knn, 'glm': expected_glm}
        spread = min(float(np.ptp(v)) for v in expected_outputs.values())
        assert spread > .1, 'degenerate synthetic predictions'
        with TemporaryDirectory(prefix='sciona-feng-knn-') as temporary:
            previous = runner.RUNS_DIR
            runner.RUNS_DIR = Path(temporary)
            try:
                result = await runner.CDGExecutionSession(None, 'synthetic-feng-knn', 'check').execute(
                    {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
                assert result['status'] == 'completed', result['status']
                for branch, expected in expected_outputs.items():
                    name = 'expanded_knn' if branch == 'knn' else 'glm'
                    actual = np.load(Path(temporary)/('check/'+branch+'/out_feng_'+name+'_segment_probabilities.npy'))
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
                    maximum = max(maximum, float(np.max(abs(actual-expected))))
                for part, expected_features in [('training', train_features), ('prediction', pred_features)]:
                    features = np.load(Path(temporary)/('check/features/out_'+part+'_features.npy'))
                    np.testing.assert_allclose(features, expected_features, rtol=1e-12, atol=1e-13)
            finally:
                runner.RUNS_DIR = previous
        cases.append(dict(seed=seed, prediction_spread=spread))
    modules = ['sciona.feng_expanded_execution', 'sciona.visualizer.runner', 'sciona.services.execution_graph_codec',
               'sciona.services.catalog_artifact_retrieval',
               'sciona.atoms.riemannian_bci.signal_processing.feng_filter',
               'sciona.atoms.riemannian_bci.signal_processing.feng_expanded_features']
    return dict(synthetic_only=True, full_runner_cases=len(cases), cases=cases, maximum_absolute_error=maximum, branch_comparisons_passed=6,
        minimum_reference_prediction_spread=min(c['prediction_spread'] for c in cases),
        graph_digest=digest, nodes=len(nodes), edges=len(edges), source_revision='00f937cc7710977dc812d9fc675864e2b8288658',
        source_hashes={'Feng_preprocessors_filtering.py': FILTER_SHA, 'Feng_preprocessors_fft.py': FFT_SHA, 'Feng_models_KNNmorefeature.py': KNN_SHA, 'Feng_models_GLM.py': GLM_SHA},
        provider_hashes={n.matched_primitive: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(runner.REGISTRY[n.matched_primitive]['impl']))).read_bytes()).hexdigest() for n in graph.nodes},
        execution_module_hashes={name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest() for name in modules},
        validation_script_hashes={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['validate_feng_expanded_execution.py', 'validate_feng_filter.py', 'validate_feng_expanded_features.py', 'validate_feng_expanded_knn.py', 'validate_feng_glm.py']},
        library_versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'scikit-learn']},
        limitations='Two source expanded-feature models for one caller-selected population. Source float32 loading, KNN prediction-batch scaling and GLM training-scaler reuse preserved. Current GLM LBFGS defaults; historical solver not reproduced. Python 2 FFT shape division adapted. Current numerical libraries, synthetic signals only; no historical-engine, clinical or competition-performance claim. Full ensemble remains incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(validate(args.reference_dir))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'full_runner_cases': report['full_runner_cases'], 'maximum_absolute_error': report['maximum_absolute_error'], 'minimum_reference_prediction_spread': report['minimum_reference_prediction_spread'], 'graph_digest': report['graph_digest']}))


if __name__ == '__main__':
    main()
