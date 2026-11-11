#!/usr/bin/env python3
"""Validate serialized Feng XGBoost signal-to-score execution with synthetic clips."""
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
from sciona.feng_xgb_execution import build_feng_xgb_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from validate_feng_filter import load_reference as filter_reference, SOURCE_SHA as FILTER_SHA
from validate_feng_fft import load_reference as fft_reference, SOURCE_SHA as FFT_SHA
from validate_feng_xgb import load_reference as xgb_reference, SOURCE_SHA as XGBoost_SHA


async def validate(directory):
    filtering, fft, xgb = filter_reference(directory), fft_reference(directory), xgb_reference(directory)
    graph = build_feng_xgb_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': 'synthetic'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'synthetic'} for e in edges]}, version_id='synthetic', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    cases = []
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
                x = np.stack([np.sin(2*np.pi*frequency*t+p) for p in [.2, .8, 1.3]], axis=1)
                noise = rng.normal(size=x.shape)
                sos = butter(4, 20 if label == 0 else 35, btype='lowpass' if label == 0 else 'highpass', fs=300, output='sos')
                result.append(.2*x + sosfilt(sos, noise, axis=0) + .01*rng.normal(size=x.shape))
            return result
        training, prediction = segments(labels), segments([0, 1, 0, 1])
        def source_features(partition):
            return np.stack([fft(filtering(x.astype(np.float32), 400, 600, .1, 180), 600, 400, 6, 30, 30, 'meanlog_std') for x in partition])
        train_features, pred_features = source_features(training), source_features(prediction)
        xgb['load_train_data_xgb'] = lambda *_: {'x': train_features.astype(np.float32), 'y': labels.copy()}
        xgb['load_test_data'] = lambda *_: {'x': pred_features.astype(np.float32), 'id': list(range(len(prediction)))}
        expected = np.array([v for _, v in xgb['predict'](None, None, xgb['train'](None, None, xgb['params']))])
        spread = float(np.ptp(expected))
        assert spread > .1, 'degenerate synthetic predictions'
        with TemporaryDirectory(prefix='sciona-feng-xgb-') as temporary:
            previous = runner.RUNS_DIR
            runner.RUNS_DIR = Path(temporary)
            try:
                result = await runner.CDGExecutionSession(None, 'synthetic-feng-xgb', 'check').execute(
                    {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
                assert result['status'] == 'completed', result['status']
                actual = np.load(Path(temporary)/'check/classifier/out_feng_xgb_segment_probabilities.npy')
                np.testing.assert_array_equal(actual, expected)
                for part, expected_features in [('training', train_features), ('prediction', pred_features)]:
                    features = np.load(Path(temporary)/('check/features/out_'+part+'_features.npy'))
                    np.testing.assert_array_equal(features, expected_features)
            finally:
                runner.RUNS_DIR = previous
        cases.append(dict(seed=seed, exact=True, prediction_spread=spread))
    modules = ['sciona.feng_xgb_execution', 'sciona.visualizer.runner', 'sciona.services.execution_graph_codec',
               'sciona.services.catalog_artifact_retrieval',
               'sciona.atoms.riemannian_bci.signal_processing.feng_filter',
               'sciona.atoms.riemannian_bci.signal_processing.feng_fft']
    return dict(synthetic_only=True, full_runner_cases=len(cases), cases=cases, maximum_absolute_error=0.,
        minimum_reference_prediction_spread=min(c['prediction_spread'] for c in cases),
        graph_digest=digest, nodes=len(nodes), edges=len(edges), source_revision='00f937cc7710977dc812d9fc675864e2b8288658',
        source_hashes={'Feng_preprocessors_filtering.py': FILTER_SHA, 'Feng_preprocessors_fft.py': FFT_SHA, 'Feng_models_XGB.py': XGBoost_SHA},
        provider_hashes={n.matched_primitive: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(runner.REGISTRY[n.matched_primitive]['impl']))).read_bytes()).hexdigest() for n in graph.nodes},
        execution_module_hashes={name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest() for name in modules},
        validation_script_hashes={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['validate_feng_xgb_execution.py', 'validate_feng_filter.py', 'validate_feng_fft.py', 'validate_feng_xgb.py']},
        library_versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'xgboost']},
        limitations='One source model for one caller-selected population. Source float32 loading, unscaled features and 500-round training preserved. Python 2 FFT shape division adapted. Current numerical libraries, synthetic signals only; no historical-engine, clinical or competition-performance claim. Full ensemble remains incomplete.')


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
