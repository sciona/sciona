#!/usr/bin/env python3
"""Run serialized Riemannian branches and compare with public source operations."""
import argparse
import ast
import asyncio
import hashlib
import inspect
import importlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import parallel_backend
from sciona.riemannian_bci_execution import build_riemannian_branch_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from validate_channel_tangent import load_reference as tangent_reference, HASHES as TANGENT_HASHES
from validate_bci_predictions import reference_model, HASHES as MODEL_HASHES
from validate_frequency_coherence import Python2Slices, SOURCE_HASH


def reference_preprocessing(directory):
    source = (directory / 'preproc.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError('pinned preprocessing source mismatch')
    namespace = {'np': SimpleNamespace(**vars(np), complex_=np.complex128), 'numpy': np,
                 'hanning': np.hanning, 'xrange': range, 'BaseEstimator': BaseEstimator,
                 'TransformerMixin': TransformerMixin,
                 'covariances': lambda X, estimator: np.stack([estimator(x) for x in X])}
    hankel_source = (directory / 'pyriemann_estimation.py').read_bytes()
    if hashlib.sha256(hankel_source).hexdigest() != '6048dd6b1ad5f59b866088aa0d781ecf5d0dd62d917a12dcbd1e61c740600a22':
        raise ValueError('pinned Hankel source mismatch')
    hankel = ast.parse(hankel_source)
    hankel.body = [n for n in hankel.body if isinstance(n, ast.ClassDef) and n.name == 'HankelCovariances']
    exec(compile(hankel, 'pinned_hankel.py', 'exec'), namespace)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'Windower', 'AutoCorrMat', 'coherences', 'Coherences'}]
    tree = ast.fix_missing_locations(Python2Slices().visit(tree))
    exec(compile(tree, 'ported_preproc.py', 'exec'), namespace)
    return namespace


async def validate(directory):
    preproc = reference_preprocessing(directory)
    tangent = tangent_reference(directory)
    graph = build_riemannian_branch_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg(
        {'cdg_nodes': [{**n, 'version_id': 'synthetic'} for n in nodes],
         'cdg_edges': [{**e, 'version_id': 'synthetic'} for e in edges]},
        version_id='synthetic', content_hash=digest, require_execution_envelope=True)
    counts, maximum = 0, 0.
    for seed in [3801, 3802]:
        rng = np.random.default_rng(seed)
        # Sixteen channels retain the public coherence reference's fixed layout.
        training = [rng.normal(size=(16, 16000)) for _ in range(12)]
        prediction = [rng.normal(size=(16, 8000 * (1 + i % 2))) for i in range(4)]
        labels = np.tile([0, 1], 6)
        expected = {}
        win = preproc['Windower'](window=20)
        train_windows = win.fit_transform(training)
        prediction_windows = win.transform(prediction)
        bands = [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]]
        preprocessors = {
            'autocorrelation': preproc['AutoCorrMat'](order=[1, 2, 4, 8, 16, 32, 64], subsample=2),
            'coherence': preproc['Coherences'](window=512, overlap=.5, fs=400, frequencies=bands, transpose=True),
        }
        for branch, feature in preprocessors.items():
            config_name = ('Alex_Gilberto_models_autocorrmat_TS_XGB.yml' if branch == 'autocorrelation'
                           else 'Alex_Gilberto_models_coherence_transposed_TS_XGB.yml')
            content = (directory / config_name).read_bytes()
            if hashlib.sha256(content).hexdigest() != MODEL_HASHES[config_name]:
                raise ValueError('source model config mismatch')
            config = yaml.safe_load(content)
            settings = config['model'][0]['CoherenceToTangent']
            transform = tangent(metric=ast.literal_eval(settings['metric']), tsupdate=settings['tsupdate'], n_jobs=8)
            with parallel_backend('threading'):
                train_features = transform.fit_transform(feature.fit_transform(train_windows))
                pred_features = transform.transform(feature.transform(prediction_windows))
            model = reference_model(config)
            model.fit(train_features, labels.repeat(2))
            scores = model.predict_proba(pred_features)[:, 1]
            expected[branch] = np.array([scores[:1].max(), scores[1:3].max(), scores[3:4].max(), scores[4:6].max()])
        with TemporaryDirectory(prefix='sciona-riemann-execution-') as temporary:
            previous = runner.RUNS_DIR
            runner.RUNS_DIR = Path(temporary)
            try:
                result = await runner.CDGExecutionSession(None, 'synthetic-riemann', 'check').execute(
                    {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
                if result['status'] != 'completed':
                    raise AssertionError('runner did not complete')
                for branch, want in expected.items():
                    actual = np.load(Path(temporary) / 'check' / (branch + '_segments') / ('out_' + branch + '_segment_probabilities.npy'))
                    np.testing.assert_allclose(actual, want, rtol=1e-10, atol=1e-11)
                    maximum = max(maximum, float(np.max(abs(actual - want))))
                    counts += 1
            finally:
                runner.RUNS_DIR = previous
    return {'synthetic_only': True, 'full_runner_cases': 2, 'branch_comparisons_passed': counts,
            'maximum_absolute_error': maximum, 'graph_digest': digest, 'nodes': len(nodes), 'edges': len(edges),
            'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658',
            'pyriemann_revision': '38c3ec1fae7b2d414ebdc669938b4d00a92d50ee',
            'source_hashes': {**TANGENT_HASHES, **MODEL_HASHES, 'preproc.py': SOURCE_HASH},
            'library_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'xgboost']},
            'validation_script_hashes': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ['validate_riemannian_execution.py', 'validate_channel_tangent.py', 'validate_bci_predictions.py', 'validate_frequency_coherence.py']},
            'execution_module_hashes': {name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest() for name in ['sciona.riemannian_bci_execution', 'sciona.visualizer.runner', 'sciona.services.execution_graph_codec', 'sciona.services.catalog_artifact_retrieval', 'sciona.atoms.riemannian_bci.signal_processing.segment_windows', 'sciona.atoms.audio_speech.atoms']},
            'provider_hashes': {n.matched_primitive: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(runner.REGISTRY[n.matched_primitive]['impl']))).read_bytes()).hexdigest() for n in graph.nodes},
            'limitations': 'Two source branches only; nine other models remain. Python 2 slice/NumPy compatibility and modern sklearn/XGBoost execution, not historical training engine parity or competition performance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(validate(args.reference_dir))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
