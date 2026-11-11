#!/usr/bin/env python3
"""Full serialized combined-feature branch versus pinned competition source."""
import argparse
import ast
import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import scipy
import yaml
from sciona.combined_feature_execution import build_combined_feature_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from validate_relative_power_execution import load_source, synthetic_segments, HASHES as POWER_HASHES
from validate_ar_standard_errors import load_reference as ar_reference, HASHES as AR_HASHES
from validate_fractal_features import load_reference as fractal_reference, HASHES as FRACTAL_HASHES
from validate_bci_predictions import reference_model

HASHES = {**POWER_HASHES, **AR_HASHES, **FRACTAL_HASHES,
    'Alex_Gilberto_models_all_flat_datasets_XGB.yml': '3df8b71abfa4caeb860ce5fbf911652297c75f859e7b7f67b2b040ba73311e55',
    'Alex_Gilberto_config_config_ARError.yml': '50b0a505833ce3b72112e051e7d6d458ae6d7ed940c6bf8582eb4e5e89d2c948',
    'Alex_Gilberto_config_config_stats.yml': '1c4f74def7676959312de035775891a608a50a41d86b236aacc5650879f6ecd9',
    'Alex_Gilberto_config_config_various.yml': '3253073d241c0c902dcbb8e027f667347d9cbc5f3648febf0683bd76ce1793a7',
}


async def validate(directory):
    for name, digest in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('pinned source mismatch: ' + name)
    source = load_source(directory)
    source['sp'] = scipy
    tree = ast.parse((directory / 'preproc.py').read_bytes())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BasicStats']
    exec(compile(tree, 'pinned_basic_stats.py', 'exec'), source)
    source['ARError'] = ar_reference(directory)
    source['VariousFeatures'] = fractal_reference(directory)
    configs = {name: yaml.safe_load((directory / ('Alex_Gilberto_config_config_' + name + '.yml')).read_text())
               for name in ['relative_log_power', 'ARError', 'stats', 'various']}
    model_config = yaml.safe_load((directory / 'Alex_Gilberto_models_all_flat_datasets_XGB.yml').read_text())
    if model_config['datasets'] != ['relative_log_power', 'arError', 'stats', 'various']:
        raise ValueError('source feature-family order differs')
    graph = build_combined_feature_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': 'synthetic'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'synthetic'} for e in edges]}, version_id='synthetic', content_hash=digest, require_execution_envelope=True)
    if len({(e.target_id, e.input_name) for e in graph.edges}) != len(graph.edges):
        raise ValueError('ambiguous graph input binding')
    maximum, minimum_spread = 0., 1.
    for seed in [7711, 7712, 7713]:
        rng = np.random.default_rng(seed)
        labels = np.tile([0, 1], 8)
        training = synthetic_segments(rng, labels, [2] * len(labels))
        counts = [1, 2, 3, 1, 2, 3]
        prediction = synthetic_segments(rng, [0, 1, 0, 1, 0, 1], counts)
        train_blocks, pred_blocks = [], []
        for name in ['relative_log_power', 'ARError', 'stats', 'various']:
            config = configs[name]
            window = source['Windower'](**config['preproc'][0]['Windower'])
            cls, kwargs = next(iter(config['preproc'][1].items()))
            transformer = source[cls](**(kwargs or {}))
            # Each source feature job has its own window arrays; Hurst mutates them.
            train_blocks.append(transformer.fit_transform(window.fit_transform(training)))
            pred_blocks.append(transformer.transform(window.transform(prediction)))
        vectorizer = source['Vectorizer']()
        train_features = vectorizer.fit_transform(np.concatenate(train_blocks, axis=-1))
        pred_features = vectorizer.transform(np.concatenate(pred_blocks, axis=-1))
        if train_features.shape[1] != 63:
            raise AssertionError('expected 21 features for each synthetic channel')
        model = reference_model(model_config)
        model.fit(train_features, labels.repeat(2))
        expected = pd.Series(model.predict_proba(pred_features)[:, 1]).groupby(np.repeat(np.arange(len(counts)), counts)).max().to_numpy()
        spread = float(np.ptp(expected))
        if spread < .01:
            raise AssertionError('degenerate synthetic reference predictions')
        minimum_spread = min(minimum_spread, spread)
        with TemporaryDirectory(prefix='sciona-combined-features-') as temporary:
            previous = runner.RUNS_DIR
            runner.RUNS_DIR = Path(temporary)
            try:
                result = await runner.CDGExecutionSession(None, 'synthetic-combined-features', 'check').execute(
                    {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
                if result['status'] != 'completed':
                    raise AssertionError('runner incomplete')
                actual_features = np.load(Path(temporary) / 'check/vectorize/out_training_features.npy')
                np.testing.assert_allclose(actual_features, train_features, rtol=1e-12, atol=1e-13)
                actual = np.load(Path(temporary) / 'check/segments/out_combined_feature_segment_probabilities.npy')
                np.testing.assert_array_equal(actual, expected)
                maximum = max(maximum, float(np.max(abs(actual - expected))))
            finally:
                runner.RUNS_DIR = previous
    modules = ['sciona.combined_feature_execution', 'sciona.relative_power_execution', 'sciona.visualizer.runner',
               'sciona.services.execution_graph_codec', 'sciona.services.catalog_artifact_retrieval',
               'sciona.atoms.riemannian_bci.signal_processing.segment_windows', 'sciona.atoms.audio_speech.atoms']
    scripts = ['validate_combined_feature_execution.py', 'validate_relative_power_execution.py', 'validate_ar_standard_errors.py',
               'validate_fractal_features.py', 'validate_bci_predictions.py']
    return dict(synthetic_only=True, full_runner_cases=3, maximum_absolute_error=maximum,
        minimum_reference_prediction_spread=minimum_spread, graph_digest=digest, nodes=len(nodes), edges=len(edges),
        source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_hashes=HASHES,
        provider_hashes={n.matched_primitive: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(runner.REGISTRY[n.matched_primitive]['impl']))).read_bytes()).hexdigest() for n in graph.nodes},
        execution_module_hashes={n: hashlib.sha256(Path(inspect.getfile(importlib.import_module(n))).read_bytes()).hexdigest() for n in modules},
        validation_script_hashes={n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest() for n in scripts},
        library_versions={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'statsmodels']},
        limitations='Single combined-feature branch only. Historical method adapters and current numerical libraries as recorded in component evidence; not a historical runtime or predictive-performance claim. Full eleven-model ensemble remains outside scope.')


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
