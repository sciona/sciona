#!/usr/bin/env python3
"""Compare serialized relative-power execution against the pinned source pipeline."""
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
import yaml
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.relative_power_execution import build_relative_power_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from validate_bci_predictions import reference_model

HASHES = {
    'preproc.py': '1cfbc71faa230e1e4ffb72f27604be557ea08cb31186457363d37eee71d44bb2',
    'mne_transformer_013.py': '469718ad7f8e26c0bb31ec20ab71d8d223a4a84f091bfdde01b4f0acee54604a',
    'Alex_Gilberto_models_relative_log_power_XGB.yml': '74b55594ae46d9d756ecaedf8a9e7524be70a37e21a3bddf968e02673d0014ce',
    'Alex_Gilberto_config_config_relative_log_power.yml': '7c762f63d3db488bfd9fb7d057bbfae083d47b71c81f46f161ba777e92abf18a',
}


def load_source(directory):
    for name, digest in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('pinned public source differs: ' + name)
    namespace = dict(np=np, welch=welch, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin, xrange=range)
    for name, selected in [('preproc.py', {'Windower', 'relative_log_power', 'RelativeLogPower'}),
                           ('mne_transformer_013.py', {'Vectorizer'})]:
        tree = ast.parse((directory / name).read_bytes())
        tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        if len(tree.body) != len(selected):
            raise ValueError('missing source definitions')
        exec(compile(tree, 'pinned_' + name, 'exec'), namespace)
    return namespace


def synthetic_segments(rng, classes, counts):
    result = []
    for label, count in zip(classes, counts):
        t = np.arange(8000 * count + 17) / 400.
        # Deliberately learnable spectral contrast; predictions must not all be equal.
        frequency = 7. if label == 0 else 43.
        signal = np.stack([np.sin(2 * np.pi * frequency * t + phase) for phase in [.1, .7, 1.2]])
        result.append(signal + .4 * rng.normal(size=signal.shape))
    return result


async def validate(directory):
    source = load_source(directory)
    feature_config = yaml.safe_load((directory / 'Alex_Gilberto_config_config_relative_log_power.yml').read_text())
    model_config = yaml.safe_load((directory / 'Alex_Gilberto_models_relative_log_power_XGB.yml').read_text())
    graph = build_relative_power_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': 'synthetic'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'synthetic'} for e in edges]}, version_id='synthetic', content_hash=digest, require_execution_envelope=True)
    maximum, minimum_spread = 0., 1.
    for seed in [6601, 6602, 6603]:
        rng = np.random.default_rng(seed)
        labels = np.tile([0, 1], 8)
        training = synthetic_segments(rng, labels, [2] * len(labels))
        counts = [1, 2, 3, 1, 2, 3]
        prediction = synthetic_segments(rng, [0, 1, 0, 1, 0, 1], counts)
        window = source['Windower'](**feature_config['preproc'][0]['Windower'])
        power = source['RelativeLogPower'](**feature_config['preproc'][1]['RelativeLogPower'])
        vectorizer = source['Vectorizer']()
        train_features = vectorizer.fit_transform(power.fit_transform(window.fit_transform(training)))
        pred_features = vectorizer.transform(power.transform(window.transform(prediction)))
        model = reference_model(model_config)
        model.fit(train_features, labels.repeat(2))
        probabilities = model.predict_proba(pred_features)[:, 1]
        expected = pd.Series(probabilities).groupby(np.repeat(np.arange(len(counts)), counts)).max().to_numpy()
        spread = float(np.ptp(expected))
        if spread < .01:
            raise AssertionError('degenerate synthetic reference predictions')
        minimum_spread = min(minimum_spread, spread)
        with TemporaryDirectory(prefix='sciona-relative-power-') as temporary:
            previous = runner.RUNS_DIR
            runner.RUNS_DIR = Path(temporary)
            try:
                result = await runner.CDGExecutionSession(None, 'synthetic-relative-power', 'check').execute(
                    {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
                if result['status'] != 'completed':
                    raise AssertionError('runner incomplete')
                actual = np.load(Path(temporary) / 'check/segments/out_relative_power_segment_probabilities.npy')
                np.testing.assert_array_equal(actual, expected)
                maximum = max(maximum, float(np.max(abs(actual - expected))))
            finally:
                runner.RUNS_DIR = previous
    modules = ['sciona.relative_power_execution', 'sciona.visualizer.runner', 'sciona.services.execution_graph_codec',
               'sciona.services.catalog_artifact_retrieval', 'sciona.atoms.riemannian_bci.signal_processing.segment_windows',
               'sciona.atoms.audio_speech.atoms']
    return dict(synthetic_only=True, full_runner_cases=3, maximum_absolute_error=maximum,
        minimum_reference_prediction_spread=minimum_spread, graph_digest=digest, nodes=len(nodes), edges=len(edges),
        source_revision='00f937cc7710977dc812d9fc675864e2b8288658', mne_reference='v0.13', source_hashes=HASHES,
        provider_hashes={n.matched_primitive: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(runner.REGISTRY[n.matched_primitive]['impl']))).read_bytes()).hexdigest() for n in graph.nodes},
        execution_module_hashes={name: hashlib.sha256(Path(inspect.getfile(importlib.import_module(name))).read_bytes()).hexdigest() for name in modules},
        validation_script_hashes={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ['validate_relative_power_execution.py', 'validate_bci_predictions.py']},
        library_versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost']},
        limitations='One source model branch only. Current numerical libraries and contemporaneous MNE reference; original environment and competition performance not reproduced. No full eleven-model ensemble approval implied.')


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
