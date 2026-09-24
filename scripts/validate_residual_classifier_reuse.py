"""Run the complete numerical graph in two synthetic non-airport applications."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def fixture(domain, seed):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(480, 3))
    query = rng.normal(size=(64, 3))
    if domain == 'manufacturing_cycle_duration':
        names = ['load_fraction', 'ambient_temperature', 'tool_wear']
        targets = 100 + 15 * features[:, 0] + 8 * features[:, 1] ** 2 + rng.normal(0, 12, 480)
        units = 'seconds'
    elif domain == 'energy_interval_consumption':
        features = np.exp(features / 2)
        query = np.exp(query / 2)
        names = ['demand_indicator', 'irradiance_indicator', 'occupancy_indicator']
        targets = 200 + 25 * features[:, 0] * features[:, 2] - 10 * features[:, 1] + rng.normal(0, 15, 480)
        units = 'watt_hours'
    else:
        raise ValueError('Unknown synthetic domain')
    return dict(features=features, targets=targets, groups=np.repeat(np.arange(120, dtype=np.int64), 4),
                feature_names=names, prediction_features=query, train_fraction=.4, seed=42,
                maximum_error=30., threshold=.5, prediction_feature_name='estimated_target'), units


def reference(p):
    """Independent sklearn/backend orchestration; no provider atom calls."""
    x = pd.DataFrame(p['features'], columns=p['feature_names'])
    y = p['targets']
    train, held = next(GroupShuffleSplit(n_splits=1, train_size=p['train_fraction'],
        random_state=p['seed']).split(x, y, p['groups']))
    internal = XGBRegressor(n_jobs=1, device='cpu').fit(x.iloc[train], y[train])
    internal_predictions = np.round(internal.predict(x)).astype(np.int32)
    retained = held[np.abs(y[held] - internal_predictions[held]) <= p['maximum_error']]
    if not 0 < len(retained) < len(held):
        raise ValueError('Fixture must exercise actual held-population filtering')
    lower = XGBRegressor(n_jobs=1, device='cpu').fit(x.iloc[retained], y[retained])
    rounded = np.round(lower.predict(x)).astype(np.int32)
    labels = (rounded < y).astype(np.int64)
    augmented = x.copy()
    augmented[p['prediction_feature_name']] = rounded.astype(np.float64)
    classifier = XGBClassifier(n_jobs=1, device='cpu').fit(augmented.iloc[held], labels[held])
    probability = classifier.predict_proba(augmented.iloc[held])[:, 1]
    residual = y[held] - rounded[held]
    offsets = np.array([np.median(residual[probability > p['threshold']]),
                        np.median(-residual[probability < p['threshold']])], dtype=np.float64)
    if not np.isfinite(offsets).all():
        raise ValueError('Fixture must exercise both calibration groups')
    query = pd.DataFrame(p['prediction_features'], columns=p['feature_names'])
    prediction = lower.predict(query)
    query[p['prediction_feature_name']] = prediction.astype(np.float64)
    probability = classifier.predict_proba(query)[:, 1]
    corrected = prediction.copy()
    corrected[probability > p['threshold']] += np.float32(offsets[0])
    corrected[probability < p['threshold']] -= np.float32(offsets[1])
    return np.round(corrected).astype(np.int32), offsets, train, retained


def validate():
    digest, nodes, edges = encode_execution_graph(build_residual_classifier_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    cases = []
    for index, domain in enumerate(['manufacturing_cycle_duration', 'energy_interval_consumption']):
        payload, units = fixture(domain, 1801 + index)
        with threadpool_limits(limits=1):
            expected, offsets, train, retained = reference(payload)
        captured = {}
        def capture(directory, node, name, value):
            if (node, name) in [('final', 'out_predictions'), ('offsets', 'out_offsets'),
                                  ('split', 'out_train'), ('retained', 'out_selection')]:
                captured[node, name] = value
        with tempfile.TemporaryDirectory(prefix='residual-reuse-') as directory:
            with threadpool_limits(limits=1), patch.object(runner, 'RUNS_DIR', Path(directory)), \
                    patch.object(runner, 'save_intermediate_value', side_effect=capture):
                result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-residual-reuse', str(index))
                                     .execute(payload, cdg=graph))
        if result['status'] != 'completed':
            raise ValueError('Reusable graph execution failed')
        np.testing.assert_array_equal(captured['final', 'out_predictions'], expected)
        np.testing.assert_array_equal(captured['offsets', 'out_offsets'], offsets)
        np.testing.assert_array_equal(np.flatnonzero(captured['split', 'out_train']), train)
        np.testing.assert_array_equal(np.flatnonzero(captured['retained', 'out_selection']), retained)
        cases.append(dict(domain=domain, target_unit=units, predictions=len(expected),
            exact_predictions=True, exact_offsets=True, exact_populations=True,
            retained_rows=len(retained), held_rows=480-len(train), completed=True))
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_sha256=digest, builder_sha256=hashlib.sha256((ROOT/'sciona/residual_classifier_graph.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scenarios=cases,
        limitations=[
            'Constructed examples establish compositional execution, not empirical model quality in these applications.',
            'Targets must support nearest-integer output quantization; error thresholds and offsets use the chosen target unit.',
            'Caller supplies ordered numeric features and aligned grouping; domain extraction and observation availability remain adapter responsibilities.',
            'In-process execution only; model persistence is intercepted and no domain tables or models are published.',
        ])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/residual_classifier_reuse.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
