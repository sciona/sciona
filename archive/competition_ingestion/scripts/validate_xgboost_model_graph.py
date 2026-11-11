"""Execute native model states through serialized graph edges, using synthetic data."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from sciona.xgboost_model_graph import build_xgboost_model_graph
import sciona.atoms.ml.xgboost.model_io as provider
from sciona.ghost.abstract import AbstractArray
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def validate():
    rng = np.random.default_rng(8403)
    features = rng.normal(size=(96, 3))
    prediction = rng.normal(size=(11, 3))
    values = 3*features[:, 0]-features[:, 2]
    names = ['axis_b', 'axis_a', 'axis_c']
    results = []
    for task in ['regression', 'binary']:
        digest, nodes, edges = encode_execution_graph(build_xgboost_model_graph(task))
        graph = decode_execution_graph(nodes, edges, digest)
        assert encode_execution_graph(graph)[0] == digest
        for domain in ['manufacturing_measurements', 'energy_forecasts']:
            targets = values if task == 'regression' else (values > 0).astype(np.int64)
            if domain == 'energy_forecasts' and task == 'regression':
                targets = targets*10
            captured = {}
            def capture(directory, node, name, value):
                if (node, name) in [('fit', 'out_state'), ('predict', 'in_state'), ('predict', 'out_predictions')]:
                    captured[(node, name)] = value
            payload = dict(features=features, targets=targets, prediction_features=prediction, feature_names=names)
            with tempfile.TemporaryDirectory(prefix='native-model-graph-') as temporary:
                with patch.object(runner, 'RUNS_DIR', Path(temporary)), patch.object(runner, 'save_intermediate_value', side_effect=capture):
                    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-model-graph', task+'-'+domain).execute(payload, cdg=graph))
            if result['status'] != 'completed':
                raise ValueError('Model graph did not complete')
            state = captured[('fit', 'out_state')]
            assert captured[('predict', 'in_state')] == state
            expected = getattr(provider, 'predict_'+task+'_model')(prediction, names, json.loads(json.dumps(state)))
            np.testing.assert_array_equal(captured[('predict', 'out_predictions')], expected)
            symbolic_state = getattr(provider, 'witness_fit_'+task+'_model')(AbstractArray(shape=(96, 3)),
                AbstractArray(shape=(96,), dtype='float64' if task == 'regression' else 'int64'), names)
            symbolic_output = getattr(provider, 'witness_predict_'+task+'_model')(AbstractArray(shape=(11, 3)), names, symbolic_state)
            assert symbolic_output.shape == (11,) and symbolic_output.dtype == 'float32'
            results.append(dict(task=task, domain=domain, completed=True, training_rows=96, prediction_rows=11,
                graph_sha256=digest, native_state_edge_exact=True, json_state_roundtrip_predictions_exact=True,
                symbolic_model_handoff_passed=True))
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0, scenarios=results,
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256((ROOT/'sciona/xgboost_model_graph.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Separate reusable fit/predict graphs; the complete corrected NASA CDG remains pending.',
            'Actual graph runner and graph codec tested; intermediate persistence intercepted and model state stayed in memory.',
            'Synthetic domain examples establish computational transfer, not empirical accuracy.',
            'Catalog binding, environment/license qualification and publication gates remain pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/xgboost_model_graph_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
