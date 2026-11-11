"""Full serialized Contrails graph execution with synthetic JSON inputs."""
import asyncio
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import sciona.atoms.dl.contrails_execution as provider
import sciona.contrails_lifecycle as lifecycle
from sciona.contrails_graph import build_contrails_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def synthetic_payload():
    rng = np.random.default_rng(627)
    def record(key):
        return dict(key=key, thermal=rng.uniform(230, 310, (4, 3, 256, 256)).astype(np.float32),
                    label=rng.integers(0, 2, (1, 256, 256)).astype(np.float32),
                    annotation_mean=rng.uniform(size=(1, 256, 256)).astype(np.float32))
    training = [record('synthetic-' + str(i)) for i in range(10)]
    scoring = [record('synthetic-score')]
    prediction = record('synthetic-prediction')
    payload = dict(version=1, training=training, scoring=scoring,
                   prediction=[{k: prediction[k] for k in ['key', 'thermal']}],
                   initialization=dict(policy='random'),
                   config=dict(seed=627, variant='v43', batch_size=1, num_workers=0, epoch_limit=1))
    return json.loads(json.dumps(payload, default=lambda a: a.tolist(), allow_nan=False))


def run_graph(graph, payload):
    captured = {}
    def capture(directory, node, name, value):
        if node == 'execute' and name == 'out_result':
            captured['result'] = value
    with tempfile.TemporaryDirectory(prefix='synthetic-contrails-') as temp:
        with patch.object(runner, 'RUNS_DIR', Path(temp)), patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-contrails', 'case').execute({'payload': payload}, cdg=graph))
    return status, captured


def validate(root):
    torch.set_num_threads(2)
    digest, nodes, edges = encode_execution_graph(build_contrails_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert len(graph.nodes) == 2 and len(graph.edges) == 1
    witness = {'payload': {}}
    for node in graph.nodes:
        fn = getattr(provider, node.matched_primitive.rsplit('.', 1)[-1])
        assert list(inspect.signature(fn).parameters) == [p.name for p in node.inputs]
        witness[node.outputs[0].name] = getattr(provider, 'witness_' + fn.__name__)(**{p.name: witness[p.name] for p in node.inputs})
    payload = synthetic_payload()
    original = lifecycle.train_fold
    completed = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(True)
        print(json.dumps({'completed_training_folds': len(completed)}), flush=True)
        return result
    with patch.object(lifecycle, 'train_fold', side_effect=observed):
        status, captured = run_graph(graph, payload)
    assert status['status'] == 'completed'
    result = captured['result']
    assert len(result['folds']) == 4 and sum(f['optimizer_updates'] for f in result['folds']) == 12
    assert np.asarray(result['probabilities']).shape == (1, 1, 256, 256)
    result_json = json.dumps(result, sort_keys=True, allow_nan=False)
    bad = dict(payload, version=True)
    try:
        run_graph(graph, bad)
    except RuntimeError:
        rejected = True
    else:
        raise AssertionError('Invalid graph payload accepted')
    paths = sorted((root / 'sciona').glob('contrails_*.py')) + [Path(__file__)]
    return {'approved': False, 'checks': {'provider_contracts': 2, 'serialized_full_model_graphs': 1,
                                        'trained_folds': 4, 'optimizer_updates': 12, 'runner_rejections': int(rejected)},
            'serialized_graph_sha256': digest, 'result_sha256': hashlib.sha256(result_json.encode()).hexdigest(),
            'provider_sha256': hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
            'hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'scope': 'Actual graph runner and providers with full models and all v43 folds; synthetic JSON data, explicit random initialization and one epoch/batch1/workers0. No served-catalog or default-schedule claim.'}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = validate(root)
    (root / 'docs/reviews/competition_contrails_graph.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['checks']), flush=True)
