"""Graph routing checks; numerical source parity is covered by reference validators."""
import importlib
import numpy as np
import pytest
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.ghost.registry import REGISTRY
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer.runner import CDGExecutionSession

populations = importlib.import_module(
    'sciona.atoms.riemannian_bci.signal_processing.andriy_populations')


@pytest.mark.parametrize('position', range(7))
def test_populations_reject_wrong_population_count(position):
    args = [[None] * 3 for _ in range(7)]
    args[position] = [None] * 2
    with pytest.raises(ValueError, match='exactly three'):
        populations.andriy_documented_populations(*args)


def test_population_witness_retains_independent_prediction_counts():
    args = [[[] for _ in range(3)] for _ in range(6)] + [[128] * 3]
    args[5] = [[None] * size for size in [1, 3, 2]]
    train, labels, pred, valid = populations.witness_andriy_documented_populations(*args)
    assert len(train) == len(labels) == 3
    assert [x.shape for x in pred] == [(19, 1965), (57, 1965), (38, 1965)]
    assert [x.shape for x in valid] == [(19,), (57,), (38,)]


@pytest.mark.asyncio
async def test_serialized_graph_routes_population_lists_and_branch_outputs(tmp_path, monkeypatch):
    graph = build_andriy_execution_graph()
    for node in graph.nodes:
        importlib.import_module(node.matched_primitive.rsplit('.', 1)[0])
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({
        'cdg_nodes': [{**n, 'version_id': 'test'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'test'} for e in edges]},
        version_id='test', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    assert len(nodes) == 4 and len(edges) == 12
    assert len({(e.target_id, e.input_name) for e in graph.edges}) == 12
    calls = []
    expected = []
    for i, size in enumerate([1, 3, 2]):
        expected.append((np.full((2, 1965), i + 1.), np.array([1, 0]),
                         np.full((19 * size, 1965), i + 10.),
                         np.arange(19 * size) % 2 == 0))

    def prepare(*values):
        i = len(calls)
        assert values == tuple(f'{name}:{i}' for name in range(7))
        calls.append(i)
        return expected[i]

    monkeypatch.setattr(populations, 'andriy_documented_population', prepare)
    seen = []
    for offset, node in enumerate(graph.nodes[1:]):
        def branch(training_features, training_labels, prediction_features,
                   valid_prediction_rows, _offset=offset):
            for actual, reference in zip(zip(training_features, training_labels,
                    prediction_features, valid_prediction_rows), expected):
                for value, wanted in zip(actual, reference):
                    np.testing.assert_array_equal(value, wanted)
            seen.append(_offset)
            return np.arange(6.) + _offset * 10, np.array([_offset + 100.])
        monkeypatch.setitem(REGISTRY, node.matched_primitive,
                            {**REGISTRY[node.matched_primitive], 'impl': branch})
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR', tmp_path)
    inputs = {port.name: [f'{i}:{j}' for j in range(3)]
              for i, port in enumerate(graph.nodes[0].inputs)}
    result = await CDGExecutionSession(None, 'synthetic-andriy-routing', 'check').execute(
        inputs, cdg=restored)
    assert result['status'] == 'completed', result
    assert calls == [0, 1, 2] and sorted(seen) == [0, 1, 2]
    for i, node in enumerate(graph.nodes[1:]):
        for port, wanted in zip(node.outputs, [np.arange(6.) + i * 10, np.array([i + 100.])]):
            np.testing.assert_array_equal(
                np.load(tmp_path / 'check' / node.node_id / f'out_{port.name}.npy'), wanted)
