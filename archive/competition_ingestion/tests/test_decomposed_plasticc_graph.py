"""Execute the serialized draft graph with complete synthetic invocations."""
import asyncio
from unittest.mock import patch
import numpy as np
import pytest
from sciona.decomposed_plasticc_graph import build_decomposed_plasticc_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.architect.models import EdgeKind
from sciona.visualizer import runner
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from scripts.validate_plasticc_runtime import synthetic_payload


@pytest.mark.parametrize('weighting,class_weighting', [('flat', 'flat'), ('redshift', 'kaggle')])
def test_serialized_complete_graph(tmp_path, weighting, class_weighting):
    graph = build_decomposed_plasticc_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    payload = synthetic_payload()
    payload['config'].update(weighting=weighting, class_weighting=class_weighting)
    expected = predict_runtime(train_runtime(prepare_runtime(payload)))
    captured = {}
    def capture(directory, node, name, value):
        if node == 'encode' and name == 'out_result': captured['result'] = value
    runner._ensure_atoms_imported()
    with patch.object(runner, 'RUNS_DIR', tmp_path), patch.object(runner, 'save_intermediate_value', side_effect=capture):
        status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-plasticc', 'decomposed').execute({'payload': payload}, cdg=graph))
    assert status['status'] == 'completed', status.get('error')
    traces = {item['node_id']: item for item in status['trace']}
    assert set(traces) == {node.node_id for node in graph.nodes}
    assert not any(item['cached'] for item in traces.values())
    callables = {edge.source_id for edge in graph.edges if edge.edge_kind == EdgeKind.CALLABLE_INJECTION}
    assert len(callables) == 12
    assert all(traces[node].get('callable_reference') is True for node in callables)
    actual = captured['result']
    assert actual['classes'] == expected['classes'] and actual['object_ids'] == expected['object_ids']
    np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-8, rtol=1e-8)
