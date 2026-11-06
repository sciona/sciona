"""A registered operation may be called repeatedly inside another graph node."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def node(identity, inputs):
    return AlgorithmicNode(node_id=identity, name=identity, description=identity,
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive='synthetic.'+identity,
        inputs=[IOSpec(name=n, type_desc='object') for n in inputs],
        outputs=[IOSpec(name='result', type_desc='object')])


def edge(source, target, parameter, kind=EdgeKind.DATA_FLOW):
    return DependencyEdge(source_id=source, target_id=target, output_name='result',
        input_name=parameter, source_type='object', target_type='object', edge_kind=kind)


@pytest.fixture
def setup_runner(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(runner, '_ensure_atoms_imported', lambda: None)
    monkeypatch.setattr(runner, 'save_intermediate_value',
                        lambda directory, identity, name, value: captured.update({(identity, name): value}))
    return captured


def test_repeated_callable_after_serialization_without_eager_invocation(monkeypatch, setup_runner):
    calls = []
    def transform(value):
        calls.append(value)
        return value*2
    def iterate(operation, initial):
        for _ in range(3):
            initial = operation(initial)
        return initial
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.transform', {'impl': transform})
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.iterate', {'impl': iterate})
    graph = CDGExport(nodes=[node('transform', ['value']), node('iterate', ['operation', 'initial'])],
        edges=[edge('transform', 'iterate', 'operation', EdgeKind.CALLABLE_INJECTION)])
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic', 'calls').execute({'initial': 3}, cdg=graph))
    assert result['status'] == 'completed' and calls == [3, 6, 12]
    assert setup_runner['iterate', 'out_result'] == 24
    assert setup_runner['iterate', 'in_operation'] is transform
    assert result['trace'][0]['callable_reference'] is True
    assert all(not t['cached'] for t in result['trace'])
    assert ('transform', 'out_result') not in setup_runner


def test_mixed_data_and_callable_consumers(monkeypatch, setup_runner):
    calls = []
    def transform(value):
        calls.append(value)
        return value+1
    def combine(operation, previous):
        return operation(previous)
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.transform', {'impl': transform})
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.combine', {'impl': combine})
    graph = CDGExport(nodes=[node('transform', ['value']), node('combine', ['operation', 'previous'])],
        edges=[edge('transform', 'combine', 'operation', EdgeKind.CALLABLE_INJECTION),
               edge('transform', 'combine', 'previous')])
    asyncio.run(runner.CDGExecutionSession(None, 'synthetic', 'mixed').execute({'value': 3}, cdg=graph))
    assert calls == [3, 4]
    assert setup_runner['combine', 'out_result'] == 5


def test_callable_only_source_rejects_silent_partial_binding(monkeypatch, setup_runner):
    for name in ('input', 'transform', 'iterate'):
        monkeypatch.setitem(runner.REGISTRY, 'synthetic.'+name, {'impl': lambda **kwargs: 0})
    graph = CDGExport(nodes=[node('input', []), node('transform', ['value']), node('iterate', ['operation'])],
        edges=[edge('input', 'transform', 'value'),
               edge('transform', 'iterate', 'operation', EdgeKind.CALLABLE_INJECTION)])
    with pytest.raises(ValueError, match='not bound closures'):
        asyncio.run(runner.CDGExecutionSession(None, 'synthetic', 'invalid').execute({}, cdg=graph))
    assert not setup_runner


def test_direct_target_executes_the_function(monkeypatch, setup_runner):
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.transform', {'impl': lambda value: value+1})
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.iterate', {'impl': lambda operation: operation(7)})
    graph = CDGExport(nodes=[node('transform', ['value']), node('iterate', ['operation'])],
        edges=[edge('transform', 'iterate', 'operation', EdgeKind.CALLABLE_INJECTION)])
    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic', 'direct').execute(
        {'value': 8}, target_node_id='transform', cdg=graph))
    assert setup_runner['transform', 'out_result'] == 9
    assert len(result['trace']) == 1


def test_loaded_callable_graph_does_not_reuse_stale_value_cache(monkeypatch, setup_runner):
    calls = []
    def operation(value):
        calls.append(value)
        return value+1
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.transform', {'impl': operation})
    monkeypatch.setitem(runner.REGISTRY, 'synthetic.iterate', {'impl': lambda operation: operation(7)})
    graph = CDGExport(nodes=[node('transform', ['value']), node('iterate', ['operation'])],
        edges=[edge('transform', 'iterate', 'operation', EdgeKind.CALLABLE_INJECTION)])
    monkeypatch.setattr(runner, 'load_cdg_from_memgraph', AsyncMock(return_value=(graph.nodes, graph.edges, {})))
    def unexpected_cache(*args):
        raise AssertionError('Callable graph must not reuse a serialized value cache')
    monkeypatch.setattr(runner, 'load_cached_outputs', unexpected_cache)
    session = runner.CDGExecutionSession(None, 'synthetic', 'loaded')
    for _ in range(2):
        assert asyncio.run(session.execute({}))['status'] == 'completed'
    assert calls == [7, 7]
    assert setup_runner['iterate', 'out_result'] == 8
