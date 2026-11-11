"""Synthetic provider and serialized graph boundaries, without model training."""
import inspect

import pytest

from sciona.atoms.ml import cassava_execution as provider
from scripts.build_cassava_execution_graph import build_graph
from sciona.services.execution_graph_codec import encode_execution_graph


def test_graph_ports_and_symbolic_composition():
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    assert len(digest) == 64 and len(nodes) == 2 and len(edges) == 1
    values = {}
    for node in graph.nodes:
        function = getattr(provider, 'cassava_' + node.node_id)
        witness = getattr(provider, 'witness_cassava_' + node.node_id)
        assert list(inspect.signature(function).parameters) == [p.name for p in node.inputs]
        assert inspect.signature(function).parameters == inspect.signature(witness).parameters
        arguments = {p.name: values.get(p.name, object()) for p in node.inputs}
        values[node.outputs[0].name] = witness(**arguments)
    assert values['result'] == {'kind': 'Cassava.ExecutionResult'}


def test_provider_preserves_pipeline_arguments_and_outputs(monkeypatch):
    import sciona.cassava_pipeline as pipeline
    plan = provider.cassava_fold_plan([f'synthetic-{i}' for i in range(25)],
                                    [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    arguments = dict(source_rows=[], efficientnet_rows=[], query_rows=[], references={},
                     torch_batch_size=1, torch_workers=0, efficientnet_batch_size=1,
                     output_directory='synthetic-output-unused')
    expected = tuple(object() for _ in range(5))
    def execute(actual_plan, **kwargs):
        assert actual_plan is plan and kwargs == arguments
        return expected
    monkeypatch.setattr(pipeline, 'execute', execute)
    result = provider.cassava_train_ensemble(plan, **arguments)
    assert tuple(result.values()) == expected
    with pytest.raises(ValueError):
        provider.cassava_fold_plan(['synthetic-duplicate'] * 25, [0] * 25, [0] * 25)
    with pytest.raises(ValueError):
        provider.witness_cassava_train_ensemble({}, **arguments)


@pytest.mark.parametrize('fault', ['missing_constraints', 'port_order', 'optional_control', 'wrong_provider'])
def test_structural_audit_rejects_invalid_graph(monkeypatch, fault):
    from scripts import audit_cassava_provider_contract as review
    graph = build_graph()
    node = graph.nodes[1]
    if fault == 'missing_constraints':
        node.inputs[1].constraints = ''
    elif fault == 'port_order':
        node.inputs[1], node.inputs[2] = node.inputs[2], node.inputs[1]
    elif fault == 'optional_control':
        node.inputs[-2].required = False
    else:
        node.matched_primitive = 'sciona.atoms.ml.cassava_execution.unregistered'
    monkeypatch.setattr(review, 'build_graph', lambda: graph)
    with pytest.raises(ValueError):
        review.audit()


def test_actual_runner_routes_plan_and_controls(monkeypatch):
    from scripts.cassava_graph_execution import execute_through_graph
    import sciona.cassava_pipeline as pipeline
    plan = provider.cassava_fold_plan([f'synthetic-{i}' for i in range(25)],
                                    [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    arguments = dict(source_rows=[], efficientnet_rows=[], query_rows=[], references={},
                     torch_batch_size=1, torch_workers=0, efficientnet_batch_size=1,
                     output_directory='synthetic-output-unused')
    expected = tuple({'synthetic_result': i} for i in range(5))
    calls = []
    def execute(actual_plan, **kwargs):
        assert actual_plan == plan and kwargs == arguments
        calls.append(True)
        return expected
    monkeypatch.setattr(pipeline, 'execute', execute)
    assert execute_through_graph(plan, **arguments) == expected
    assert calls == [True]
