from copy import deepcopy
import pytest
from scripts.temporal_sparse_synthetic import payload
from sciona.temporal_sparse_selected_execution import prepare, execute
from sciona.temporal_sparse_selected_graph import build_selected_temporal_graph
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('selection', [dict(max_epochs=1, patience=2, min_delta=0.),
    dict(max_epochs=4, patience=2, min_delta=0.), dict(max_epochs=8, patience=2, min_delta=1e6)])
def test_native_selected_lifecycle(selection, tmp_path, monkeypatch):
    config = payload(tmp_path)
    config['selection'] = selection
    original = deepcopy(config)
    expected = execute(prepare(config))
    graph = build_selected_temporal_graph()
    actual = run(graph, {'payload': config}, tmp_path, monkeypatch)
    assert actual['prepare', 'out_result'] is actual['execute', 'in_prepared']
    assert actual['execute', 'out_result'] == expected
    assert config == original
    assert all(p.constraints for node in graph.nodes for p in node.inputs + node.outputs)


@pytest.mark.parametrize('mode,value', [('min', -2.), ('max', 2.)])
def test_generic_stopping_native_state_edge(mode, value, tmp_path, monkeypatch):
    from sciona.architect.handoff import CDGExport
    from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
    nodes = []
    for name, ports in [('initialize', [('patience', 'int'), ('min_delta', 'float'), ('mode', 'str')]),
                        ('observe', [('state', 'object'), ('value', 'float')])]:
        nodes.append(AlgorithmicNode(node_id=name, name=name, description='Generic metric stopping policy',
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.ml.validation_stopping.' + name,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in ports],
            outputs=[IOSpec(name='result', type_desc='object')]))
    graph = CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='initialize', target_id='observe',
        output_name='result', input_name='state', source_type='object', target_type='object')])
    captured = run(graph, dict(patience=2, min_delta=0., mode=mode, value=value), tmp_path, monkeypatch)
    initial = captured['initialize', 'out_result']
    result = captured['observe', 'out_result']
    assert captured['observe', 'in_state'] is initial
    assert initial.evaluations == 0 and initial.best is None
    assert result.evaluations == 1 and result.best == value and result.best_index == 0
