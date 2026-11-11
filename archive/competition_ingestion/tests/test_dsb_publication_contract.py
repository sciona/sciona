"""Publication must preserve shared inputs, side outputs and callable defaults."""
import pytest
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec
from scripts.promote_dsb_execution import boundary_ports, check_hashes, provider_ports


def fixture():
    def step(value, topk=5):
        return value, 0
    nodes = [AlgorithmicNode(node_id=name, name=name, description='Synthetic publication fixture', concept_type='custom',
        inputs=[IOSpec(name='value', type_desc='object'), IOSpec(name='topk', type_desc='int', required=False)],
        outputs=[IOSpec(name='result', type_desc='object'), IOSpec(name='metrics' if name == 'a' else 'score', type_desc='object')])
        for name in ['a', 'b']]
    graph = CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='a', target_id='b', output_name='result', input_name='value', source_type='object', target_type='object')])
    return graph, {n.node_id: provider_ports(n, step) for n in nodes}, step


def test_shared_default_and_unconsumed_side_output():
    graph, ports, _ = fixture()
    result = boundary_ports(graph, ports)
    assert [p.name for p in result['input']] == ['value', 'topk']
    assert result['input'][1].default_value_repr == '5'
    assert [p.name for p in result['output']] == ['metrics', 'result', 'score']


def test_conflicting_shared_default_rejects():
    graph, ports, _ = fixture()
    ports['b']['input'][1].default_value_repr = '7'
    with pytest.raises(ValueError, match='Ambiguous'):
        boundary_ports(graph, ports)


def test_duplicate_unconsumed_output_rejects():
    graph, ports, _ = fixture()
    ports['b']['output'][1].name = 'metrics'
    with pytest.raises(ValueError, match='Ambiguous'):
        boundary_ports(graph, ports)


def test_callable_requiredness_rejects():
    graph, _, fn = fixture()
    graph.nodes[0].inputs[1].required = True
    with pytest.raises(ValueError, match='default'):
        provider_ports(graph.nodes[0], fn)


def test_changed_or_external_evidence_rejects(tmp_path):
    (tmp_path / 'synthetic.txt').write_text('synthetic')
    with pytest.raises(ValueError, match='drift'):
        check_hashes(tmp_path, {'synthetic.txt': 'incorrect'})
    with pytest.raises(ValueError, match='drift'):
        check_hashes(tmp_path, {'../outside.txt': 'incorrect'})
