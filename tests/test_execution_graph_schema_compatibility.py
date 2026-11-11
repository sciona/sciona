"""Synthetic historical envelopes remain hash-bound after model expansion."""
import hashlib
import json

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, DependencyEdge
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph


def historical_envelope():
    graph = CDGExport(nodes=[
        AlgorithmicNode(node_id='a', name='Source', description='Synthetic', concept_type='arithmetic',
                        outputs=[IOSpec(name='x', type_desc='float')]),
        AlgorithmicNode(node_id='b', name='Sink', description='Synthetic', concept_type='arithmetic',
                        inputs=[IOSpec(name='x', type_desc='float')]),
    ], edges=[DependencyEdge(source_id='a', target_id='b', output_name='x', input_name='x',
                             source_type='float', target_type='float')])
    _, nodes, edges = encode_execution_graph(graph)
    payload = json.loads(nodes[0]['type_signature'])
    for node in payload['graph']['nodes']:
        for field in ('phase', 'state_ports', 'semantic_role', 'time_complexity',
                      'space_complexity', 'aliases', 'common_patterns', 'map_hop_size'):
            node.pop(field, None)
        for port in node['inputs'] + node['outputs']:
            port.pop('port_type', None)
    return rebind(payload, nodes), nodes, edges


def rebind(payload, nodes):
    digest = hashlib.sha256(json.dumps(payload['graph'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for index, row in enumerate(nodes):
        envelope = payload if index == 0 else json.loads(row['type_signature'])
        envelope['snapshot_hash'] = digest
        row['type_signature'] = json.dumps(envelope, sort_keys=True, separators=(',', ':'))
    return digest


def test_historical_defaults_do_not_invalidate_authenticated_envelope():
    digest, nodes, edges = historical_envelope()
    original = json.dumps(nodes, sort_keys=True)
    graph = decode_execution_graph(nodes, edges, digest)
    assert graph.nodes[0].map_hop_size == 0
    assert graph.nodes[0].outputs[0].type_desc == 'float'
    assert json.dumps(nodes, sort_keys=True) == original
    assert encode_execution_graph(graph)[0] != digest


@pytest.mark.parametrize('fault', ['tamper', 'unknown', 'coercion', 'projection', 'edge', 'duplicate'])
def test_historical_envelope_corruption_is_rejected(fault):
    digest, nodes, edges = historical_envelope()
    payload = json.loads(nodes[0]['type_signature'])
    if fault == 'tamper':
        payload['graph']['nodes'][0]['name'] = 'Tampered'
        nodes[0]['type_signature'] = json.dumps(payload)
    elif fault in ('unknown', 'coercion'):
        field, value = ('unknown_field', True) if fault == 'unknown' else ('map_hop_size', '0')
        payload['graph']['nodes'][0][field] = value
        digest = rebind(payload, nodes)
    elif fault == 'projection':
        nodes[0]['name'] = 'Tampered'
    elif fault == 'edge':
        edges = []
    else:
        nodes.append(dict(nodes[0]))
    with pytest.raises(ValueError):
        decode_execution_graph(nodes, edges, digest)


def test_current_envelope_roundtrip():
    graph = CDGExport(nodes=[AlgorithmicNode(node_id='a', name='Synthetic', description='Fixture', concept_type='arithmetic')], edges=[])
    digest, nodes, edges = encode_execution_graph(graph)
    assert decode_execution_graph(nodes, edges, digest) == graph
