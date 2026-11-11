"""Lossless, hash-bound execution graphs in canonical CDG projections."""
import hashlib
import json
from sciona.architect.handoff import CDGExport

FORMAT='sciona.execution_graph.v1'
REFERENCE_FORMAT='sciona.execution_graph_ref.v1'


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def encode_execution_graph(graph:CDGExport):
    return _encode_snapshot(graph, graph.model_dump(mode='json'))


def _encode_snapshot(graph, snapshot):
    if not graph.nodes or len({n.node_id for n in graph.nodes})!=len(graph.nodes):
        raise ValueError('execution graph needs unique nonempty nodes')
    digest=hashlib.sha256(_json(snapshot).encode()).hexdigest()
    rows=[]
    for index,node in enumerate(graph.nodes):
        payload={'format':FORMAT if index==0 else REFERENCE_FORMAT,'snapshot_hash':digest}
        if index==0:payload['graph']=snapshot
        rows.append({'node_id':node.node_id,'parent_node_id':node.parent_id or '',
                     'name':node.name,'description':node.description,'concept_type':node.concept_type.value,
                     'status':node.status.value,'matched_primitive':node.matched_primitive or None,
                     'type_signature':_json(payload)})
    edges=[{key:getattr(edge,key) for key in ['source_id','target_id','output_name','input_name']} for edge in graph.edges]
    if len({_json(e) for e in edges})!=len(edges):raise ValueError('duplicate projected edges')
    return digest,rows,edges


def decode_execution_graph(nodes,edges,expected_hash):
    """Return None for legacy rows; reject partial or corrupted execution envelopes."""
    payloads=[]
    for row in nodes:
        try:payload=json.loads(row.get('type_signature') or '')
        except (ValueError,TypeError):payload=None
        payloads.append(payload if isinstance(payload,dict) else {})
    if not any(p.get('format') in {FORMAT,REFERENCE_FORMAT} for p in payloads):return None
    if not expected_hash or any(p.get('format') not in {FORMAT,REFERENCE_FORMAT} for p in payloads):
        raise ValueError('execution graph requires a version hash and complete envelopes')
    snapshots=[p for p in payloads if p.get('format')==FORMAT]
    if len(snapshots)!=1 or any(p.get('snapshot_hash')!=expected_hash for p in payloads):
        raise ValueError('execution snapshot identity mismatch')
    snapshot=snapshots[0]['graph']
    # Authenticate the stored representation before adding defaults introduced
    # by newer model schemas. Historical version identities must stay stable.
    if hashlib.sha256(_json(snapshot).encode()).hexdigest()!=expected_hash:
        raise ValueError('execution snapshot content hash mismatch')
    graph=CDGExport.model_validate(snapshot)
    # Reject ignored fields and coerced values. Only omitted model defaults may
    # be added in memory; explicit historical data must survive validation.
    if _json(graph.model_dump(mode='json',exclude_unset=True))!=_json(snapshot):
        raise ValueError('execution snapshot fields changed during validation')
    digest,expected_nodes,expected_edges=_encode_snapshot(graph,snapshot)
    if digest!=expected_hash:raise ValueError('execution snapshot content hash mismatch')
    actual_nodes=[{key:row.get(key) for key in expected_nodes[0]} for row in nodes]
    actual_edges=[{key:row.get(key) for key in ['source_id','target_id','output_name','input_name']} for row in edges]
    if sorted(actual_nodes,key=lambda r:r['node_id'])!=sorted(expected_nodes,key=lambda r:r['node_id']) or sorted(actual_edges,key=_json)!=sorted(expected_edges,key=_json):
        raise ValueError('execution graph projection drift')
    return graph
