"""Sharded video graph using frozen catalog interfaces without runtime data."""
import json
from pathlib import Path
from sciona.architect.models import IOSpec
from sciona.decomposed_video_multilabel_graph import build_decomposed_video_multilabel_graph


def build_catalog_video_multilabel_graph():
    frozen=json.loads((Path(__file__).resolve().parents[1]/'docs/reviews/video_graph_contracts.json').read_text())
    records={r['node_id']:r for r in frozen['records']}
    graph=build_decomposed_video_multilabel_graph()
    if set(records)!={n.node_id for n in graph.nodes}:raise ValueError('Exact graph bindings required')
    for node in graph.nodes:
        record=records[node.node_id]
        if node.matched_primitive!=record['runtime_fqdn']:raise ValueError('Runtime binding differs')
        for direction,field in [('input','inputs'),('output','outputs')]:
            ports=sorted([p for p in record['ports'] if p['direction']==direction],key=lambda p:p['ordinal'])
            if [p['ordinal'] for p in ports]!=list(range(len(ports))):raise ValueError('Contiguous ordered ports required')
            setattr(node,field,[IOSpec(**{k:p[k] for k in ('name','type_desc','required','default_value_repr','constraints')}) for p in ports])
    nodes={n.node_id:n for n in graph.nodes}
    for edge in graph.edges:
        # All source nodes are callable providers; their first output identifies
        # the callable edge, while the runtime passes the function itself.
        edge.output_name=nodes[edge.source_id].outputs[0].name
    graph.metadata.update(artifact_source='video_multilabel_catalog_contracts',
        scope='Full configured sharded training with nine explicit numerical providers and exact frozen catalog interfaces.',
        limits=['Recipe and original CDG publication are pending.',
                'Caller supplies private in-process shard stores; no data or paths are embedded.'])
    return graph
