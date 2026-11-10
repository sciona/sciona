"""Complete HuBMAP recipe with explicit reusable interfaces and source corrections."""
import json
from pathlib import Path
from sciona.hubmap_execution import build_hubmap_lifecycle_graph
from sciona.architect.models import IOSpec


def build_catalog_hubmap_graph():
    root=Path(__file__).resolve().parents[1]
    contracts=json.loads((root/'docs/reviews/hubmap_provider_contracts.json').read_text())
    scope=json.loads((root/'docs/reviews/hubmap_original_scope.json').read_text())
    if contracts.get('passed') is not True or scope.get('passed') is not True:
        raise ValueError('Passing source and provider qualifications required')
    graph=build_hubmap_lifecycle_graph()
    providers={r['runtime_fqdn']:r for r in contracts['providers']}
    if len(graph.nodes)!=6 or len(graph.edges)!=5 or set(providers)!={n.matched_primitive for n in graph.nodes}:
        raise ValueError('Complete six-node lifecycle and five reviewed providers required')
    for node in graph.nodes:
        record=providers[node.matched_primitive]
        for direction,field in [('input','inputs'),('output','outputs')]:
            ports=sorted((p for p in record['ports'] if p['direction']==direction),key=lambda p:p['ordinal'])
            if [(p.name,p.type_desc) for p in getattr(node,field)]!=[(p['name'],p['type_desc']) for p in ports]:
                raise ValueError('Reviewed callable ports differ')
            setattr(node,field,[IOSpec(**{k:p[k] for k in ('name','type_desc','constraints','required','default_value_repr')}) for p in ports])
        node.description=record['plain_description']
    graph.metadata.update(artifact_source='corrected_original_competition_execution',
        original_source_version_id=scope['source_version_id'],original_source_content_hash=scope['source_content_hash'],
        catalog_output_sources={
            'initial_metrics':dict(node_id='initial_training',output_name='initial_metrics'),
            'retraining_metrics':dict(node_id='retraining',output_name='retraining_metrics'),
            'predictions':dict(node_id='final_prediction',output_name='predictions')},
        source_corrections={r['stage_id']:r['realization']+' '+r['correction'] for r in scope['source_stage_review']},
        scope='Full grouped raster preparation, initial fold training, pseudo-mask generation, fresh retraining and final prediction. Caller justifies foreground meaning, physical scale, independent populations, pseudo-source isolation and explicit checkpoint selection.',
        reuse_decision=scope['reuse_decision'],limits=list(dict.fromkeys(scope['limitations']+contracts['limitations'])),
        qualification_status='Corrected stored original execution and publication remain pending.')
    return graph
