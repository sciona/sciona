"""Full paired-video ensemble with reviewed interfaces and explicit reuse bounds."""
import json
from pathlib import Path
from sciona.dfdc_graph import build_dfdc_graph
from sciona.architect.models import IOSpec


def build_catalog_dfdc_graph():
    root=Path(__file__).resolve().parents[1]
    contracts=json.loads((root/'docs/reviews/dfdc_provider_contracts.json').read_text())
    scope=json.loads((root/'docs/reviews/dfdc_original_scope.json').read_text())
    graph=build_dfdc_graph()
    providers={r['runtime_fqdn']:r for r in contracts['providers']}
    if set(providers)!={n.matched_primitive for n in graph.nodes}:
        raise ValueError('Exact reviewed DFDC providers required')
    for node in graph.nodes:
        record=providers[node.matched_primitive]
        for direction,field in [('input','inputs'),('output','outputs')]:
            ports=sorted([p for p in record['ports'] if p['direction']==direction],key=lambda p:p['ordinal'])
            if [(p.name,p.type_desc) for p in getattr(node,field)]!=[(p['name'],p['type_desc']) for p in ports]:
                raise ValueError('Callable port names/types changed')
            setattr(node,field,[IOSpec(**{k:p[k] for k in ('name','type_desc','constraints','required','default_value_repr')}) for p in ports])
        node.description=record['plain_description']
    graph.metadata.update(artifact_source='corrected_original_competition_execution',
        original_source_version_id=scope['source_version_id'],original_source_content_hash=scope['source_content_hash'],
        catalog_output_sources=dict(result=dict(node_id='execute',output_name='result')),
        source_corrections={r['stage_id']:r['realization']+' '+r['correction'] for r in scope['source_stage_review']},
        scope='Full face-video preparation, B7 training and selected-checkpoint aggregation; caller-justified paired targets, population independence, landmark conventions and explicit executable checkpoint plan.',reuse_decision=scope['reuse_decision'],limits=scope['limitations'],
        qualification_status='Original corrected graph execution and publication remain pending.')
    return graph
