"""Full text-pair lifecycle with approved catalog interfaces and recipe policy."""
import json
from pathlib import Path

from sciona.architect.models import IOSpec
from sciona.text_pair_complete_graph import build_complete_text_pair_graph


def build_catalog_text_pair_graph():
    path = Path(__file__).resolve().parents[1]/'docs/reviews/text_pair_approved_provider_state.json'
    frozen = json.loads(path.read_text())
    records = {r['runtime_fqdn']: r for r in frozen['records'] if r['recipe_adapter']}
    graph = build_complete_text_pair_graph()
    if set(records) != {n.matched_primitive for n in graph.nodes}:
        raise ValueError('Exact six approved recipe bindings required')
    for node in graph.nodes:
        record = records[node.matched_primitive]
        for direction, field in [('input', 'inputs'), ('output', 'outputs')]:
            ports = sorted([p for p in record['ports'] if p['direction'] == direction], key=lambda p: p['ordinal'])
            if [p['ordinal'] for p in ports] != list(range(len(ports))):
                raise ValueError('Contiguous ordered ports required')
            previous = getattr(node, field)
            if [(p.name, p.type_desc) for p in previous] != [(p['name'], p['type_desc']) for p in ports]:
                raise ValueError('Qualified runtime port names/types changed')
            setattr(node, field, [IOSpec(**{k: p[k] for k in
                ('name', 'type_desc', 'required', 'default_value_repr', 'constraints')}) for p in ports])
    graph.metadata.update(artifact_source='text_pair_complete_catalog_contracts',
        catalog_output_sources=dict(result=dict(node_id='format_rows', output_name='result')),
        scope='Separate title/description binary relevance using six approved recipe adapters and eleven registered numerical/text dependencies.',
        qualification_status='Original CDG intake, stored execution and publication remain pending.')
    return graph
