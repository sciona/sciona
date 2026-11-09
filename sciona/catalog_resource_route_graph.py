"""Explicit usage contracts for the existing approved generic route stages."""
from sciona.resource_route_graph import build_resource_route_graph
from scripts.complete_resource_route_provider_metadata import qualified_plan


def build_catalog_resource_route_graph():
    plan = qualified_plan()
    records = {r['runtime_fqdn']:r for r in plan['records']}
    graph = build_resource_route_graph()
    for node in graph.nodes:
        record = records[node.matched_primitive]
        for direction,ports in [('input',node.inputs),('output',node.outputs)]:
            expected = [p for p in record['ports'] if p['direction']==direction]
            if len(ports)!=len(expected):raise ValueError('Route interface arity differs')
            for port,reviewed in zip(ports,expected,strict=True):
                for field in ('name','type_desc','required','default_value_repr'):
                    if getattr(port,field)!=reviewed[field]:raise ValueError('Exact retained provider signature differs')
                port.constraints = reviewed['constraints']
        node.description = record['plain_description']
    graph.metadata.update(
        artifact_source='corrected_original_competition_execution',
        publication_status='candidate_pending_stored_execution',
        catalog_output_sources={'result':{'node_id':'select','output_name':'result'}},
        provider_contract_sha256=plan['qualification_sha256'],
        provider_contract_policy=plan['contract_policy'],
        reuse_decision='Costs, resource dimensions, units, graph identities and closures are caller policy; all five stages are generic.',
        qualification_limits=records[graph.nodes[0].matched_primitive]['validity_bounds'])
    return graph
