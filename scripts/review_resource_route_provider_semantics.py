"""Explicit reusable contracts for the five existing route-stage providers."""
import inspect
import json

from scripts.validate_resource_route_current_runtime import ROOT, sha, source_inventory, runtime_review
from sciona.resource_route_graph import build_resource_route_graph
import sciona.atoms.ml.resource_route_execution as adapters

STATE = ('Trusted immutable State produced by encode_state: unique nonempty node/edge identities, known endpoints, '
    'nonnegative integer costs and resource vectors aligned to nonempty finite budgets, explicit closures and positive expansion limit. '
    'Caller supplies cost, resource and closure meaning; no intrinsic physical units or domain schema.')
CANDIDATES = ('Trusted Candidates produced by generate_candidates, retaining original State and all directed edges not blocked '
    'and individually within budgets. Cumulative resource feasibility is determined during search; candidates are transitions, not complete routes.')
SEARCHED = ('Trusted SearchResult produced by search_routes. Status optimal, infeasible or search_limit; original Candidates retained. '
    'Optimal contains ordered edge identities, integer cost, consumed-resource tuple and expansion count. '
    'Infeasible means queue exhaustion or blocked boundary. Search_limit makes no feasibility claim. '
    'Intermediate non-success cost/resource placeholders must not be interpreted as a feasible zero-cost route.')
VALIDATED = (SEARCHED+' Optimal paths have independently checked continuity, closures, goal, totals and budgets against the original State. '
    'This checks feasibility, not independent optimality; optimality relies on nonnegative-cost expanded-state search.')
PAYLOAD = ('Exact dict keys nodes, edges, start, goal, budgets, blocked_nodes, blocked_edges, max_expansions. '
    'Nodes and closure lists contain unique nonempty strings; endpoints refer to known nodes. '
    'Edges are dicts with exactly id, source, target, cost, resources; edge IDs unique, cost and resource entries nonnegative Python integers excluding bool. '
    'Budgets are a nonempty nonnegative integer list, aligned with every resource vector; max_expansions is a positive integer. '
    'Closures reference existing identities. Cost and resources are additive. Input is detached into immutable state.')
RESULT = ('Strict JSON dict with status, edge_ids, cost, resources, expansions. Optimal returns ordered edges and exact integer totals; '
    'start equals goal permits an empty route unless blocked. Other statuses return empty edge_ids and null cost/resources. '
    'Equal-cost ties follow transition/insertion order; no secondary resource optimum. No named competition submission format.')
CONTRACTS = {
    'encode': (PAYLOAD, STATE, 'Turn an explicit graph, budgets and closures into a checked immutable planning state.'),
    'generate': (STATE, CANDIDATES, 'Keep graph transitions allowed by closures and individual resource budgets.'),
    'search': (CANDIDATES, SEARCHED, 'Find a least-cost feasible route, prove none exists by exhausting the queue, or report that the search limit was reached.'),
    'validate': (SEARCHED, VALIDATED, 'Check the selected route against original edges, closures, resource limits and reported totals.'),
    'select': (VALIDATED, RESULT, 'Return the route and totals, or an explicit non-success status without a proposed route.'),
}
LIMITS = [
    'Finite directed graph with nonnegative integer additive costs and nonnegative integer resource consumption only.',
    'Budget-state products may be large. Expansion limits do not bound peak queue memory or wall-clock latency.',
    'Trusted in-process stage objects only. Arbitrary manually forged states and portable intermediate interchange are not qualified.',
    'Node labels, costs, units and closures are caller policy. No automatic domain ingestion or arbitrary simulation.',
    'Synthetic automated Tier 3 evidence; no historical competition accuracy, human Tier 1 review or production scale claim.',
]


def review():
    path = ROOT/'docs/reviews/resource_route_current_runtime.json'
    runtime = json.loads(path.read_text())
    if runtime['passed'] is not True or runtime['tests'] != dict(tests=21,failures=0,errors=0,skipped=0):
        raise ValueError('Complete current runtime qualification required')
    if runtime['providers'] != source_inventory() or runtime['runtime'] != runtime_review():
        raise ValueError('Provider sources or provisioned environment changed')
    for name,digest in runtime['software_sha256'].items():
        if sha(ROOT/name) != digest:raise ValueError('Executed software changed')
    records = {p['runtime_fqdn']:p for p in runtime['providers']}
    reviews = []
    graph = build_resource_route_graph()
    for node in graph.nodes:
        fn = getattr(adapters,'resource_route_'+node.node_id)
        signature = inspect.signature(fn)
        if list(signature.parameters) != [p.name for p in node.inputs]:
            raise ValueError('Runtime input names differ')
        if any(p.default is not p.empty for p in signature.parameters.values()):
            raise ValueError('Unexpected optional provider argument')
        input_contract,output_contract,plain = CONTRACTS[node.node_id]
        ports = [dict(direction=d,name=p.name,ordinal=i,type_desc=p.type_desc,required=p.required,
                      default_value_repr=p.default_value_repr,constraints=contract)
            for d,values,contract in [('input',node.inputs,input_contract),('output',node.outputs,output_contract)]
            for i,p in enumerate(values)]
        reviews.append(dict(**records[node.matched_primitive],node_id=node.node_id,
            plain_description=plain,ports=ports,dependencies=[],validity_bounds=LIMITS))
    return dict(format='resource-route-provider-semantics.v1',passed=True,approved=False,
        proposed_tier=3,review_source='automated',catalog_mutations=0,catalog_state_checked=False,
        runtime_sha256=sha(path),reviewer_sha256=sha(__file__),providers=reviews,
        reuse_decision='All five stages are generic constrained-route operations with explicit caller policy; no task-specific data or units.',
        remaining_requirements=['Reconcile richer contracts with exact versions in both catalogs without rewriting retained qualification history.',
            'Verify rollback and native publication eligibility for metadata changes.',
            'Execute and publish the corrected original under the qualified contracts.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/resource_route_provider_semantics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(report['providers']),approved=False)))
