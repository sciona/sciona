"""Bind every original planning stage to the explicit reusable route realization."""
import hashlib
import json

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.plan_flavours_provider_intake import sha
from scripts.review_text_pair_original_scope import anchor

ORIGINAL = '65947cba-17e3-5a60-ac2b-7a3545885a39'
ORIGINAL_HASH = 'd07d9ca3d6cf5dc01c2b487384d68cc218797da1504dc97483cbeb69b0f0ee1f'
IDENTITY = 'da422553-55e1-580c-b5a0-0e85f5b8ae12'
PARENT = '3c10fde8-db29-5c97-bdbe-d173e7468201'
PARENT_HASH = '641f783f01a3e5b5247a0e6fd86969663f48ad0b9c2fae2f037395d84fba191e'
STAGES = {
    'state_encoder': dict(symbol='encode_state',
        realization='Encode a finite directed graph with explicit node and edge identities, start and goal, nonnegative integer costs, nonnegative integer resource vectors, finite budgets, closures and an expansion limit. Detach the input into immutable tuples and frozen state objects.',
        limits='Caller supplies graph construction, cost meaning, resource units and closure policy. No map, game, scheduling or robot input adapter is implied. Intermediate objects are trusted in-process values.'),
    'candidate_generator': dict(symbol='generate_candidates',
        realization='Retain every edge not excluded by an explicit closure or by individual resource consumption exceeding a budget. Nonnegative resource increments make this pruning sound for the chosen additive model.',
        limits='Candidates are directed transitions, not whole routes. Feasibility is checked cumulatively during search. No learned policy proposal, spatial control feature or domain heuristic is required or claimed.'),
    'search_or_rollout': dict(symbol='search_routes',
        realization='Run exact shortest-path search on node plus consumed-resource labels with nonnegative integer costs. Retain distinct resource states and strictly improving costs. Return the first popped goal, queue-exhaustion infeasibility, or explicit search_limit. Zero-cost zero-resource cycles terminate through strict label improvement.',
        limits='This selects deterministic finite resource-constrained search. No stochastic simulator, MCTS, PPO, self-play, continuous dynamics, path merging or parallel C++ execution is claimed. State-space size grows with the product of budgets; expansion limits do not bound peak queue memory.'),
    'constraint_repair': dict(symbol='validate_route',
        realization='Select the source-permitted constraint-checking branch. Independently check returned edge identities, continuity, closures, goal, accumulated costs, resources and budgets against the original encoded graph. Reject invalid paths without inventing repairs or relaxing constraints.',
        limits='Validation establishes feasibility and reported totals of an optimal-status route, not independent optimality. Search establishes optimality under the declared cost model. Non-success statuses carry no selected route; forged intermediate objects are outside the input boundary.'),
    'plan_selection': dict(symbol='select_plan',
        realization='Return strict JSON status, ordered edge identities, total cost, resource totals and expansion count. Successful routes include the zero-edge start-equals-goal plan. Infeasible and limited searches return empty edge lists and null costs/resources.',
        limits='Generic route result only; a named competition submission format requires a separate adapter. Equal-cost ties follow input transition order and queue insertion order, with no unique-route or secondary-resource optimum guarantee. Search-limit status makes no feasibility or infeasibility claim.'),
}


def review():
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'agent_simulation_search_planning_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
                    bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest,_id(fqdn+'/'+digest),_id(fqdn)) != (ORIGINAL_HASH,ORIGINAL,IDENTITY):
        raise ValueError('Exact original source reconstruction required')
    stages = snapshot['template']['stages']
    if len(stages)!=len(STAGES) or {s['stage_id'] for s in stages}!=set(STAGES):
        raise ValueError('Every source stage must be reviewed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original = db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(ORIGINAL,)).fetchone()
        if original != dict(artifact_id=IDENTITY,fqdn=fqdn,status='draft',is_publishable=False,content_hash=ORIGINAL_HASH,is_latest=True):
            raise ValueError('Original candidate state changed')
        stored = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(stored)!=1 or stored[0]['details']['snapshot']!=snapshot:
            raise ValueError('Stored original snapshot differs')
        parent = db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(PARENT,)).fetchone()
        if parent != dict(artifact_id='fe968b7f-e0eb-5492-90f6-100b914b3442',fqdn=fqdn+'.execution',status='approved',is_publishable=True,content_hash=PARENT_HASH,is_latest=True,trust_tier=3):
            raise ValueError('Retained derivative state changed')
    return dict(format='resource-route-original-scope.v1',passed=True,approved=False,catalog_mutations=0,
        catalog_state_checked=True,source_snapshot_reconstructed=True,source_version_id=ORIGINAL,
        source_content_hash=ORIGINAL_HASH,artifact_id=IDENTITY,fqdn=fqdn,parent_version_id=PARENT,
        parent_content_hash=PARENT_HASH,reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'],limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor('resource_route_planning',STAGES[s['stage_id']]['symbol'])]) for s in stages],
        reuse_decision='Five generic route stages share explicit immutable state. Graph labels, costs, resource meanings and closures are caller policy; no competition-specific units, schema adapter or embedded task data. Reuse requires the same finite nonnegative additive model.',
        remaining_requirements=['Qualify current runtime and independent numerical/reuse tests.',
            'Verify exact approved provider contracts and dependencies in both catalogs.',
            'Bind and execute the corrected stored original; test publication rollback and served selection.'],
        limits=['Source-stage review and current catalog identity only; this report does not confer publication approval.',
            'Selected deterministic route realization covers all five stages; optional search families remain alternatives.',
            'Automated Tier 3 candidate only. No human Tier 1 review or empirical deployment qualification.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/resource_route_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,stages=len(report['source_stage_review']),approved=False)))
