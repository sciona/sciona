"""Automated Tier 3 review of the complete generic resource-route topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='65947cba-17e3-5a60-ac2b-7a3545885a39'
SOURCE_HASH='d07d9ca3d6cf5dc01c2b487384d68cc218797da1504dc97483cbeb69b0f0ee1f'
SCOPE='Generic five-stage agent planning topology; constrained-route search is an explicit permitted realization, not a historical winning solution.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community only; no human Tier 1 certification or Tier 2 usage qualification. Original intake remains draft with mandatory provenance.',
 'Complete five-stage generic topology realized as finite directed resource-constrained route planning. No historical winning solution, arbitrary game, stochastic simulation or continuous control claim.',
 'Costs and resource consumption are nonnegative integers; budgets are finite integer vectors. Closures are explicit. No domain-specific map ingestion or resource estimation.',
 'Dijkstra operates on node plus consumed-resource state. Nonnegative costs justify stopping at the first popped goal; exhaustion proves no feasible route only after the queue empties. Exact resource labels are retained without dominance pruning.',
 'An explicit expansion limit returns search_limit without a feasible-route or infeasibility claim. Budget products can make state spaces large; this does not qualify peak memory or production latency.',
 'Candidate generation filters impossible edges; constraint validation independently checks the selected route against original edges, closures and totals. No invented repaired edges or relaxed constraints.',
 'Synthetic evidence includes independent exhaustive bounded-walk comparisons and actual five-stage serialized runner execution for optimal, infeasible, limited and identity routes. No real-world generalization claim.',
 'Algorithm uses Python standard library; graph schema uses pinned existing project dependencies. Qualified only in the recorded provisioned environment; no clean installation, other-platform or process-resume qualification.',
 'Provider objects are trusted in-process intermediates. Only the input payload and final result are strict JSON boundaries; intermediate persistence/interchange is not qualified.'
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_resource_route_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    assert source['source_stages']==['state_encoder','candidate_generator','search_or_rollout','constraint_repair','plan_selection']
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==12 and tests['exhaustively_compared_synthetic_graphs']==30
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']==dict(state_encoder='encode_state',candidate_generator='generate_candidates',search_or_rollout='search_routes',constraint_repair='validate_route',plan_selection='select_plan')
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']==dict(actual_runner_nodes=5,serialized_edges=4,executed_cases=4,optimal=True,infeasible=True,search_limit=True,identity_route=True,witness_chain=True,strict_json_output=True)
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('resource_route_*.py')}<=set(execution['code_sha256'])
    from sciona.resource_route_graph import build_resource_route_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.resource_route_execution as provider
    graph=build_resource_route_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'resource_route_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_resource_route_'+node.node_id)(witness)
    assert witness=={'kind':'ResourceRoute.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/resource-route-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['graph_schema_dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'pydantic','pydantic-core','typing-extensions','typing-inspection','annotated-types'}
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['architecture']==platform.machine()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    auxiliary=['scripts/review_resource_route_execution.py','requirements/resource-route-execution.txt']
    return dict(format='resource-route-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_resource_route_{s}.json':sha(reviews/f'competition_resource_route_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_resource_route_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
