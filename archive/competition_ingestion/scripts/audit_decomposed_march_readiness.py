"""Check immutable March scope and existing reuse contracts without promotion."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import networkx as nx
import numpy as np
from sciona.atoms.algorithmic.graph.utilities.atoms import pagerank
from sciona.atoms.scoring.atoms import elo_rating_update
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.competition_graph import load_competition_graph
from sciona.march_features import season_features
from scripts.march_synthetic import game,CONTROLS
from scripts.import_flavours_provider_drafts import ROOT,connect

ORIGINAL='26d177bb-7d6d-5ee1-b881-2bafe69064cd'
ORIGINAL_HASH='a32fed30707dd363c95f39462a6bd6db505102f4449f3dccf24fb62d71af34ee'
IDENTITY='ff29169b-d168-57eb-8c5c-a4c939e13373'
PARENT='7ed0edcf-3ce4-5f07-9a9a-4e0af736e61c'
PARENT_HASH='6e821a2dc2ca4abdd54bdb29046e2a0953cac7ebbfab837f284e90f81be19619'
STAGES=['season_aggregation','graph_centralities','elo_ratings','pairwise_concatenation','logistic_regression']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_probes():
    adjacency=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    generic=pagerank(adjacency.T,damping=.85,max_iter=1,tol=1e-13)
    try:nx.pagerank(nx.from_numpy_array(adjacency,create_using=nx.DiGraph),alpha=.85,max_iter=1,tol=1e-13)
    except nx.PowerIterationFailedConvergence:pass
    else:raise ValueError('Expected reference convergence failure')
    if not np.isfinite(generic).all():raise ValueError('Generic PageRank probe changed')
    reference=nx.pagerank(nx.from_numpy_array(adjacency,create_using=nx.DiGraph),alpha=.85,max_iter=1000,tol=1e-13)
    transposed=pagerank(adjacency.T,damping=.85,max_iter=1000,tol=3e-13)
    if not np.allclose(transposed,[reference[i] for i in range(3)],rtol=0,atol=1e-12):
        raise ValueError('Transposed adjacency PageRank comparison failed')
    if np.allclose(pagerank(adjacency,damping=.85,max_iter=1000,tol=3e-13),transposed,rtol=0,atol=1e-12):
        raise ValueError('Directed orientation probe must distinguish transpose')
    try:elo_rating_update(0.,1_000_000.,1.,k=32.)
    except OverflowError:pass
    else:raise ValueError('Existing Elo overflow behavior changed; reassess compatibility')
    if 'scale' in inspect.signature(elo_rating_update).parameters:
        raise ValueError('Existing Elo now accepts scale; reassess reuse')
    games=[game(0,0,1,80,70),game(1,0,1,70,80)]
    stable=season_features(games,**dict(CONTROLS,elo_initial=0.,elo_k=1_000_000.))
    if not all(np.isfinite(v).all() for v in stable[1]['teams'].values()):
        raise ValueError('Reference extreme-difference rating behavior changed')
    a=season_features(games,**dict(CONTROLS,elo_scale=200.))
    b=season_features(games,**dict(CONTROLS,elo_scale=400.))
    if a[1]['teams'][0][4]==b[1]['teams'][0][4]:
        raise ValueError('Caller rating scale must affect the actual recipe')
    rng=np.random.default_rng(712634)
    cases=0
    for size in (1,2,17,129):
        for kind in ('random','zeros','ones','alternating'):
            forward=rng.random(size);reverse=rng.random(size)
            if kind=='zeros':forward[:]=reverse[:]=0
            elif kind=='ones':forward[:]=reverse[:]=1
            elif kind=='alternating':forward[:]=np.arange(size)%2;reverse[:]=1-forward
            before=forward.copy(),reverse.copy()
            actual=.5*ordered_weighted_sum(np.stack([forward,np.ones(size),reverse]),np.array([1.,1.,-1.]))
            expected=.5*(forward+1-reverse)
            if actual.tobytes()!=expected.tobytes() or any(a.tobytes()!=b.tobytes() for a,b in zip(before,(forward,reverse))):
                raise ValueError('Generic ordered sum changes orientation projection')
            cases+=1
    return dict(synthetic_only=True,pagerank_exhaustion_contract_differs=True,
        pagerank_requires_adjacency_transpose=True,pagerank_tolerance_conventions_differ=True,
        transposed_pagerank_agrees_within_1e_12=True,
        elo_fixed_scale=400,elo_extreme_difference_overflows=True,
        reference_extreme_difference_finite=True,reference_custom_scale_affects_ratings=True,
        ordered_sum_cases=cases,orientation_projection_byte_equal=True,inputs_unchanged=True,
        scope='Reduction parity only. Existing PageRank and Elo are not drop-in replacements for the current feature recipe; no new implementation is approved.')


def audit():
    old=json.loads((ROOT/'docs/reviews/competition_march_graph_execution.json').read_text())
    changed=[p for p,h in old['code_sha256'].items() if sha(ROOT/p)!=h]
    if changed:raise ValueError('Historical reference code changed')
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/march_execution.py'
    if sha(provider)!=old['provider_sha256']:raise ValueError('Historical provider changed')
    runtime={n:dict(recorded=v,installed=metadata.version(n)) for n,v in
        (line.split('==') for line in (ROOT/'requirements/march-execution.txt').read_text().splitlines())}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original=db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(ORIGINAL,)).fetchone()
        if original!=dict(artifact_id=IDENTITY,fqdn='cdg.competition.solution.kaggle.march_mania_benchmark',status='draft',is_publishable=False,content_hash=ORIGINAL_HASH,is_latest=True):
            raise ValueError('Original identity or publication state differs')
        parent=db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(PARENT,)).fetchone()
        if parent!=dict(artifact_id='ef505a91-b2cf-5fd5-8a03-972135bf9c05',
                fqdn='cdg.competition.solution.kaggle.march_mania_benchmark.execution',status='approved',
                is_publishable=True,content_hash=PARENT_HASH,is_latest=True,trust_tier=3):
            raise ValueError('Approved reference identity differs')
        graph=load_competition_graph(db,version_id=ORIGINAL)
        if {n.node_id for n in graph.nodes}!=set(STAGES):raise ValueError('Original five-stage scope differs')
        dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(PARENT,)).fetchall()
        if not any(d==dict(dependency_artifact_fqdn=original['fqdn'],dependency_content_hash=ORIGINAL_HASH,dependency_role='cdg',optional=False) for d in dependencies):
            raise ValueError('Mandatory source lineage missing')
        candidates=db.execute('''SELECT a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,v.version_id::text,v.content_hash
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE v.is_latest AND
            a.source_symbol=ANY(%s) ORDER BY a.fqdn''',(['pagerank','elo_rating_update','ordered_weighted_sum','march_prepare','march_execute'],)).fetchall()
    return dict(format='decomposed-march-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,reference=parent,source_stages=STAGES,derivative_dependencies=dependencies,
        source_scope='Generic benchmark topology with no identified historical winning implementation. Logistic regression is an explicit allowed branch; PageRank is a graph feature, not certified true strength of schedule.',
        prior_execution_changed_files=changed,prior_provider_hash_matches=True,runtime_versions=runtime,
        reuse_candidates=candidates,reuse_probes=reuse_probes(),
        reference_code_sha256=old['code_sha256'],prior_provider_sha256=sha(provider),
        inspected_provider_sha256={name:sha(inspect.getfile(inspect.unwrap(fn))) for name,fn in
            [('pagerank',pagerank),('elo_rating_update',elo_rating_update),('ordered_weighted_sum',ordered_weighted_sum)]},
        auditor_sha256=sha(__file__),next_steps=[
            'Re-execute the unchanged full baseline on the current compatible runtime without overwriting historical evidence or changing the shared environment.',
            'Preserve aggregation, directed rank, ordered ratings, paired features, training-only scaling, logistic fitting, frozen-base calibration and symmetric prediction.',
            'Expose reusable operations with explicit contracts. Retain caller scale, stable Elo arithmetic and PageRank failure semantics; do not silently substitute existing incompatible atoms or add duplicate implementations.',
            'Use the existing generic ordered sum for orientation projection if full extracted execution retains reference behavior.',
            'Qualify new provider interfaces, source semantics, stored execution and Tier 3 publication before selecting the original.'],
        limits=['No source, provider or original approval follows from readiness probes.',
            'Runtime metadata and unchanged source hashes do not establish installed-file integrity or native execution.',
            'Box-score feature construction remains domain-specific; matrix fitting and generic numeric reduction admit broader reuse within documented contracts.'])


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_march_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original_status=result['original']['status'],source_stages=len(result['source_stages']),reuse_probes=result['reuse_probes'],runtime_versions=result['runtime_versions'])))
