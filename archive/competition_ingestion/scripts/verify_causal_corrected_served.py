"""Execute the original CDG selected through production catalog conversion."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from scripts import stage_causal_corrected_intake as intake
from scripts import promote_causal_corrected_intake as publisher
from scripts import validate_causal_full_decomposition as oracle


def selection(plan,providers):
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.check_candidate(db,plan,providers,approved=True)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',(plan['artifact_id'],)).fetchall()
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(plan['artifact_id'],)).fetchall()
        if len(rows)!=1 or len(versions)!=1 or str(versions[0]['version_id'])!=plan['version_id']:
            raise ValueError('Served original or latest revision selection differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(plan['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        if candidate.cdg!=intake.build_graph() or candidate.content_hash!=plan['graph_sha256'] or candidate.trust_tier!=3:
            raise ValueError('Production candidate differs')
        if candidate.description!=publisher.DESCRIPTION or candidate.conceptual_summary!=publisher.PLAIN or candidate.verified_leaf_coverage!=1:
            raise ValueError('Served descriptions or coverage differ')
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(plan['artifact_id'],)).fetchone()['eligible']:
            raise ValueError('Native eligibility lost')
    return candidate.cdg


async def verify(reference_dir):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    plan,providers,qualification=publisher.review()
    graph=selection(plan,providers)
    previous=oracle.build_explicit_causal_feature_graph
    oracle.build_explicit_causal_feature_graph=lambda:graph.model_copy(deep=True)
    try:result=await oracle.validate(reference_dir)
    finally:oracle.build_explicit_causal_feature_graph=previous
    retained=json.loads((intake.ROOT/'docs/reviews/causal_corrected_intake_execution.json').read_text())
    for field in ('graph_hash','nodes','edges','cases','source_commit','source_hashes','implementation_hashes','provider_hashes','library_versions','resource_policy'):
        if result[field]!=retained[field]:raise ValueError('Served execution differs: '+field)
    if publisher.review()[2]!=qualification or selection(plan,providers)!=graph:
        raise ValueError('Publication changed during execution')
    result.update(approved=True,trust_tier=3,catalog_mutations=0,
        original_artifact_id=plan['artifact_id'],selected_version_id=plan['version_id'],
        production_candidate_conversion=True,selected_candidate_executed=True,
        original_and_parent_histories_preserved=True,native_eligibility=True,
        verifier_sha256=intake.sha(__file__),plan_sha256=intake.sha(intake.PLAN),
        retrieval_source_sha256=intake.sha(intake.ROOT/'sciona/services/catalog_artifact_retrieval.py'),
        limitations=qualification['limitations'],
        selection_scope='Served SQL view, latest-version predicate, document RPC and production converter. HTTP transport and semantic-search ranking are not covered.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(verify(args.reference_dir))
    (intake.ROOT/'docs/reviews/causal_corrected_served_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','approved','selected_version_id','cases')}),flush=True)
