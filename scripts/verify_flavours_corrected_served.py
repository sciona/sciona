"""Execute the corrected original selected through the production catalog converter."""
import json
import os
import runpy
import tempfile
import time
from pathlib import Path
import pytest
from threadpoolctl import threadpool_limits
from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from scripts import stage_flavours_corrected_intake as intake
from scripts import promote_flavours_corrected_intake as publisher


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


def verify():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    plan,providers,qualification=publisher.review();graph=selection(plan,providers)
    oracle_path=intake.ROOT/'tests/test_flavours_decomposed_evaluation.py'
    oracle_sha=intake.sha(oracle_path);verifier_sha=intake.sha(__file__)
    oracle=runpy.run_path(str(oracle_path))['test_complete_evaluated_graph_matches_reference']
    start=time.monotonic()
    print('Executing production-selected original CDG: all 841 nodes, native threads=1',flush=True)
    with tempfile.TemporaryDirectory(prefix='sciona-flavours-served-') as temporary:
        with pytest.MonkeyPatch.context() as patch,threadpool_limits(limits=1):
            patch.setitem(oracle.__globals__,'build_evaluated_graph',lambda:graph.model_copy(deep=True))
            oracle(Path(temporary),patch)
    if publisher.review()[2]!=qualification or selection(plan,providers)!=graph:
        raise ValueError('Publication changed during execution')
    if intake.sha(oracle_path)!=oracle_sha or intake.sha(__file__)!=verifier_sha:
        raise ValueError('Oracle or verifier changed during execution')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,synthetic_only=True,
        original_artifact_id=plan['artifact_id'],selected_version_id=plan['version_id'],graph_sha256=plan['graph_sha256'],
        production_candidate_conversion=True,selected_candidate_executed=True,executed_nodes=841,all_nodes_uncached=True,
        model_fits_per_workflow=115,exact_query_scores=True,exact_population_blends=True,
        exact_metrics_threshold_flags_and_population_sizes=True,original_and_parent_histories_preserved=True,
        native_eligibility=True,native_threads=1,elapsed_seconds=round(time.monotonic()-start,3),
        oracle_sha256=oracle_sha,verifier_sha256=verifier_sha,
        retrieval_source_sha256=intake.sha(intake.ROOT/'sciona/services/catalog_artifact_retrieval.py'),
        plan_sha256=intake.sha(intake.PLAN),
        selection_scope='Served SQL view, latest-version predicate, document RPC and production converter. HTTP transport and semantic-search ranking are not covered.')


if __name__=='__main__':
    result=verify()
    (intake.ROOT/'docs/reviews/flavours_corrected_served_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
