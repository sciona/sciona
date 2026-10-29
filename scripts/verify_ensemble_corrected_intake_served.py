"""Execute the complete candidate obtained through production served selection."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from threadpoolctl import threadpool_limits

from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from scripts.plan_ensemble_corrected_intake import ROOT,plan,build_graph,sha,require
from scripts.check_ensemble_corrected_intake import check_candidate
from scripts.promote_ensemble_corrected_intake import DESCRIPTION,PLAIN_DESCRIPTION
from scripts.review_ensemble_corrected_intake_publication import review
from scripts import validate_ensemble_execution as oracle


def selection(proposed):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed,approved=True)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',
            (proposed['artifact_id'],)).fetchall()
        require(len(rows)==1,'Original CDG must have exactly one served row')
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',
            (proposed['artifact_id'],)).fetchall()
        require(len(versions)==1 and str(versions[0]['version_id'])==proposed['version_id'],'Latest selection chose wrong version')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        require(candidate.cdg==build_graph() and candidate.content_hash==proposed['graph_sha256']
            and candidate.trust_tier==3,'Runtime candidate differs from corrected graph')
        require(candidate.description==DESCRIPTION and candidate.conceptual_summary==PLAIN_DESCRIPTION,'Served descriptions differ')
        require(candidate.verified_leaf_coverage==1,'Complete binding coverage required')
    return candidate.cdg


async def verify(args):
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        require(os.environ.get(key)=='1','Explicit single-thread environment required')
    require(Path(os.environ['R_LIBS_USER']).resolve()==Path(os.environ.get('SCIONA_ENSEMBLE_R_LIBRARY',str(ROOT/'.venv/ensemble-r-runtime/library'))).resolve(),
        'Selected R runtime differs')
    proposed=plan();qualification=review();graph=selection(proposed)
    verifier_sha=sha(__file__);retrieval_sha=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py')
    with threadpool_limits(limits=1),patch.object(oracle,'build_ensemble_execution_graph',return_value=graph):
        await oracle.validate(args)
    result=json.loads(args.output.read_text())
    require(result['graph_digest']==proposed['graph_sha256'] and result['executed_nodes']==155
        and result['exact_final_source_blend'] and result['synthetic_only'],'Selected full execution incomplete')
    require(plan()==proposed and review()==qualification and selection(proposed)==graph,'Serving/qualification changed during execution')
    require(verifier_sha==sha(__file__) and retrieval_sha==sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'),'Selection implementation changed')
    result.update(passed=True,approved=True,trust_tier=3,catalog_mutations=0,
        selected_version_id=proposed['version_id'],artifact_id=proposed['artifact_id'],
        production_candidate_conversion=True,selected_candidate_executed=True,
        verifier_sha256=verifier_sha,retrieval_source_sha256=retrieval_sha,
        publication_review_sha256=sha(ROOT/'docs/reviews/ensemble_corrected_intake_publication_review.json'),
        native_threads=1,
        selection_scope='Actual served SQL view and latest-version predicate, document RPC and production candidate converter. HTTP transport and semantic-search ranking are not covered.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,selected_candidate_executed=True,nodes=155)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    asyncio.run(verify(parser.parse_args()))
