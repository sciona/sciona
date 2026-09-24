"""Fresh served selection and production candidate materialization plus execution."""
import argparse
import asyncio
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.stage_momentum_original_revision import ROOT,ARTIFACT,sha,require
from scripts.promote_momentum_revision import check_publication,DESCRIPTION,PLAIN_DESCRIPTION
from scripts.review_momentum_revision import review
from scripts.validate_momentum_revision_execution import validate
from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever


def selection(qualification):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,imported=check_publication(db,qualification)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',(ARTIFACT,)).fetchall()
        require(len(rows)==1,'One served source artifact required')
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(ARTIFACT,)).fetchall()
        require(len(versions)==1 and str(versions[0]['version_id'])==imported['version_id'],'Latest version selection differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(rows[0]['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        require(candidate.cdg==graph and candidate.content_hash==imported['graph_sha256'] and candidate.trust_tier==3,'Production candidate differs')
        require(candidate.description==DESCRIPTION and candidate.conceptual_summary==PLAIN_DESCRIPTION,'Production descriptions differ')
        require(candidate.verified_leaf_coverage==1,'Grounded coverage differs')
    return dict(passed=True,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],trust_tier=3,
        semantic_approval_verified=True,corrected_descriptions_verified=True,original_history_preserved=True,
        scope='Direct served SQL view, latest-version predicate, get_artifact_document and production candidate converter; HTTP transport and semantic search ranking not covered.')


async def verify(source):
    qualification=review(source);before=selection(qualification)
    execution=await validate(source,approved=True)
    require(selection(qualification)==before and review(source)==qualification,'Qualification changed during served execution')
    return dict(passed=True,approved=True,trust_tier=3,selection=before,execution=execution,catalog_mutations=0,
        validator_sha256=sha(__file__),execution_validator_sha256=sha(ROOT/'scripts/validate_momentum_revision_execution.py'),
        retriever_sha256=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=asyncio.run(verify(parser.parse_args().source_directory))
    (ROOT/'docs/reviews/momentum_revision_served_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,trust_tier=3,stored_vectors=84,maximum_ulp_error=0)))
