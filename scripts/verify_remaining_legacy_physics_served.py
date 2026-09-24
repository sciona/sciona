"""Fresh served catalog selection and execution for all five legacy revisions."""
import argparse
import asyncio
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require
from scripts.review_remaining_legacy_physics_revisions import review
from scripts.promote_remaining_legacy_physics_revisions import check_publication,publication_fields
from scripts.validate_remaining_legacy_physics_execution import validate
from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever


def selection(qualification):
    results=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        selected=check_publication(db,qualification)
        for item,(graph,record) in zip(qualification['graphs'],selected):
            rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',(item['artifact_id'],)).fetchall()
            require(len(rows)==1,'Unique served legacy artifact required')
            versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(item['artifact_id'],)).fetchall()
            require(len(versions)==1 and str(versions[0]['version_id'])==item['version_id'],'Served latest legacy version differs')
            document=db.execute('SELECT get_artifact_document(%s) AS d',(rows[0]['fqdn'],)).fetchone()['d']
            candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
            technical,plain,_,_=publication_fields(db,item)
            require(candidate.cdg==graph and candidate.content_hash==item['graph_sha256'] and candidate.trust_tier==3,'Materialized legacy candidate differs')
            require(candidate.description==technical and candidate.conceptual_summary==plain and candidate.verified_leaf_coverage==1,'Served legacy descriptions or coverage differ')
            results.append(dict(family=item['family'],version_id=item['version_id'],graph_sha256=item['graph_sha256'],trust_tier=3,semantic_approval_verified=True,original_history_preserved=True))
    return results


async def verify(source):
    qualification=review(source);before=selection(qualification)
    execution=await validate(source,approved=True)
    require(selection(qualification)==before and review(source)==qualification,'Qualification or selection changed during served execution')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,selection=before,execution=execution,
        scope='Direct served SQL view, latest-version predicate, get_artifact_document and production candidate converter plus actual stored execution. HTTP transport and semantic search ranking are not covered.',
        validator_sha256=sha(__file__),execution_validator_sha256=sha(ROOT/'scripts/validate_remaining_legacy_physics_execution.py'),retriever_sha256=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=asyncio.run(verify(parser.parse_args().source_directory))
    (ROOT/'docs/reviews/remaining_legacy_physics_served_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,graphs=len(report['selection']),trust_tier=3)))
