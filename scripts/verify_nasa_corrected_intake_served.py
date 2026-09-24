"""Fresh SQL catalog selection, runtime candidate materialization and served execution."""
import argparse
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from scripts.plan_nasa_corrected_intake import ROOT, plan, require, sha
from scripts.check_nasa_corrected_intake import check_candidate
from scripts.promote_nasa_corrected_intake import DESCRIPTION, PLAIN_DESCRIPTION
from scripts.validate_nasa_corrected_intake_catalog import validate


def selection(proposed):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed,approved=True)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',
            (proposed['artifact_id'],)).fetchall()
        require(len(rows)==1,'Original CDG must have one served row')
        # Same predicate as CatalogMacroArtifactRetriever._latest_version;
        # use the configured direct catalog connection, not an HTTP mock.
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',
            (proposed['artifact_id'],)).fetchall()
        require(len(versions)==1 and str(versions[0]['version_id'])==proposed['version_id'],'Latest selection chose wrong version')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        require(candidate.cdg==build_complete_workflow_graph(10) and candidate.content_hash==proposed['graph_sha256']
            and candidate.trust_tier==3,'Runtime candidate differs from corrected graph')
        require(candidate.description==DESCRIPTION and candidate.conceptual_summary==PLAIN_DESCRIPTION,'Served descriptions differ')
        require(candidate.verified_leaf_coverage==1,'Complete binding coverage required')
    return dict(passed=True,selected_version_id=proposed['version_id'],trust_tier=3,nodes=len(candidate.cdg.nodes),
        corrected_descriptions_verified=True,
        scope='Actual served SQL view and latest-version predicate, get_artifact_document, and production candidate converter. HTTP transport and semantic search ranking are not covered.')


def verify(source):
    proposed=plan();before=selection(proposed)
    execution=validate(source,approved=True)
    require(selection(proposed)==before,'Selection changed during execution')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,selection=before,execution=execution,
        validator_sha256=sha(Path(__file__)),catalog_validator_sha256=sha(ROOT/'scripts/validate_nasa_corrected_intake_catalog.py'),
        retrieval_source_sha256=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=verify(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/nasa_corrected_intake_served_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,trust_tier=3,stored_predictions=report['execution']['stored_predictions_compared'])))
