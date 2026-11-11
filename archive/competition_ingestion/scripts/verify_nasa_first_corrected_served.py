"""Verify served selection, immutable approval evidence and selected-graph execution."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from sciona.nasa_first_complete_graph import build_nasa_first_complete_graph
from scripts.plan_nasa_first_corrected_intake import ROOT,plan,require,sha
from scripts.check_nasa_first_corrected_intake import check_candidate,PUBLICATION_RUNNER
from scripts.promote_nasa_first_corrected_intake import DESCRIPTION,PLAIN_DESCRIPTION
from scripts.review_nasa_first_corrected_publication import review
import scripts.validate_nasa_first_complete_graph as complete


def selection(proposed):
    qualification=review()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed,approved=True)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchall()
        require(len(rows)==1,'Exactly one served original CDG required')
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(proposed['artifact_id'],)).fetchall()
        require(len(versions)==1 and str(versions[0]['version_id'])==proposed['version_id'],'Served latest version differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        require(candidate.cdg==build_nasa_first_complete_graph() and candidate.content_hash==proposed['graph_sha256'] and candidate.trust_tier==3,'Served runtime candidate differs')
        require(candidate.description==DESCRIPTION and candidate.conceptual_summary==PLAIN_DESCRIPTION and candidate.verified_leaf_coverage==1,'Served description or coverage differs')
        audits=db.execute('SELECT passed,status,source_kind,details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',(proposed['version_id'],PUBLICATION_RUNNER)).fetchall()
        require(audits==[dict(passed=True,status='completed',source_kind='automated',details=dict(qualification=qualification,publication_tier=3,
            publisher_sha256=sha(ROOT/'scripts/promote_nasa_first_corrected_intake.py'),description=DESCRIPTION,plain_description=PLAIN_DESCRIPTION))],'Immutable approval evidence differs')
    return candidate.cdg


def verify(source,checkpoint):
    proposed=plan();graph=selection(proposed)
    with patch.object(complete,'build_nasa_first_complete_graph',side_effect=lambda:graph.model_copy(deep=True)):
        execution=complete.validate(source,checkpoint)
    require(selection(proposed)==graph,'Served selection changed during execution')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'],served_sql_selection=True,production_candidate_converter=True,
        corrected_descriptions_verified=True,original_history_preserved=True,execution=execution,
        verifier_sha256=sha(Path(__file__)),retrieval_source_sha256=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'),
        limitations=['Served SQL selection and production candidate execution; HTTP transport and semantic search ranking are not qualified.',
                     'Compositional native fitting evidence: all fit operands are checked before qualified checkpoint replay.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=verify(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_corrected_served.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,trust_tier=3,nodes=report['execution']['executed_outer_nodes'],exact_predictions=report['execution']['exact_native_prediction_comparisons'])))
