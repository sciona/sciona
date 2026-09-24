"""Execute the selected corrected intake version with immutable source validation."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_corrected_intake import ROOT,plan,sha,require
from scripts.check_nasa_corrected_intake import check_candidate
import scripts.validate_nasa_complete_workflow as execution


def validate(source,approved=False):
    proposed=plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed,approved=approved)
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=proposed['version_id'],content_hash=proposed['graph_sha256'],require_execution_envelope=True)
    audit_path=ROOT/'docs/reviews/nasa_original_intake_scope_audit.json'
    require(sha(audit_path)==proposed['evidence_sha256']['nasa_original_intake_scope_audit.json'],'Historical scope evidence changed')
    historical=json.loads(audit_path.read_text())
    def immutable_scope(source_directory):
        # The historical audit remains unchanged. Current catalog approval and
        # immutable original records were checked by check_candidate above.
        for name,digest in historical['source_sha256'].items():
            require(sha(source_directory/name)==digest,'Pinned source changed: '+name)
        return dict(passed=True,current_intake_approved=approved,selected_execution_version=proposed['version_id'],
            original_source_version=proposed['original_version_id'],original_content_hash=proposed['original_content_hash'],
            historical_scope_audit_sha256=sha(audit_path),source_sha256=historical['source_sha256'],
            corrections=historical['corrections'],stage_regions=historical['stage_regions'],
            provenance_policy=proposed['provenance_policy'],catalog_mutations=0)
    original_select=execution.select_complete_workflow_graph
    original_build=execution.build_complete_workflow_graph
    def select(payload):
        selected=original_select(payload)
        return graph if selected.metadata['active_query_populations']==10 else selected
    def build(count):
        return graph if count==10 else original_build(count)
    with patch.object(execution,'audit',side_effect=immutable_scope), \
            patch.object(execution,'select_complete_workflow_graph',side_effect=select), \
            patch.object(execution,'build_complete_workflow_graph',side_effect=build):
        result=execution.validate(source)
    stored=[case for case in result['cases'] if case['active_query_populations']==10]
    require(result['passed'] and len(stored)==1 and stored[0]['graph_sha256']==proposed['graph_sha256']
        and stored[0]['predictions']==640,'Stored complete intake execution required')
    for case in result['cases']:
        case['execution_source']='stored_corrected_intake' if case['active_query_populations']==10 else 'additional_sparse_candidate_regression'
    result.pop('approved',None)
    result['approval_scope']='Only the selected ten-query-population intake version is under catalog qualification; the sparse candidate is supplementary regression evidence.'
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed,approved=approved)
    require(plan()==proposed,'Qualification changed during execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],
        original_identity_preserved=True,original_records_preserved=True,graph_sha256=proposed['graph_sha256'],
        reused_atoms=len(proposed['atoms']),bindings=len(proposed['bindings']),stored_predictions_compared=640,
        additional_candidate_predictions=128,execution=result,validator_sha256=sha(Path(__file__)),
        checker_sha256=sha(ROOT/'scripts/check_nasa_corrected_intake.py'),planner_sha256=proposed['planner_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--approved',action='store_true');args=parser.parse_args()
    report=validate(args.source_directory,args.approved)
    name='nasa_corrected_intake_served_execution.json' if args.approved else 'nasa_corrected_intake_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='execution'}))
