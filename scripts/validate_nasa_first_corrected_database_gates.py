"""Reject actual original-workflow catalog corruption inside rolled-back savepoints."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_nasa_first_corrected_intake import ROOT,plan,sha,require
from scripts.check_nasa_first_corrected_intake import check_candidate
from scripts.import_nasa_first_corrected_intake import RUNNER


def validate():
    proposed=plan();v=proposed['version_id'];identity=proposed['artifact_id'];original=proposed['original_version_id']
    faults=[
        ('version_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,v),'Candidate selection state differs'),
        ('version_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(v,),'Candidate selection state differs'),
        ('original_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(original,),'Original latest version changed'),
        ('publishability','UPDATE artifacts SET is_publishable=true WHERE artifact_id=%s',(identity,),'Candidate selection state differs'),
        ('root_contract',"UPDATE artifact_io_specs SET constraints='' WHERE version_id=%s AND direction='input' AND name='raw_populations'",(v,),'Candidate root contracts differ'),
        ('output_contract',"UPDATE artifact_io_specs SET constraints='' WHERE version_id=%s AND direction='output' AND name='state'",(v,),'Candidate root contracts differ'),
        ('node_binding',"UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id='training/bind_state'",('0'*64,v),'Candidate node bindings differ'),
        ('inference_binding',"DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id='inference/predict'",(v,),'Candidate node bindings differ'),
        ('runtime_binding',"UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s AND node_id='inference/unbind_state'",(v,),'Candidate node bindings differ'),
        ('nested_evidence',"UPDATE artifact_audit_evidence SET details=details-'nested_bindings' WHERE version_id=%s AND runner_version=%s",(v,RUNNER),'Candidate staging evidence differs'),
        ('output_evidence',"UPDATE artifact_audit_evidence SET details=details-'output_bindings' WHERE version_id=%s AND runner_version=%s",(v,RUNNER),'Candidate staging evidence differs'),
        ('failed_evidence','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s AND runner_version=%s',(v,RUNNER),'Candidate staging evidence differs'),
        ('historical_failure',"UPDATE artifact_audit_evidence SET passed=true WHERE version_id=%s AND runner_version='competition-callable-preflight.v1'",(original,),'Historical failed preflight evidence changed'),
        ('graph_edge',"DELETE FROM artifact_cdg_edges WHERE version_id=%s AND source_id='training/bind_state' AND target_id='inference/unbind_state'",(v,),'execution graph projection drift'),
    ]
    for index,atom in enumerate(proposed['atoms']):
        faults.append((f'dependency_{index}','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s',
                       (v,atom['fqdn']),'Candidate dependency closure differs'))
    rejected=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,));check_candidate(db,proposed)
        for name,sql,args,expected in faults:
            db.execute('SAVEPOINT fault')
            try:
                require(db.execute(sql,args).rowcount>0,'Fault changed no rows')
                try:check_candidate(db,proposed)
                except ValueError as error:
                    require(str(error)==expected,'Unexpected rejection: '+str(error));rejected.append(name)
                else:raise ValueError('Catalog corruption accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT fault');db.execute('RELEASE SAVEPOINT fault')
        check_candidate(db,proposed);db.rollback()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on') as db:check_candidate(db,proposed)
    return dict(passed=True,approved=False,committed_catalog_mutations=0,rollback_verified=True,rejected_faults=rejected,
        version_id=v,graph_sha256=proposed['graph_sha256'],plan_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_corrected_intake_plan.json'),
        implementation_sha256={name:sha(ROOT/name) for name in ['scripts/validate_nasa_first_corrected_database_gates.py','scripts/check_nasa_first_corrected_intake.py']})


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_nasa_first_corrected_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(report['rejected_faults']),rollback_verified=True)))
