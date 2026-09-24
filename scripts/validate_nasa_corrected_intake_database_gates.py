"""Exercise corrected-intake integrity checks with rollback-only catalog faults."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from scripts.plan_nasa_corrected_intake import ROOT, plan, require, sha
from scripts.check_nasa_corrected_intake import check_candidate


def fault_cases(p):
    version=p['version_id']; original=p['original_version_id']
    cases=[
        ('original_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,original),'Immutable intake identity differs'),
        ('original_snapshot',"UPDATE artifact_audit_evidence SET details=jsonb_set(details,'{snapshot}','{}'::jsonb) WHERE version_id=%s AND runner_version='competition-intake.v1'",(original,),'Immutable intake snapshot differs'),
        ('original_ports',"UPDATE artifact_io_specs SET type_desc=type_desc || ' changed' WHERE version_id=%s",(original,),'Original history differs'),
        ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Corrected version differs'),
        ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,),'Corrected version differs'),
        ('candidate_latest','UPDATE artifact_versions SET is_latest=true WHERE version_id=%s',(version,),'Corrected version differs'),
        ('original_not_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(original,),'Intake latest-version selection differs'),
        ('premature_approval',"UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(p['artifact_id'],),'Intake approval state differs'),
        ('root_contract',"UPDATE artifact_io_specs SET constraints=constraints || ' changed' WHERE version_id=%s",(version,),'Corrected root contracts differ'),
        ('optional_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,),'Corrected provenance differs'),
        ('provenance_hash','UPDATE artifact_dependencies SET dependency_content_hash=%s WHERE dependent_version_id=%s',('0'*64,version),'Corrected provenance differs'),
        ('provenance_metadata',"UPDATE artifact_dependencies SET binding_metadata='{}'::jsonb WHERE dependent_version_id=%s",(version,),'Corrected provenance differs'),
        ('failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,),'Unresolved corrected-version failure'),
        ('runtime_binding',"UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s",(version,),'Corrected bindings differ'),
    ]
    edge=build_complete_workflow_graph(10).edges[0]
    cases.append(('missing_edge','DELETE FROM artifact_cdg_edges WHERE version_id=%s AND source_id=%s AND target_id=%s AND input_name=%s',
        (version,edge.source_id,edge.target_id,edge.input_name),'execution graph projection drift'))
    for index,atom in enumerate(p['atoms']):
        cases.extend([
            (f'provider_hash_{index}','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,atom['version_id']),'Approved provider version differs'),
            (f'provider_approval_{index}',"UPDATE artifacts SET status='draft',is_publishable=false WHERE artifact_id=%s",(atom['artifact_id'],),'Approved provider identity differs'),
        ])
    for key,parent in p['parent_graphs'].items():
        cases.append((key+'_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,parent['version_id']),'Approved parent version differs'))
    return cases


def validate():
    proposed=plan(); rejected=[]
    connection=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(connection,row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        for name in ['residual-classifier-draft.v1','nasa-domain-draft.v1','nasa-lifecycle-draft.v1','nasa-population-draft.v1','nasa-corrected-intake-draft.v1']:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(name,))
        check_candidate(db,proposed)
        cases=fault_cases(proposed)
        for index,(name,statement,params,expected) in enumerate(cases):
            db.execute('SAVEPOINT fault')
            try:
                require(db.execute(statement,params).rowcount>0,'Fault injection changed no rows: '+name)
                try:check_candidate(db,proposed)
                except ValueError as error:
                    require(str(error)==expected,'Unexpected rejection for '+name+': '+str(error))
                    rejected.append(name)
                else:raise ValueError('Fault accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT fault')
                db.execute('RELEASE SAVEPOINT fault')
            if (index+1)%10==0:
                print(json.dumps(dict(rejected=index+1,total=len(cases))),flush=True)
        check_candidate(db,proposed)
        db.rollback()
    with psycopg.connect(connection,row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed)
    require(plan()==proposed,'Qualification inputs changed during validation')
    return dict(passed=True,approved=False,committed_catalog_mutations=0,rollback_verified=True,
        rejected_faults=rejected,artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'],original_version_id=proposed['original_version_id'],
        original_rows_sha256=proposed['original_rows_sha256'],
        validator_sha256=sha(Path(__file__)),checker_sha256=sha(ROOT/'scripts/check_nasa_corrected_intake.py'),
        planner_sha256=sha(ROOT/'scripts/plan_nasa_corrected_intake.py'),
        scope='Application integrity checker against rollback-only database faults, including all 39 reused provider hashes and approvals. Baseline checks all 485 bindings. Successful publication and latest-version activation require separate transaction and served-selection qualification.')


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_corrected_intake_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(report['rejected_faults']),rollback_verified=True,committed_catalog_mutations=0)))
