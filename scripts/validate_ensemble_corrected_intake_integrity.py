"""Reject corrupted draft candidate state, rolling back every injected change."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_ensemble_corrected_intake import check_candidate
from scripts.plan_ensemble_corrected_intake import ROOT,plan,sha,require


def validate():
    proposed=plan();version=proposed['version_id'];identity=proposed['artifact_id']
    first=proposed['bindings'][0]
    cases=[
        ('approval',"UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(identity,), 'Intake approval state'),
        ('trust_tier','UPDATE artifact_versions SET trust_tier=2 WHERE version_id=%s',(version,), 'Corrected version'),
        ('lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,), 'Corrected version'),
        ('content_hash',"UPDATE artifact_versions SET content_hash=repeat('0',64) WHERE version_id=%s",(version,), 'Corrected version'),
        ('root_contract',"UPDATE artifact_io_specs SET type_desc='corrupt' WHERE version_id=%s AND direction='input' AND ordinal=0",(version,), 'Corrected root contracts'),
        ('binding_runtime',"UPDATE artifact_cdg_bindings SET evidence_summary=jsonb_set(evidence_summary,'{runtime_fqdn}','\"corrupt\"') WHERE version_id=%s AND node_id=%s",(version,first['node_id']), 'Corrected bindings'),
        ('binding_version',"UPDATE artifact_cdg_bindings SET evidence_summary=jsonb_set(evidence_summary,'{provider_version_id}','\"corrupt\"') WHERE version_id=%s AND node_id=%s",(version,first['node_id']), 'Corrected bindings'),
        ('binding_alias',"UPDATE artifact_cdg_bindings SET evidence_summary=jsonb_set(evidence_summary,'{output_aliases_by_ordinal}','[]') WHERE version_id=%s AND node_id=%s",(version,first['node_id']), 'Corrected bindings'),
        ('provenance','DELETE FROM artifact_dependencies WHERE dependent_version_id=%s',(version,), 'Corrected provenance'),
        ('staging_evidence',"UPDATE artifact_audit_evidence SET details=jsonb_set(details,'{planner_sha256}','\"corrupt\"') WHERE version_id=%s AND runner_version='ensemble-corrected-intake-draft.v1'",(version,), 'Corrected staging evidence'),
        ('failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,), 'Unresolved corrected-version failure'),
    ]
    results={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            autocommit=True,options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        check_candidate(db,proposed)
        for name,sql,params,reason in cases:
            with db.transaction():
                changed=db.execute(sql,params).rowcount
                require(changed>0,'Fault did not mutate a row')
                try:check_candidate(db,proposed)
                except ValueError as error:
                    require(reason in str(error),'Wrong rejection reason: '+str(error))
                    results[name]=dict(rejected=True,reason=reason)
                else:raise ValueError('Corrupted state accepted: '+name)
                raise psycopg.Rollback()
            check_candidate(db,proposed)
            print('Rejected and rolled back: '+name,flush=True)
    require(plan()==proposed,'Plan changed during corruption checks')
    report=dict(passed=True,approved=False,committed_catalog_mutations=0,cases=results,
        candidate_version_id=version,graph_sha256=proposed['graph_sha256'],
        checker_sha256=sha(ROOT/'scripts/check_ensemble_corrected_intake.py'),
        validator_sha256=sha(__file__),plan_sha256=sha(ROOT/'docs/reviews/ensemble_corrected_intake_plan.json'))
    (ROOT/'docs/reviews/ensemble_corrected_intake_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    report=validate()
    print(json.dumps(dict(passed=True,faults_rejected=len(report['cases']),committed_catalog_mutations=0)))
