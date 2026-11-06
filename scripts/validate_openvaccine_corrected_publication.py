"""Reject corrupted candidate state and prove pre/post-activation rollback."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from scripts import promote_openvaccine_corrected_intakes as publisher
from scripts import stage_openvaccine_corrected_intakes as staging


def snapshot(db):
    result={}
    ids=[t['identity'] for t in staging.TARGETS.values()]
    for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_parameters','artifact_references','artifact_audit_rollups','artifact_audit_evidence','artifact_io_specs']:
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=ANY(%s::uuid[]) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    for table in ['artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_dependencies','artifact_validity_bounds']:
        key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key} IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s::uuid[])) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    refs=['openvaccine-'+key+'-corrected-intake-execution-v1' for key in staging.TARGETS]
    rows=db.execute('SELECT to_jsonb(t) AS row FROM references_registry t WHERE ref_id=ANY(%s) ORDER BY to_jsonb(t)::text',(refs,)).fetchall()
    result['reference']=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def validate(target):
    intake=staging.Intake(target)
    plan,providers,review=publisher.review(target);version=plan['version_id'];rejected=[]
    test_path='tests/test_openvaccine_corrected_publication.py'
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-m','pytest','-q',test_path],cwd=staging.ROOT,
        env=environment,capture_output=True,text=True,check=True)
    match=re.search(r'(\d+) passed',result.stdout)
    if not match or int(match.group(1))!=publisher.TEST_COUNT:raise ValueError('Complete publication evidence tests required')
    tests=dict(passed=True,count=publisher.TEST_COUNT,sha256=staging.sha(staging.ROOT/test_path))
    with staging.connect() as db:before=snapshot(db)
    def unchanged():
        with staging.connect() as db:
            if snapshot(db)!=before:raise ValueError('Publication changes survived rollback')
            intake.check_candidate(db,plan,providers)
    with staging.connect() as db:
        faults=[
            ('other_original_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.other['original'],)),
            ('other_source_dependency',"DELETE FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s",(version,intake.other['fqdn'])),
            ('binding_alias',"UPDATE artifact_cdg_bindings SET evidence_summary=jsonb_set(evidence_summary,'{output_aliases_by_ordinal}','[\"wrong\"]'::jsonb) WHERE version_id=%s AND node_id='blend'",(version,)),
            ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version)),
            ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,)),
            ('candidate_lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,)),
            ('original_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(intake.ORIGINAL,)),
            ('original_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.ORIGINAL,)),
            ('parent_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(staging.PARENT,)),
            ('parent_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(staging.PARENT,)),
            ('parent_publication','UPDATE artifacts SET is_publishable=false WHERE artifact_id=(SELECT artifact_id FROM artifact_versions WHERE version_id=%s)',(staging.PARENT,)),
            ('binding_hash',"UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id='fit_forecast'",('0'*64,version)),
            ('binding_missing',"DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id='blend'",(version,)),
            ('dependency_optional','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,)),
            ('root_input_type',"UPDATE artifact_io_specs SET type_desc='float' WHERE version_id=%s AND direction='input' AND name='payload'",(version,)),
            ('score_output_missing',"DELETE FROM artifact_io_specs WHERE version_id=%s AND direction='output' AND name='result'",(version,)),
            ('failed_staging_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,)),
            ('serialized_node',"UPDATE artifact_cdg_nodes SET description='corrupted' WHERE version_id=%s AND node_id='fit_forecast'",(version,)),
            ('serialized_edge',"DELETE FROM artifact_cdg_edges WHERE version_id=%s AND target_id='blend' AND input_name='coefficients'",(version,)),
            ('provider_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(providers['records'][0]['version_id'],)),
        ]
        for name,sql,args in faults:
            db.execute('SAVEPOINT corrupted')
            try:
                changed=db.execute(sql,args).rowcount
                if changed<1:raise ValueError('Fault did not modify state: '+name)
                try:intake.check_candidate(db,plan,providers)
                except ValueError:rejected.append(name)
                else:raise AssertionError('Corrupt candidate accepted: '+name)
            finally:db.execute('ROLLBACK TO SAVEPOINT corrupted');db.execute('RELEASE SAVEPOINT corrupted')
        db.rollback()
    unchanged();dry=publisher.publish(target);unchanged()
    for point in ('metadata','activation'):
        try:publisher.publish(target,fail_at=point)
        except publisher.InjectedFailure:pass
        else:raise ValueError('Expected publication failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,rejected_faults=rejected,dry_run=dry,
        execution_evidence_tests=tests,
        all_rollbacks_complete=True,injected_failure_points=['metadata','activation'],
        publisher_sha256=staging.sha(publisher.__file__),validator_sha256=staging.sha(__file__),
        execution_sha256=review['execution_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=list(staging.TARGETS))
    target=parser.parse_args().target
    result=validate(target)
    staging.Intake(target).report_path('publication_gates').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(target=target,**result)))
