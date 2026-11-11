"""Reject corrupted candidate state and prove pre/post-activation rollback."""
import hashlib
import json
from scripts import promote_causal_corrected_intake as publisher
from scripts import stage_causal_corrected_intake as intake


def snapshot(db):
    result={}
    for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_references','artifact_audit_rollups','artifact_audit_evidence','artifact_io_specs']:
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',(intake.IDENTITY,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    rows=db.execute('SELECT to_jsonb(t) AS row FROM references_registry t WHERE ref_id=%s',(publisher.REFERENCE,)).fetchall()
    result['reference']=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def validate():
    plan,providers,review=publisher.review();version=plan['version_id'];rejected=[]
    with intake.connect() as db:before=snapshot(db)
    def unchanged():
        with intake.connect() as db:
            if snapshot(db)!=before:raise ValueError('Publication changes survived rollback')
            intake.check_candidate(db,plan,providers)
    with intake.connect() as db:
        faults=[
            ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version)),
            ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,)),
            ('candidate_lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,)),
            ('original_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(intake.ORIGINAL,)),
            ('original_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.ORIGINAL,)),
            ('parent_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.PARENT,)),
            ('binding_hash',"UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id='ensemble'",('0'*64,version)),
            ('binding_missing',"DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id='fit_direction'",(version,)),
            ('dependency_optional','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,)),
            ('root_input_type',"UPDATE artifact_io_specs SET type_desc='dict' WHERE version_id=%s AND direction='input' AND name='pair_labels'",(version,)),
            ('score_output_missing',"DELETE FROM artifact_io_specs WHERE version_id=%s AND direction='output' AND name='causal_scores'",(version,)),
            ('failed_staging_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,)),
            ('serialized_node',"UPDATE artifact_cdg_nodes SET description='corrupted' WHERE version_id=%s AND node_id='ensemble'",(version,)),
            ('serialized_edge',"DELETE FROM artifact_cdg_edges WHERE version_id=%s AND target_id='model_roles' AND input_name='direction_model'",(version,)),
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
    unchanged();dry=publisher.publish();unchanged()
    for point in ('metadata','activation'):
        try:publisher.publish(fail_at=point)
        except publisher.InjectedFailure:pass
        else:raise ValueError('Expected publication failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,rejected_faults=rejected,dry_run=dry,
        all_rollbacks_complete=True,injected_failure_points=['metadata','activation'],
        publisher_sha256=intake.sha(publisher.__file__),validator_sha256=intake.sha(__file__),
        execution_sha256=review['execution_sha256'])


if __name__=='__main__':
    result=validate()
    (intake.ROOT/'docs/reviews/causal_corrected_publication_gates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
