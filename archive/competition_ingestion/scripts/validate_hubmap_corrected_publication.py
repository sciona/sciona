"""Reject corrupted candidate state and prove pre/post-activation rollback."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
from scripts import promote_hubmap_corrected_intake as publisher
from scripts import stage_hubmap_corrected_intake as intake


def snapshot(db):
    result={}
    for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_references','artifact_audit_rollups','artifact_audit_evidence','artifact_io_specs','artifact_parameters']:
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',(intake.IDENTITY,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    rows=db.execute('SELECT to_jsonb(t) AS row FROM references_registry t WHERE ref_id=%s',(publisher.REFERENCE,)).fetchall()
    result['reference']=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def validate():
    plan,providers,review=publisher.review();version=plan['version_id'];rejected=[]
    test_path='tests/test_hubmap_publication_evidence.py'
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-m','pytest','-q',test_path],cwd=intake.ROOT,
        env=environment,capture_output=True,text=True,check=True)
    match=re.search(r'(\d+) passed',result.stdout)
    if not match or int(match.group(1))!=37:raise ValueError('Complete publication evidence tests required')
    tests=dict(passed=True,count=37,sha256=intake.sha(intake.ROOT/test_path))
    with intake.connect() as db:before=snapshot(db)
    def unchanged():
        with intake.connect() as db:
            if snapshot(db)!=before:raise ValueError('Publication changes survived rollback')
            intake.check(db,plan)
    with intake.connect() as db:
        faults=[
            ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version)),
            ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,)),
            ('candidate_lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,)),
            ('original_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(intake.ORIGINAL,)),
            ('original_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.ORIGINAL,)),
            ('parent_history',"UPDATE artifact_versions SET semver='corrupted' WHERE version_id=%s",(intake.PARENT,)),
            ('parent_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(intake.PARENT,)),
            ('parent_publication','UPDATE artifacts SET is_publishable=false WHERE artifact_id=(SELECT artifact_id FROM artifact_versions WHERE version_id=%s)',(intake.PARENT,)),
            ('dependency_optional','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,)),
            ('failed_candidate_audit',"INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,source_kind,runner_version,details) VALUES(gen_random_uuid(),%s,%s,'asset_integrity_check',false,'completed','automated','synthetic-corruption','{}'::jsonb)",(intake.IDENTITY,version)),
        ]
        for binding in plan['bindings']:
            node=binding['node_id']
            faults.extend([
                ('binding_hash:'+node,'UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id=%s',('0'*64,version,node)),
                ('binding_missing:'+node,'DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id=%s',(version,node)),
                ('serialized_node:'+node,"UPDATE artifact_cdg_nodes SET description='corrupted' WHERE version_id=%s AND node_id=%s",(version,node))])
        graph=intake.build_catalog_hubmap_graph()
        for edge in graph.edges:
            faults.append(('serialized_edge:'+edge.target_id,
                'DELETE FROM artifact_cdg_edges WHERE version_id=%s AND source_id=%s AND target_id=%s AND output_name=%s AND input_name=%s',
                (version,edge.source_id,edge.target_id,edge.output_name,edge.input_name)))
        for port in plan['ports']:
            direction,name=port['direction'],port['name']
            faults.append(('root_contract:'+direction+':'+name,
                "UPDATE artifact_io_specs SET constraints='corrupted' WHERE version_id=%s AND direction=%s AND name=%s",
                (version,direction,name)))
            if direction=='input':
                faults.extend([
                    ('root_required:'+name,"UPDATE artifact_io_specs SET required=NOT required WHERE version_id=%s AND direction='input' AND name=%s",(version,name)),
                    ('root_default:'+name,"UPDATE artifact_io_specs SET default_value_repr='synthetic-invalid-default' WHERE version_id=%s AND direction='input' AND name=%s",(version,name))])
            else:
                faults.append(('root_output_missing:'+name,"DELETE FROM artifact_io_specs WHERE version_id=%s AND direction='output' AND name=%s",(version,name)))
        for provider in providers['providers']:
            for prefix in ('artifact','atom'):
                faults.append(('provider_tier:'+prefix+':'+provider['runtime_fqdn'],
                    f'UPDATE {prefix}_versions SET trust_tier=1 WHERE version_id=%s',(provider['version_id'],)))
        for name,sql,args in faults:
            db.execute('SAVEPOINT corrupted')
            try:
                changed=db.execute(sql,args).rowcount
                if changed<1:raise ValueError('Fault did not modify state: '+name)
                try:intake.check(db,plan)
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
        execution_evidence_tests=tests,
        all_rollbacks_complete=True,injected_failure_points=['metadata','activation'],
        publisher_sha256=intake.sha(Path(publisher.__file__)),validator_sha256=intake.sha(Path(__file__)),
        execution_sha256=review['execution_sha256'],qualification=review)


if __name__=='__main__':
    result=validate()
    (intake.ROOT/'docs/reviews/hubmap_corrected_publication_gates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
