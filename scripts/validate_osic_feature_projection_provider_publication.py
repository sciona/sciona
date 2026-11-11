"""Exercise database corruption gates and complete publication rollback."""
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from scripts import promote_osic_feature_projection_providers as publisher
from scripts.plan_flavours_provider_intake import sha


def validate():
    plan=publisher.qualify_plan()
    identities=[UUID(r['artifact_id']) for r in plan['records']]
    def unchanged():
        with publisher.connect() as db:
            if publisher.snapshot(db,identities)!=plan['before']:raise ValueError('Publication did not roll back completely')
            if db.execute('SELECT 1 FROM references_registry WHERE ref_id=%s',(publisher.REFERENCE,)).fetchone():raise ValueError('Rollback left reference registry entry')
    unchanged();rejected=[]
    with publisher.connect() as db:
        writes=publisher.apply_transaction(db,plan)
        def fault(name,sql,args):
            db.execute('SAVEPOINT corruption')
            try:
                if db.execute(sql,args).rowcount < 1:raise ValueError('Fault modified no rows: '+name)
                try:publisher.check(db,plan,published=True)
                except ValueError:rejected.append(name)
                else:raise AssertionError('Corruption accepted: '+name)
            finally:db.execute('ROLLBACK TO SAVEPOINT corruption');db.execute('RELEASE SAVEPOINT corruption')
        for i,r in enumerate(plan['records']):
            identity,version=UUID(r['artifact_id']),UUID(r['version_id'])
            for table in ('atom_versions','artifact_versions'):
                fault(f'{table}_hash_{i}',f'UPDATE {table} SET content_hash=%s WHERE version_id=%s',('0'*64,version))
                fault(f'{table}_tier_{i}',f'UPDATE {table} SET trust_tier=1 WHERE version_id=%s',(version,))
                fault(f'{table}_latest_{i}',f'UPDATE {table} SET is_latest=false WHERE version_id=%s',(version,))
            for table in ('atom_io_specs','artifact_io_specs'):
                fault(f'{table}_type_{i}',f"UPDATE {table} SET type_desc='invalid' WHERE version_id=%s AND direction='output'",(version,))
            for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
                fault(f'{table}_publication_{i}',f'UPDATE {table} SET is_publishable=false WHERE {key}=%s',(identity,))
            fault(f'failed_audit_{i}',"INSERT INTO artifact_audit_evidence (evidence_id,artifact_id,version_id,audit_type,passed,status,source_kind,runner_version,details) VALUES (%s,%s,%s,'semantic_audit',false,'completed','automated','synthetic-fault',%s)",
                  (uuid5(version,'osic_feature_projection-provider-fault'),identity,version,Jsonb({'scope':'Rollback-only injected failure'})))
            fault(f'unexpected_dependency_{i}',"INSERT INTO artifact_dependencies (dependent_version_id,dependency_artifact_fqdn,dependency_content_hash,port_name,dependency_role,optional,binding_metadata) VALUES (%s,%s,%s,'','logic_atom',false,%s)",
                  (version,r['fqdn'],r['content_hash'],Jsonb({'scope':'Rollback-only unexpected dependency'})))
        print('Provider identity, version, interface and failed-audit faults rejected:',len(rejected),flush=True)
        for i,r in enumerate(plan['records']):
            identity,version=UUID(r['artifact_id']),UUID(r['version_id'])
            fault(f'canonical_bounds_{i}',"UPDATE artifact_audit_rollups SET review_limitations='{}' WHERE artifact_id=%s",(identity,))
            fault(f'legacy_review_{i}',"UPDATE atom_audit_rollups SET review_status='pending' WHERE atom_id=%s",(identity,))
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                fault(f'{prefix}_description_{i}',f"DELETE FROM {prefix}_descriptions WHERE {key}=%s AND kind='dejargonized'",(identity,))
                if any(p['direction']=='input' for p in r['interfaces']):
                    fault(f'{prefix}_parameters_{i}',f'DELETE FROM {prefix}_parameters WHERE version_id=%s',(version,))
                else:
                    fault(f'{prefix}_unexpected_parameter_{i}',
                        f'''INSERT INTO {prefix}_parameters ({key},version_id,name,position,kind,type_desc,
                            required,default_value_repr,technical_description,dejargonized_description,constraints_json)
                            VALUES (%s,%s,'unexpected',0,'positional_or_keyword','int',true,'',
                            'Rollback-only invented parameter','Invented parameter',%s)''',
                        (identity,version,Jsonb({})))
                fault(f'{prefix}_references_{i}',f'DELETE FROM {prefix}_references WHERE {key}=%s',(identity,))
            fault(f'evidence_missing_{i}','DELETE FROM artifact_audit_evidence WHERE evidence_id=%s',(uuid5(version,publisher.RUNNER),))
        print('All database faults rejected:',len(rejected),flush=True)
        publisher.check(db,plan,published=True)
        db.rollback()
    unchanged()
    # Existing port rows are no-op inserts; fault offsets count actual writes.
    points=sorted({1, max(1, writes//3), max(1, 2*writes//3), writes})
    for point in points:
        try:
            with publisher.connect() as db:publisher.apply_transaction(db,plan,fail_after=point)
        except publisher.InjectedFailure:pass
        else:raise ValueError('Expected injected publication failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,rejected_faults=rejected,fault_count=len(rejected),
        publication_writes=writes,injected_failure_points=points,all_rollbacks_complete=True,
        native_eligibility_verified=True,staged_provider_histories_preserved=len(identities),
        publisher_sha256=sha(publisher.__file__),validator_sha256=sha(__file__),plan_sha256=sha(publisher.PLAN))


if __name__=='__main__':
    result=validate()
    (publisher.ROOT/'docs/reviews/osic_feature_projection_provider_publication_gates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rejected_faults'}))
