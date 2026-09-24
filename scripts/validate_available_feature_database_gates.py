"""Verify staged providers and reject rollback-only catalog corruption."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_available_feature_providers import ROOT,plan,sha
from scripts.stage_available_feature_providers import RUNNER


def require(condition,message):
    if not condition:raise ValueError(message)


def check_staged(db,proposed,*,approved=False):
    source=db.execute('SELECT a.artifact_id,a.fqdn,v.version_id,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',
        (proposed['source_provenance']['version_id'],)).fetchone()
    require(source and {key:str(value) for key,value in source.items()}==proposed['source_provenance'],'Source provenance differs')
    for atom in proposed['atoms']:
        for identities,versions,ports,key,draft in [('artifacts','artifact_versions','artifact_io_specs','artifact_id','draft'),
                ('atoms','atom_versions','atom_io_specs','atom_id','flagged')]:
            row=db.execute(f'SELECT fqdn,status,is_publishable FROM {identities} WHERE {key}=%s',(atom['artifact_id'],)).fetchone()
            require(row==dict(fqdn=atom['fqdn'],status='approved' if approved else draft,is_publishable=approved),'Provider identity/state differs')
            row=db.execute(f'SELECT {key} AS identity,content_hash,is_latest,trust_tier FROM {versions} WHERE version_id=%s',(atom['version_id'],)).fetchone()
            require(row and str(row['identity'])==atom['artifact_id'] and row['content_hash']==atom['content_hash']
                and row['is_latest'] and row['trust_tier']==3,'Provider version differs')
            columns=['direction','name','ordinal','type_desc','constraints','required','default_value_repr']
            if key=='artifact_id':columns.append('dim_signature')
            actual=db.execute(f'SELECT {",".join(columns)} FROM {ports} WHERE version_id=%s ORDER BY direction,ordinal',(atom['version_id'],)).fetchall()
            expected=[]
            for direction,selected in [('input',atom['inputs']),('output',atom['outputs'])]:
                for ordinal,port in enumerate(selected):
                    expected.append({column:direction if column=='direction' else ordinal if column=='ordinal' else port[column] for column in columns})
            require(actual==expected,'Provider port contracts differ')
        runtime=db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(atom['artifact_id'],)).fetchone()
        require(runtime and runtime['import_module']+'.'+runtime['source_symbol']==atom['runtime_fqdn'],'Provider runtime differs')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(atom['version_id'],)).fetchone(),'Unresolved failed audit')
        audits=db.execute('SELECT details,status,source_kind,passed FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',
            (atom['version_id'],RUNNER)).fetchall()
        expected=dict(details=dict(scope='Non-publishable draft identity, ports and evidence binding only.',
            qualification=proposed['evidence'],source_provenance=proposed['source_provenance']),status='completed',source_kind='automated',passed=True)
        require(audits==[expected],'Staging evidence differs')
        if not approved:
            for view,key in [('catalog_artifacts_served','artifact_id'),('catalog_atoms_served','atom_id')]:
                require(not db.execute(f'SELECT 1 FROM {view} WHERE {key}=%s',(atom['artifact_id'],)).fetchone(),'Draft unexpectedly served')


def validate():
    proposed=plan();faults=[]
    for i,atom in enumerate(proposed['atoms']):
        version=atom['version_id'];port=atom['inputs'][0]['name']
        for table in ['artifact_versions','atom_versions']:
            faults.extend([(f'{table}_hash_{i}',f'UPDATE {table} SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Provider version differs'),
                (f'{table}_tier_{i}',f'UPDATE {table} SET trust_tier=1 WHERE version_id=%s',(version,),'Provider version differs')])
        for table in ['artifact_io_specs','atom_io_specs']:
            faults.append((f'{table}_contract_{i}',f"UPDATE {table} SET constraints='' WHERE version_id=%s AND direction='input' AND name=%s",(version,port),'Provider port contracts differ'))
    first=proposed['atoms'][0];version=first['version_id'];identity=first['artifact_id']
    faults.extend([
        ('missing_staging_evidence','DELETE FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',(version,RUNNER),'Staging evidence differs'),
        ('evidence_payload',"UPDATE artifact_audit_evidence SET details='{}'::jsonb WHERE version_id=%s AND runner_version=%s",(version,RUNNER),'Staging evidence differs'),
        ('failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,),'Unresolved failed audit'),
        ('runtime_pointer',"UPDATE atoms SET source_symbol='synthetic_wrong_runtime' WHERE atom_id=%s",(identity,),'Provider runtime differs'),
        ('canonical_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(version,),'Provider version differs'),
        ('legacy_latest','UPDATE atom_versions SET is_latest=false WHERE version_id=%s',(version,),'Provider version differs'),
        ('canonical_publishable','UPDATE artifacts SET is_publishable=true WHERE artifact_id=%s',(identity,),'Provider identity/state differs'),
        ('legacy_publishable','UPDATE atoms SET is_publishable=true WHERE atom_id=%s',(identity,),'Provider identity/state differs'),
        ('source_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('1'*64,proposed['source_provenance']['version_id']),'Source provenance differs')])
    rejected=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        check_staged(db,proposed)
        for name,statement,params,expected in faults:
            db.execute('SAVEPOINT injected_fault')
            try:
                require(db.execute(statement,params).rowcount>0,'Fault injection changed no rows')
                try:check_staged(db,proposed)
                except ValueError as error:
                    require(str(error)==expected,'Unexpected fault rejection: '+str(error));rejected.append(name)
                else:raise ValueError('Catalog fault accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT injected_fault');db.execute('RELEASE SAVEPOINT injected_fault')
        check_staged(db,proposed);db.rollback()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:check_staged(db,proposed)
    return dict(passed=True,approved=False,committed_catalog_mutations=0,provider_count=13,
        rejected_faults=rejected,rollback_verified=True,
        validator_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/reviews/available_feature_provider_plan.json'),
        scope='Application verifier exercised against actual rollback-only database faults, not a claim about trigger enforcement.')


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/available_feature_provider_database_gates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(result['rejected_faults']),rollback_verified=True,approved=False)))
