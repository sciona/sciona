"""Rollback-only rehearsal of provider versions; never persist unreviewed rows."""
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import UUID

from dotenv import dotenv_values
import psycopg
from psycopg.rows import dict_row

from scripts.plan_osic_relocated_provider_versions import ROOT, plan

PLAN_REVISION='76dc1ee2'
PLAN_PATH='docs/reviews/osic_relocated_provider_version_plan.json'


def connect():
    return psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
        row_factory=dict_row,options='-c statement_timeout=30000 -c lock_timeout=5000')


def snapshot(db, ids):
    result={}
    for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
        for suffix in ('s','_versions','_io_specs','_audit_rollups'):
            table=prefix+suffix
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s)',(ids,)).fetchall()
            result[table]=sorted((r['row'] for r in rows),key=lambda r:json.dumps(r,sort_keys=True))
    for table in ('artifact_dependencies','artifact_audit_evidence'):
        where=('dependent_version_id IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s))'
               if table=='artifact_dependencies' else 'artifact_id=ANY(%s)')
        result[table]=sorted((r['row'] for r in db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {where}',(ids,))),
                             key=lambda r:json.dumps(r,sort_keys=True))
    return result


class InjectedFailure(RuntimeError):
    pass


def validate():
    raw=subprocess.check_output(['git','show',PLAN_REVISION+':'+PLAN_PATH],cwd=ROOT)
    expected=json.loads(raw)
    if raw!=(ROOT/PLAN_PATH).read_bytes() or plan()!=expected:
        raise ValueError('Reviewed version plan changed')
    records=[r for r in expected['records'] if r['action']=='new_provider_version_required']
    ids=[UUID(r['artifact_id']) for r in records]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        before=snapshot(db,ids)
        proposed={r['proposed']['version_id'] for r in records}
        if any(row['version_id'] in proposed for table in ('atom_versions','artifact_versions')
               for row in before[table]):
            raise ValueError('Proposed versions already exist; reconcile before rehearsal')

    def rehearse(fail_after=None):
        operations=0
        inserted=0
        exposed=0
        def tick(count):
            nonlocal operations,inserted
            operations+=1
            inserted+=count
            if operations==fail_after:
                raise InjectedFailure('Injected rollback-only rehearsal failure')
        with connect() as db:
            try:
                db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('osic-relocated-version-rehearsal.v1',))
                for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                    db.execute(f'SELECT {key} FROM {prefix}s WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
                    db.execute(f'SELECT version_id FROM {prefix}_versions WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
                if snapshot(db,ids)!=before:
                    raise ValueError('Catalog changed since rehearsal snapshot')
                for r in records:
                    new=r['proposed'];old=r['previous_version_id'];identity=r['artifact_id']
                    for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                        tick(db.execute(f'''INSERT INTO {prefix}_versions
                            (version_id,{key},content_hash,semver,fingerprint,trust_tier,s3_key,is_latest,derives_from)
                            VALUES (%s,%s,%s,%s,%s,3,'',false,%s) ON CONFLICT DO NOTHING''',
                            (new['version_id'],identity,new['content_hash'],new['semver'],new['fingerprint'],old)).rowcount)
                        actual=db.execute(f'''SELECT content_hash,semver,fingerprint,trust_tier,is_latest,derives_from::text
                            FROM {prefix}_versions WHERE version_id=%s AND {key}=%s''',(new['version_id'],identity)).fetchone()
                        if actual!={k:new[k] for k in ('content_hash','semver','fingerprint')}|dict(trust_tier=3,is_latest=False,derives_from=old):
                            raise ValueError('Inserted version differs')
                        fields='direction,name,type_desc,constraints,required,default_value_repr,ordinal'
                        if prefix=='artifact':fields+=',dim_signature'
                        tick(db.execute(f'''INSERT INTO {prefix}_io_specs ({key},version_id,{fields})
                            SELECT {key},%s,{fields} FROM {prefix}_io_specs
                            WHERE version_id=%s AND {key}=%s ON CONFLICT DO NOTHING''',
                            (new['version_id'],old,identity)).rowcount)
                        previous=db.execute(f'SELECT {fields} FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal,name',(old,)).fetchall()
                        current=db.execute(f'SELECT {fields} FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal,name',(new['version_id'],)).fetchall()
                        if not previous or current!=previous:
                            raise ValueError('Copied interfaces differ')
                        latest=db.execute(f'SELECT version_id::text FROM {prefix}_versions WHERE {key}=%s AND is_latest',(identity,)).fetchall()
                        if latest!=[dict(version_id=old)]:
                            raise ValueError('Existing latest version changed')
                        if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(identity,)).fetchone():
                            raise ValueError('Existing served artifact changed')
                    fields='dependency_artifact_fqdn,dependency_content_hash,dependency_role,port_name,optional,binding_metadata'
                    tick(db.execute(f'''INSERT INTO artifact_dependencies (dependent_version_id,{fields})
                        SELECT %s,{fields} FROM artifact_dependencies WHERE dependent_version_id=%s
                        ON CONFLICT DO NOTHING''',(new['version_id'],old)).rowcount)
                    previous=db.execute(f'SELECT {fields} FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn,port_name',(old,)).fetchall()
                    current=db.execute(f'SELECT {fields} FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn,port_name',(new['version_id'],)).fetchall()
                    if current!=previous:
                        raise ValueError('Copied dependencies differ')
                    # The document RPC exposes historical IO even while the old
                    # version remains latest. Never commit this staging state.
                    document=db.execute('SELECT get_artifact_document(%s) AS d',(r['runtime_fqdn'],)).fetchone()['d']
                    if any(row.get('version_id')==new['version_id'] for row in document.get('io_specs') or []):
                        exposed+=1
                if fail_after is not None:
                    raise ValueError('Failure point not reached')
            finally:
                db.rollback()
        return dict(logical_operations=operations,inserted_rows_before_rollback=inserted,
                    proposed_versions_exposed_by_document_rpc=exposed)

    def unchanged():
        with connect() as db:
            db.execute('SET TRANSACTION READ ONLY')
            if snapshot(db,ids)!=before:
                raise ValueError('Rollback did not restore catalog snapshot')
    dry=rehearse();unchanged()
    points=sorted({1,dry['logical_operations']//2,dry['logical_operations']})
    for point in points:
        try:rehearse(point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected failure')
        unchanged()
    repeat=rehearse();unchanged()
    if repeat!=dry or plan()!=expected:
        raise ValueError('Repeat rehearsal or source identities changed')
    return dict(passed=True,approved=False,catalog_mutations=0,rollback_only=True,
        providers=len(records),rehearsal=dry,fault_after_operations=points,
        all_rollbacks_complete=True,repeat_rehearsal_exact=True,
        plan_revision=PLAN_REVISION,plan_sha256=hashlib.sha256(raw).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            'No apply mode: proposed non-latest versions remain unreviewed and are rolled back.',
            'Historical version interfaces appear in the document RPC; non-latest is not a publication isolation boundary.',
            'Final replacement must atomically publish reviewed versions after immutable source anchoring; replacement graph versions need exact new bindings while historical versions remain unchanged.',
            'Rollback rehearsal does not establish semantic approval, committed idempotency or full workflow completion.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_relocated_version_rehearsal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
