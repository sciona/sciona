"""Atomically supersede the encoder draft while preserving historical evidence."""
import argparse
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from scripts.plan_typed_inputs_correction import ROOT,sha,evidence,immutable
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable,ensure_row
from scripts.promote_openvaccine_decomposed_providers import snapshot

PLAN=ROOT/'docs/reviews/typed_inputs_correction_plan.json'
GATES=ROOT/'docs/reviews/typed_inputs_correction_transactions.json'
COMMIT='08b6da70'
RUNNER='typed-inputs-unicode-correction.v1'


def reviewed_plan():
    p=immutable(str(PLAN.relative_to(ROOT)),COMMIT)
    if sha(ROOT/'scripts/plan_typed_inputs_correction.py')!=p['planner_sha256']:raise ValueError('Planner changed')
    import hashlib
    if hashlib.sha256(json.dumps(evidence(),sort_keys=True,separators=(',',':')).encode()).hexdigest()!=p['evidence_sha256']:
        raise ValueError('Corrected evidence changed')
    return p


def check(db,p,applied):
    for r in p['providers']:
        changed=r['new_version_required'];old=r['previous']
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            base=db.execute(f'SELECT fqdn,status,is_publishable,description FROM {prefix}s WHERE {key}=%s',(r['atom_id'],)).fetchone()
            expected=dict(fqdn=r['runtime_fqdn'],status='flagged' if prefix=='atom' else 'draft',is_publishable=False,
                description=(r if applied else old)['seed_fields']['description'])
            if base!=expected:raise ValueError('Non-serving draft metadata differs')
            oldrow=db.execute(f'SELECT to_jsonb(v) AS row FROM {prefix}_versions v WHERE version_id=%s',(old['version_id'],)).fetchone()
            wanted=dict(r['previous_versions'][prefix],is_latest=not(applied and changed))
            if not oldrow or comparable(oldrow['row'])!=wanted:raise ValueError('Historical version changed')
            versions=db.execute(f'SELECT version_id::text FROM {prefix}_versions WHERE {key}=%s ORDER BY version_id',(r['atom_id'],)).fetchall()
            expected_ids=sorted({old['version_id'],r['version_id']} if applied else {old['version_id']})
            if [v['version_id'] for v in versions]!=expected_ids:raise ValueError('Version inventory differs')
            current=db.execute(f'SELECT version_id::text,content_hash,semver,fingerprint,trust_tier FROM {prefix}_versions WHERE {key}=%s AND is_latest',(r['atom_id'],)).fetchall()
            now=r if applied else old
            if comparable(current)!=[dict(version_id=now['version_id'],content_hash=now['content_hash'],semver=now['semver'],fingerprint=now['fingerprint'],trust_tier=3)]:
                raise ValueError('Current version differs')
            for record in ([old,r] if applied and changed else [old]):
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(record['version_id'],)).fetchall()
                if ports!=sorted(record['interfaces'],key=lambda x:(x['direction'],x['ordinal'])):raise ValueError('Versioned ports differ')
            if db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(r['atom_id'],)).fetchone():
                raise ValueError('Correction unexpectedly serves a draft')


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    p=reviewed_plan();ids=[UUID(r['atom_id']) for r in p['providers']]
    if apply:
        gate=json.loads(GATES.read_text())
        if not gate['passed'] or not gate['all_rollbacks_complete'] or gate['plan_sha256']!=sha(PLAN) or gate['stager_sha256']!=sha(__file__):
            raise ValueError('Successful current rollback gates required')
    writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            db.execute(f'SELECT {key} FROM {prefix}s WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
            db.execute(f'SELECT version_id FROM {prefix}_versions WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
        marker=uuid5(UUID(p['providers'][0]['version_id']),RUNNER)
        repeated=db.execute('SELECT 1 FROM artifact_audit_evidence WHERE evidence_id=%s',(marker,)).fetchone() is not None
        check(db,p,repeated)
        if not repeated and snapshot(db,ids)!=p['before']:raise ValueError('Catalog changed since correction plan')
        old_audits=db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s)',(ids,)).fetchall()
        def tick(n):
            nonlocal writes
            writes+=n
            if fail_after is not None and writes>=fail_after:raise InjectedFailure('Injected correction failure')
        def ensure(table,key,row):tick(int(ensure_row(db,table,key,row)))
        for r in p['providers']:
            identity=UUID(r['atom_id']);version=UUID(r['version_id']);old=UUID(r['previous']['version_id'])
            if r['new_version_required'] and not repeated:
                for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                    tick(db.execute(f'UPDATE {prefix}_versions SET is_latest=false WHERE version_id=%s AND is_latest',(old,)).rowcount)
                    ensure(prefix+'_versions',dict(version_id=version),dict(version_id=version,**{key:identity},
                        content_hash=r['content_hash'],semver=r['semver'],fingerprint=r['fingerprint'],trust_tier=3,s3_key='',is_latest=True))
                    for port in r['interfaces']:
                        row={key:identity,'version_id':version,**port}
                        ensure(prefix+'_io_specs',{k:row[k] for k in (key,'version_id','direction','name')},row)
                    tick(db.execute(f'UPDATE {prefix}s SET description=%s WHERE {key}=%s',(r['seed_fields']['description'],identity)).rowcount)
            if r['new_version_required']:
                eid=uuid5(old,RUNNER+'-counterexample')
                ensure('artifact_audit_evidence',dict(evidence_id=eid),dict(evidence_id=eid,artifact_id=identity,version_id=old,
                    audit_type='semantic_audit',passed=False,status='completed',source_kind='automated',runner_version=RUNNER+'-counterexample',
                    details=Jsonb(dict(evidence_commit=p['evidence_commit'],evidence_sha256=p['evidence_sha256'],
                        scope='Frozen encoder changes explicit surrogate pairs during standard JSON round trip; this failure applies only to the superseded encoder version.'))))
            eid=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=eid),dict(evidence_id=eid,artifact_id=identity,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(plan_commit=COMMIT,plan_sha256=sha(PLAN),provider_sha256=r['provider_sha256'],
                    evidence_sha256=p['evidence_sha256'],scope='Corrected non-serving draft only; semantic publication remains pending.'))))
        check(db,p,True)
        after_audits=db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s)',(ids,)).fetchall()
        if any(row not in after_audits for row in old_audits):raise ValueError('Historical audit evidence changed')
        if reviewed_plan()!=p:raise ValueError('Correction inputs changed during transaction')
        if fail_after is not None:raise ValueError('Failure point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,repeated=repeated,rows_changed=writes,served_providers=0,
                plan_sha256=sha(PLAN),stager_sha256=sha(__file__))


def validate():
    p=reviewed_plan();ids=[UUID(r['atom_id']) for r in p['providers']]
    def unchanged():
        with connect() as db:
            if snapshot(db,ids)!=p['before']:raise ValueError('Rollback incomplete')
            check(db,p,False)
    unchanged();dry=stage();unchanged()
    points=sorted({1,dry['rows_changed']//2,dry['rows_changed']})
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected failure not raised')
        unchanged()
    return dict(passed=True,all_rollbacks_complete=True,catalog_mutations=0,dry_run=dry,
                fault_after_writes=points,plan_sha256=sha(PLAN),stager_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','apply','repeat'])
    action=parser.parse_args().action
    result=validate() if action=='validate' else stage(apply=True)
    if action=='repeat' and (not result['repeated'] or result['rows_changed']):raise ValueError('Repeat changed state')
    output=GATES if action=='validate' else ROOT/f'docs/reviews/typed_inputs_correction_{action}.json'
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
