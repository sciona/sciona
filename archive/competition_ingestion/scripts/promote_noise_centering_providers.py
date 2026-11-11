"""Publish the existing noise_centering providers with exact Tier 3 evidence."""
import argparse
import inspect
import json
import subprocess
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb

from scripts.import_flavours_provider_drafts import connect, ROOT
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.promote_openvaccine_decomposed_providers import snapshot, PORT_FIELDS
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.review_noise_centering_semantics import review as current_review, atoms, sha

RUNNER = 'noise_centering-provider-community.v1'
REFERENCE = 'noise_centering-provider-review-v1'
COMMIT = '2fc46848'
REVIEW = 'docs/reviews/noise_centering_semantics.json'
PLAN = ROOT/'docs/reviews/noise_centering_provider_publication_plan.json'
GATES = ROOT/'docs/reviews/noise_centering_provider_publication_gates.json'


def review():
    committed = subprocess.check_output(['git','show',COMMIT+':'+REVIEW], cwd=ROOT)
    if (ROOT/REVIEW).read_bytes() != committed: raise ValueError('Immutable review changed')
    result = current_review()
    if result != json.loads(committed): raise ValueError('Current qualification differs')
    return result


def histories(db, ids):
    result = {}
    for prefix, key in [('atom','atom_id'),('artifact','artifact_id')]:
        for suffix in ('versions','parameters','references','io_specs'):
            table = prefix+'_'+suffix
            # New metadata is checked separately; preserve every preexisting row.
            result[table] = comparable(db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
    result['artifact_audit_evidence'] = comparable(db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
    return result


def build_plan():
    qualified = review(); records = qualified['providers']; ids = [UUID(r['artifact_id']) for r in records]
    url = 'https://github.com/sciona/sciona/blob/'+COMMIT+'/'+REVIEW
    inserts = [dict(table='references_registry', keys={'ref_id':REFERENCE}, row=dict(ref_id=REFERENCE,
        ref_type='web',title='Explicit additive noise and median centering implementation and synthetic qualification',url=url))]
    rollups = []; states = {}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        before = snapshot(db, ids); preserved = histories(db, ids)
        for r in records:
            identity = r['artifact_id']; version = r['version_id']
            fn = getattr(atoms, r['fqdn'].rsplit('.',1)[1]); signature = inspect.signature(fn)
            states[identity] = {}
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                states[identity][prefix] = db.execute(f'SELECT status,is_publishable FROM {prefix}s WHERE {key}=%s',(identity,)).fetchone()
                expected = dict(status='flagged' if prefix=='atom' else 'draft',is_publishable=False)
                if states[identity][prefix] != expected: raise ValueError('Expected unpublished existing provider')
                for p in r['interfaces']:
                    if prefix == 'atom':
                        row = dict(atom_id=identity,version_id=version,**p)
                        inserts.append(dict(table='atom_io_specs',keys={k:row[k] for k in ('atom_id','version_id','direction','name')},row=row))
                    if p['direction'] != 'input': continue
                    row = {key:identity,'version_id':version,'name':p['name'],'position':p['ordinal'],
                        'kind':signature.parameters[p['name']].kind.name.lower(),'type_desc':p['type_desc'],
                        'required':p['required'],'default_value_repr':p['default_value_repr'],
                        'technical_description':p['constraints'],'dejargonized_description':'Caller supplies '+p['name'].replace('_',' ')+'.',
                        'constraints_json':dict(parameter_contract=p['constraints'],limitations=qualified['limitations'])}
                    inserts.append(dict(table=prefix+'_parameters',keys={k:row[k] for k in (key,'version_id','name')},row=row))
                for kind,content in [('technical',r['operation']+' '+r['validity_bounds']+' '+r['reuse']),
                                     ('dejargonized',r['plain_description']+' Accuracy on new problems must be checked separately.')]:
                    score = round(estimate_jargon_score(content),6)
                    if kind=='dejargonized' and score>=.4: raise ValueError('Plain description requires revision')
                    keys = {key:identity,'kind':kind,'language':'en'}
                    inserts.append(dict(table=prefix+'_descriptions',keys=keys,row=dict(keys,content=content,
                        generated_by=RUNNER,reviewed=False,jargon_score=score)))
                keys = {key:identity,'ref_key':REFERENCE}
                inserts.append(dict(table=prefix+'_references',keys=keys,row=dict(keys,ref_id=REFERENCE,
                    title='Explicit additive noise and median centering implementation and synthetic qualification',url=url,source='llm_extracted',
                    verified=True,confidence='high',relevance_note=r['operation']+' '+r['validity_bounds'])))
                old = db.execute(f'SELECT to_jsonb(t) AS row FROM {prefix}_audit_rollups t WHERE {key}=%s',(identity,)).fetchone()
                old = comparable(old['row']) if old else None
                limits = list(dict.fromkeys((old or {}).get('review_limitations',[])+qualified['limitations']+[r['validity_bounds']]))
                fields = dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',
                    semantic_status='pass_with_limits',developer_semantics_status='pass_with_limits',review_status='approved',
                    review_semantic_verdict='pass_with_limits',review_developer_semantics_verdict='pass_with_limits',
                    trust_readiness='ready',review_limitations=limits,review_required_actions=[],trust_blockers=[],
                    acceptability_band='acceptable_with_limits')
                rollups.append(dict(table=prefix+'_audit_rollups',key=key,identity=identity,before=old,fields=fields))
    return dict(records=records,qualification=qualified,inserts=inserts,rollups=rollups,states=states,
        before=before,preserved=preserved,publisher_sha256=sha(__file__),review_sha256=sha(ROOT/REVIEW))


def qualify_plan():
    p = json.loads(PLAN.read_text())
    if p['publisher_sha256'] != sha(__file__) or p['review_sha256'] != sha(ROOT/REVIEW) or p['qualification'] != review():
        raise ValueError('Publication qualification changed')
    return p


def check(db, plan, published=False):
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    now = histories(db, ids)
    for table,rows in plan['preserved'].items():
        if any(row not in now[table] for row in rows): raise ValueError('Prior history modified: '+table)
    for r in plan['records']:
        identity,version = r['artifact_id'],r['version_id']
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',(identity,)).fetchall()
            expected = dict(fqdn=r['fqdn'],version_id=version,content_hash=r['content_hash'],trust_tier=3,
                **(dict(status='approved',is_publishable=True) if published else plan['states'][identity][prefix]))
            if actual != [expected]: raise ValueError('Identity/version/publication differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
            if ports != r['interfaces']: raise ValueError('Exact interface differs')
            if published and not db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
                raise ValueError('Native eligibility failed')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(version,)).fetchone():
            raise ValueError('Completed failed audit exists')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchone():
            raise ValueError('Unexpected registered provider dependency')
        if published:
            evidence = db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE evidence_id=%s',(uuid5(UUID(version),RUNNER),)).fetchone()
            if not evidence or evidence['passed'] is not True or evidence['details'] != evidence_details(plan,r):
                raise ValueError('Exact publication evidence differs')
    if published:
        for item in plan['inserts']:
            rows = db.execute('SELECT '+','.join(item['row'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchall()
            if comparable(rows) != [item['row']]: raise ValueError('Published metadata differs: '+item['table'])
        for item in plan['rollups']:
            row = db.execute('SELECT '+','.join(item['fields'])+' FROM '+item['table']+' WHERE '+item['key']+'=%s',(item['identity'],)).fetchone()
            if row != item['fields']: raise ValueError('Published review or limitations differ')


def evidence_details(plan, record):
    return dict(plan_sha256=sha(PLAN),review_sha256=plan['review_sha256'],publication_tier=3,
        provider=record,limitations=plan['qualification']['limitations'],
        previous_rollups=[r for r in plan['rollups'] if r['identity']==record['artifact_id'] and r['before']])


class InjectedFailure(RuntimeError): pass


def apply_transaction(db,plan,fail_after=None):
    check(db,plan); writes=0
    def tick(n):
        nonlocal writes
        writes+=n
        if fail_after is not None and writes>=fail_after: raise InjectedFailure('Injected publication rollback')
    for item in plan['inserts']:
        row = dict(item['row'])
        if 'constraints_json' in row: row['constraints_json']=Jsonb(row['constraints_json'])
        tick(int(ensure_row(db,item['table'],item['keys'],row)))
    for item in plan['rollups']:
        table,key,identity=item['table'],item['key'],item['identity']
        old=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=%s FOR UPDATE',(identity,)).fetchone()
        if (comparable(old['row']) if old else None)!=item['before']: raise ValueError('Previous rollup changed')
        if old:
            db.execute(f'UPDATE {table} SET '+','.join(k+'=%s' for k in item['fields'])+f' WHERE {key}=%s',(*item['fields'].values(),identity));tick(1)
        else: tick(int(ensure_row(db,table,{key:identity},{key:identity,**item['fields']})))
    for r in plan['records']:
        identity,version=r['artifact_id'],r['version_id']; eid=uuid5(UUID(version),RUNNER)
        tick(int(ensure_row(db,'artifact_audit_evidence',{'evidence_id':eid},dict(evidence_id=eid,
            artifact_id=identity,version_id=version,audit_type='semantic_audit',passed=True,status='completed',
            source_kind='automated',runner_version=RUNNER,details=Jsonb(evidence_details(plan,r))))))
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
                raise ValueError('Native eligibility failed before activation')
            tick(db.execute(f"UPDATE {prefix}s SET status='approved',is_publishable=true WHERE {key}=%s AND (status<>'approved' OR NOT is_publishable)",(identity,)).rowcount)
    check(db,plan,published=True)
    if fail_after is not None: raise ValueError('Fault point not reached')
    return writes


def publish():
    plan=qualify_plan(); gates=json.loads(GATES.read_text())
    if (gates['passed'] is not True or gates['all_rollbacks_complete'] is not True
            or gates['plan_sha256']!=sha(PLAN) or gates['publisher_sha256']!=sha(__file__)
            or gates['validator_sha256']!=sha(ROOT/'scripts/validate_noise_centering_provider_publication.py')
            or gates['fault_count']!=21): raise ValueError('Matching complete publication gates required')
    ids=[UUID(r['artifact_id']) for r in plan['records']]
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            db.execute(f'SELECT {key} FROM {prefix}s WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
        already=db.execute("SELECT count(*) AS n FROM artifacts WHERE artifact_id=ANY(%s) AND status='approved' AND is_publishable",(ids,)).fetchone()['n']==1
        if already: check(db,plan,published=True); writes=0
        else:
            if snapshot(db,ids)!=plan['before']: raise ValueError('Catalog changed since plan')
            writes=apply_transaction(db,plan)
        for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
            if db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone()['n']!=1:
                raise ValueError('The noise-centering provider must be served')
        if qualify_plan()!=plan: raise ValueError('Evidence changed during publication')
    return dict(applied=True,already_approved=already,rows_changed=writes,served_providers=1,trust_tier=3,plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','apply','repeat'])
    action=parser.parse_args().action
    result=build_plan() if action=='plan' else publish()
    if action=='repeat' and result['rows_changed']: raise ValueError('Repeat changed rows')
    path=PLAN if action=='plan' else ROOT/('docs/reviews/noise_centering_provider_publication_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_written=True,providers=1) if action=='plan' else result))
