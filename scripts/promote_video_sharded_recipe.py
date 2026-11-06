"""Publish the existing video_sharded_recipe providers with exact Tier 3 evidence."""
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
from scripts.stage_video_sharded_recipe import preserved
from scripts.stage_video_sharded_recipe import sha
from scripts.stage_video_sharded_recipe import reviewed_plan
from sciona.atoms.ml import video_multilabel_sharded_execution as atoms

RUNNER = 'video_sharded_recipe-adapter-community.v1'
REFERENCE = 'video_sharded_recipe-adapter-review-v1'
COMMIT = '6ab52c5b'
REVIEW = 'docs/reviews/video_sharded_recipe_semantics.json'
PLAN = ROOT/'docs/reviews/video_sharded_recipe_publication_plan.json'
GATES = ROOT/'docs/reviews/video_sharded_recipe_publication_gates.json'


def review():
    committed = subprocess.check_output(['git','show',COMMIT+':'+REVIEW], cwd=ROOT)
    if (ROOT/REVIEW).read_bytes() != committed: raise ValueError('Immutable review changed')
    result = json.loads(committed)
    record = result['adapter']
    provider = dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
        version_id=record['version_id'], content_hash=record['content_hash'], interfaces=record['interfaces'],
        operation='Run full configured sharded training, held-out threshold fitting and query prediction with nine explicit numerical functions.',
        plain_description='Train models, combine their scores and choose labels using separate training, tuning and prediction inputs.',
        validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
        reuse='Domain recipe with separately selectable numeric operations; model, optimization and population policies remain explicit.')
    result['providers'] = [provider]
    intake = reviewed_plan()
    if sha(ROOT/'docs/reviews/video_sharded_recipe_intake_plan.json')!=result['intake_plan_sha256']:
        raise ValueError('Qualified intake changed')
    if sha(ROOT/'scripts/review_video_sharded_recipe_semantics.py') != result['reviewer_sha256']:
        raise ValueError('Qualified reviewer changed')
    if sha(ROOT/'docs/reviews/catalog_video_multilabel_execution.json') != result['execution_evidence_sha256']:
        raise ValueError('Qualified execution report changed')
    for path,digest in result['execution']['software_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Executed software changed')
    if sha(inspect.getfile(atoms)) != record['provider_sha256']: raise ValueError('Recipe source changed')
    for provider,record in zip(result['providers'],intake['providers'],strict=True):
        if (provider['artifact_id']!=record['atom_id'] or provider['version_id']!=record['version_id']
            or provider['content_hash']!=record['content_hash'] or provider['interfaces']!=record['interfaces']):
            raise ValueError('Qualified provider or interfaces changed')
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
        ref_type='web',title='Configured sharded video recipe synthetic qualification',url=url))]
    for dependency in qualified['registered_provider_dependencies']:
        row = dict(dependent_version_id=records[0]['version_id'],
            **{k:dependency[k] for k in ('dependency_artifact_fqdn','dependency_content_hash','port_name','dependency_role','optional')},
            binding_metadata=dict(runtime_fqdn=dependency['runtime_fqdn'], version_id=dependency['version_id'],
                interfaces=dependency['interfaces'], qualification_sha256=sha(ROOT/REVIEW)))
        keys = {k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','port_name','dependency_role')}
        inserts.append(dict(table='artifact_dependencies',keys=keys,row=row))
    rollups = []; states = {}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        before = snapshot(db, ids); history = histories(db, ids); originals = preserved(db)
        for r in records:
            identity = r['artifact_id']; version = r['version_id']
            fn = getattr(atoms, r['fqdn'].rsplit('.',1)[1]); signature = inspect.signature(fn)
            states[identity] = {}
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                states[identity][prefix] = db.execute(f'SELECT status,is_publishable FROM {prefix}s WHERE {key}=%s',(identity,)).fetchone()
                expected = dict(status='flagged' if prefix=='atom' else 'draft',is_publishable=False)
                if states[identity][prefix] != expected: raise ValueError('Unexpected existing provider state')
                for p in r['interfaces']:
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
                    title='Configured sharded video recipe synthetic qualification',url=url,source='llm_extracted',
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
        before=before,preserved=history,originals_sha256=originals,publisher_sha256=sha(__file__),review_sha256=sha(ROOT/REVIEW))


def qualify_plan():
    p = json.loads(PLAN.read_text())
    if p['publisher_sha256'] != sha(__file__) or p['review_sha256'] != sha(ROOT/REVIEW) or p['qualification'] != review():
        raise ValueError('Publication qualification changed')
    return p


def check_dependencies(db, plan, published):
    fields = ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash',
              'port_name','dependency_role','optional','binding_metadata')
    actual = comparable(db.execute('SELECT '+','.join(fields)+' FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY port_name',
        (plan['records'][0]['version_id'],)).fetchall())
    expected = sorted([item['row'] for item in plan['inserts'] if item['table']=='artifact_dependencies'], key=lambda r:r['port_name']) if published else []
    if actual != expected: raise ValueError('Exact dependency closure differs')
    for dependency in plan['qualification']['registered_provider_dependencies']:
        version = dependency['version_id']
        rows = db.execute("""SELECT a.artifact_id,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=%s AND v.version_id=%s""", (dependency['dependency_artifact_fqdn'],version)).fetchall()
        if len(rows)!=1: raise ValueError('Dependency identity missing')
        row=rows[0]
        if (row['status']!='approved' or not row['is_publishable'] or not row['is_latest']
                or row['trust_tier']!=3 or row['content_hash']!=dependency['dependency_content_hash']):
            raise ValueError('Dependency approval or version changed')
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(row['artifact_id'],)).fetchone():
            raise ValueError('Dependency not served')
        ports=db.execute('SELECT '+','.join(PORT_FIELDS)+' FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
        if ports!=dependency['interfaces']: raise ValueError('Dependency interface changed')
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            state=db.execute(f'''SELECT a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest,
                {prefix}_is_publishable(a.{key}) AS eligible FROM {prefix}s a
                JOIN {prefix}_versions v USING({key}) WHERE a.fqdn=%s AND v.version_id=%s''',
                (dependency['dependency_artifact_fqdn'],version)).fetchall()
            if state != [dict(status='approved',is_publishable=True,content_hash=dependency['dependency_content_hash'],
                              trust_tier=3,is_latest=True,eligible=True)]:
                raise ValueError('Dependency native eligibility or legacy version differs')
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE fqdn=%s',
                              (dependency['dependency_artifact_fqdn'],)).fetchone():
                raise ValueError('Dependency not served in both catalogs')
            actual_ports=db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (version,)).fetchall()
            if actual_ports != dependency['interfaces']:raise ValueError('Dependency catalog contracts differ')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(version,)).fetchone():
            raise ValueError('Failed dependency audit')


def check(db, plan, published=False):
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    if preserved(db) != plan['originals_sha256']: raise ValueError('Original CDG history changed')
    check_dependencies(db, plan, published)
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
        for field in ('constraints_json','binding_metadata'):
            if field in row: row[field]=Jsonb(row[field])
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
            or gates['validator_sha256']!=sha(ROOT/'scripts/validate_video_sharded_recipe_publication.py')
            or gates['fault_count']!=84): raise ValueError('Matching complete publication gates required')
    ids=[UUID(r['artifact_id']) for r in plan['records']]
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            db.execute(f'SELECT {key} FROM {prefix}s WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
        already=db.execute("SELECT count(*) AS n FROM artifacts WHERE artifact_id=ANY(%s) AND status='approved' AND is_publishable",(ids,)).fetchone()['n']==len(ids)
        if already: check(db,plan,published=True); writes=0
        else:
            if snapshot(db,ids)!=plan['before']: raise ValueError('Catalog changed since plan')
            writes=apply_transaction(db,plan)
        for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
            if db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone()['n']!=len(ids):
                raise ValueError('Existing provider must be served')
        if qualify_plan()!=plan: raise ValueError('Evidence changed during publication')
    return dict(applied=True,already_approved=already,rows_changed=writes,served_providers=len(ids),trust_tier=3,plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','apply','repeat'])
    action=parser.parse_args().action
    result=build_plan() if action=='plan' else publish()
    if action=='repeat' and result['rows_changed']: raise ValueError('Repeat changed rows')
    path=PLAN if action=='plan' else ROOT/('docs/reviews/video_sharded_recipe_publication_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_written=True,providers=1) if action=='plan' else result))
