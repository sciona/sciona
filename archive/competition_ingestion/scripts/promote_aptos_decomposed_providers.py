"""Publish exact reviewed providers at Tier 3 with native catalog eligibility."""
import argparse
import importlib
import inspect
import json
import hashlib
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb

from scripts.import_flavours_provider_drafts import ROOT,connect
from scripts.import_residual_execution_drafts import comparable,ensure_row
from scripts.review_aptos_decomposed_providers import review
from scripts.stage_aptos_decomposed_providers import preserved as outside_history
PORT_FIELDS = ('direction','name','ordinal','type_desc','required','default_value_repr','constraints')
def expected_dependencies(records):
    return []

from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.plan_flavours_provider_intake import sha

PLAN=ROOT/'docs/reviews/aptos_decomposed_provider_publication_plan.json'
RUNNER='aptos-decomposed-provider-community.v1'
REFERENCE='aptos-decomposed-provider-review-v1'
TABLES=[('atoms','atom_id'),('artifacts','artifact_id'),('atom_versions','atom_id'),
        ('artifact_versions','artifact_id'),('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id'),
        ('atom_parameters','atom_id'),('artifact_parameters','artifact_id'),
        ('atom_descriptions','atom_id'),('artifact_descriptions','artifact_id'),
        ('atom_references','atom_id'),('artifact_references','artifact_id'),
        ('atom_audit_rollups','atom_id'),('artifact_audit_rollups','artifact_id'),
        ('artifact_audit_evidence','artifact_id')]


def snapshot(db,identities):
    result={}
    for table,key in TABLES:
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text',(identities,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def preserved_history(db, identities):
    """Protect versions, interfaces and all prior versioned audit evidence."""
    result = {}
    for table, key in TABLES:
        if table not in ('atom_versions', 'artifact_versions', 'atom_io_specs',
                         'artifact_io_specs', 'artifact_audit_evidence'):
            continue
        condition = " AND runner_version IS DISTINCT FROM %s" if table == 'artifact_audit_evidence' else ''
        args = (identities, RUNNER) if condition else (identities,)
        rows = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s)' + condition + ' ORDER BY to_jsonb(t)::text', args).fetchall()
        result[table] = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return result


def build_plan():
    qualification=review();current={'providers':qualification['providers']}
    records=[];rollups=[];parameter_updates=[];metadata_expectations=[]
    inserts=[dict(table='references_registry',keys={'ref_id':REFERENCE},row=dict(ref_id=REFERENCE,
        ref_type='web',title='Exact provider semantics and synthetic qualification',
        url='https://github.com/sciona/sciona/blob/6aea87899fd79f15ca5bfe40a355b0a65e8a8210/docs/reviews/aptos_decomposed_provider_semantics.json'))]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for semantic,record in zip(qualification['providers'],current['providers'],strict=True):
            if semantic['runtime_fqdn']!=record['runtime_fqdn']:raise ValueError('Provider order differs')
            version=UUID(record['version_id'])
            identity=db.execute('SELECT artifact_id FROM artifact_versions WHERE version_id=%s',(version,)).fetchone()['artifact_id']
            existing=False
            item=dict(semantic,artifact_id=str(identity),existing_approved=existing,states={},ports={})
            for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
                item['states'][table]=db.execute(f'SELECT status,is_publishable FROM {table} WHERE {key}=%s',(identity,)).fetchone()
                expected_status='flagged' if table=='atoms' else 'draft'
                if item['states'][table]!=dict(status=expected_status,is_publishable=False):
                    raise ValueError('Expected unpublished staged provider')
            for table in ('atom_io_specs','artifact_io_specs'):
                item['ports'][table]=db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {table} WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
                if item['ports'][table]!=sorted(semantic['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Staged ports differ from qualified provider interface')
            records.append(item)
            module,name=semantic['runtime_fqdn'].rsplit('.',1);function=getattr(importlib.import_module(module),name)
            signature=inspect.signature(function)
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                for port in item['ports'][prefix+'_io_specs']:
                    if port['direction']!='input':continue
                    parameter=signature.parameters[port['name']]
                    row={key:str(identity),'version_id':str(version),'name':port['name'],
                         'position':port['ordinal'],'kind':parameter.kind.name.lower(),
                         'type_desc':port['type_desc'],'required':port['required'],
                         'default_value_repr':port['default_value_repr'],
                         'technical_description':semantic['parameters'][port['name']],
                         'dejargonized_description':'Caller supplies '+port['name'].replace('_',' ')+'.',
                         'constraints_json':dict(validity_bounds=semantic['validity_bounds'],parameter_contract=semantic['parameters'][port['name']])}
                    keys={k:row[k] for k in (key,'version_id','name')}
                    existing_parameter=db.execute(f'SELECT type_desc,required,default_value_repr,position,kind FROM {prefix}_parameters WHERE {key}=%s AND version_id=%s AND name=%s',(identity,version,port['name'])).fetchone()
                    if existing_parameter:
                        if comparable(existing_parameter)!={k:row[k] for k in existing_parameter}:raise ValueError('Existing parameter differs')
                        before=db.execute(f'SELECT constraints_json FROM {prefix}_parameters WHERE {key}=%s AND version_id=%s AND name=%s',(identity,version,port['name'])).fetchone()['constraints_json']
                        after=dict(before or {},validity_bounds=semantic['validity_bounds'],review_status='automated_tier_3_pass_with_limits')
                        parameter_updates.append(dict(table=prefix+'_parameters',keys=keys,before=before,after=after))
                    else:inserts.append(dict(table=prefix+'_parameters',keys=keys,row=row))
                for kind,content in [('technical',semantic['operation']+' '+semantic['validity_bounds']+' '+semantic['outputs']),
                                     ('dejargonized',semantic['operation']+' '+semantic['plain_limits'])]:
                    old=db.execute(f'SELECT content,generated_by,reviewed,jargon_score FROM {prefix}_descriptions WHERE {key}=%s AND kind=%s AND language=%s',(identity,kind,'en')).fetchone()
                    if old:
                        keys={key:str(identity),'kind':kind,'language':'en'}
                        metadata_expectations.append(dict(table=prefix+'_descriptions',keys=keys,row=comparable(old)))
                        continue
                    score=round(estimate_jargon_score(content),6)
                    if kind=='dejargonized' and score>=.4:raise ValueError('Description needs plainer language')
                    keys={key:str(identity),'kind':kind,'language':'en'}
                    inserts.append(dict(table=prefix+'_descriptions',keys=keys,row=dict(keys,content=content,
                        generated_by=RUNNER,reviewed=False,jargon_score=score)))
                keys={key:str(identity),'ref_key':REFERENCE}
                inserts.append(dict(table=prefix+'_references',keys=keys,row=dict(keys,
                    ref_id=REFERENCE,
                    title='Exact provider semantics and synthetic qualification',
                    url='https://github.com/sciona/sciona/blob/6aea87899fd79f15ca5bfe40a355b0a65e8a8210/docs/reviews/aptos_decomposed_provider_semantics.json',
                    source='llm_extracted',verified=True,confidence='high',
                    relevance_note=semantic['operation']+' '+semantic['validity_bounds'])))
                previous=db.execute(f'SELECT to_jsonb(t) AS row FROM {prefix}_audit_rollups t WHERE {key}=%s',(identity,)).fetchone()
                before=comparable(previous['row']) if previous else None
                limits=list(dict.fromkeys((before or {}).get('review_limitations',[])+qualification['limitations']+[semantic['validity_bounds']]))
                fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',
                    semantic_status='pass_with_limits',developer_semantics_status='pass_with_limits',
                    review_status='approved',review_semantic_verdict='pass_with_limits',
                    review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',
                    review_limitations=limits,review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
                rollups.append(dict(table=prefix+'_audit_rollups',key=key,identity=str(identity),before=before,fields=fields))
        identities=[UUID(r['artifact_id']) for r in records]
        preserved=[UUID(r['artifact_id']) for r in records]
        before=snapshot(db,identities);unchanged=preserved_history(db,preserved)
        outside=outside_history(db)
    if len(records)!=4 or len(preserved)!=4:raise ValueError('Expected four exact provider promotions')
    return dict(passed=True,approved=False,qualification=qualification,records=records,inserts=inserts,rollups=rollups,
        parameter_updates=parameter_updates,metadata_expectations=metadata_expectations,
        dependencies=expected_dependencies(current['providers']),before=before,preserved_approved=unchanged,
        outside_preserved=outside,review_sha256=sha(ROOT/'docs/reviews/aptos_decomposed_provider_semantics.json'))


def check(db,plan,published=False):
    if outside_history(db)!=plan['outside_preserved']:
        raise ValueError('Original intake or existing preparation changed')
    for record in plan['records']:
        identity,version=UUID(record['artifact_id']),UUID(record['version_id'])
        for table,versions,key in [('atoms','atom_versions','atom_id'),('artifacts','artifact_versions','artifact_id')]:
            rows=db.execute(f'SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier FROM {table} a JOIN {versions} v USING({key}) WHERE a.{key}=%s AND v.is_latest',(identity,)).fetchall()
            expected=dict(fqdn=record['catalog_fqdn'],version_id=str(version),content_hash=record['content_hash'],trust_tier=3,
                          **(dict(status='approved',is_publishable=True) if published else record['states'][table]))
            if rows!=[expected]:raise ValueError('Exact provider version or publication state differs: '+table)
        for table,expected in record['ports'].items():
            actual=db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {table} WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
            if actual!=expected:raise ValueError('Provider ports differ: '+table)
        failed=db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(version,)).fetchone()
        if failed:raise ValueError('Completed failed audit exists')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchone():
            raise ValueError('Unexpected internal provider dependency')
        if published:
            for function in ('atom_is_publishable','artifact_is_publishable'):
                if not db.execute(f'SELECT {function}(%s) AS eligible',(identity,)).fetchone()['eligible']:
                    raise ValueError('Native catalog eligibility failed: '+function)
            evidence=db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE evidence_id=%s',(uuid5(version,RUNNER),)).fetchone()
            if not evidence or not evidence['passed'] or evidence['details']['plan_sha256']!=sha(PLAN):raise ValueError('Publication evidence missing')
    for expected in plan['dependencies']:
        actual=db.execute('SELECT dependent_version_id::text,dependency_artifact_fqdn,dependency_content_hash,port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s',
                          (expected['dependent_version_id'],expected['dependency_artifact_fqdn'])).fetchall()
        if actual!=[expected]:raise ValueError('Required exact dependency differs')
    if published:
        for item in plan['parameter_updates']:
            actual=db.execute('SELECT constraints_json FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchall()
            if comparable(actual)!=[{'constraints_json':item['after']}]:raise ValueError('Parameter validity constraints differ')
        for item in plan['metadata_expectations']:
            actual=db.execute('SELECT '+','.join(item['row'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchall()
            if comparable(actual)!=[item['row']]:raise ValueError('Retained provider descriptions differ')
        for item in plan['inserts']:
            actual=db.execute('SELECT '+','.join(item['row'])+f" FROM {item['table']} WHERE "+
                              ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchall()
            if comparable(actual)!=[item['row']]:raise ValueError('Published metadata differs: '+item['table'])
        for item in plan['rollups']:
            actual=db.execute('SELECT '+','.join(item['fields'])+f" FROM {item['table']} WHERE {item['key']}=%s",(item['identity'],)).fetchone()
            if actual!=item['fields']:raise ValueError('Published validity bounds or review differ')


class InjectedFailure(RuntimeError):pass


def apply_transaction(db,plan,fail_after=None):
    check(db,plan)
    writes=0
    def tick(count):
        nonlocal writes
        writes+=count
        if fail_after is not None and writes>=fail_after:raise InjectedFailure('Injected publication failure')
    for item in plan['inserts']:
        row=dict(item['row'])
        if 'constraints_json' in row:row['constraints_json']=Jsonb(row['constraints_json'])
        tick(int(ensure_row(db,item['table'],item['keys'],row)))
    for item in plan['parameter_updates']:
        predicate=' AND '.join(k+'=%s' for k in item['keys']);args=tuple(item['keys'].values())
        actual=db.execute('SELECT constraints_json FROM '+item['table']+' WHERE '+predicate+' FOR UPDATE',args).fetchone()
        if actual is None or comparable(actual['constraints_json'])!=item['before']:
            raise ValueError('Previous parameter constraints changed')
        db.execute('UPDATE '+item['table']+' SET constraints_json=%s WHERE '+predicate,(Jsonb(item['after']),*args))
        tick(1)
    for item in plan['rollups']:
        table,key,identity=item['table'],item['key'],item['identity']
        old=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=%s',(identity,)).fetchone()
        if (comparable(old['row']) if old else None)!=item['before']:raise ValueError('Prior rollup changed')
        if old:
            db.execute(f'UPDATE {table} SET '+','.join(f'{k}=%s' for k in item['fields'])+f' WHERE {key}=%s',(*item['fields'].values(),identity))
            tick(1)
        else:tick(int(ensure_row(db,table,{key:identity},{key:identity,**item['fields']})))
    for record in plan['records']:
        identity,version=UUID(record['artifact_id']),UUID(record['version_id'])
        prior=[r for r in plan['rollups'] if r['identity']==str(identity) and r['before']]
        details=dict(plan_sha256=sha(PLAN),review_sha256=plan['review_sha256'],publication_tier=3,
                     provider=record,limitations=plan['qualification']['limitations'],previous_rollups=prior)
        evidence_id=uuid5(version,RUNNER)
        tick(int(ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},dict(
            evidence_id=evidence_id,artifact_id=identity,version_id=version,audit_type='semantic_audit',
            passed=True,status='completed',source_kind='automated',runner_version=RUNNER,details=Jsonb(details)))))
        for table,key,function in [('atoms','atom_id','atom_is_publishable'),('artifacts','artifact_id','artifact_is_publishable')]:
            if not db.execute(f'SELECT {function}(%s) AS eligible',(identity,)).fetchone()['eligible']:
                raise ValueError('Native eligibility not satisfied before publication: '+function)
            result=db.execute(f"UPDATE {table} SET status='approved',is_publishable=true WHERE {key}=%s AND (status<>'approved' OR NOT is_publishable)",(identity,))
            tick(result.rowcount)
    check(db,plan,published=True)
    preserved=[UUID(r['artifact_id']) for r in plan['records']]
    if preserved_history(db,preserved)!=plan['preserved_approved']:raise ValueError('Staged provider versions, interfaces or prior audit changed')
    if fail_after is not None:raise ValueError('Fault point not reached')
    return writes


def qualify_plan():
    plan=json.loads(PLAN.read_text())
    if review()!=plan['qualification'] or sha(ROOT/'docs/reviews/aptos_decomposed_provider_semantics.json')!=plan['review_sha256']:
        raise ValueError('Semantic review changed')
    return plan


def publish(apply=False):
    plan=qualify_plan()
    if apply:
        gates=json.loads((ROOT/'docs/reviews/aptos_decomposed_provider_publication_gates.json').read_text())
        expected_faults=len(plan['records'])*12+len(plan['dependencies'])*2+sum(9 if any(p['direction']=='input' for p in r['ports']['atom_io_specs']) else 7 for r in plan['records'])
        if (not gates['passed'] or gates['publisher_sha256']!=sha(__file__) or gates['plan_sha256']!=sha(PLAN)
                or gates['validator_sha256']!=sha(ROOT/'scripts/validate_aptos_decomposed_provider_publication.py')
                or not gates['all_rollbacks_complete'] or not gates['native_eligibility_verified']
                or gates['fault_count']!=expected_faults):raise ValueError('Matching publication gates required')
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        ids=[UUID(r['artifact_id']) for r in plan['records']]
        for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
            db.execute(f'SELECT {key} FROM {table} WHERE {key}=ANY(%s) FOR UPDATE',(ids,)).fetchall()
        already=db.execute("SELECT count(*) AS n FROM artifacts WHERE artifact_id=ANY(%s) AND status='approved' AND is_publishable",(ids,)).fetchone()['n']==4
        if already:check(db,plan,published=True);writes=0
        else:
            if snapshot(db,ids)!=plan['before']:raise ValueError('Staged catalog changed since plan')
            writes=apply_transaction(db,plan)
        for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
            n=db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone()['n']
            if n!=4:raise ValueError('Not all providers served')
        if qualify_plan()!=plan:raise ValueError('Publication evidence changed during transaction')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_changed=writes,exact_providers_approved=4,
        newly_approved_canonical_providers=0 if already else 4,trust_tier=3,served_providers=4,plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','dry','apply','repeat'])
    action=parser.parse_args().action
    result=build_plan() if action=='plan' else publish(apply=action in ('apply','repeat'))
    if action=='repeat' and result['rows_changed']:raise ValueError('Repeat changed rows')
    path=PLAN if action=='plan' else ROOT/('docs/reviews/aptos_decomposed_provider_publication_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_written=True,providers=len(result['records'])) if action=='plan' else result))
