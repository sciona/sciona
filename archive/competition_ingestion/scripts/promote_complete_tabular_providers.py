"""Publish qualified tabular providers with exact Tier 3 evidence."""
import argparse
import inspect
import importlib
import json
import subprocess
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb

from scripts.import_flavours_provider_drafts import connect, ROOT
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.promote_openvaccine_decomposed_providers import snapshot, PORT_FIELDS
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.stage_complete_tabular_providers import preserved
from scripts.stage_complete_tabular_providers import sha
from scripts.stage_complete_tabular_providers import reviewed as reviewed_plan
from scripts.stage_complete_tabular_providers import legacy_snapshot, served_snapshot

RUNNER = 'complete-tabular-provider-community.v1'
REFERENCE = 'complete-tabular-provider-review-v1'
COMMIT = '0125379e'
REVIEW = 'docs/reviews/complete_tabular_provider_semantics.json'
PLAN = ROOT/'docs/reviews/complete_tabular_provider_publication_plan.json'
GATES = ROOT/'docs/reviews/complete_tabular_provider_publication_gates.json'


def review():
    committed = subprocess.check_output(['git', 'show', COMMIT+':'+REVIEW], cwd=ROOT)
    if (ROOT/REVIEW).read_bytes() != committed: raise ValueError('Immutable semantic review changed')
    result = json.loads(committed)
    if not result['passed'] or result['proposed_tier'] != 3 or len(result['providers']) != 10:
        raise ValueError('Complete ten-provider Tier 3 review required')
    for path, digest in result['software_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Semantic source changed')
    native_path = 'docs/reviews/tabular_provider_native_qualification.json'
    native_raw = (ROOT/native_path).read_bytes()
    if native_raw != subprocess.check_output(['git','show','02be96ce:'+native_path], cwd=ROOT):
        raise ValueError('Immutable native evidence changed')
    native = json.loads(native_raw)
    if not native['passed'] or native['uncached_native_executions'] != 16:
        raise ValueError('Complete direct native evidence required')
    for path, digest in native['software_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Native execution source changed')
    from scripts.review_video_numeric_runtime import review as runtime_review
    if runtime_review() != native['runtime']: raise ValueError('Qualified runtime changed')
    intake = reviewed_plan()
    if sha(ROOT/'docs/reviews/complete_tabular_provider_intake_plan.json') != result['intake_plan_sha256']:
        raise ValueError('Qualified intake changed')
    by_id = {r['atom_id']:r for r in intake['providers']}
    for r in result['providers']:
        source = by_id[r['artifact_id']]
        if r['version_id'] != source['version_id'] or r['content_hash'] != source['content_hash'] or r['fqdn'] != source['catalog_fqdn']:
            raise ValueError('Qualified provider identity differs')
        r.update(runtime_fqdn=source['runtime_fqdn'], intake_interfaces=source['interfaces'],
            operation=r['plain_description'], reuse='Numerical semantics are reusable across domains; recipe choices remain explicit.',
            registered_provider_dependencies=[dict(d, fqdn=by_id[d['artifact_id']]['catalog_fqdn'],
                runtime_fqdn=by_id[d['artifact_id']]['runtime_fqdn']) for d in r['dependencies']])
    result['limitations'] = [v for v in result['limits'] if 'remain pending' not in v] + [
        'Six numerical providers have 16 uncached native executions bound to immutable evidence.',
        'Automated Tier 3; no human Tier 1 review, independent accuracy or named submission claim.']
    result['native_evidence_sha256'] = sha(ROOT/native_path)
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
        ref_type='web',title='Expanded tabular provider qualification',url=url))]
    rollups = []; states = {}; port_updates=[]; metadata_updates=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake=reviewed_plan()
        if legacy_snapshot(db,intake)!=intake['legacy_sha256'] or served_snapshot(db,intake)!=intake['served_sha256']:
            raise ValueError('Reviewed legacy state changed')
        before = snapshot(db, ids); history = histories(db, ids); originals = preserved(db)
        for r in records:
            identity = r['artifact_id']; version = r['version_id']
            module,symbol=r['runtime_fqdn'].rsplit('.',1)
            fn = getattr(importlib.import_module(module),symbol); signature = inspect.signature(fn)
            states[identity] = {}
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                states[identity][prefix] = db.execute(f'SELECT status,is_publishable FROM {prefix}s WHERE {key}=%s',(identity,)).fetchone()
                expected = dict(status='draft',is_publishable=False)
                if prefix=='artifact' and states[identity][prefix]!=expected:
                    raise ValueError('Expected staged canonical draft')
                keys={key:identity}
                old_description=db.execute(f'SELECT description FROM {prefix}s WHERE {key}=%s',(identity,)).fetchone()
                metadata_updates.append(dict(table=prefix+'s',keys=keys,before=old_description,
                    after=dict(description=r['operation']+' '+r['validity_bounds'])))
                for port in r['interfaces']:
                    keys={key:identity,'version_id':version,'direction':port['direction'],'name':port['name']}
                    before_port=db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE '+
                        ' AND '.join(k+'=%s' for k in keys),tuple(keys.values())).fetchone()
                    if before_port is None:
                        inserts.append(dict(table=prefix+'_io_specs',keys=keys,row={key:identity,'version_id':version,**port}))
                    else:
                        if {k:v for k,v in before_port.items() if k!='constraints'}!={k:v for k,v in port.items() if k!='constraints'}:
                            raise ValueError('Current callable port differs')
                        port_updates.append(dict(table=prefix+'_io_specs',keys=keys,
                            before=before_port['constraints'],after=port['constraints']))
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
                    after=dict(content=content,generated_by=RUNNER,reviewed=False,jargon_score=score)
                    old_description=db.execute('SELECT '+','.join(after)+f' FROM {prefix}_descriptions WHERE '+
                        ' AND '.join(k+'=%s' for k in keys),tuple(keys.values())).fetchone()
                    if old_description is None:inserts.append(dict(table=prefix+'_descriptions',keys=keys,row=dict(keys,**after)))
                    else:metadata_updates.append(dict(table=prefix+'_descriptions',keys=keys,before=comparable(old_description),after=after))
                keys = {key:identity,'ref_key':REFERENCE}
                inserts.append(dict(table=prefix+'_references',keys=keys,row=dict(keys,ref_id=REFERENCE,
                    title='Expanded tabular provider qualification',url=url,source='llm_extracted',
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
    for r in records:
        for dep in r['registered_provider_dependencies']:
            dependency=dep
            row=dict(dependent_version_id=r['version_id'],dependency_artifact_fqdn=dependency['fqdn'],
                dependency_content_hash=dep['content_hash'],port_name='',dependency_role='logic_atom',optional=False,
                binding_metadata=dict(dependency_artifact_id=dep['artifact_id'],dependency_version_id=dep['version_id']))
            keys={k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name')}
            inserts.append(dict(table='artifact_dependencies',keys=keys,row=row))
    return dict(records=records,qualification=qualified,inserts=inserts,rollups=rollups,states=states,
        port_updates=port_updates,metadata_updates=metadata_updates,
        before=before,preserved=history,originals_sha256=originals,publisher_sha256=sha(__file__),review_sha256=sha(ROOT/REVIEW))


def qualify_plan():
    p = json.loads(PLAN.read_text())
    if p['publisher_sha256'] != sha(__file__) or p['review_sha256'] != sha(ROOT/REVIEW) or p['qualification'] != review():
        raise ValueError('Publication qualification changed')
    return p


def check_dependencies(db, plan):
    dependencies={d['artifact_id']:d for r in plan['records'] for d in r['registered_provider_dependencies']}
    for identity,dep in dependencies.items():
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            rows=db.execute(f"""SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                WHERE a.{key}=%s AND v.is_latest AND a.status='approved' AND a.is_publishable
                AND {prefix}_is_publishable(a.{key})""",(identity,)).fetchall()
            if rows!=[dict(fqdn=dep['fqdn'],version_id=dep['version_id'],content_hash=dep['content_hash'],trust_tier=3)]:
                raise ValueError('Exact approved dependency unavailable')
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(identity,)).fetchone():
                raise ValueError('Dependency not served')
        module,symbol=dep['runtime_fqdn'].rsplit('.',1)
        if db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(identity,)).fetchone()!=dict(import_module=module,source_symbol=symbol):
            raise ValueError('Dependency runtime binding changed')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(dep['version_id'],)).fetchone():
            raise ValueError('Failed dependency audit')


def check(db, plan, published=False):
    if published: check_dependencies(db,plan)
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    if preserved(db) != plan['originals_sha256']: raise ValueError('Original CDG history changed')
    now = histories(db, ids)
    for table,rows in plan['preserved'].items():
        for row in rows:
            expected=dict(row['row'])
            if published:
                for update in plan['port_updates']:
                    if update['table']==table and all(str(expected.get(k))==str(v) for k,v in update['keys'].items()):
                        expected['constraints']=update['after']
            if dict(row=expected) not in now[table]:raise ValueError('Prior history modified: '+table)
    for r in plan['records']:
        identity,version = r['artifact_id'],r['version_id']
        module,symbol=r['runtime_fqdn'].rsplit('.',1)
        actual_source=db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(identity,)).fetchone()
        if actual_source!=dict(import_module=module,source_symbol=symbol):raise ValueError('Runtime binding differs')
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',(identity,)).fetchall()
            expected = dict(fqdn=r['fqdn'],version_id=version,content_hash=r['content_hash'],trust_tier=3,
                **(dict(status='approved',is_publishable=True) if published else plan['states'][identity][prefix]))
            if actual != [expected]: raise ValueError('Identity/version/publication differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
            if published and ports != r['interfaces']: raise ValueError('Exact interface differs')
            if not published and prefix=='artifact' and ports != r['intake_interfaces']:
                raise ValueError('Staged interface differs')
            if published and not db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
                raise ValueError('Native eligibility failed')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(version,)).fetchone():
            raise ValueError('Completed failed audit exists')
        expected_deps=[item['row'] for item in plan['inserts'] if item['table']=='artifact_dependencies' and item['row']['dependent_version_id']==version] if published else []
        dep_fields=('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name','dependency_role','optional','binding_metadata')
        actual_deps=comparable(db.execute('SELECT '+','.join(dep_fields)+' FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(version,)).fetchall())
        if actual_deps!=sorted(expected_deps,key=lambda d:d['dependency_artifact_fqdn']):raise ValueError('Exact provider dependencies differ')
        if published:
            for dep in r['registered_provider_dependencies']:
                ready=db.execute("SELECT 1 FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s AND v.content_hash=%s AND v.is_latest AND a.status='approved' AND a.is_publishable",(dep['artifact_id'],dep['version_id'],dep['content_hash'])).fetchone()
                if not ready:raise ValueError('Required exact dependency unavailable')
        if published:
            evidence = db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE evidence_id=%s',(uuid5(UUID(version),RUNNER),)).fetchone()
            if not evidence or evidence['passed'] is not True or evidence['details'] != evidence_details(plan,r):
                raise ValueError('Exact publication evidence differs')
    if published:
        for item in plan['inserts']:
            rows = db.execute('SELECT '+','.join(item['row'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchall()
            if comparable(rows) != [item['row']]: raise ValueError('Published metadata differs: '+item['table'])
        for item in plan['metadata_updates']:
            actual=db.execute('SELECT '+','.join(item['after'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']),tuple(item['keys'].values())).fetchone()
            if comparable(actual)!=item['after']:raise ValueError('Updated descriptions differ')
        for item in plan['rollups']:
            row = db.execute('SELECT '+','.join(item['fields'])+' FROM '+item['table']+' WHERE '+item['key']+'=%s',(item['identity'],)).fetchone()
            if row != item['fields']: raise ValueError('Published review or limitations differ')


def evidence_details(plan, record):
    return dict(plan_sha256=sha(PLAN),review_sha256=plan['review_sha256'],publication_tier=3,
        provider=record,limitations=plan['qualification']['limitations'],
        previous_rollups=[r for r in plan['rollups'] if r['identity']==record['artifact_id'] and r['before']],
        previous_port_constraints=[u for u in plan['port_updates'] if u['keys'].get('version_id')==record['version_id']],
        previous_descriptions=[u for u in plan['metadata_updates'] if record['artifact_id'] in u['keys'].values()])


class InjectedFailure(RuntimeError): pass


def apply_transaction(db,plan,fail_after=None):
    check(db,plan); writes=0
    def tick(n):
        nonlocal writes
        writes+=n
        if fail_after is not None and writes>=fail_after: raise InjectedFailure('Injected publication rollback')
    for item in plan['inserts']:
        row = dict(item['row'])
        for key in ('constraints_json','binding_metadata'):
            if key in row:row[key]=Jsonb(row[key])
        tick(int(ensure_row(db,item['table'],item['keys'],row)))
    for item in plan['port_updates']:
        where=' AND '.join(k+'=%s' for k in item['keys'])
        old=db.execute('SELECT constraints FROM '+item['table']+' WHERE '+where+' FOR UPDATE',tuple(item['keys'].values())).fetchone()
        if old!=dict(constraints=item['before']):raise ValueError('Prior port contract changed')
        tick(db.execute('UPDATE '+item['table']+' SET constraints=%s WHERE '+where,(item['after'],*item['keys'].values())).rowcount)
    for item in plan['metadata_updates']:
        where=' AND '.join(k+'=%s' for k in item['keys'])
        old=db.execute('SELECT '+','.join(item['before'])+' FROM '+item['table']+' WHERE '+where+' FOR UPDATE',tuple(item['keys'].values())).fetchone()
        if comparable(old)!=item['before']:raise ValueError('Prior description changed')
        tick(db.execute('UPDATE '+item['table']+' SET '+','.join(k+'=%s' for k in item['after'])+' WHERE '+where,(*item['after'].values(),*item['keys'].values())).rowcount)
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
            or gates['validator_sha256']!=sha(ROOT/'scripts/validate_complete_tabular_provider_publication.py')
            or gates['fault_count']!=250): raise ValueError('Matching complete publication gates required')
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
    path=PLAN if action=='plan' else ROOT/('docs/reviews/complete_tabular_provider_publication_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_written=True,providers=10) if action=='plan' else result))
