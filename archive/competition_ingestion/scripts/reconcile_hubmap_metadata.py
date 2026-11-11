"""Repair reviewed HuBMAP metadata with exact-state guards and rollback qualification."""
import argparse
import json
import subprocess
from pathlib import Path
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.import_residual_execution_drafts import ensure_row, comparable
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.review_hubmap_provider_contracts import review, READINESS
from scripts.audit_hubmap_original_readiness import sha

RUNNER = 'hubmap-reviewed-metadata.v1'
COMMIT = '73e673280320835527e9e55c4f8bdf127a129728'
REPORT = 'docs/reviews/hubmap_provider_contracts.json'
GATES = ROOT/'docs/reviews/hubmap_metadata_gates.json'


def qualified():
    if not isinstance(COMMIT,str) or len(COMMIT)!=40 or any(c not in '0123456789abcdef' for c in COMMIT):
        raise ValueError('Committed complete-graph provider contract review required')
    raw = subprocess.check_output(['git','show',COMMIT+':'+REPORT], cwd=ROOT)
    if raw != (ROOT/REPORT).read_bytes() or json.loads(raw) != review(json.loads(raw)['evidence_commit']):
        raise ValueError('Immutable semantic qualification changed')
    semantic = json.loads(raw)
    records = json.loads((ROOT/READINESS).read_text())['records']
    by_name = {r['runtime_fqdn']:r for r in semantic['providers']}
    targets = []
    for record in records[1:]:
        identity = record['identity']
        contract = by_name.get(identity['fqdn'], semantic['derivative'])
        targets.append(dict(identity=identity, catalogs=record['catalogs'], contract=contract))
    return semantic, targets, records[0]


def snapshot(db, records, *, preserved=False):
    ids = [UUID(r['identity']['artifact_id']) for r in records]
    result = {}
    for prefix, key in [('artifact','artifact_id'), ('atom','atom_id')]:
        suffixes = ['versions','references','audit_rollups']
        if prefix=='atom':suffixes.append('audit_evidence')
        if not preserved:
            suffixes += ['descriptions','parameters','io_specs']
        for suffix in suffixes:
            table = prefix+'_'+suffix
            result[table] = comparable(db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
        projection = "to_jsonb(t)-'updated_at'" if preserved else 'to_jsonb(t)'
        result[prefix+'s'] = comparable(db.execute(f'SELECT {projection} AS row FROM {prefix}s t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
    for table in ('artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_dependencies'):
        key = 'dependent_version_id' if table.endswith('dependencies') else 'version_id'
        result[table] = comparable(db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key} IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s)) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
    if not preserved:
        result['artifact_audit_evidence'] = comparable(db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall())
    return result


def check_identity(db, target):
    r = target['identity']
    for prefix, key in [('artifact','artifact_id'), ('atom','atom_id')]:
        if prefix not in target['catalogs']:
            continue
        actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
            FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''', (r['artifact_id'],)).fetchall()
        wanted = {k:r[k] for k in ('fqdn','status','is_publishable','version_id','content_hash','trust_tier')}
        if actual != [wanted]:
            raise ValueError('Retained provider or graph identity changed')
        if db.execute(f'SELECT 1 FROM {prefix}_audit_evidence WHERE version_id=%s AND NOT passed',(r['version_id'],)).fetchone():
            raise ValueError('Failed retained audit in '+prefix+' catalog')
    if r['artifact_kind'] == 'atom':
        module, symbol = r['fqdn'].rsplit('.',1)
        actual = db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s', (r['artifact_id'],)).fetchone()
        if actual != dict(import_module=module, source_symbol=symbol):
            raise ValueError('Runtime binding changed')


class InjectedFailure(RuntimeError):
    pass


def parameter_row(port,contract,semantic):
    """Preserve each callable's requiredness, default, position and all outputs."""
    if port['direction']!='input':raise ValueError('Input parameter required')
    outputs=[dict(name=p['name'],type_desc=p['type_desc'],constraints=p['constraints'])
             for p in sorted(contract['ports'],key=lambda p:p['ordinal']) if p['direction']=='output']
    if not outputs:raise ValueError('Complete output contracts required')
    return dict(type_desc=port['type_desc'],required=port['required'],
        default_value_repr=port['default_value_repr'],position=port['ordinal'],kind='positional_or_keyword',
        technical_description=port['constraints'],dejargonized_description=contract['plain_description'],
        constraints_json=dict(input_contract=port['constraints'],output_contracts=outputs,
            validity_bounds=semantic['limitations'],reuse=semantic['reuse_decision']))


def reconcile(apply=False, fail_after=None):
    semantic, targets, original = qualified()
    if apply:
        gates = json.loads(GATES.read_text())
        if gates.get('passed') is not True or gates['script_sha256'] != sha(Path(__file__)) or gates['review_sha256'] != sha(ROOT/REPORT):
            raise ValueError('Current rollback gates required')
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        for target in targets:
            db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE', (target['identity']['artifact_id'],)).fetchone()
            if 'atom' in target['catalogs']:
                db.execute('SELECT atom_id FROM atoms WHERE atom_id=%s FOR UPDATE',(target['identity']['artifact_id'],)).fetchone()
            check_identity(db, target)
        preserved = snapshot(db, targets, preserved=True)
        original_before = snapshot(db, [original])
        def changed(count):
            nonlocal writes
            writes += count
            if fail_after is not None and writes >= fail_after:
                raise InjectedFailure('Injected metadata rollback')
        for target in targets:
            r, contract = target['identity'], target['contract']
            plain = contract['plain_description']
            score = round(estimate_jargon_score(plain),6)
            if score >= .4:
                raise ValueError('Plain description exceeds native jargon bound')
            for prefix, key in [('artifact','artifact_id'), ('atom','atom_id')]:
                if prefix not in target['catalogs']:
                    continue
                keys = {key:r['artifact_id'], 'kind':'dejargonized', 'language':'en'}
                changed(int(ensure_row(db,prefix+'_descriptions',keys,dict(keys,content=plain,generated_by=RUNNER,reviewed=False,jargon_score=score))))
                for port in contract['ports']:
                    keys = {key:r['artifact_id'],'version_id':r['version_id'],'direction':port['direction'],'name':port['name']}
                    where = ' AND '.join(k+'=%s' for k in keys)
                    old = db.execute('SELECT direction,name,type_desc,ordinal,required,default_value_repr,constraints FROM '+prefix+'_io_specs WHERE '+where, tuple(keys.values())).fetchone()
                    baseline = next(p for p in target['catalogs'][prefix]['ports'] if p['direction']==port['direction'] and p['name']==port['name'])
                    if old not in (baseline, port):
                        raise ValueError('Unexpected port metadata drift')
                    if old != port:
                        changed(db.execute('UPDATE '+prefix+'_io_specs SET constraints=%s WHERE '+where, (port['constraints'],*keys.values())).rowcount)
                    if port['direction'] == 'input':
                        keys = {key:r['artifact_id'],'version_id':r['version_id'],'name':port['name']}
                        row=parameter_row(port,contract,semantic)
                        row['constraints_json']=Jsonb(row['constraints_json'])
                        changed(int(ensure_row(db,prefix+'_parameters',keys,dict(keys,**row))))
                if db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes', (r['artifact_id'],)).fetchone()['yes'] is not True:
                    raise ValueError('Native eligibility remains false')
            evidence_id = uuid5(UUID(r['version_id']),RUNNER)
            changed(int(ensure_row(db,'artifact_audit_evidence',dict(evidence_id=evidence_id),dict(evidence_id=evidence_id,
                artifact_id=r['artifact_id'],version_id=r['version_id'],audit_type='semantic_audit',passed=True,status='completed',
                source_kind='automated',runner_version=RUNNER,details=Jsonb(dict(review_commit=COMMIT,review_sha256=sha(ROOT/REPORT),
                    publisher_sha256=sha(Path(__file__)),trust_tier=3,scope='Existing provider and derivative metadata only; original approval remains pending.'))))))
        if preserved != snapshot(db, targets, preserved=True) or original_before != snapshot(db,[original]):
            raise ValueError('Preserved source, graph or approval state changed')
        if fail_after is not None:
            raise ValueError('Fault point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, writes=writes, native_eligible_artifacts=len(targets),
                native_eligible_atoms=sum('atom' in t['catalogs'] for t in targets),
                original_unchanged=True,serialized_graph_and_versions_unchanged=True,trust_tier=3)


def validate():
    _, targets, original = qualified()
    with connect() as db:
        before = snapshot(db, targets+[original])
    rejected = []
    with connect() as db:
        for target in targets:
            r = target['identity']
            faults = []
            for prefix,key in [('artifact','artifact_id'),('atom','atom_id')]:
                if prefix not in target['catalogs']:continue
                rejected_status = 'draft' if prefix == 'artifact' else 'flagged'
                faults.extend([
                    (prefix+'_hash',f'UPDATE {prefix}_versions SET content_hash=%s WHERE version_id=%s',('0'*64,r['version_id'])),
                    (prefix+'_tier',f'UPDATE {prefix}_versions SET trust_tier=1 WHERE version_id=%s',(r['version_id'],)),
                    (prefix+'_latest',f'UPDATE {prefix}_versions SET is_latest=false WHERE version_id=%s',(r['version_id'],)),
                    (prefix+'_status',f"UPDATE {prefix}s SET status=%s WHERE {key}=%s",(rejected_status,r['artifact_id'])),
                    (prefix+'_publishable',f'UPDATE {prefix}s SET is_publishable=false WHERE {key}=%s',(r['artifact_id'],)),
                    (prefix+'_failed_audit',f"INSERT INTO {prefix}_audit_evidence({key},version_id,audit_type,passed,status,source_kind,runner_version) VALUES(%s,%s,'semantic_audit',false,'completed','automated','synthetic-rollback-test')",(r['artifact_id'],r['version_id'])),
                ])
            if r['artifact_kind']=='atom':
                faults.append(('runtime',"UPDATE atoms SET source_symbol='incorrect' WHERE atom_id=%s",(r['artifact_id'],)))
            for name, sql, args in faults:
                db.execute('SAVEPOINT corrupted')
                try:
                    if db.execute(sql,args).rowcount != 1:
                        raise ValueError('Fault did not modify exactly one target')
                    try:
                        check_identity(db,target)
                    except ValueError:
                        rejected.append(r['fqdn']+':'+name)
                    else:
                        raise ValueError('Corrupted identity was accepted')
                finally:
                    db.execute('ROLLBACK TO SAVEPOINT corrupted')
                    db.execute('RELEASE SAVEPOINT corrupted')
        db.rollback()
    dry = reconcile()
    points = [1, dry['writes']//2, dry['writes']]
    for point in points:
        with connect() as db:
            if snapshot(db, targets+[original]) != before:
                raise ValueError('Dry run survived rollback')
        try:
            reconcile(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Expected injected failure')
        with connect() as db:
            if snapshot(db, targets+[original]) != before:
                raise ValueError('Injected changes survived rollback')
    return dict(passed=True,dry_run=dry,rollback_points=points,rejected_faults=rejected,catalog_mutations=0,
                script_sha256=sha(Path(__file__)),review_sha256=sha(ROOT/REPORT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validate','apply','repeat'])
    action = parser.parse_args().action
    result = validate() if action=='validate' else reconcile(apply=True)
    if action=='repeat' and result['writes']:
        raise ValueError('Repeat changed metadata')
    path = GATES if action=='validate' else ROOT/('docs/reviews/hubmap_metadata_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
