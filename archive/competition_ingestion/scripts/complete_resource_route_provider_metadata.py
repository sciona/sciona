"""Add reviewed route contracts without replacing existing ports or approval history."""
import argparse
import hashlib
import json
import subprocess
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb

from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.plan_flavours_provider_intake import sha
from scripts.review_resource_route_provider_semantics import review as current_review
from scripts.promote_openvaccine_decomposed_providers import snapshot, PORT_FIELDS
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

RUNNER = 'resource-route-provider-contract-metadata.v1'
PLAN = ROOT/'docs/reviews/resource_route_provider_metadata_plan.json'
GATES = ROOT/'docs/reviews/resource_route_provider_metadata_gates.json'
SEMANTICS = 'docs/reviews/resource_route_provider_semantics.json'
COMMIT = '51723116'
ANCHORS = ['65947cba-17e3-5a60-ac2b-7a3545885a39','3c10fde8-db29-5c97-bdbe-d173e7468201']


def qualification():
    raw = subprocess.check_output(['git','show',COMMIT+':'+SEMANTICS],cwd=ROOT)
    if raw != (ROOT/SEMANTICS).read_bytes() or json.loads(raw) != current_review():
        raise ValueError('Immutable reviewed provider qualification changed')
    return json.loads(raw)


def preserved(db, records):
    ids = [UUID(r['artifact_id']) for r in records]
    result = {}
    for table,key in [('atoms','atom_id'),('artifacts','artifact_id'),('atom_versions','atom_id'),
                      ('artifact_versions','artifact_id'),('atom_io_specs','atom_id'),
                      ('artifact_io_specs','artifact_id'),('atom_audit_rollups','atom_id'),
                      ('artifact_audit_rollups','artifact_id'),('artifact_audit_evidence','artifact_id')]:
        value = "to_jsonb(t)-'updated_at'" if table in ('atoms','artifacts') else 'to_jsonb(t)'
        where = ' AND runner_version IS DISTINCT FROM %s' if table=='artifact_audit_evidence' else ''
        args = (ids,RUNNER) if where else (ids,)
        rows = db.execute(f'SELECT {value} AS row FROM {table} t WHERE {key}=ANY(%s)'+where+f' ORDER BY {value}::text',args).fetchall()
        result[table] = hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    for version in ANCHORS:result[version] = history(db,version)
    return result


def check(db, records, require_eligible=False):
    for record in records:
        identity,version = record['artifact_id'],record['version_id']
        module,symbol = record['runtime_fqdn'].rsplit('.',1)
        if db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(identity,)).fetchone()!=dict(import_module=module,source_symbol=symbol):
            raise ValueError('Exact runtime binding changed')
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            rows = db.execute(f'SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest',(identity,)).fetchall()
            if rows != [dict(fqdn=record['catalog_fqdn'],status='approved',is_publishable=True,
                version_id=version,content_hash=record['content_hash'],trust_tier=3)]:
                raise ValueError('Exact approved source version changed')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
            if ports != record[prefix+'_ports']:raise ValueError('Retained native ports changed')
            eligible = db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes',(identity,)).fetchone()['yes']
            if require_eligible and not eligible:
                raise ValueError('Provider native eligibility failed')
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(identity,)).fetchone():
                raise ValueError('Provider is not served')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchone():
            raise ValueError('Unexpected registered provider dependency')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(version,)).fetchone():
            raise ValueError('Failed provider audit')


def build_plan():
    reviewed = qualification();records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for semantic in reviewed['providers']:
            identity = db.execute('SELECT artifact_id::text FROM artifact_versions WHERE version_id=%s',(semantic['version_id'],)).fetchone()['artifact_id']
            record = dict(semantic,artifact_id=identity)
            for prefix in ('atom','artifact'):
                ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(semantic['version_id'],)).fetchall()
                expected = [dict(p,constraints='') for p in sorted(semantic['ports'],key=lambda p:(p['direction'],p['ordinal']))]
                if ports!=expected:raise ValueError('Expected exact original interfaces with empty annotations')
                record[prefix+'_ports'] = ports
            records.append(record)
        if len(records)!=5 or len({r['artifact_id'] for r in records})!=5:raise ValueError('Five exact providers required')
        check(db,records)
        prior = preserved(db,records)
    return dict(passed=True,catalog_mutations=0,records=records,preserved=prior,
        qualification_sha256=sha(ROOT/SEMANTICS),writer_sha256=sha(__file__),
        contract_policy='Add versioned parameter constraints and audit evidence; preserve original IO rows and source versions. Graph usage contracts must be explicitly serialized and executed.')


def qualified_plan():
    reviewed=qualification();plan = json.loads(PLAN.read_text())
    if plan['qualification_sha256']!=sha(ROOT/SEMANTICS) or plan['writer_sha256']!=sha(__file__):
        raise ValueError('Metadata plan qualification changed')
    if [{k:v for k,v in r.items() if k not in ('artifact_id','atom_ports','artifact_ports')} for r in plan['records']] != reviewed['providers']:
        raise ValueError('Planned provider contracts differ from immutable review')
    for r in plan['records']:
        expected=[dict(p,constraints='') for p in sorted(r['ports'],key=lambda p:(p['direction'],p['ordinal']))]
        if r['atom_ports']!=expected or r['artifact_ports']!=expected:raise ValueError('Planned retained ports differ')
    return plan


class InjectedFailure(RuntimeError):pass


def apply(apply=False,fail_after=None):
    plan = qualified_plan();writes = 0
    if apply:
        gates=json.loads(GATES.read_text())
        if gates.get('passed') is not True or gates['plan_sha256']!=sha(PLAN) or gates['writer_sha256']!=sha(__file__):
            raise ValueError('Current rollback gates required')
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for record in plan['records']:
            db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(record['artifact_id'],)).fetchone()
        check(db,plan['records'])
        if preserved(db,plan['records'])!=plan['preserved']:raise ValueError('Retained history changed')
        def ensure(table,keys,row):
            nonlocal writes
            writes += int(ensure_row(db,table,keys,row))
            if writes==fail_after:raise InjectedFailure('Injected metadata transaction failure')
        for record in plan['records']:
            identity,version = UUID(record['artifact_id']),UUID(record['version_id'])
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                for kind,content in [('technical',record['plain_description']+' '+ ' '.join(p['constraints'] for p in record['ports'])),('dejargonized',record['plain_description'])]:
                    keys = {key:identity,'kind':kind,'language':'en'}
                    ensure(prefix+'_descriptions',keys,dict(keys,content=content,generated_by=RUNNER,reviewed=False,
                        jargon_score=round(estimate_jargon_score(content),6)))
                for port in record['ports']:
                    if port['direction']!='input':continue
                    keys = {key:identity,'version_id':version,'name':port['name']}
                    ensure(prefix+'_parameters',keys,dict(keys,type_desc=port['type_desc'],required=port['required'],
                        default_value_repr=port['default_value_repr'],position=port['ordinal'],kind='positional_or_keyword',
                        technical_description=port['constraints'],dejargonized_description=record['plain_description'],
                        constraints_json=Jsonb(dict(input_contract=port['constraints'],
                            output_contracts=[p for p in record['ports'] if p['direction']=='output'],
                            validity_bounds=record['validity_bounds'],review_source='automated',trust_tier=3))))
            evidence_id = uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=evidence_id),dict(evidence_id=evidence_id,artifact_id=identity,version_id=version,
                audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(qualification_sha256=plan['qualification_sha256'],plan_sha256=sha(PLAN),
                    contract=record,scope=plan['contract_policy']))))
        check(db,plan['records'],require_eligible=True)
        if preserved(db,plan['records'])!=plan['preserved']:raise ValueError('Metadata update changed retained history')
        if fail_after is not None:raise ValueError('Failure point was not reached')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=writes,approved_providers=5,native_eligibility_verified=True,ports_and_history_preserved=True,
        original_promoted=False,plan_sha256=sha(PLAN))


def validate():
    plan=qualified_plan();ids=[UUID(r['artifact_id']) for r in plan['records']]
    with connect() as db:before=snapshot(db,ids)
    def unchanged():
        with connect() as db:
            if snapshot(db,ids)!=before:raise ValueError('Rollback left provider changes')
            if preserved(db,plan['records'])!=plan['preserved']:raise ValueError('Rollback changed retained history')
            check(db,plan['records'])
    dry=apply();unchanged();points=[1,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:apply(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected transaction failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,injected_failure_points=points,
        all_rollbacks_complete=True,plan_sha256=sha(PLAN),writer_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    if action=='plan':result=build_plan();path=PLAN
    elif action=='validate':result=validate();path=GATES
    else:
        result=apply(apply=True)
        if action=='repeat' and result['rows_created']:raise ValueError('Repeated write changed rows')
        path=ROOT/('docs/reviews/resource_route_provider_metadata_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=5) if action=='plan' else result))
