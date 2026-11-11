"""Mirror exact legacy causal providers as canonical drafts without serving changes."""
import argparse
import importlib
import inspect
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import connect,ROOT
from scripts.import_residual_execution_drafts import ensure_row,comparable
from scripts.plan_flavours_provider_intake import sha
from scripts.reconcile_flavours_legacy_drafts import history,served_history
from scripts.review_causal_provider_interfaces import interfaces
from scripts.stage_causal_provider_drafts import qualified_plan

PLAN=ROOT/'docs/reviews/causal_legacy_reconciliation_plan.json'
GATES=ROOT/'docs/reviews/causal_legacy_reconciliation_transactions.json'
RUNNER='causal-legacy-canonical-draft.v1'


def native_ports(function):
    result=[]
    for item in interfaces(function,[]):
        row=dict(item);nullable=row.pop('nullable_annotation',False)
        row['constraints']='Exact native signature; nullable annotation: '+str(nullable)+'. Qualified graph modes require explicit validity bounds.'
        result.append(row)
    return result


def plan():
    qualified_plan()
    review_path=ROOT/'docs/reviews/causal_provider_interface_review.json'
    reviewed=json.loads(review_path.read_text())
    candidates=[p for p in reviewed['providers'] if not p['canonical_latest'] and p['legacy_latest']]
    if len(candidates)!=12:raise ValueError('Expected 12 legacy-only providers')
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for candidate in candidates:
            atom=db.execute('SELECT * FROM atoms WHERE fqdn=%s',(candidate['catalog_fqdn'],)).fetchone()
            version=db.execute('SELECT * FROM atom_versions WHERE version_id=%s',(candidate['version_id'],)).fetchone()
            if (not atom or not version or version['atom_id']!=atom['atom_id'] or version['content_hash']!=candidate['content_hash']
                    or not version['is_latest'] or version['trust_tier']!=3 or atom['status']!='approved' or not atom['is_publishable']):
                raise ValueError('Exact approved legacy version required')
            if db.execute('SELECT 1 FROM artifacts WHERE artifact_id=%s OR fqdn=%s',(atom['atom_id'],atom['fqdn'])).fetchone():raise ValueError('Canonical identity already exists')
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(version['version_id'],)).fetchone():raise ValueError('Canonical version collision')
            module,name=candidate['runtime_fqdn'].rsplit('.',1)
            function=getattr(importlib.import_module(module),name)
            if sha(inspect.getsourcefile(inspect.unwrap(function)))!=candidate['provider_sha256']:raise ValueError('Provider source drift')
            fields={key:atom[key] for key in ['fqdn','namespace_root','namespace_path','source_module_path','source_symbol',
                'description','source_kind','is_ffi','owner_id','source_repo_id','source_package','visibility_tier','stateful_kind','is_stochastic']}
            fields.update(artifact_id=atom['atom_id'],artifact_kind='atom',status='draft',is_publishable=False)
            v={key:version[key] for key in ['version_id','content_hash','semver','is_latest','trust_tier','s3_key','fingerprint','derives_from']}
            v['artifact_id']=atom['atom_id']
            records.append(dict(runtime=candidate['runtime_fqdn'],artifact_id=str(atom['atom_id']),version_id=candidate['version_id'],
                source_sha256=candidate['provider_sha256'],artifact=comparable(fields),version=comparable(v),ports=native_ports(function),
                legacy_history=history(db,atom['atom_id']),served_before=served_history(db,atom['atom_id'])))
    return dict(approved=False,catalog_mutations=0,providers=records,interface_review_sha256=sha(review_path),
        execution_evidence_sha256=reviewed['execution_evidence_sha256'],
        limitations=['Canonical drafts only; existing legacy approvals and serving preserved',
                    'Canonical output alias result is new metadata; existing legacy ports are retained unchanged'])


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    qualified_plan()
    reviewed=json.loads(PLAN.read_text());writes=0
    if sha(ROOT/'docs/reviews/causal_provider_interface_review.json')!=reviewed['interface_review_sha256']:
        raise ValueError('Interface evidence drift')
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        def ensure(table,key,row):
            nonlocal writes
            writes+=int(ensure_row(db,table,key,row))
            if fail_after is not None and writes==fail_after:raise InjectedFailure('Injected mirror failure')
        for record in reviewed['providers']:
            identity=UUID(record['artifact_id']);version=UUID(record['version_id'])
            if history(db,identity)!=record['legacy_history']:raise ValueError('Legacy history drift')
            module,name=record['runtime'].rsplit('.',1)
            function=getattr(importlib.import_module(module),name)
            if sha(inspect.getsourcefile(inspect.unwrap(function)))!=record['source_sha256'] or native_ports(function)!=record['ports']:
                raise ValueError('Exact source/interface drift')
            ensure('artifacts',{'artifact_id':identity},record['artifact'])
            ensure('artifact_versions',{'version_id':version},record['version'])
            for port in record['ports']:
                row=dict(port,artifact_id=identity,version_id=version)
                ensure('artifact_io_specs',{k:row[k] for k in ['artifact_id','version_id','direction','name']},row)
            eid=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':eid},dict(evidence_id=eid,artifact_id=identity,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(scope='Unpublished canonical draft mirror under unchanged legacy identity',plan_sha256=sha(PLAN),
                    source_sha256=record['source_sha256'],legacy_history=record['legacy_history']))))
            if history(db,identity)!=record['legacy_history'] or served_history(db,identity)!=record['served_before']:
                raise ValueError('Legacy history or serving changed')
        if fail_after is not None:raise ValueError('Failure point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,canonical_drafts=len(reviewed['providers']),rows_created=writes,
        legacy_history_preserved=True,existing_serving_preserved=True,newly_served_rows=0,plan_sha256=sha(PLAN))


def assert_absent():
    reviewed=json.loads(PLAN.read_text())
    ids=[UUID(r['artifact_id']) for r in reviewed['providers']]
    with connect() as db:
        for table in ['artifacts','artifact_versions','artifact_io_specs','artifact_audit_evidence']:
            if db.execute(f'SELECT 1 FROM {table} WHERE artifact_id=ANY(%s)',(ids,)).fetchone():raise ValueError('Rollback left canonical rows')
        for record in reviewed['providers']:
            identity=UUID(record['artifact_id'])
            if history(db,identity)!=record['legacy_history'] or served_history(db,identity)!=record['served_before']:
                raise ValueError('Rollback changed legacy history or serving')


def validate():
    assert_absent();dry=stage();assert_absent()
    points=[3,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected failure')
        assert_absent()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,fault_after_writes=points,
        all_rollbacks_complete=True,plan_sha256=sha(PLAN),importer_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat']);args=parser.parse_args()
    if args.action=='plan':result=plan();path=PLAN
    elif args.action=='validate':result=validate();path=GATES
    else:
        gates=json.loads(GATES.read_text())
        if not gates['passed'] or not gates['all_rollbacks_complete'] or gates['plan_sha256']!=sha(PLAN) or gates['importer_sha256']!=sha(__file__):
            raise ValueError('Matching transaction validation required')
        result=stage(apply=True)
        if args.action=='repeat' and result['rows_created']:raise ValueError('Repeat not idempotent')
        path=ROOT/('docs/reviews/causal_legacy_reconciliation_'+args.action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='providers'},indent=2))
