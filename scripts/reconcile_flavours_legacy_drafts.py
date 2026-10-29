"""Stage missing canonical mirrors of six existing exact legacy providers."""
import argparse
import ast
import hashlib
import inspect
import importlib
import json
import textwrap
from pathlib import Path
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import connect,ROOT
from scripts.import_residual_execution_drafts import ensure_row,comparable
from scripts.plan_flavours_provider_intake import sha,kind
from typing import get_type_hints

PLAN=ROOT/'docs/reviews/flavours_legacy_reconciliation_plan.json'
RUNNER='flavours-legacy-canonical-draft.v1'


def history(db,identity):
    result={}
    for table in ('atoms','atom_versions','atom_io_specs'):
        rows=db.execute(f"SELECT to_jsonb(t) AS row FROM {table} t WHERE atom_id=%s ORDER BY to_jsonb(t)::text",(identity,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def served_history(db,identity):
    rows=db.execute('SELECT to_jsonb(t) AS row FROM catalog_artifacts_served t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',(identity,)).fetchall()
    return dict(rows=len(rows),sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest())


def ports(function):
    original=inspect.unwrap(function);signature=inspect.signature(function);hints=get_type_hints(function)
    definition=next(n for n in ast.parse(textwrap.dedent(inspect.getsource(original))).body if isinstance(n,ast.FunctionDef))
    annotations={a.arg:ast.unparse(a.annotation) for a in (*definition.args.args,*definition.args.kwonlyargs)}
    result=[]
    for ordinal,(name,p) in enumerate(signature.parameters.items()):
        try:description=kind(hints[name])
        except ValueError:description=annotations[name]
        result.append(dict(direction='input',name=name,ordinal=ordinal,type_desc=description,
            required=p.default is inspect.Parameter.empty,default_value_repr='' if p.default is inspect.Parameter.empty else repr(p.default),
            constraints='Exact native signature. Flavours qualification covers the explicit dense finite/runtime regime; broader API review remains separate.'))
    output={'compute_ks_agreement':'ks','compute_cvm_mass_decorrelation':'cvm','roc_auc_truncated_weighted':'auc',
        'standard_scaler_fit':'state','standard_scaler_partial_fit':'state','standard_scaler_transform':'transformed'}[original.__name__]
    try:description=kind(hints['return'])
    except ValueError:description=ast.unparse(definition.returns)
    result.append(dict(direction='output',name=output,ordinal=0,type_desc=description,required=True,default_value_repr='',
        constraints='Numeric metric, native scaler state, or transformed matrix according to the exact implementation.'))
    return result


def plan():
    prior=json.loads((ROOT/'docs/reviews/flavours_provider_intake_plan.json').read_text())
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for candidate in prior['existing_providers']:
            if candidate['canonical_latest']:continue
            atom=db.execute('SELECT * FROM atoms WHERE fqdn=%s',(candidate['catalog_fqdn'],)).fetchone()
            version=db.execute('SELECT * FROM atom_versions WHERE version_id=%s',(candidate['version_id'],)).fetchone()
            if not atom or not version or version['atom_id']!=atom['atom_id'] or version['content_hash']!=candidate['content_hash'] or not version['is_latest'] or version['trust_tier']!=3:
                raise ValueError('Exact existing legacy version required')
            if db.execute('SELECT 1 FROM artifacts WHERE artifact_id=%s OR fqdn=%s',(atom['atom_id'],atom['fqdn'])).fetchone():raise ValueError('Canonical identity already exists')
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(version['version_id'],)).fetchone():raise ValueError('Canonical version collision')
            module,name=candidate['runtime_fqdn'].rsplit('.',1);function=getattr(importlib.import_module(module),name)
            if sha(inspect.getsourcefile(inspect.unwrap(function)))!=candidate['provider_sha256']:raise ValueError('Provider source changed')
            fields={key:atom[key] for key in ['fqdn','namespace_root','namespace_path','source_module_path','source_symbol',
                'description','source_kind','is_ffi','owner_id','source_repo_id','source_package',
                'visibility_tier','stateful_kind','is_stochastic']}
            fields.update(artifact_id=atom['atom_id'],artifact_kind='atom',status='draft',is_publishable=False)
            v={key:version[key] for key in ['version_id','content_hash','semver','is_latest','trust_tier','s3_key','fingerprint','derives_from']}
            v['artifact_id']=atom['atom_id']
            records.append(dict(runtime=candidate['runtime_fqdn'],artifact_id=str(atom['atom_id']),version_id=candidate['version_id'],
                source_sha256=candidate['provider_sha256'],artifact=comparable(fields),version=comparable(v),ports=ports(function),
                legacy_history_sha256=history(db,atom['atom_id']),served_before=served_history(db,atom['atom_id'])))
    if len(records)!=6:raise ValueError('Expected six missing canonical mirrors')
    return dict(passed=True,approved=False,catalog_mutations=0,providers=records,
        execution_evidence_sha256=sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'),
        limitations=['Unpublished canonical drafts only. Legacy identity, versions, ports and publication states remain unchanged.',
            'Existing native scaler interfaces include broader modes than the qualified dense finite Flavours use. Publication requires explicit validity bounds.'])


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    reviewed=json.loads(PLAN.read_text());writes=0
    if reviewed['execution_evidence_sha256']!=sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'):raise ValueError('Execution evidence drift')
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        def ensure(table,key,row):
            nonlocal writes
            writes+=int(ensure_row(db,table,key,row))
            if fail_after is not None and writes==fail_after:raise InjectedFailure('Injected mirror intake failure')
        for record in reviewed['providers']:
            identity=UUID(record['artifact_id']);version=UUID(record['version_id'])
            if history(db,identity)!=record['legacy_history_sha256']:raise ValueError('Legacy history changed')
            module,name=record['runtime'].rsplit('.',1);function=getattr(importlib.import_module(module),name)
            if sha(inspect.getsourcefile(inspect.unwrap(function)))!=record['source_sha256'] or ports(function)!=record['ports']:raise ValueError('Exact provider interface/source changed')
            ensure('artifacts',{'artifact_id':identity},record['artifact'])
            ensure('artifact_versions',{'version_id':version},record['version'])
            for port in record['ports']:
                row=dict(port,artifact_id=identity,version_id=version)
                ensure('artifact_io_specs',{k:row[k] for k in ('artifact_id','version_id','direction','name')},row)
            evidence_id=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':evidence_id},dict(evidence_id=evidence_id,artifact_id=identity,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(scope='Canonical draft mirror under preserved legacy identity; not approval.',
                    plan_sha256=sha(PLAN),source_sha256=record['source_sha256'],legacy_history_sha256=record['legacy_history_sha256']))))
            if history(db,identity)!=record['legacy_history_sha256']:raise ValueError('Legacy history was modified')
            # The union view also serves already-approved legacy atoms. Preserve
            # those rows exactly; the new canonical draft must add no rows.
            if served_history(db,identity)!=record['served_before']:raise ValueError('Existing serving changed')
            if db.execute("SELECT 1 FROM artifacts WHERE artifact_id=%s AND (status<>'draft' OR is_publishable)",(identity,)).fetchone():raise ValueError('Canonical mirror is not an unpublished draft')
        if fail_after is not None:raise ValueError('Failure injection point was not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,canonical_drafts=6,rows_created=writes,legacy_history_preserved=True,existing_serving_preserved=True,newly_served_rows=0,plan_sha256=sha(PLAN))


def validate():
    reviewed=json.loads(PLAN.read_text())
    def absent():
        with connect() as db:
            if db.execute('SELECT 1 FROM artifacts WHERE artifact_id=ANY(%s)',([UUID(r['artifact_id']) for r in reviewed['providers']],)).fetchone():raise ValueError('Rollback left canonical drafts')
    absent();dry=stage();absent();points=[3,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected rollback')
        absent()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,fault_after_writes=points,all_rollbacks_complete=True,
        importer_sha256=sha(__file__),plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    args=parser.parse_args()
    if args.action=='plan':result=plan();path=PLAN
    elif args.action=='validate':result=validate();path=ROOT/'docs/reviews/flavours_legacy_reconciliation_transactions.json'
    else:
        gates=json.loads((ROOT/'docs/reviews/flavours_legacy_reconciliation_transactions.json').read_text())
        if not gates['passed'] or gates['plan_sha256']!=sha(PLAN) or gates['importer_sha256']!=sha(__file__):raise ValueError('Matching transaction qualification required')
        result=stage(apply=True)
        if args.action=='repeat' and result['rows_created']!=0:raise ValueError('Repeat was not idempotent')
        path=ROOT/('docs/reviews/flavours_legacy_reconciliation_'+args.action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='providers'}))
