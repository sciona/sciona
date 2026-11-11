"""Stage frozen new-provider drafts atomically, without publication changes."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
from uuid import UUID,uuid5,NAMESPACE_URL
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.plan_flavours_provider_intake import ROOT,sha,interfaces
from scripts.import_residual_execution_drafts import ensure_row,comparable

PLAN=ROOT/'docs/reviews/flavours_provider_intake_plan.json'
RUNNER='flavours-decomposed-provider-intake.v1'


class InjectedFailure(RuntimeError):pass


def reviewed_plan():
    plan=json.loads(PLAN.read_text());evidence=ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'
    if sha(evidence)!=plan['execution_evidence_sha256']:raise ValueError('Execution evidence drift')
    report=json.loads(evidence.read_text());graph=build_evaluated_graph()
    if not report['passed'] or encode_execution_graph(graph)[0]!=plan['graph_digest']:raise ValueError('Qualified graph drift')
    for path,digest in report['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Qualified execution source changed')
    directory=ROOT.parent/'sciona-atoms-ml'
    specs={s.import_module+'.'+s.source_symbol:s for s in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms')}
    grouped={}
    for node in graph.nodes:grouped.setdefault(node.matched_primitive,[]).append(node)
    if len(plan['new_providers'])!=31:raise ValueError('Expected frozen 31-provider intake')
    for record in plan['new_providers']:
        spec=specs[record['runtime_fqdn']]
        if (str(spec.version_id)!=record['version_id'] or spec.content_hash!=record['content_hash']
                or sha(spec.file_path)!=record['provider_sha256'] or comparable(spec.fingerprint)!=record['fingerprint']
                or str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn))!=record['atom_id']):
            raise ValueError('Provider source/version/identity drift')
        if comparable({k:getattr(spec,k) for k in record['seed_fields']})!=record['seed_fields']:raise ValueError('Provider seed fields changed')
        function=getattr(importlib.import_module(spec.import_module),spec.source_symbol)
        if interfaces(function,grouped.get(record['runtime_fqdn'],[]))!=record['interfaces']:raise ValueError('Provider interface drift')
    return plan


def connect():
    return psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
        options='-c statement_timeout=30000 -c lock_timeout=10000')


def intake(apply=False,fail_after=None):
    plan=reviewed_plan();counts=Counter();writes=0
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('flavours-decomposed-provider-intake.v1'))")
        owners=db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if owners!=[plan['owner']]:raise ValueError('Provider ownership changed')
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        def ensure(table,key,row):
            nonlocal writes
            changed=ensure_row(db,table,key,row);counts[table]+=int(changed);writes+=int(changed)
            if fail_after is not None and writes==fail_after:raise InjectedFailure('Injected failure after '+str(writes)+' writes')
        for record in plan['new_providers']:
            identity=UUID(record['atom_id']);version=UUID(record['version_id'])
            fields=dict(record['seed_fields'],source_package=record['seed_fields']['namespace_root'],status='flagged',is_publishable=False)
            atom=AtomSeedRow(**fields).as_dict(owner_id=plan['owner']['owner_id'],source_repo_id=plan['owner']['source_repo_id'])
            atom['atom_id']=identity
            ensure('atoms',{'atom_id':identity},atom)
            canonical={k:v for k,v in atom.items() if k in columns}
            canonical.update(artifact_id=identity,artifact_kind='atom',status='draft',is_publishable=False)
            ensure('artifacts',{'artifact_id':identity},canonical)
            for table,key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
                ensure(table,{'version_id':version},dict(version_id=version,**{key:identity},content_hash=record['content_hash'],
                    semver=record['semver'],is_latest=True,trust_tier=3,s3_key='',fingerprint=record['fingerprint']))
            for table,key in [('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id')]:
                for port in record['interfaces']:
                    row={key:identity,'version_id':version,**port}
                    ensure(table,{k:row[k] for k in [key,'version_id','direction','name']},row)
            evidence_id=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':evidence_id},dict(evidence_id=evidence_id,artifact_id=identity,
                version_id=version,audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(runtime_fqdn=record['runtime_fqdn'],source_sha256=record['provider_sha256'],
                    interfaces=record['interfaces'],plan_sha256=sha(PLAN),execution_evidence_sha256=plan['execution_evidence_sha256'],
                    dependency_runtimes=record['dependencies'],scope='Exact source/version/interface draft intake only; no approval or serving.'))))
        identities=[UUID(r['atom_id']) for r in plan['new_providers']]
        for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)',(identities,)).fetchone():raise ValueError('Draft unexpectedly served')
        if fail_after is not None:raise ValueError('Injected failure point was not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,providers=31,rows_created=writes,counts=dict(counts),served_drafts=0,
        plan_sha256=sha(PLAN),publication='pending_review')


def assert_absent(plan):
    ids=[UUID(r['atom_id']) for r in plan['new_providers']]
    with connect() as db:
        for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone():raise ValueError('Rollback left draft identities behind')


def validate_transactions():
    plan=reviewed_plan();assert_absent(plan)
    dry=intake();assert_absent(plan)
    points=[3,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:intake(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected transaction failure')
        assert_absent(plan)
    return dict(passed=True,catalog_mutations=0,dry_run=dry,fault_after_writes=points,all_rollbacks_complete=True,plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true');parser.add_argument('--validate-transactions',action='store_true')
    args=parser.parse_args()
    if args.apply and args.validate_transactions:parser.error('Choose validation or apply')
    if args.apply:
        check=json.loads((ROOT/'docs/reviews/flavours_provider_intake_transactions.json').read_text())
        if not check['passed'] or not check['all_rollbacks_complete'] or check['plan_sha256']!=sha(PLAN):
            raise ValueError('Matching successful transaction validation required')
    result=validate_transactions() if args.validate_transactions else intake(apply=args.apply)
    filename='flavours_provider_intake_transactions.json' if args.validate_transactions else 'flavours_provider_intake_applied.json' if args.apply else 'flavours_provider_intake_dry_run.json'
    (ROOT/'docs/reviews'/filename).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
