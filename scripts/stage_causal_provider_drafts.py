"""Freeze and atomically stage the 15 missing causal providers without approval."""
import argparse
from collections import Counter
import importlib
import inspect
import json
from uuid import UUID,uuid5,NAMESPACE_URL
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.causal_decomposed_features import build_explicit_causal_feature_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import ensure_row,comparable
from scripts.plan_flavours_provider_intake import sha
from scripts.review_causal_provider_interfaces import review,interfaces,ROOT

PLAN=ROOT/'docs/reviews/causal_provider_intake_plan.json'
GATES=ROOT/'docs/reviews/causal_provider_intake_transactions.json'
RUNNER='causal-decomposed-provider-intake.v1'


def parsed():
    result={}
    for name in ['sciona-atoms','sciona-atoms-ml']:
        directory=ROOT.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name,directory),artifact_root=directory/'src/sciona/atoms'):
            result[spec.import_module+'.'+spec.source_symbol]=spec
    return result


def existing_history(db,identities):
    result={}
    for table,key in [('atoms','atom_id'),('atom_versions','atom_id'),('atom_io_specs','atom_id'),
                      ('artifacts','artifact_id'),('artifact_versions','artifact_id'),('artifact_io_specs','artifact_id'),
                      ('artifact_audit_evidence','artifact_id')]:
        rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text',
            ([UUID(i) for i in identities],)).fetchall()
        import hashlib
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def plan():
    current=review();specs=parsed()
    candidates=[p for p in current['providers'] if not p['canonical_latest'] and not p['legacy_latest']]
    if len(candidates)!=15:raise ValueError('Expected 15 missing providers')
    records=[];owners={}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for repo in ['sciona-atoms','sciona-atoms-ml']:
            rows=db.execute('SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s',(repo,)).fetchall()
            if len(rows)!=1:raise ValueError('Unique ownership required for '+repo)
            owners[repo]=rows[0]
        for record in candidates:
            spec=specs[record['runtime_fqdn']]
            for table in ['atoms','artifacts']:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s',(spec.fqdn,)).fetchone():raise ValueError('Existing identity cannot be staged as new')
            for table in ['atom_versions','artifact_versions']:
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s',(spec.version_id,)).fetchone():raise ValueError('Version identity collision')
            fields={key:getattr(spec,key) for key in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path',
                'import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
            records.append(dict(record,atom_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),
                seed_fields=fields,semver=spec.semver,fingerprint=comparable(spec.fingerprint)))
        existing=[p for p in current['providers'] if p not in candidates]
        identities=[]
        for record in existing:
            row=db.execute('SELECT atom_id::text FROM atoms WHERE fqdn=%s',(record['catalog_fqdn'],)).fetchone()
            if not row:raise ValueError('Expected preserved legacy identity')
            identities.append(row['atom_id'])
        for fqdn in ['cdg.competition.solution.kaggle.cause_effect_2nd','cdg.competition.solution.kaggle.cause_effect_2nd.execution']:
            row=db.execute('SELECT artifact_id::text FROM artifacts WHERE fqdn=%s',(fqdn,)).fetchone()
            if not row:raise ValueError('Missing original/derivative anchor')
            identities.append(row['artifact_id'])
        preserved=existing_history(db,identities)
    return dict(approved=False,catalog_mutations=0,new_providers=records,owners=owners,
        graph_digest=current['graph_digest'],execution_evidence_sha256=current['execution_evidence_sha256'],
        interface_review_sha256=sha(ROOT/'docs/reviews/causal_provider_interface_review.json'),
        preserved_identities=identities,preserved_history=preserved,
        limitations=['Unpublished drafts only; semantic review and promotion remain separate',
                    'Existing providers, original and derivative records are protected by history digests'])


def qualified_plan():
    result=json.loads(PLAN.read_text())
    evidence_path=ROOT/'docs/reviews/causal_full_decomposition_source_parity.json'
    if sha(evidence_path)!=result['execution_evidence_sha256'] or sha(ROOT/'docs/reviews/causal_provider_interface_review.json')!=result['interface_review_sha256']:
        raise ValueError('Qualification evidence drift')
    evidence=json.loads(evidence_path.read_text())
    graph=build_explicit_causal_feature_graph()
    if not evidence['passed'] or encode_execution_graph(graph)[0]!=result['graph_digest']:raise ValueError('Graph qualification drift')
    for path,digest in evidence['implementation_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Implementation drift')
    for module,digest in evidence['provider_hashes'].items():
        if sha(inspect.getfile(importlib.import_module(module)))!=digest:raise ValueError('Provider source drift')
    specs=parsed();grouped={}
    for node in graph.nodes:grouped.setdefault(node.matched_primitive,[]).append(node)
    if len(result['new_providers'])!=15:raise ValueError('Frozen intake count changed')
    for record in result['new_providers']:
        spec=specs[record['runtime_fqdn']]
        if (str(spec.version_id)!=record['version_id'] or spec.content_hash!=record['content_hash']
                or sha(spec.file_path)!=record['provider_sha256'] or comparable(spec.fingerprint)!=record['fingerprint']
                or str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn))!=record['atom_id']
                or comparable({k:getattr(spec,k) for k in record['seed_fields']})!=record['seed_fields']):
            raise ValueError('Exact provider identity/source drift')
        function=getattr(importlib.import_module(spec.import_module),spec.source_symbol)
        if interfaces(function,grouped.get(record['runtime_fqdn'],[]))!=record['interfaces']:raise ValueError('Provider interface drift')
    return result


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    reviewed=qualified_plan();writes=0;counts=Counter()
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        if existing_history(db,reviewed['preserved_identities'])!=reviewed['preserved_history']:raise ValueError('Protected history drift')
        for repo,owner in reviewed['owners'].items():
            rows=db.execute('SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s',(repo,)).fetchall()
            if rows!=[owner]:raise ValueError('Provider ownership drift')
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        def ensure(table,key,row):
            nonlocal writes
            count=int(ensure_row(db,table,key,row));writes+=count;counts[table]+=count
            if fail_after is not None and writes==fail_after:raise InjectedFailure('Injected failure')
        for record in reviewed['new_providers']:
            identity=UUID(record['atom_id']);version=UUID(record['version_id'])
            owner=reviewed['owners'][record['seed_fields']['repo_name']]
            fields=dict(record['seed_fields'],source_package=record['seed_fields']['namespace_root'],status='flagged',is_publishable=False)
            atom=AtomSeedRow(**fields).as_dict(**owner);atom['atom_id']=identity
            ensure('atoms',{'atom_id':identity},atom)
            canonical={k:v for k,v in atom.items() if k in columns}
            canonical.update(artifact_id=identity,artifact_kind='atom',status='draft',is_publishable=False)
            ensure('artifacts',{'artifact_id':identity},canonical)
            for table,key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
                ensure(table,{'version_id':version},dict(version_id=version,**{key:identity},content_hash=record['content_hash'],
                    semver=record['semver'],is_latest=True,trust_tier=3,s3_key='',fingerprint=record['fingerprint']))
            for table,key in [('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id')]:
                for port in record['interfaces']:
                    native=dict(port);nullable=native.pop('nullable_annotation',False)
                    native['constraints']='Exact native signature and runtime contract; nullable annotation: '+str(nullable)+'.'
                    row=dict(native,**{key:identity},version_id=version)
                    ensure(table,{k:row[k] for k in [key,'version_id','direction','name']},row)
            eid=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':eid},dict(evidence_id=eid,artifact_id=identity,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(runtime_fqdn=record['runtime_fqdn'],plan_sha256=sha(PLAN),
                    source_sha256=record['provider_sha256'],interfaces=record['interfaces'],dependency_runtimes=record['dependencies'],
                    scope='Exact unpublished provider draft intake; not semantic approval'))))
        identities=[UUID(r['atom_id']) for r in reviewed['new_providers']]
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s)',(identities,)).fetchone():raise ValueError('Draft unexpectedly served')
        if db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=ANY(%s)',(identities,)).fetchone():raise ValueError('Legacy draft unexpectedly served')
        if existing_history(db,reviewed['preserved_identities'])!=reviewed['preserved_history']:raise ValueError('Protected history changed')
        if fail_after is not None:raise ValueError('Failure point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,providers=15,rows_created=writes,counts=dict(counts),
        served_drafts=0,protected_history_preserved=True,plan_sha256=sha(PLAN))


def assert_absent():
    reviewed=qualified_plan();ids=[UUID(r['atom_id']) for r in reviewed['new_providers']]
    with connect() as db:
        for table,key in [('atoms','atom_id'),('artifacts','artifact_id'),('atom_versions','atom_id'),('artifact_versions','artifact_id'),
                          ('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id'),('artifact_audit_evidence','artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone():raise ValueError('Rollback left rows')
        if existing_history(db,reviewed['preserved_identities'])!=reviewed['preserved_history']:raise ValueError('Rollback changed protected history')


def validate():
    assert_absent();dry=stage();assert_absent()
    points=[3,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected rollback')
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
            raise ValueError('Matching transaction qualification required')
        result=stage(apply=True)
        if args.action=='repeat' and result['rows_created']:raise ValueError('Repeat not idempotent')
        path=ROOT/('docs/reviews/causal_provider_intake_'+args.action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'new_providers','owners','preserved_identities','preserved_history'}},indent=2))
