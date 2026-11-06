"""Stage nineteen implementation-reviewed recipe providers without approval.

Freeze exact source fingerprints and ordered interfaces, then validate rollback
and idempotence. Existing preparation and original intake remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import subprocess
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.decomposed_webtraffic_graph import build_decomposed_webtraffic_graph
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.import_flavours_provider_drafts import ROOT
from scripts.audit_decomposed_webtraffic_readiness import sha
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_future_sales_interfaces import interfaces

RUNNER = 'webtraffic-decomposed-provider-intake.v1'
PLAN = ROOT/'docs/reviews/webtraffic_decomposed_provider_intake_plan.json'
GATES = ROOT/'docs/reviews/webtraffic_decomposed_provider_intake_gates.json'
FIELDS = ('fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
          'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')
PRESERVED = [UUID(value) for value in (
    '7217a6e8-5fcf-58a0-b0b1-caad0e40a20a', '40e129cc-15bd-558c-b13c-1fea2320e2aa',
    '0742efc1-47cb-5327-8338-2b1616424b6c', 'e3e7bd54-486c-5e58-a748-8c8e0313766d',
    'dba8aea7-fc48-50ad-a652-0377d4f67fdf')]



def require(condition, message):
    if not condition: raise ValueError(message)


def review():
    relative = 'docs/reviews/decomposed_webtraffic_semantics.json'
    raw = (ROOT/relative).read_bytes()
    require(raw == subprocess.check_output(['git', 'show', '6e341888:'+relative], cwd=ROOT),
            'Committed semantic review changed')
    result = json.loads(raw)
    require(result['passed'] and result['tests'] == dict(tests=52,failures=0,errors=0,skipped=0),
            'Complete semantic qualification required')
    for path, digest in result['software_sha256'].items():
        file = ROOT.parent/path if path.split('/')[0] in ('sciona-atoms-dl','sciona-atoms-ml','sciona-atoms-signal') else ROOT/path
        require(sha(file) == digest, 'Qualified software changed')
    require(all(sha(ROOT/p) == h for p,h in result['evidence_sha256'].items()), 'Pinned evidence changed')
    require(encode_execution_graph(build_decomposed_webtraffic_graph())[0] == result['execution']['graph_sha256'],
            'Executed graph changed')
    for name,digest in result['execution']['provider_sha256'].items():
        module,symbol = name.rsplit('.',1)
        fn = getattr(importlib.import_module(module),symbol)
        require(sha(inspect.getfile(inspect.unwrap(fn))) == digest, 'Executed provider changed')
    return dict(new_providers=result['providers'],graph_digest=result['execution']['graph_sha256'])


def local_plan():
    qualification = review()
    graph = build_decomposed_webtraffic_graph()
    wanted = {r['runtime_fqdn']: r for r in qualification['new_providers']
              if r['runtime_fqdn'].startswith('sciona.atoms.dl.decomposed_webtraffic_execution.')}
    records = []
    for repo_name in ('sciona-atoms-dl',):
        directory = ROOT.parent/repo_name
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo_name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime not in wanted:
                continue
            record = wanted[runtime]
            require(str(spec.version_id) == record['version_id'] and spec.content_hash == record['content_hash']
                    and sha(spec.file_path) == record['provider_sha256'], 'Provider identity changed')
            function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
            nodes = [n for n in graph.nodes if n.matched_primitive == runtime]
            records.append(dict(record, atom_id=str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)),
                seed_fields=comparable({k: getattr(spec, k) for k in FIELDS}), semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), interfaces=interfaces(function, nodes[0])))
    require(len(records) == 19, 'Expected nineteen new providers')
    return dict(approved=False, graph_digest=qualification['graph_digest'],
        review_sha256=sha(ROOT/'docs/reviews/decomposed_webtraffic_semantics.json'),
        intake_script_sha256=sha(ROOT/'scripts/stage_webtraffic_decomposed_providers.py'),
        providers=sorted(records, key=lambda r: r['runtime_fqdn']),
        limitations=['Draft source/interface intake only; semantic publication and dependency gates remain.',
                    'Nineteen providers expose complete preparation, feature blocks, three-model training with callable updates and checkpoint retention, prediction and aligned output. Numeric correlations, normalization and suffix alignment are independently reusable; recipe assumptions are explicit. Original, derivative and provider histories are preserved.'])


def ownership(db):
    result = {}
    for repo in ('sciona-atoms-dl',):
        rows = db.execute('''SELECT DISTINCT a.owner_id::text,a.source_repo_id::text
            FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s''', (repo,)).fetchall()
        require(len(rows) == 1, 'Unique provider repository ownership required')
        result[repo] = rows[0]
    return result


def preserved(db):
    state = {}
    for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
        for suffix in ('s', '_versions', '_io_specs', '_parameters', '_descriptions', '_references', '_audit_rollups'):
            table = prefix+suffix
            state[table] = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (PRESERVED,)).fetchall()
    state['audit_evidence'] = db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text', (PRESERVED,)).fetchall()
    for table in ('artifact_cdg_nodes', 'artifact_cdg_edges', 'artifact_cdg_bindings', 'artifact_dependencies'):
        key = 'dependent_version_id' if table == 'artifact_dependencies' else 'version_id'
        state[table] = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key} IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s)) ORDER BY to_jsonb(t)::text', (PRESERVED,)).fetchall()
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def plan():
    result = local_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        result['owners'] = ownership(db)
        result['preserved_sha256'] = preserved(db)
        for record in result['providers']:
            for table in ('atoms', 'artifacts'):
                require(not db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s', (record['catalog_fqdn'],)).fetchone(), 'Provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                require(not db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (UUID(record['version_id']),)).fetchone(), 'Provider version already exists')
    return result


def reviewed_plan():
    frozen = json.loads(PLAN.read_text())
    require({k: v for k, v in frozen.items() if k not in ('owners', 'preserved_sha256')} == local_plan(),
            'Frozen provider intake changed')
    return frozen


class InjectedFailure(RuntimeError):
    pass


def stage(apply=False, fail_after=None):
    frozen = reviewed_plan()
    if apply:
        gates = json.loads(GATES.read_text())
        require(gates['passed'] and gates['all_rollbacks_complete'] and gates['plan_sha256'] == sha(PLAN),
                'Matching rollback gates required')
    counts = Counter()
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        require(ownership(db) == frozen['owners'], 'Repository ownership changed')
        require(preserved(db) == frozen['preserved_sha256'], 'Existing preparation/intake changed')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        def ensure(table, keys, row):
            nonlocal writes
            changed = int(ensure_row(db, table, keys, row))
            counts[table] += changed
            writes += changed
            if fail_after is not None and writes == fail_after:
                raise InjectedFailure('Injected draft transaction failure')
        for record in frozen['providers']:
            identity, version = UUID(record['atom_id']), UUID(record['version_id'])
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                duplicate = db.execute(f'''SELECT 1 FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE a.{key}<>%s AND (a.fqdn=%s OR v.version_id=%s OR v.content_hash=%s)''',
                    (identity,record['catalog_fqdn'],version,record['content_hash'])).fetchone()
                require(not duplicate, 'Duplicate provider identity or implementation exists')
            duplicate = db.execute('SELECT 1 FROM atoms WHERE atom_id<>%s AND import_module=%s AND source_symbol=%s',
                (identity,record['seed_fields']['import_module'],record['seed_fields']['source_symbol'])).fetchone()
            require(not duplicate, 'Runtime implementation already cataloged elsewhere')
            fields = dict(record['seed_fields'], source_package=record['seed_fields']['namespace_root'],
                          status='flagged', is_publishable=False)
            owner = frozen['owners'][record['seed_fields']['repo_name']]
            atom = AtomSeedRow(**fields).as_dict(**owner)
            atom['atom_id'] = identity
            ensure('atoms', {'atom_id': identity}, atom)
            artifact = {k: v for k, v in atom.items() if k in columns}
            artifact.update(artifact_id=identity, artifact_kind='atom', status='draft', is_publishable=False)
            ensure('artifacts', {'artifact_id': identity}, artifact)
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                ensure(prefix+'_versions', {'version_id': version}, dict(version_id=version, **{key: identity},
                    content_hash=record['content_hash'], semver=record['semver'], is_latest=True,
                    trust_tier=3, s3_key='', fingerprint=record['fingerprint']))
                for port in record['interfaces']:
                    row = {key: identity, 'version_id': version, **port}
                    ensure(prefix+'_io_specs', {k: row[k] for k in (key, 'version_id', 'direction', 'name')}, row)
                actual = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                require(actual == sorted(record['interfaces'], key=lambda p: (p['direction'], p['ordinal'])), 'Stored interface differs')
            evidence_id = uuid5(version, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id,
                artifact_id=identity, version_id=version, audit_type='asset_integrity_check', passed=True,
                status='completed', source_kind='automated', runner_version=RUNNER,
                details=Jsonb(dict(plan_sha256=sha(PLAN), source_sha256=record['provider_sha256'],
                    review_sha256=frozen['review_sha256'], interfaces=record['interfaces'],
                    scope='Exact source/interface draft intake only; no publication approval.'))))
        ids = [UUID(r['atom_id']) for r in frozen['providers']]
        for table, key in [('catalog_atoms_served', 'atom_id'), ('catalog_artifacts_served', 'artifact_id')]:
            require(not db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)', (ids,)).fetchone(), 'Draft provider unexpectedly served')
        require(preserved(db) == frozen['preserved_sha256'], 'Existing preparation/intake changed')
        require(reviewed_plan() == frozen, 'Intake evidence changed during transaction')
        require(fail_after is None, 'Failure injection not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, approved=False, providers=19, rows_created=writes, counts=dict(counts),
                served_drafts=0, existing_records_preserved=True, plan_sha256=sha(PLAN))


def absent():
    frozen = reviewed_plan()
    ids = [UUID(r['atom_id']) for r in frozen['providers']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
            require(not db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)', (ids,)).fetchone(), 'Rollback left provider identity')
        require(preserved(db) == frozen['preserved_sha256'], 'Rollback changed existing records')


def validate():
    absent()
    dry = stage()
    absent()
    points = [3, dry['rows_created']//2, dry['rows_created']]
    for point in points:
        try:
            stage(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Failure injection not reached')
        absent()
    return dict(passed=True, catalog_mutations=0, dry_run=dry, rollback_points=points,
                all_rollbacks_complete=True, plan_sha256=sha(PLAN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'validate', 'apply'])
    action = parser.parse_args().action
    if action == 'plan':
        result, path = plan(), PLAN
    elif action == 'validate':
        result, path = validate(), GATES
    else:
        gates = json.loads(GATES.read_text())
        require(gates['passed'] and gates['all_rollbacks_complete'] and gates['plan_sha256'] == sha(PLAN), 'Matching rollback gates required')
        first, repeat = stage(apply=True), stage(apply=True)
        require(repeat['rows_created'] == 0, 'Repeated staging changed catalog')
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/webtraffic_decomposed_provider_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'action': action, 'report': path.name, 'approved': False}))
