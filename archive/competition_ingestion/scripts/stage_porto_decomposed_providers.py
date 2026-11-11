"""Stage four implementation-reviewed recipe providers without approval.

Freeze exact source fingerprints and ordered interfaces, then validate rollback
and idempotence. Existing preparation and original intake remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
from typing import get_type_hints, get_origin, get_args
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.decomposed_porto_graph import build_decomposed_porto_graph
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.review_decomposed_porto_implementation import ROOT, review, require, sha, interfaces

RUNNER = 'porto-decomposed-provider-intake.v1'
PLAN = ROOT/'docs/reviews/porto_decomposed_provider_intake_plan.json'
GATES = ROOT/'docs/reviews/porto_decomposed_provider_intake_gates.json'
FIELDS = ('fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
          'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')
PRESERVED = [UUID(value) for value in (
    '6a048426-5b6a-5804-bd43-c738b5f50291', '5e94f305-a2ca-5805-a963-f5d19952daaa',
    '42645e5b-6d6a-51ce-bc17-2fd202e7794a', 'a4600020-903b-590a-ae0e-53d1af9c90f9',
    '4f486f1c-82db-50db-b5d2-188aeaa8cc4a')]


def local_plan():
    qualification = review()
    frozen = json.loads((ROOT/'docs/reviews/decomposed_porto_implementation_review.json').read_text())
    require(qualification == frozen, 'Frozen integration review changed')
    graph = build_decomposed_porto_graph()
    wanted = {r['runtime_fqdn']: r for r in qualification['providers']
              if r['runtime_fqdn'].startswith('sciona.atoms.ml.decomposed_porto_execution.')}
    records = []
    for repo_name in ('sciona-atoms', 'sciona-atoms-ml'):
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
    require(len(records) == 4, 'Expected four new providers')
    return dict(approved=False, graph_digest=qualification['graph_digest'],
        review_sha256=sha(ROOT/'docs/reviews/decomposed_porto_implementation_review.json'),
        intake_script_sha256=sha(ROOT/'scripts/stage_porto_decomposed_providers.py'),
        providers=sorted(records, key=lambda r: r['runtime_fqdn']),
        limitations=['Draft source/interface intake only; semantic publication and dependency gates remain.',
                    'Four recipe adapters retain all five neural branches, tree training, and probability averaging. Existing generic ordered weighted combination remains independently approved and unchanged. Exact source-detail qualification remains pending.'])


def ownership(db):
    result = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
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
    return dict(applied=apply, approved=False, providers=4, rows_created=writes, counts=dict(counts),
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
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/porto_decomposed_provider_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'action': action, 'report': path.name, 'approved': False}))
