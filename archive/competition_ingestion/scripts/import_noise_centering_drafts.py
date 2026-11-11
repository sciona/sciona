"""Atomically import reviewed noise centering providers as non-serving drafts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from uuid import UUID, NAMESPACE_URL, uuid5

from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'docs/reviews/noise_centering_intake_plan.json'
GATES = ROOT/'docs/reviews/noise_centering_intake_transactions.json'
COMMIT = '209785de'
RUNNER = 'noise-centering-draft-intake.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_plan():
    raw = subprocess.check_output(['git', 'show', COMMIT+':'+str(PLAN.relative_to(ROOT))], cwd=ROOT)
    if raw != PLAN.read_bytes():
        raise ValueError('Immutable draft plan changed')
    plan = json.loads(raw)
    if sha(ROOT/'scripts/plan_noise_centering_intake.py') != plan['planner_sha256']:
        raise ValueError('Reviewed planner changed')
    from scripts.plan_noise_centering_intake import evidence, EVIDENCE_COMMIT
    evidence_bytes = evidence()
    if (plan['execution_evidence_commit'] != EVIDENCE_COMMIT
            or hashlib.sha256(evidence_bytes).hexdigest() != plan['execution_evidence_sha256']):
        raise ValueError('Qualified arithmetic evidence changed')
    directory = ROOT.parent/'sciona-atoms-ml'
    provider = directory/'src/sciona/atoms/ml/noise_median_centering.py'
    committed = subprocess.check_output(['git', 'show', plan['provider_revision']+':src/sciona/atoms/ml/noise_median_centering.py'], cwd=directory)
    if committed != provider.read_bytes():
        raise ValueError('Provider differs from immutable revision')
    parsed = {s.import_module+'.'+s.source_symbol: s for s in _parse_registered_atoms(
        repo=ProviderRepo(directory.name, directory), artifact_root=directory/'src/sciona/atoms')}
    expected = {'sciona.atoms.ml.noise_median_centering.'+n for n in ('add_noise_median_center',)}
    if len(plan['new_providers']) != 1 or {p['runtime_fqdn'] for p in plan['new_providers']} != expected:
        raise ValueError('Exactly one reviewed provider required')
    for row in plan['new_providers']:
        spec = parsed[row['runtime_fqdn']]
        if (str(spec.version_id) != row['version_id'] or spec.content_hash != row['content_hash']
                or sha(spec.file_path) != row['provider_sha256'] or spec.semver != row['semver']
                or comparable(spec.fingerprint) != row['fingerprint']
                or str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)) != row['atom_id']
                or comparable({k: getattr(spec, k) for k in row['seed_fields']}) != row['seed_fields']):
            raise ValueError('Provider identity or implementation changed')
    return plan


class InjectedFailure(RuntimeError):
    pass


def intake(apply=False, fail_after=None):
    plan = reviewed_plan()
    if apply:
        gate = json.loads(GATES.read_text())
        if (gate.get('passed') is not True or gate.get('all_rollbacks_complete') is not True
                or gate['plan_sha256'] != sha(PLAN)
                or gate['importer_sha256'] != sha(Path(__file__))):
            raise ValueError('Current successful transaction gates required')
    counts = Counter()
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a "
            "JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if owners != [plan['owner']]:
            raise ValueError('Provider ownership changed')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='artifacts'")}

        def ensure(table, key, row):
            nonlocal writes
            changed = int(ensure_row(db, table, key, row))
            counts[table] += changed
            writes += changed
            if fail_after is not None and writes == fail_after:
                raise InjectedFailure('Injected draft transaction failure')

        for record in plan['new_providers']:
            identity, version = UUID(record['atom_id']), UUID(record['version_id'])
            fields = dict(record['seed_fields'], source_package=record['seed_fields']['namespace_root'],
                          status='flagged', is_publishable=False)
            atom = AtomSeedRow(**fields).as_dict(**plan['owner'])
            atom['atom_id'] = identity
            ensure('atoms', dict(atom_id=identity), atom)
            canonical = {k: v for k, v in atom.items() if k in columns}
            canonical.update(artifact_id=identity, artifact_kind='atom', status='draft', is_publishable=False)
            ensure('artifacts', dict(artifact_id=identity), canonical)
            for table, key in [('atom_versions', 'atom_id'), ('artifact_versions', 'artifact_id')]:
                ensure(table, dict(version_id=version), dict(version_id=version, **{key: identity},
                    content_hash=record['content_hash'], semver=record['semver'], is_latest=True,
                    trust_tier=3, s3_key='', fingerprint=record['fingerprint']))
            for table, key in [('atom_io_specs', 'atom_id'), ('artifact_io_specs', 'artifact_id')]:
                for port in record['interfaces']:
                    row = {key: identity, 'version_id': version, **port}
                    ensure(table, {k: row[k] for k in (key, 'version_id', 'direction', 'name')}, row)
            evidence_id = uuid5(version, RUNNER)
            ensure('artifact_audit_evidence', dict(evidence_id=evidence_id), dict(
                evidence_id=evidence_id, artifact_id=identity, version_id=version,
                audit_type='asset_integrity_check', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(dict(plan_commit=COMMIT,
                    plan_sha256=sha(PLAN), provider_sha256=record['provider_sha256'],
                    execution_evidence_sha256=plan['execution_evidence_sha256'],
                    scope='Exact source/version/interface draft import only. Dependency qualification, full metadata and semantic publication gates remain pending.'))))
        ids = [UUID(r['atom_id']) for r in plan['new_providers']]
        for table, key in [('catalog_atoms_served', 'atom_id'), ('catalog_artifacts_served', 'artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)', (ids,)).fetchone():
                raise ValueError('Draft unexpectedly served')
        if fail_after is not None:
            raise ValueError('Injected failure point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, approved=False, providers=1, rows_created=writes, counts=dict(counts),
                served_drafts=0, plan_sha256=sha(PLAN), importer_sha256=sha(Path(__file__)))


def assert_absent(plan):
    ids = [UUID(r['atom_id']) for r in plan['new_providers']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            for suffix in ('s', '_versions', '_io_specs'):
                if db.execute(f'SELECT 1 FROM {prefix+suffix} WHERE {key}=ANY(%s)', (ids,)).fetchone():
                    raise ValueError('Draft rows survived rollback')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE artifact_id=ANY(%s)', (ids,)).fetchone():
            raise ValueError('Draft evidence survived rollback')


def validate():
    plan = reviewed_plan()
    assert_absent(plan)
    dry = intake()
    assert_absent(plan)
    points = [1, dry['rows_created']//2, dry['rows_created']]
    for point in points:
        try:
            intake(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Expected transaction failure')
        assert_absent(plan)
    return dict(passed=True, catalog_mutations=0, dry_run=dry, fault_after_writes=points,
        all_rollbacks_complete=True, plan_sha256=sha(PLAN), importer_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'dry', 'apply', 'repeat'])
    action = parser.parse_args().action
    result = validate() if action == 'validate' else intake(apply=action in ('apply', 'repeat'))
    if action == 'repeat' and result['rows_created']:
        raise ValueError('Repeated draft import changed rows')
    output = GATES if action == 'validate' else ROOT/f'docs/reviews/noise_centering_intake_{action}.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
