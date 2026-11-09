"""Stage source-matched temporal_selected providers as drafts, preserving existing identities."""
import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
from types import SimpleNamespace
from uuid import UUID, NAMESPACE_URL, uuid5

from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from scripts.validate_temporal_selected_complete_runtime import ROOT, ML, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.stage_text_pair_complete_adapters import (
    FIELDS, interfaces, ownership, legacy_snapshot, served_snapshot, InjectedFailure, require,
)

RUNNER = 'complete-temporal_selected-provider-intake.v1'
PLAN = ROOT/'docs/reviews/complete_temporal_selected_provider_intake_plan.json'
GATES = ROOT/'docs/reviews/complete_temporal_selected_provider_intake_gates.json'
EVIDENCE = 'docs/reviews/temporal_selected_complete_runtime.json'
PRESERVED = [UUID('5acf5fba-a858-5d4a-9832-77b99bea6714'), UUID('6d2beee4-72c5-59c8-b20e-b9c963d4954b')]


def preserved(db):
    rows = {}
    for suffix in ('s', '_versions', '_io_specs', '_parameters', '_descriptions', '_references', '_audit_rollups', '_audit_evidence'):
        table = 'artifact'+suffix
        rows[table] = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text', (PRESERVED,)).fetchall()
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def evidence():
    raw = (ROOT/EVIDENCE).read_bytes()
    require(raw == subprocess.check_output(['git', 'show', '76b98767:'+EVIDENCE], cwd=ROOT), 'Immutable qualification changed')
    report = json.loads(raw)
    require(report['passed'] and report['tests'] == dict(tests=76, failures=0, errors=0, skipped=0), 'Complete runtime evidence required')
    for path, digest in report['software_sha256'].items():
        require(sha(ROOT/path) == digest, 'Qualified software changed')
    from scripts.review_video_numeric_runtime import review as runtime_review
    require(runtime_review() == report['runtime'], 'Qualified runtime changed')
    return report


def plan():
    report = evidence(); wanted = {r['runtime_fqdn']: r for r in report['providers']}
    records = []
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML/'src/sciona/atoms'):
        name = spec.import_module+'.'+spec.source_symbol
        if name not in wanted: continue
        old = wanted[name]
        require(str(spec.version_id) == old['version_id'] and spec.content_hash == old['content_hash'], 'Source identity changed')
        fn = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        signature = inspect.signature(fn)
        def annotation(a): return a if isinstance(a, str) else inspect.formatannotation(a)
        node = SimpleNamespace(inputs=[SimpleNamespace(name=n, type_desc=annotation(p.annotation)) for n,p in signature.parameters.items()],
            outputs=[SimpleNamespace(type_desc=annotation(signature.return_annotation), name='result')])
        ports = interfaces(fn, node)
        for port in ports: port['constraints'] = inspect.getdoc(fn)
        existing = []  # All four provider identities are new; verified below.
        records.append(dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, legacy_existing=bool(existing),
            atom_id=existing[0]['artifact_id'] if existing else str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)),
            version_id=str(spec.version_id), content_hash=spec.content_hash, semver=spec.semver,
            fingerprint=comparable(spec.fingerprint), seed_fields=comparable({k:getattr(spec,k) for k in FIELDS}),
            interfaces=ports, provider_sha256=sha(spec.file_path)))
    require(len(records) == len(wanted) == 4, 'Four exact providers required')
    frozen = dict(providers=sorted(records, key=lambda r:r['runtime_fqdn']), review_sha256=sha(ROOT/EVIDENCE),
        intake_script_sha256=sha(__file__), approved=False)
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        from sciona.competition_import import _canonical
        from scripts.review_temporal_selected_original_scope import ORIGINAL, ORIGINAL_HASH, review as scope_review
        scope = scope_review()
        source = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'", (ORIGINAL,)).fetchall()
        require(len(source) == 1 and hashlib.sha256(_canonical(source[0]['details']['snapshot']).encode()).hexdigest() == ORIGINAL_HASH, 'Original catalog snapshot differs')
        anchors = db.execute('SELECT a.artifact_id::text,a.status,a.is_publishable,v.version_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=ANY(%s) AND v.is_latest ORDER BY a.artifact_id', (PRESERVED,)).fetchall()
        require(anchors == [dict(artifact_id=str(PRESERVED[0]),status='draft',is_publishable=False,version_id=ORIGINAL,content_hash=ORIGINAL_HASH), dict(artifact_id=str(PRESERVED[1]),status='approved',is_publishable=True,version_id='4384168e-6c35-5d00-b070-1af1eaa86783',content_hash='a09b614fbdc7f3ca46db3cbbb746c45b8a6d691d8be0b9ad0e36cd4f779c0f26')], 'Original or retained derivative state differs')
        frozen.update(source_scope_sha256=sha(ROOT/'docs/reviews/temporal_selected_original_scope.json'), source_snapshot_verified=True, preserved_anchors=anchors)
        require(scope == json.loads((ROOT/'docs/reviews/temporal_selected_original_scope.json').read_text()), 'Source review changed')
        for record in records:
            require(not db.execute('SELECT 1 FROM artifacts WHERE fqdn=%s', (record['catalog_fqdn'],)).fetchone(), 'Canonical identity already exists')
            actual = db.execute('''SELECT a.atom_id::text AS artifact_id,a.status,a.is_publishable,v.version_id::text,
                v.content_hash,v.trust_tier,atom_is_publishable(a.atom_id) AS native_eligible,
                EXISTS(SELECT 1 FROM catalog_atoms_served s WHERE s.atom_id=a.atom_id) AS served
                FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''', (record['catalog_fqdn'],)).fetchall()
            require(not actual and not db.execute('SELECT 1 FROM atoms WHERE fqdn=%s', (record['catalog_fqdn'],)).fetchone(), 'Legacy identity already exists')
        frozen.update(owners=ownership(db), preserved_sha256=preserved(db),
            legacy_sha256=legacy_snapshot(db, frozen), served_sha256=served_snapshot(db, frozen))
    return frozen


def reviewed():
    frozen = json.loads(PLAN.read_text())
    evidence()
    require(frozen['intake_script_sha256'] == sha(__file__) and frozen['review_sha256'] == sha(ROOT/EVIDENCE), 'Intake source changed')
    return frozen


def check_preserved(db, frozen):
    require(ownership(db) == frozen['owners'] and preserved(db) == frozen['preserved_sha256'], 'Source CDG or ownership changed')
    require(legacy_snapshot(db, frozen) == frozen['legacy_sha256'], 'Existing atoms changed')
    require(served_snapshot(db, frozen) == frozen['served_sha256'], 'Served selection changed')


def stage(apply=False, fail_after=None):
    frozen = reviewed(); writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        check_preserved(db, frozen)
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        def ensure(table, keys, row):
            nonlocal writes
            writes += int(ensure_row(db, table, keys, row))
            if fail_after is not None and writes == fail_after: raise InjectedFailure('Draft transaction rollback check')
        for r in frozen['providers']:
            identity, version = UUID(r['atom_id']), UUID(r['version_id'])
            atom = AtomSeedRow(**dict(r['seed_fields'], source_package=r['seed_fields']['namespace_root'],
                status='flagged', is_publishable=False)).as_dict(**frozen['owners'][r['seed_fields']['repo_name']])
            atom['atom_id'] = identity
            if not r['legacy_existing']: ensure('atoms', {'atom_id':identity}, atom)
            artifact = {k:v for k,v in atom.items() if k in columns}
            artifact.update(artifact_id=identity, artifact_kind='atom', status='draft', is_publishable=False)
            ensure('artifacts', {'artifact_id':identity}, artifact)
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                if prefix == 'atom' and r['legacy_existing']: continue
                ensure(prefix+'_versions', {'version_id':version}, dict(version_id=version, **{key:identity},
                    content_hash=r['content_hash'], semver=r['semver'], is_latest=True, trust_tier=3, s3_key='', fingerprint=r['fingerprint']))
                for port in r['interfaces']:
                    row = {key:identity, 'version_id':version, **port}
                    ensure(prefix+'_io_specs', {k:row[k] for k in (key, 'version_id', 'direction', 'name')}, row)
                actual = db.execute(f'SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (version,)).fetchall()
                require(actual == r['interfaces'], 'Stored interface differs')
            eid = uuid5(version, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id':eid}, dict(evidence_id=eid, artifact_id=identity,
                version_id=version, audit_type='asset_integrity_check', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(dict(plan_sha256=sha(PLAN), review_sha256=frozen['review_sha256'],
                    source_sha256=r['provider_sha256'], scope='Draft source/interface intake only; no approval.'))))
        check_preserved(db, frozen)
        require(reviewed() == frozen and fail_after is None, 'Evidence changed or fault not reached')
        if not apply: db.rollback()
    return dict(applied=apply, approved=False, rows_created=writes, providers=4, newly_served_rows=0)


def absent():
    frozen = reviewed()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY'); check_preserved(db, frozen)
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            ids = [UUID(r['atom_id']) for r in frozen['providers'] if prefix == 'artifact' or not r['legacy_existing']]
            for suffix in ('s', '_versions', '_io_specs'):
                require(not db.execute(f'SELECT 1 FROM {prefix+suffix} WHERE {key}=ANY(%s)', (ids,)).fetchone(), 'Rollback left provider rows')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE runner_version=%s', (RUNNER,)).fetchone(), 'Rollback left evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['plan', 'validate', 'apply'])
    action = parser.parse_args().action
    if action == 'plan': result, path = plan(), PLAN
    elif action == 'validate':
        absent(); dry = stage(); absent()
        points = [3, dry['rows_created']//2, dry['rows_created']]
        for point in points:
            try: stage(fail_after=point)
            except InjectedFailure: pass
            else: raise ValueError('Fault not reached')
            absent()
        result, path = dict(passed=True, dry_run=dry, rollback_points=points, plan_sha256=sha(PLAN)), GATES
    else:
        gates = json.loads(GATES.read_text())
        require(gates['passed'] and gates['plan_sha256'] == sha(PLAN), 'Matching rollback checks required')
        first, repeat = stage(apply=True), stage(apply=True)
        require(repeat['rows_created'] == 0, 'Repeated intake changed catalog')
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/complete_temporal_selected_provider_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(action=action, approved=False, report=path.name)))
