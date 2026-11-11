"""Mirror existing trackml_nearest identities into draft canonical records, preserving legacy history.

This is source/interface intake only. It neither changes existing atoms nor
approves their canonical counterparts for pipeline selection.
"""
import argparse
import hashlib
import inspect
import json
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb

from scripts.audit_trackml_nearest_intake import audit, atoms, ROOT, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row

RUNNER = 'trackml_nearest-canonical-intake.v1'
PLAN = ROOT/'docs/reviews/trackml_nearest_canonical_intake_plan.json'
GATES = ROOT/'docs/reviews/trackml_nearest_canonical_intake_gates.json'
FIELDS = ('fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
          'description', 'domain_tags', 'source_kind', 'is_ffi', 'source_package', 'owner_id', 'source_repo_id')


def legacy_digest(db, ids):
    state = {}
    for suffix in ('s', '_versions', '_io_specs', '_parameters', '_descriptions', '_references', '_audit_rollups'):
        table = 'atom'+suffix
        state[table] = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE atom_id=ANY(%s) ORDER BY to_jsonb(t)::text', (ids,)).fetchall()
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def served_digest(db,ids):
    rows=db.execute('SELECT to_jsonb(t) AS row FROM catalog_artifacts_served t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
    return hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()


def interfaces(name):
    if name != 'helix_nearest_point_distance':
        raise ValueError('Unexpected nearest-point provider')
    ports = []
    signature = inspect.signature(getattr(atoms, name))
    for i, (parameter_name, parameter) in enumerate(signature.parameters.items()):
        required = parameter.default is inspect.Parameter.empty
        if parameter_name == 'iterations':
            description = 'Positive integer fixed Newton iteration count; default 3. This approximation does not guarantee a global minimum or convergence.'
        elif parameter_name in ('x', 'y', 'z'):
            description = 'Nonempty aligned finite float64 target coordinate vector, in a consistent caller-defined length unit.'
        else:
            description = 'Finite scalar or aligned float64 helix parameter vector, in the same caller-defined length unit as target coordinates. Pitch is axial displacement per full revolution and must be nonzero; radius must be nonnegative. Initial point must lie on the specified helix.'
        description += ' Qualified domain excludes nonfinite intermediate arithmetic and outputs.'
        ports.append(dict(direction='input', name=parameter_name, ordinal=i,
            type_desc='ndarray' if required else 'int', required=required,
            default_value_repr='' if required else repr(parameter.default), constraints=description))
    for i, output in enumerate(('x1', 'y1', 'z1', 'dist')):
        ports.append(dict(direction='output', name=output, ordinal=i, type_desc='ndarray',
            required=True, default_value_repr='',
            constraints='Aligned finite float64 vector in the caller-defined length unit. Fixed-iteration projected point coordinate or Euclidean target distance; distance is nonnegative. No global-optimum guarantee.'))
    return ports


def plan():
    qualification = audit()
    if not qualification['metadata_only_identity_eligible']:
        raise ValueError('Exact existing identities required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        atom_columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='atoms'")}
        ids = [UUID(p['existing']['atom_id']) for p in qualification['providers']]
        for provider in qualification['providers']:
            if provider['canonical']:
                raise ValueError('Canonical identity already exists; use repeat verification')
            old = provider['existing']; identity = old['atom_id']; version = old['version_id']
            source = db.execute('SELECT '+','.join(k for k in FIELDS if k in atom_columns)+' FROM atoms WHERE atom_id=%s', (identity,)).fetchone()
            artifact = {k: v for k, v in source.items() if k in columns}
            artifact.update(artifact_id=identity, artifact_kind='atom', status='draft', is_publishable=False)
            version_row = db.execute('''SELECT version_id::text,content_hash,semver,s3_key,fingerprint,trust_tier,is_latest
                FROM atom_versions WHERE version_id=%s''', (version,)).fetchone()
            version_row['artifact_id'] = identity
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s OR content_hash=%s',
                          (version, old['content_hash'])).fetchone():
                raise ValueError('Canonical version or implementation already exists')
            records.append(dict(artifact=comparable(artifact), version=comparable(version_row),
                interfaces=interfaces(old['source_symbol']), source_sha256=provider['source_sha256']))
        preserved = legacy_digest(db, ids)
        served = served_digest(db,ids)
    return dict(approved=False, records=records, legacy_sha256=preserved,
        served_sha256=served,
        stager_sha256=sha(__file__), audit_sha256=qualification['audit_sha256'],
        source_state_sha256=qualification['state_source_sha256'])


def reviewed_plan():
    frozen = json.loads(PLAN.read_text())
    if frozen['stager_sha256'] != sha(__file__): raise ValueError('Stager changed')
    current = audit()
    if (not current['metadata_only_identity_eligible'] or current['audit_sha256'] != frozen['audit_sha256']
            or current['state_source_sha256'] != frozen['source_state_sha256']):
        raise ValueError('Qualified identity/source changed')
    for p, record in zip(current['providers'], frozen['records'], strict=True):
        old = p['existing']
        if (old['atom_id'] != record['artifact']['artifact_id'] or old['version_id'] != record['version']['version_id']
                or old['content_hash'] != record['version']['content_hash'] or p['source_sha256'] != record['source_sha256']
                or interfaces(old['source_symbol']) != record['interfaces']):
            raise ValueError('Provider identity or interface changed')
    return frozen


class InjectedFailure(RuntimeError): pass


def stage(apply=False, fail_after=None):
    frozen = reviewed_plan()
    if apply:
        gates = json.loads(GATES.read_text())
        if not gates['passed'] or gates['plan_sha256'] != sha(PLAN) or not gates['all_rollbacks_complete']:
            raise ValueError('Matching rollback qualification required')
    ids = [UUID(r['artifact']['artifact_id']) for r in frozen['records']]
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        db.execute('SELECT atom_id FROM atoms WHERE atom_id=ANY(%s) FOR SHARE', (ids,)).fetchall()
        if legacy_digest(db, ids) != frozen['legacy_sha256']: raise ValueError('Legacy history changed')
        def ensure(table, keys, row):
            nonlocal writes
            writes += int(ensure_row(db, table, keys, row))
            if fail_after is not None and writes >= fail_after: raise InjectedFailure('Injected rollback')
        for record in frozen['records']:
            identity = record['artifact']['artifact_id']; version = record['version']['version_id']
            ensure('artifacts', {'artifact_id': identity}, record['artifact'])
            ensure('artifact_versions', {'version_id': version}, record['version'])
            for port in record['interfaces']:
                row = dict(artifact_id=identity, version_id=version, **port)
                ensure('artifact_io_specs', {k: row[k] for k in ('artifact_id','version_id','direction','name')}, row)
            actual = db.execute('''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
            if actual != record['interfaces']: raise ValueError('Stored interface differs')
            evidence_id = uuid5(UUID(version), RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id,
                artifact_id=identity, version_id=version, audit_type='asset_integrity_check', passed=True,
                status='completed', source_kind='automated', runner_version=RUNNER,
                details=Jsonb(dict(plan_sha256=sha(PLAN), source_sha256=record['source_sha256'],
                    scope='Exact existing identity mirrored as draft with explicit supported interface; not publication approval.'))))
        if db.execute("SELECT 1 FROM artifacts WHERE artifact_id=ANY(%s) AND (status<>'draft' OR is_publishable)",(ids,)).fetchone():
            raise ValueError('Canonical draft unexpectedly approved or publishable')
        if served_digest(db,ids)!=frozen['served_sha256']:raise ValueError('Legacy serving changed')
        if legacy_digest(db, ids) != frozen['legacy_sha256']: raise ValueError('Legacy history modified')
        if reviewed_plan() != frozen: raise ValueError('Evidence changed during transaction')
        if fail_after is not None: raise ValueError('Fault point not reached')
        if not apply: db.rollback()
    return dict(applied=apply, approved=False, rows_created=writes, legacy_preserved=True, legacy_serving_preserved=True, served_drafts=0)


def absent():
    frozen = reviewed_plan(); ids = [UUID(r['artifact']['artifact_id']) for r in frozen['records']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for table in ('artifacts','artifact_versions','artifact_io_specs','artifact_audit_evidence'):
            if db.execute(f'SELECT 1 FROM {table} WHERE artifact_id=ANY(%s)', (ids,)).fetchone():
                raise ValueError('Rollback left canonical rows')
        if legacy_digest(db, ids) != frozen['legacy_sha256']: raise ValueError('Rollback changed legacy history')
        if served_digest(db,ids)!=frozen['served_sha256']:raise ValueError('Rollback changed legacy serving')


def validate():
    absent(); dry = stage(); absent()
    points = [1, dry['rows_created']//2, dry['rows_created']]
    for point in points:
        try: stage(fail_after=point)
        except InjectedFailure: pass
        else: raise ValueError('Fault not reached')
        absent()
    return dict(passed=True, catalog_mutations=0, dry_run=dry, rollback_points=points,
                all_rollbacks_complete=True, plan_sha256=sha(PLAN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','validate','apply'])
    action = parser.parse_args().action
    if action == 'plan': result, path = plan(), PLAN
    elif action == 'validate': result, path = validate(), GATES
    else:
        first = stage(apply=True); repeat = stage(apply=True)
        if repeat['rows_created']: raise ValueError('Repeat changed rows')
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/trackml_nearest_canonical_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(action=action, approved=False, report=path.name)))
