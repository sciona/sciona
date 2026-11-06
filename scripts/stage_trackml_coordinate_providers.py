"""Stage two source-qualified coordinate projection providers without approval.

Freeze exact source fingerprints and ordered interfaces, then validate rollback
and idempotence. Existing tracking derivative and original intake remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
from typing import get_type_hints
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.decomposed_trackml_coordinate_graph import build_decomposed_trackml_coordinate_graph
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.audit_decomposed_trackml_readiness import ROOT, sha
import subprocess

def require(condition, message):
    if not condition: raise ValueError(message)

RUNNER = 'trackml-coordinate-provider-intake.v1'
PLAN = ROOT/'docs/reviews/trackml_coordinate_provider_intake_plan.json'
GATES = ROOT/'docs/reviews/trackml_coordinate_provider_intake_gates.json'
FIELDS = ('fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
          'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')
PRESERVED = [UUID('a2067934-0f90-5083-81b3-71025d158b6a'), UUID('ae6f41f2-cb0b-51ce-a567-0ab855ece4ce')]


def interfaces(function, nodes):
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    kinds = {np.ndarray: 'ndarray', float: 'float'}
    result = []
    for ordinal, (name, parameter) in enumerate(signature.parameters.items()):
        require(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, 'Unsupported parameter kind')
        require(hints[name] in kinds, 'Unsupported parameter annotation')
        required = parameter.default is inspect.Parameter.empty
        default = '' if required else repr(parameter.default)
        for node in nodes:
            require(list(signature.parameters) == [p.name for p in node.inputs], 'Input order differs')
            port = node.inputs[ordinal]
            require(port.type_desc == kinds[hints[name]] and port.required == required
                    and port.default_value_repr in (None, '', default), 'Input contract differs')
        result.append(dict(direction='input', name=name, ordinal=ordinal,
            type_desc=kinds[hints[name]], required=required, default_value_repr=default, constraints=''))
    outputs = [(p.name, p.type_desc) for p in nodes[0].outputs]
    require(all([(p.name, p.type_desc) for p in n.outputs] == outputs for n in nodes), 'Repeated output contract differs')
    require(len(outputs) == 1 and outputs[0][1] == kinds[hints['return']], 'Return annotation differs')
    result.extend(dict(direction='output', name=n, ordinal=i, type_desc=t,
                       required=True, default_value_repr='', constraints='') for i, (n, t) in enumerate(outputs))
    return result


def local_plan():
    evidence = 'docs/reviews/decomposed_trackml_coordinate_graph_execution.json'
    raw = (ROOT/evidence).read_bytes()
    require(raw == subprocess.check_output(['git', 'show', '9583e463:'+evidence], cwd=ROOT), 'Immutable graph evidence changed')
    report = json.loads(raw)
    require(report['passed'] and report['exact_registered_bindings'] and report['explicit_callable_edges'] == 6,
            'Qualified explicit graph bindings required')
    for name, digest in report['implementation_sha256'].items():
        require(sha(ROOT/name) == digest, 'Qualified graph implementation changed')
    graph = build_decomposed_trackml_coordinate_graph()
    from sciona.services.execution_graph_codec import encode_execution_graph
    require(encode_execution_graph(graph)[0] == report['graph_sha256'], 'Qualified graph changed')
    wanted = {'sciona.atoms.physics.coordinate_projection.'+name for name in ('cylindrical_projection', 'plane_projection')}
    records = []
    directory = ROOT.parent/'sciona-atoms-physics'
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name, directory), artifact_root=directory/'src/sciona/atoms'):
        runtime = spec.import_module+'.'+spec.source_symbol
        if runtime not in wanted: continue
        require(sha(spec.file_path) == report['all_provider_sha256'][runtime], 'Executed provider changed')
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        nodes = [n for n in graph.nodes if n.matched_primitive == runtime]
        ports = interfaces(function, nodes)
        constraints = {
            'coords': 'Finite float64 array (N,3), including empty batches. Consistent caller-defined length units. Cylindrical projection requires finite nonzero transverse norm; plane projection requires nonzero axial coordinate. Nonfinite arithmetic rejects.',
            'reference_radius': 'Positive finite real scalar in the coordinate length unit.',
            'transverse_weight': 'Finite real dimensionless weight for the first two projected coordinates; caller policy, default 1.0.',
            'reference_height': 'Finite real axial plane coordinate in the coordinate length unit; either sign or zero is allowed.',
            'projected': 'Finite float64 array (N,3) for cylindrical projection or (N,2) for plane projection; source arithmetic order retained. No detector or neighbor-selection policy.'}
        for port in ports: port['constraints'] = constraints[port['name']]
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
            atom_id=str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)),
            version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(spec.file_path),
            seed_fields=comparable({k: getattr(spec, k) for k in FIELDS}), semver=spec.semver,
            fingerprint=comparable(spec.fingerprint), interfaces=ports))
    require(len(records) == 2, 'Expected two new projection providers')
    return dict(approved=False, graph_digest=report['graph_sha256'], review_sha256=sha(ROOT/evidence),
        intake_script_sha256=sha(__file__), providers=sorted(records, key=lambda r: r['runtime_fqdn']),
        limitations=['Draft source/interface intake only; semantic publication and dependency gates remain.',
                    'Generic numerical projections with caller-defined units and reference geometry; tracking policy stays in the recipe.'])


def ownership(db):
    result = {}
    for repo in ('sciona-atoms-physics',):
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
    return dict(applied=apply, approved=False, providers=2, rows_created=writes, counts=dict(counts),
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
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/trackml_coordinate_provider_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'action': action, 'report': path.name, 'approved': False}))
