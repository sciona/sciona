"""Stage six qualified separate-field recipe adapters without approval.

Freeze exact source fingerprints and ordered interfaces, then validate rollback
and idempotence. Existing text-pair derivative and original intake remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.text_pair_complete_graph import build_complete_text_pair_graph
from scripts.validate_complete_text_pair_graph import provider_inventory
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.audit_decomposed_trackml_readiness import ROOT, sha
import subprocess

def require(condition, message):
    if not condition: raise ValueError(message)

RUNNER = 'text-pair-complete-adapter-intake.v1'
PLAN = ROOT/'docs/reviews/text_pair_complete_adapter_intake_plan.json'
GATES = ROOT/'docs/reviews/text_pair_complete_adapter_intake_gates.json'
FIELDS = ('fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
          'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')
PRESERVED = [UUID('e81e178e-524d-5e8b-8986-c183cea57949'), UUID('dcfe4361-b150-54dd-84ef-fb4526490fe7')]


def interfaces(function, node):
    signature = inspect.signature(function)
    require(list(signature.parameters) == [p.name for p in node.inputs], 'Input order differs')
    result = []
    for ordinal, (name, parameter) in enumerate(signature.parameters.items()):
        require(parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY),
                'Unsupported parameter kind')
        annotation = parameter.annotation if isinstance(parameter.annotation, str) else inspect.formatannotation(parameter.annotation)
        require(node.inputs[ordinal].type_desc == annotation, 'Graph annotation differs')
        required = parameter.default is inspect.Parameter.empty
        # Native qualification explicitly supplied every optional argument. The
        # catalog still preserves the actual callable's optional/default contract.
        result.append(dict(direction='input', name=name, ordinal=ordinal, type_desc=annotation,
            required=required, default_value_repr='' if required else repr(parameter.default), constraints=''))
    returns = signature.return_annotation
    annotation = returns if isinstance(returns, str) else inspect.formatannotation(returns)
    require(len(node.outputs)==1 and node.outputs[0].type_desc==annotation, 'Return annotation differs')
    result.append(dict(direction='output', name=node.outputs[0].name, ordinal=0, type_desc=annotation,
                       required=True, default_value_repr='', constraints=''))
    return sorted(result,key=lambda p:(p['direction'],p['ordinal']))


def local_plan():
    evidence='docs/reviews/text_pair_complete_graph_qualification.json'
    raw=(ROOT/evidence).read_bytes()
    require(raw==subprocess.check_output(['git','show','8c09afaa:'+evidence],cwd=ROOT), 'Immutable graph evidence changed')
    report=json.loads(raw)
    require(report['passed'] and report['native_executions']==3, 'Complete native evidence required')
    require(report['tests']==dict(tests=5,failures=0,errors=0,skipped=0), 'Complete native cases required')
    for path,digest in report['software_sha256'].items():
        require(sha(ROOT/path)==digest, 'Executed software changed')
    from scripts.review_video_numeric_runtime import review
    require(review()==report['runtime'], 'Installed runtime changed')
    from sciona.services.execution_graph_codec import encode_execution_graph
    graph=build_complete_text_pair_graph()
    digest=encode_execution_graph(graph)[0]
    require(digest==report['graph_sha256'], 'Executed graph changed')
    adapters,dependencies=provider_inventory()
    require(adapters==report['adapters'] and dependencies==report['approved_dependencies'],
            'Qualified adapter or dependency identities changed')
    nodes={node.matched_primitive:node for node in graph.nodes}
    frozen={r['runtime_fqdn']:r for r in report['adapters']}
    require(set(nodes)==set(frozen) and len(nodes)==6, 'Exact six adapters required')
    directory=ROOT.parent/'sciona-atoms-ml'
    records=[]
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms'):
        runtime=spec.import_module+'.'+spec.source_symbol
        if runtime not in nodes:continue
        evidence_record=frozen[runtime]
        require(spec.fqdn==evidence_record['catalog_fqdn'] and str(spec.version_id)==evidence_record['version_id']
                and spec.content_hash==evidence_record['content_hash'], 'Executed source identity changed')
        function=getattr(importlib.import_module(spec.import_module),spec.source_symbol)
        require(str(inspect.signature(function))==evidence_record['signature'], 'Signature changed')
        ports=interfaces(function,nodes[runtime])
        for port in ports:port['constraints']=inspect.getdoc(function)
        records.append(dict(runtime_fqdn=runtime,catalog_fqdn=spec.fqdn,
            atom_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=sha(spec.file_path),
            seed_fields=comparable({k:getattr(spec,k) for k in FIELDS}),semver=spec.semver,
            fingerprint=comparable(spec.fingerprint),interfaces=ports))
    require(len(records)==6, 'Exact source records required')
    return dict(approved=False,graph_digest=digest,review_sha256=sha(ROOT/evidence),
        intake_script_sha256=sha(__file__),approved_dependencies=dependencies,
        providers=sorted(records,key=lambda r:r['runtime_fqdn']),
        limitations=['Draft recipe adapter intake only; no independent numerical primitive or publication approval.',
                    'Full stage contracts and exact dependency bindings require semantic review before promotion.',
                    'Concrete execution-local state only; arbitrary forged or concurrently mutated intermediates are unqualified.',
                    'Existing text-pair original and derivative remain unchanged.'])


def ownership(db):
    result = {}
    for repo in ('sciona-atoms-ml',):
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
            require(not db.execute('SELECT 1 FROM artifacts WHERE fqdn=%s', (record['catalog_fqdn'],)).fetchone(), 'Canonical provider already exists')
            legacy=db.execute('''SELECT a.atom_id::text,v.version_id::text,v.content_hash FROM atoms a
                JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(record['catalog_fqdn'],)).fetchall()
            record['legacy_existing']=bool(legacy)
            if legacy:
                require(len(legacy)==1 and legacy[0]['version_id']==record['version_id']
                        and legacy[0]['content_hash']==record['content_hash'], 'Legacy source version differs')
                record['atom_id']=legacy[0]['atom_id']
            else:
                require(not db.execute('SELECT 1 FROM atoms WHERE fqdn=%s',(record['catalog_fqdn'],)).fetchone(), 'Legacy identity missing latest version')
            for table in (('artifact_versions',) if legacy else ('atom_versions', 'artifact_versions')):
                require(not db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (UUID(record['version_id']),)).fetchone(), 'Provider version already exists')
        result['legacy_sha256']=legacy_snapshot(db,result)
        result['served_sha256']=served_snapshot(db,result)
    return result


def legacy_snapshot(db, frozen):
    ids=[UUID(r['atom_id']) for r in frozen['providers'] if r.get('legacy_existing')]
    state={}
    for suffix in ('s','_versions','_io_specs','_parameters','_descriptions','_references','_audit_rollups'):
        table='atom'+suffix
        state[table]=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE atom_id=ANY(%s) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
    return hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()


def served_snapshot(db,frozen):
    ids=[UUID(r['atom_id']) for r in frozen['providers']]
    state={}
    for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
        state[table]=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
    return hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()


def reviewed_plan():
    frozen = json.loads(PLAN.read_text())
    local=local_plan()
    comparison={k:v for k,v in frozen.items() if k not in ('owners','preserved_sha256','legacy_sha256','served_sha256')}
    comparison['providers']=[{k:v for k,v in r.items() if k!='legacy_existing'} for r in frozen['providers']]
    for record in comparison['providers']:
        record['atom_id']=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+record['catalog_fqdn']))
    require(comparison==local,'Frozen provider intake changed')
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
        require(legacy_snapshot(db,frozen)==frozen['legacy_sha256'], 'Legacy provider rows changed')
        require(served_snapshot(db,frozen)==frozen['served_sha256'], 'Existing served selection changed')
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
            if not record['legacy_existing']:
                ensure('atoms', {'atom_id': identity}, atom)
            else:
                existing=db.execute('SELECT atom_id::text FROM atoms WHERE fqdn=%s',(record['catalog_fqdn'],)).fetchall()
                require(existing==[dict(atom_id=record['atom_id'])], 'Legacy identity differs')
            artifact = {k: v for k, v in atom.items() if k in columns}
            artifact.update(artifact_id=identity, artifact_kind='atom', status='draft', is_publishable=False)
            ensure('artifacts', {'artifact_id': identity}, artifact)
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                if prefix=='atom' and record['legacy_existing']:continue
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
        require(served_snapshot(db,frozen)==frozen['served_sha256'], 'Staging changed served selection')
        require(preserved(db) == frozen['preserved_sha256'], 'Existing preparation/intake changed')
        require(legacy_snapshot(db,frozen)==frozen['legacy_sha256'], 'Staging changed legacy rows')
        require(reviewed_plan() == frozen, 'Intake evidence changed during transaction')
        require(fail_after is None, 'Failure injection not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, approved=False, providers=len(frozen['providers']), rows_created=writes, counts=dict(counts),
                newly_served_rows=0, existing_records_preserved=True, legacy_records_preserved=True, plan_sha256=sha(PLAN))


def absent():
    frozen = reviewed_plan()
    ids = [UUID(r['atom_id']) for r in frozen['providers']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for table, key in [(prefix+suffix, key) for prefix,key in [('atom','atom_id'),('artifact','artifact_id')] for suffix in ('s','_versions','_io_specs')] + [('artifact_audit_evidence','artifact_id')]:
            checked_ids=([UUID(r['atom_id']) for r in frozen['providers'] if not r['legacy_existing']] if key=='atom_id' else ids)
            require(not db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)', (checked_ids,)).fetchone(), 'Rollback left provider identity')
        require(preserved(db) == frozen['preserved_sha256'], 'Rollback changed existing records')
        require(legacy_snapshot(db,frozen)==frozen['legacy_sha256'], 'Rollback changed legacy providers')
        require(served_snapshot(db,frozen)==frozen['served_sha256'], 'Rollback changed served selection')


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
        result, path = dict(first=first, repeat=repeat), ROOT/'docs/reviews/text_pair_complete_adapter_intake_applied.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'action': action, 'report': path.name, 'approved': False}))
