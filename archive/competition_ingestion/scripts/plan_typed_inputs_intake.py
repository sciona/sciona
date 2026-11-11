"""Freeze generic conversion contracts and synthetic and graph qualification evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_REVISION = 'a4b3000'
EVIDENCE_COMMIT = '90c85fa6'
MODULE = 'sciona.atoms.ml.pipeline.typed_inputs'
PROVIDER_PATH = 'src/sciona/atoms/ml/pipeline/typed_inputs.py'
EVIDENCE_FILES = ('docs/reviews/typed_inputs_provider.json','docs/reviews/osic_quantile_replay_graph.json')
CONSTRAINTS = {
    'values': ('numpy.ndarray', 'Nonempty plain ndarray with finite native float32 or float64 values. Any shape, including scalar arrays, is structurally supported; preserve axes and element order. Integer, complex, object and subclass arrays are rejected. Caller owns coordinate units, axis meanings and precision requirements.'),
    'dtype': ('str', 'Explicit exact Python string float32 or float64. Always allocate independent storage, including same-dtype conversion. Native NumPy narrowing may round or underflow to zero; nonfinite output including overflow is rejected. No implicit dtype inference, clipping, normalization or unit conversion.'),
    'descriptors': ('list[list[str]]', 'Nonempty plain list of nonempty plain lists of nonblank Python strings. Complete descriptors must be unique. Preserve descriptor and component order and exact spelling, including surrounding whitespace. Caller owns feature meanings and array-column alignment; real feature labels remain private runtime metadata.'),
}
OUTPUTS = {
    'cast_finite_array': ('numpy.ndarray', 'Independent finite array in the requested float32/float64 dtype, preserving shape and element order without input mutation. Native rounding and underflow semantics are retained; precision loss must be acceptable to the caller.'),
    'encode_feature_descriptors': ('list[str]', 'New list of unique compact JSON strings with standard ASCII escaping and no separator whitespace. Each string round-trips to the corresponding ordered descriptor without delimiter collisions. Input descriptors are not mutated; no schema inference, sorting or array conversion.'),
}
COMMON = ' Concrete in-process execution only; abstract witness rejects. Synthetic motion and power examples demonstrate constructed conversion reuse, not empirical effectiveness or domain transfer. No HTTP transport or whole-CDG approval claim.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    reports={}
    for name in EVIDENCE_FILES:
        raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+name],cwd=ROOT)
        if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable qualification evidence changed')
        reports[name]=json.loads(raw)
    analytic,graph=[reports[name] for name in EVIDENCE_FILES]
    if (not all(r.get('passed') is True and r.get('synthetic_only') is True for r in reports.values())
            or len(analytic['cast_cases'])!=8 or len(analytic['rejected_contracts'])!=18
            or not all(all(c[key] for key in ('exact_native_cast','shape_and_order_preserved',
                'independent_storage','input_unchanged')) for c in analytic['cast_cases'])
            or not all(analytic[key] for key in ('descriptor_roundtrip_exact','delimiter_collisions_avoided',
                'unicode_escaped','input_descriptors_unchanged'))):
        raise ValueError('Complete generic conversion qualification required')
    execution=graph['graph_execution']
    if execution['nodes']!=6 or execution['edges']!=8 or execution['missing_edges_rejected']!=8:
        raise ValueError('Complete quantile graph qualification required')
    replay=graph['raw_replay_qualification']['fresh_process_replay']
    if not all(replay[k] for k in ('features_exact','predictions_exact','feature_fitting_disabled','model_updates_disabled')):
        raise ValueError('Exact graph replay required')
    if sha(ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH)!=analytic['provider_sha256']:
        raise ValueError('Qualified conversion provider changed')
    if sha(ROOT/'scripts/validate_typed_inputs_provider.py')!=analytic['validator_sha256']:
        raise ValueError('Qualified conversion validator changed')
    for name,digest in graph['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified graph implementation changed')
    for module,digest in execution['provider_sha256'].items():
        relative=module.removeprefix('sciona.atoms.ml.').replace('.','/')+'.py'
        if sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/relative)!=digest:
            raise ValueError('Qualified graph provider changed')
    return json.dumps(reports,sort_keys=True,separators=(',',':')).encode()


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 2 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly two generic conversion providers required')
    nodes = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for spec in sorted(specs, key=lambda s: s.source_symbol):
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s', (spec.fqdn, identity)).fetchone():
                    raise ValueError('Provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            ports = [dict(direction='input', ordinal=i, name=a.arg, type_desc=CONSTRAINTS[a.arg][0],
                required=True, default_value_repr='', constraints=CONSTRAINTS[a.arg][1])
                for i, a in enumerate(nodes[spec.source_symbol].args.args)]
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=OUTPUTS[spec.source_symbol][0],
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol][1]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='typed-inputs-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'NumPy casting and standard JSON encoding invoke no registered atom dependencies. Quantile graph evidence is a composed use, not a fixed dependency.',
            'Caller controls acceptable precision loss, coordinate meanings, descriptor privacy and array-column alignment.',
            'Quantile graph and original-CDG approval remain pending.'])





if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/typed_inputs_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
