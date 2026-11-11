"""Freeze feature replay contracts and their synthetic qualification evidence."""
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
PROVIDER_REVISION = '0e11572'
EVIDENCE_COMMIT = 'f4094d4a'
MODULE = 'sciona.atoms.ml.domain_adapters.osic_feature_replay'
PROVIDER_PATH = 'src/sciona/atoms/ml/domain_adapters/osic_feature_replay.py'
EVIDENCE_FILES = ('docs/reviews/osic_feature_replay_provider.json',)
CONSTRAINTS = {
    'state': ('dict', 'Private caller-owned frozen preparation state with exact format, coordinates, ordered categories, ranges, arithmetic dtypes, feature_columns, observations and query_scope. Require unique nonblank ordered categories without cross-field collisions; feature order must match category order then numeric,time_offset,baseline_target. Numeric ranges are finite and nondegenerate. Stored observations must uniquely cover queried entities and contain finite numeric values and fitted categories. Supported saved arithmetic dtypes are float32/float64 and signed/unsigned 8/16/32/64-bit integers. Caller establishes provenance and ensures intended arithmetic is representable; structural checks do not authenticate state. Never publish saved observations or query scope.'),
    'queries': ('pandas.DataFrame', 'Nonempty explicitly mapped entity/time table with nonblank string entities and finite real numeric times, complete rows, and exact equality to saved query_scope records in supplied order. Caller supplies unambiguous unique column names. No query population change, reordering, baseline inference or extrapolation to an undeclared population. Query time is converted to saved time-offset arithmetic dtype; caller establishes representability and compatible units.'),
    'feature_coordinate': ('str', 'Explicit nonblank Python string exactly matching saved feature_coordinate. Names bind caller-declared feature meanings and order; this operation does not infer units or permit a changed coordinate system.'),
    'response_coordinate': ('str', 'Explicit nonblank Python string exactly matching saved response_coordinate. Caller establishes compatible response units and actual availability of stored observations; matching labels do not establish provenance.'),
}
OUTPUTS = {'transform_frozen_queries': 'Mapping with features (M,D) numeric array, structured ordered feature_columns, boolean observed_mask (M,), and observed_values (M,). Features preserve fitted source arithmetic before column stacking; output dtype may be promoted by categorical columns. Observed values contain saved observations where query time exactly equals observed_time, and zero elsewhere. Column metadata is copied; state and query inputs are not mutated. No model-input float32 cast, descriptor serialization, model loading, fitting, extrema discovery, prediction reduction or branch blending.'}
COMMON = ' Concrete in-process execution only; abstract witness rejects. This is a source-specific frozen transductive feature recipe, not an unrestricted train-only scaler. Qualification covers synthetic float32/float64 arithmetic. Broader integer configurations require caller validation. No empirical effectiveness, HTTP transport or original-CDG approval claim.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    reports={}
    for name in EVIDENCE_FILES:
        raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+name],cwd=ROOT)
        if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable qualification evidence changed')
        reports[name]=json.loads(raw)
    analytic=reports[EVIDENCE_FILES[0]]
    if (analytic.get('passed') is not True or analytic.get('synthetic_only') is not True
            or len(analytic['cases'])!=2 or len(analytic['rejected_contracts'])!=13
            or {c['arithmetic_dtype'] for c in analytic['cases']}!={'float32','float64'}
            or not all(all(c[key] for key in ('exact_fitted_features','exact_observation_mask',
                'exact_observation_values','state_and_queries_unchanged','output_columns_independent',
                'fitting_disabled')) for c in analytic['cases'])):
        raise ValueError('Complete frozen feature replay qualification required')
    if sha(ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH)!=analytic['provider_sha256']:
        raise ValueError('Qualified feature replay provider changed')
    for name,digest in analytic['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified feature replay implementation changed')
    return json.dumps(reports,sort_keys=True,separators=(',',':')).encode()


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 1 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly one frozen feature replay provider required')
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
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='dict',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='osic-feature-replay-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'The provider delegates to frozen preparation-state software using NumPy and pandas; no registered atom computation is invoked during replay.',
            'Transductive query scope, coordinate meanings, numeric representability, availability and private state provenance remain caller-owned.',
            'Input casting, descriptor serialization, model replay, reduction and blending are separate graph boundaries; whole-CDG approval remains pending.'])




if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_feature_replay_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
