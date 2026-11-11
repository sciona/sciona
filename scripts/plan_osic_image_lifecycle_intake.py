"""Freeze separate image lifecycle contracts and synthetic numeric-state evidence."""
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
PROVIDER_REVISION = 'bc4642e'
EVIDENCE_COMMIT = 'dc5718f3'
MODULE = 'sciona.atoms.ml.domain_adapters.osic_image_lifecycle'
PROVIDER_PATH = 'src/sciona/atoms/ml/domain_adapters/osic_image_lifecycle.py'
EVIDENCE_FILES = ('docs/reviews/osic_image_lifecycle_providers.json',)
CONSTRAINTS = {
    'model': ('tf_keras.Model', 'Exact fixed local B5 topology with a positive optimizer update counter and a complete finite float32 weight inventory. The counter is not evidence of complete training. Caller controls authenticity and invokes export outside live training; validation consumes RNG state.'),
    'directory': ('str | pathlib.Path', 'Caller-owned private filesystem location. Export requires a new directory and writes numeric weights and a strict manifest; failed I/O may leave partial files. Restore requires stable existing files, validates identities and uses NPZ with pickle disabled. Caller controls permissions, path trust and provenance. No remote transport or atomic snapshot guarantee.'),
    'image_coordinate': ('str', 'Explicit nonempty image-coordinate declaration, exact-match bound into the numeric state. Caller establishes intensity normalization, shape (N,512,512,1), units and source policy; labels alone do not establish equivalence.'),
    'metadata_coordinate': ('str', 'Explicit nonempty metadata-coordinate declaration, exact-match bound into the numeric state. Caller establishes ordered four-component feature meaning, units and encoding; no mapping or normalization is inferred.'),
    'response_coordinate': ('str', 'Explicit nonempty scalar slope-coordinate declaration, exact-match bound into the numeric state. Caller establishes units and compatibility with downstream baseline and query time coordinates.'),
}
OUTPUTS = {
    'build_image_model': ('tf_keras.Model', 'Uncompiled random-initialized fixed EfficientNetB5 image/metadata model with one scalar linear output, GaussianNoise(0.2) and Dropout(0.5). No fitting, optimizer, pretrained weights or input normalization. Construction consumes caller RNG state. Require qualified efficientnet 1.1.0 and legacy Keras; caller establishes seeds and bounds compute threads.'),
    'export_image_prediction_state': ('dict', 'Strict manifest describing private finite float32 numeric weights written into the new directory. Fixed local graph, model-code identity, ordered shapes, archive digest and coordinate labels are bound. No optimizer, callback or RNG state: inference only, not training resumption. Constructor validation consumes RNG state; I/O is not atomic.'),
    'restore_image_prediction_state': ('tf_keras.Model', 'Uncompiled fixed local model with exact finite float32 weights restored after manifest, coordinate, graph, model-code, inventory and archive checks. No fitting, optimizer state or executable graph deserialization. Construction consumes RNG state. Digests do not establish provenance or defend against concurrent file changes.'),
}
COMMON = ' Concrete provisioned runtime only; abstract witness rejects. Fixed source architecture is adapter policy, separate from generic bounded prediction. Intermediate synthetic-checkpoint qualification only; no complete-training, original-CDG, empirical transfer, clean-install or HTTP transport claim.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    name=EVIDENCE_FILES[0]
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+name],cwd=ROOT)
    if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable lifecycle evidence changed')
    report=json.loads(raw)
    state=report['state_qualification']; execution=state['execution']
    replay=execution['fresh_process_replay']
    if (report['passed'] is not True or report['synthetic_only'] is not True
            or report['private_source_unchanged'] is not True or report['abstract_execution_rejected'] is not True
            or execution['intermediate_optimizer_updates']!=96 or execution['weight_arrays']!=738
            or execution['parameters']!=28514709 or len(execution['corruption_cases_rejected'])!=10
            or replay!=dict(passed=True,queries=3,exact_predictions=True,fitting_disabled=True,
                hdf5_loading_disabled=True,optimizer_state_absent=True)
            or report['provider_calls']!=dict(
                export=dict(build_image_model=3,export_image_prediction_state=1,restore_image_prediction_state=10),
                fresh_replay=dict(build_image_model=1,export_image_prediction_state=0,restore_image_prediction_state=1))):
        raise ValueError('Complete exact synthetic lifecycle evidence required')
    if (sha(ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH)!=report['provider_sha256']
            or sha(ROOT/'scripts/validate_osic_image_lifecycle_providers.py')!=report['validator_sha256']):
        raise ValueError('Qualified provider or validator changed')
    for name,digest in state['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified lifecycle implementation changed')
    return json.dumps({EVIDENCE_FILES[0]:report},sort_keys=True,separators=(',',':')).encode()


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 3 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly three lifecycle providers required')
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
    return dict(format='osic-image-lifecycle-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256={name:sha(ROOT/name) for name in ('sciona/osic_image_model.py','sciona/osic_image_state.py')},
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'Fixed model construction and state implementation are hash-bound software calls, not registered provider-to-provider calls. No new registered atom dependency is asserted.',
            'Caller owns private state paths, coordinates, authenticity and stable file access. Inference state cannot resume training.',
            'Intermediate synthetic qualification does not complete image training, the stored original CDG or empirical transfer.'])



if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_image_lifecycle_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
