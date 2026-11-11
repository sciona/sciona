"""Freeze the reusable bounded Keras prediction contract and qualification evidence."""
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
PROVIDER_REVISION = '80fb9de'
EVIDENCE_COMMIT = '10a56033'
MODULE = 'sciona.atoms.ml.keras_bounded_prediction'
PROVIDER_PATH = 'src/sciona/atoms/ml/keras_bounded_prediction.py'
EVIDENCE_FILES = ('docs/reviews/keras_bounded_provider.json','docs/reviews/keras_bounded_b5.json')
CONSTRAINTS = {
    'model': ('tf_keras.Model',
        'Caller-provisioned trusted legacy-Keras model in the qualified installed runtime. Model input geometry, custom code, state authenticity, feature meanings and units remain caller responsibilities. No model loading or fitting is performed. Custom predict/model code is not sandboxed; automatic graph/cache creation can occur.'),
    'inputs': ('list[numpy.ndarray]',
        'Nonempty list or tuple of finite float32/float64 arrays, each rank>=1, sharing a nonzero leading dimension N. Preserve caller row and input order. Native Keras conversion remains in effect; model-defined input geometry and dtype conversion are not inferred by this provider. Inputs and model are private in-process runtime values.'),
    'batch_size': ('int',
        'Positive Python integer excluding bool. Explicit native Keras batch size, including the final partial batch. Caller bounds model-compute threads and reserves one additional private dataset worker. Dataset autotuning is disabled. Batch size does not independently bound memory or model-compute concurrency.'),
}
OUTPUTS = {
    'predict_bounded': 'One finite floating numpy.ndarray with rank>=1 and leading dimension N matching input rows. Remaining shape, output dtype and units follow the caller model. Multiple outputs, nonfinite output and row-count mismatch are rejected after prediction. No feature preparation, key joining, loss, confidence calibration or unit conversion.'
}
COMMON = ' Concrete provisioned execution only; abstract witness rejects. No JSON/HTTP transport, clean-install or empirical effectiveness qualification. Synthetic motion and electrical-power examples establish constructed reuse, and a full-size B5 check establishes native batch parity.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    reports={}
    for name in EVIDENCE_FILES:
        raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+name],cwd=ROOT)
        if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable qualification evidence changed')
        reports[name]=json.loads(raw)
    analytic,b5=[reports[name] for name in EVIDENCE_FILES]
    if (not all(r.get('passed') is True and r.get('synthetic_only') is True for r in reports.values())
            or analytic['analytic_cases']!=48 or len(analytic['cases'])!=48
            or not all(c['exact_analytic_predictions'] for c in analytic['cases'])
            or len(analytic['rejected_contracts'])!=17
            or not analytic['batch_normalization_inference_exact'] or not analytic['stored_weights_unchanged']
            or b5['generic_provider_calls']!=4):
        raise ValueError('Complete analytic, contract and inference-state qualification required')
    batch=b5['source_batch_qualification']['execution']
    if (batch['queries']!=33 or batch['source_default_batch_size']!=32 or batch['partial_batch_size']!=1
            or not batch['exact_source_predictions'] or not batch['learned_weights_unchanged']):
        raise ValueError('Full and partial native B5 batches required')
    provider=ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH
    if any(sha(provider)!=r['provider_sha256'] for r in reports.values()):
        raise ValueError('Qualified provider changed')
    if (sha(ROOT/'sciona/keras_bounded_prediction.py')!=analytic['implementation_sha256']
            or sha(ROOT/'scripts/validate_keras_bounded_provider.py')!=analytic['validator_sha256']
            or sha(ROOT/'scripts/validate_keras_bounded_b5.py')!=b5['validator_sha256']):
        raise ValueError('Qualified implementation or validator changed')
    for name,digest in b5['source_batch_qualification']['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified B5 implementation changed')
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
        raise ValueError('Exactly one bounded prediction provider required')
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
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='numpy.ndarray',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='keras-bounded-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(ROOT/'sciona/keras_bounded_prediction.py'),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'TensorFlow and legacy Keras are software dependencies, not registered atom dependencies. The matcher helper is separately hash-bound.',
            'Caller-provisioned custom model code remains trusted and may have side effects; this is not arbitrary-code isolation.',
            'The bounded prediction operation does not qualify image-model training, the complete OSIC workflow or empirical transfer.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/keras_bounded_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
