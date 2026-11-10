"""Prepare exact non-serving sparse-neural provider identities and port contracts."""
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
EVIDENCE_COMMIT = 'c372ca09'
EVIDENCE = 'docs/reviews/mercari_registered_model_graph.json'
MODULE = 'sciona.atoms.ml.sparse_neural_prediction'
CONSTRAINTS = {
    'features': 'Finite canonical float32 CSR matrix with positive feature width; training requires at least one row, prediction supports empty rows. Explicit zeros count as present in binary mode. Caller normalizes units and preserves fitted preprocessing, vocabulary and feature ordering. No application-specific preprocessing is implicit.',
    'targets': 'Finite aligned nonempty float32 or float64 training response vector in the declared normalized response coordinate. Training-only inputs; calibration and query labels must be excluded. No implicit logarithm, inverse transform or output clamp.',
    'feature_columns': 'Nonempty ordered unique nonblank feature identifiers exactly matching CSR width. Must identify the same fitted feature semantics and order at prediction; equal width alone is insufficient.',
    'feature_coordinate': 'Nonblank identity of caller-defined normalized feature coordinates, matching fitted state exactly. Different domains require explicit compatible feature adapters; labels alone do not prove semantic equivalence.',
    'response_coordinate': 'Nonblank identity of normalized response coordinates, matching fitted state exactly. Caller owns physical units, transformations and any inverse transformation. Predictions retain this coordinate.',
    'mode': 'Exactly mse or quantile. MSE minimizes squared error; quantile fits soft quantile probabilities and decodes their expectation. No Huber objective is implied.',
    'controls': 'Exact dictionary: n_hidden (two or three positive integer widths), activation (relu/prelu), learning_rate and lr_decay (positive finite), n_epoch and batch_size and bs_decay and n_bins (positive integers), decay_epochs (nonnegative integer), reg_l1 and reg_l2 (finite nonnegative first-layer weight penalties), seed (unsigned 32-bit integer), binary_X and use_target_scaling (booleans), quantile_precision and classification_loss_multiplier (positive finite). Quantile requires use_target_scaling. Precision depends on response-coordinate squared scale. Partial batches are retained; learning-rate and batch schedules are verified. One synchronous CPU worker; caller must reserve its core in shared budget.',
    'state': 'Numeric JSON prediction state with exact format sciona.sparse_neural_prediction.v1, mode, feature_columns, feature_coordinate, response_coordinate, n_hidden, activation, binary_X, use_target_scaling, n_bins, decoder and weights. Validated finite graph weights and fitted decoder only; no executable deserialization or optimizer state. Prediction-only restoration, no training resumption. Learned state is private runtime material.',
}



def plan():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed connected execution evidence changed')
    evidence = json.loads(raw)
    if (evidence.get('passed') is not True or evidence.get('synthetic_only') is not True
            or len(evidence['fits']) != 12 or len(evidence['blend_rounds']) != 3
            or not all(all(r[k] is True for k in ('exact_source_calibration', 'exact_source_query',
                'full_worker_schedule_verified')) for r in evidence['fits'])
            or evidence['exact_original_final_query'] is not True
            or evidence['fresh_process_replay'] != dict(passed=True, restored_models=12,
                exact_final_predictions=True, training_executed=False)):
        raise ValueError('Successful connected registered-model execution required')
    expected_models = {(g, g * 100 + i, 'mse' if i < 2 else 'quantile')
                       for g in (1, 2, 3) for i in range(4)}
    if {(r['group'], r['seed'], r['mode']) for r in evidence['fits']} != expected_models:
        raise ValueError('Exact declared model inventory required')
    if evidence['blend_rounds'] != [dict(after_group=g, model_count=n, finite=True)
                                    for g, n in ((1, 4), (3, 8), (2, 12))]:
        raise ValueError('Exact connected blend rounds required')
    for relative, digest in evidence['software_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Execution implementation changed: '+relative)
    repo = ROOT.parent/'sciona-atoms-ml'
    for module, digest in evidence['provider_sha256'].items():
        path = repo/'src'/Path(*module.split('.')).with_suffix('.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Connected provider implementation changed: '+module)
    provider_path = repo/'src/sciona/atoms/ml/sparse_neural_prediction.py'
    provider = dict(revision='30723cc', sha256=evidence['provider_sha256'][MODULE],
                    functions=['fit_sparse_neural', 'predict_sparse_neural'])
    committed = subprocess.check_output(['git', 'show', provider['revision']+':src/sciona/atoms/ml/sparse_neural_prediction.py'], cwd=repo)
    if committed != provider_path.read_bytes() or hashlib.sha256(committed).hexdigest() != provider['sha256']:
        raise ValueError('Immutable provider implementation changed')
    definitions = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    specs = {s.source_symbol: s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE}
    if set(specs) != set(provider['functions']):
        raise ValueError('Exact provider inventory required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for name in provider['functions']:
            spec = specs[name]
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s', (spec.fqdn, identity)).fetchone():
                    raise ValueError('Provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            node = definitions[name]
            ports = [dict(direction='input', ordinal=i, name=a.arg, type_desc=ast.unparse(a.annotation),
                          required=True, default_value_repr='', constraints=CONSTRAINTS[a.arg])
                     for i, a in enumerate(node.args.args)]
            output = CONSTRAINTS['state'] if name.startswith('fit_') else 'Finite response vector aligned to input rows: float32 for MSE, float64 for quantile, including empty queries. Declared response coordinate is preserved. No inverse transform or clipping. Model restoration performs no training.'
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
                              required=True, default_value_repr='', constraints=output))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+name, atom_id=str(identity), version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=provider['sha256'], semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='sparse-neural-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=provider['revision'], planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Draft identities only; canonical draft and legacy flagged, both unpublished, until separate publication qualification.',
            'Witnesses establish metadata compatibility only; caller owns training-only fitting, feature vocabulary and semantic correctness of coordinate labels. No cross-domain accuracy is implied.',
            'Requires installed sciona runtime, TensorFlow and numeric dependencies; one synchronous single-core worker per operation. Dependency and license qualification remains pending.',
            'Complete metadata, dependency qualification, transaction rollback tests and served-selection verification remain required.',
            'No original competition CDG approval or historical winning-ensemble completeness is implied.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/sparse_neural_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), ports_per_catalog=sum(len(p['interfaces']) for p in result['new_providers']), applied=False)))
