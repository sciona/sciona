"""Prepare exact non-serving blend-provider identities and port contracts."""
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
EVIDENCE_COMMIT = '33eb355b'
EVIDENCE = 'docs/reviews/competition_mercari_tensorflow_synthetic_graph.json'
MODULE = 'sciona.atoms.ml.positive_log_blending'
CONSTRAINTS = {
    'predictions': 'Finite float64 matrix (rows, model columns) in one declared normalized numeric response coordinate. Exact ordered model-column identity required. Fit requires at least two rows; apply permits zero rows. Values clip to explicit fitted bounds before log1p.',
    'targets': 'Aligned finite nonnegative float64 vector from the calibration population. Caller must ensure calibration scores are held-out or out-of-fold and have the same normalized response coordinate as targets.',
    'model_columns': 'Nonempty ordered list of unique nonempty model-output identifiers matching matrix width. Exact identity and order must agree with fitted state; these identify models, not data rows.',
    'response_coordinate': 'Nonblank identifier for an explicitly chosen common normalized numeric response coordinate. Physical measurements must be normalized before log1p. Exact identifier must match fitted state.',
    'clip_bounds': 'Two finite numeric bounds with 0 <= lower < upper, expressed in the declared response coordinate; only model predictions are clipped, not calibration targets.',
    'controls': 'Dictionary with exactly alpha, max_iter, tolerance, seed. Positive finite alpha and tolerance, positive integer max_iter, integer 0 <= seed < 2**32. Positive Lasso with intercept, precomputed Gram matrix and random coordinate selection. Nonconvergence is rejected.',
    'state': 'Portable JSON dictionary with exactly format, model_columns, response_coordinate, coefficients, intercept, clip_bounds. Format sciona.positive_log_blend.v1; finite nonnegative coefficients, finite intercept, valid bounds, exact model and coordinate identities. No pickle.',
}


def plan():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed connected execution evidence changed')
    evidence = json.loads(raw)
    if not evidence['passed'] or not all(r['registered_provider_json_parity'] for r in evidence['blend_rounds']):
        raise ValueError('Registered-provider execution qualification required')
    for relative, digest in evidence['software_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Execution implementation changed: '+relative)
    repo = ROOT.parent/'sciona-atoms-ml'
    provider_path = repo/'src/sciona/atoms/ml/positive_log_blending.py'
    provider = evidence['registered_blend_provider']
    committed = subprocess.check_output(['git', 'show', provider['revision']+':src/sciona/atoms/ml/positive_log_blending.py'], cwd=repo)
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
            output = CONSTRAINTS['state'] if name.startswith('fit_') else 'Finite float64 vector aligned with input rows, in the declared normalized response coordinate. Intercept is unconstrained: outputs may be negative; there is no silent final clamp. Inverse-transform overflow is rejected.'
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
                              required=True, default_value_repr='', constraints=output))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+name, atom_id=str(identity), version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=provider['sha256'], semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='positive-log-blend-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=provider['revision'], planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Draft identities only; canonical draft and legacy flagged, both unpublished, until separate publication qualification.',
            'Witnesses establish shape and coordinate compatibility only; caller owns population independence and semantic correctness of coordinate labels.',
            'Complete metadata, dependency qualification, transaction rollback tests and served-selection verification remain required.',
            'No original competition CDG approval or historical winning-ensemble completeness is implied.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/positive_log_blend_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), ports_per_catalog=sum(len(p['interfaces']) for p in result['new_providers']), applied=False)))
