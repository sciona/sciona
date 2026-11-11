"""Prepare exact non-serving quantile response provider identities and port contracts."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_COMMIT = '64c60ec3'
EVIDENCE = 'docs/reviews/quantile_checkpoint_replay.json'
MODULE = 'sciona.atoms.ml.quantile_response_encoding'
CONSTRAINTS = {
    'targets': 'Nonempty finite float32 or float64 target vector in an explicitly chosen normalized numeric response coordinate. Fit uses training targets only, promotes to float64 and retains repeated quantiles. Encoding must use the same coordinate and fitted boundaries.',
    'response_coordinate': 'Nonblank normalized numeric response-coordinate identifier, matching fitted state exactly. Caller owns meaning, upstream normalization and any inverse transform; labels do not prove physical equivalence.',
    'n_bins': 'Positive integer bin count excluding booleans. Source-style floating percentile grid must produce exactly n_bins + 1 boundaries. Linear quantile interpolation; repeated edges and classes are retained.',
    'state': 'Portable JSON dictionary with exactly format, response_coordinate and boundaries. Format sciona.quantile_response.v1; at least two finite nondecreasing numeric boundaries with finite midpoint computations. Exact response-coordinate identity required. Fitted boundaries are runtime state, not public evaluation data.',
    'soft': 'Explicit boolean. True returns float64 normalized exponential distance weights. False returns float32 left-open/right-closed indicators, preserving all-zero rows at the minimum or outside the fitted range.',
    'precision': 'Positive finite numeric precision, excluding booleans, in inverse squared response-coordinate units. Used only in soft mode but required valid in both modes. Scale changes require inverse-square precision adjustment. Direct exponential normalization underflow or overflow is rejected.',
    'probabilities': 'Nonempty finite nonnegative float32 or float64 matrix with one column per fitted bin, in exact fitted bin order. Rows sum to one within rtol 1e-6 and atol 1e-8; all-zero hard-encoding rows are rejected. Caller owns probability calibration and bin-column semantics.',
}
OUTPUTS = {
    'fit_quantile_response': CONSTRAINTS['state'],
    'encode_quantile_response': 'Finite matrix (target rows, fitted bins): float64 normalized soft weights or float32 hard indicators. Hard rows may sum to zero; they are not necessarily probability distributions. No refit.',
    'decode_quantile_response': 'Finite float64 vector of probability-weighted bin midpoints in the declared normalized response coordinate. No inverse response transform or calibration is inferred; overflow is rejected.',
}



def plan():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed connected execution evidence changed')
    evidence = json.loads(raw)
    if (not evidence['measurement_completed'] or not evidence['exact_equivalence']
            or evidence['restored_models'] != 12 or evidence['quantile_models'] != 6
            or not all(r['full_training_encoding_exact'] for r in evidence['cases'] if r['quantile_model'])):
        raise ValueError('Registered-provider connected qualification required')
    for relative, digest in evidence['software_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Execution implementation changed: '+relative)
    for module, digest in evidence['provider_sha256'].items():
        if hashlib.sha256(Path(importlib.import_module(module).__file__).read_bytes()).hexdigest() != digest:
            raise ValueError('Connected provider implementation changed: '+module)
    repo = ROOT.parent/'sciona-atoms-ml'
    provider_path = repo/'src/sciona/atoms/ml/quantile_response_encoding.py'
    provider = dict(revision='11395be', sha256=evidence['provider_sha256'][MODULE],
                    functions=['fit_quantile_response', 'encode_quantile_response', 'decode_quantile_response'])
    committed = subprocess.check_output(['git', 'show', provider['revision']+':src/sciona/atoms/ml/quantile_response_encoding.py'], cwd=repo)
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
            output = OUTPUTS[name]
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
                              required=True, default_value_repr='', constraints=output))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+name, atom_id=str(identity), version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=provider['sha256'], semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='quantile-response-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=provider['revision'], planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Draft identities only; canonical draft and legacy flagged, both unpublished, until separate publication qualification.',
            'Witnesses establish shape and coordinate compatibility only; caller owns training-only fitting, bin ordering, response transforms and precision-unit conversion.',
            'Complete metadata, dependency qualification, transaction rollback tests and served-selection verification remain required.',
            'No original competition CDG approval or historical winning-ensemble completeness is implied.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/quantile_response_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), ports_per_catalog=sum(len(p['interfaces']) for p in result['new_providers']), applied=False)))
