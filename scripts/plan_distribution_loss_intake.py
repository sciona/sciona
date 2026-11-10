"""Freeze concrete differentiable loss ports and source-qualified provider identity."""
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
PROVIDER_REVISION = '1f5da19'
EVIDENCE_COMMIT = '98113163'
MODULE = 'sciona.atoms.ml.tensorflow_distribution_losses'
PROVIDER_PATH = 'src/sciona/atoms/ml/tensorflow_distribution_losses.py'
EVIDENCE = 'docs/reviews/osic_quantile_objective_validation.json'
CONSTRAINTS = {
    'targets': ('TensorFlow float32 Tensor or convertible float32 array',
        'Finite nonempty shape (N,1), aligned row-for-row with predictions. Explicit float32 required; no automatic float64 conversion. Caller supplies target units, row identity and observation availability.'),
    'predictions': ('TensorFlow float32 Tensor or Variable or convertible float32 array',
        'Finite shape (N,Q) for pinball or (N,3) for clipped score. Same units as targets. Pinball columns align with quantiles; score columns are low, point, high. Crossing is permitted. Tensor/Variable conversion preserves the TensorFlow gradient path when used within the caller gradient tape.'),
    'quantiles': ('float32-convertible vector',
        'Shape (Q,), Q>0; every quantile strictly between zero and one after float32 conversion. Explicit column correspondence; sorting and uniqueness are not required or performed. Quantiles are dimensionless caller policy.'),
    'minimum_scale': ('float or int',
        'Positive finite Python scalar, excluding bool, remaining finite and positive in float32. Floor on high-minus-low spread, in target units. Crossed or negative spreads encounter this floor.'),
    'maximum_error': ('float or int',
        'Positive finite Python scalar, excluding bool, remaining finite and positive in float32. Cap on absolute point error, in target units. No unit conversion is performed.'),
}
OUTPUTS = {
    'mean_pinball': 'Scalar float32 Tensor: arithmetic mean over all N*Q pinball terms, in target units. Differentiable almost everywhere; TensorFlow defines subgradients at ties. No calibration or prediction ordering guarantee.',
    'clipped_laplace_score': 'Scalar float32 Tensor: row mean of sqrt(2)*min(abs(target-point),cap)/max(high-low,floor) + log(sqrt(2)*max(high-low,floor)). Numeric score depends on the caller unit convention; the logarithm uses the numerical scale in those units. Piecewise differentiable, with TensorFlow subgradients at boundaries. No calibration guarantee.',
}
COMMON = ' Concrete provisioned TensorFlow execution only; abstract witness rejects. No training, optimizer or state mutation. Extreme finite float32 inputs can overflow intermediate arithmetic; callers must check loss and gradients before updates. TensorFlow tensors and gradient tapes are in-process objects; JSON/HTTP serialization is not qualified.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Immutable source evidence changed')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('synthetic_only') is not True
            or len(report['source_comparisons']) != 9
            or not all(c['value_exact'] and c['gradient_exact'] for c in report['source_comparisons'])
            or len(report['rejected_contract_cases']) != 8):
        raise ValueError('Source value, gradient and rejection evidence required')
    for name, digest in report['software_sha256'].items():
        path = (ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH if name == 'distribution_loss_provider'
                else ROOT/name.removeprefix('matcher/'))
        if sha(path) != digest:
            raise ValueError('Qualified source changed')
    return raw


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 2 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly two distribution loss providers required')
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
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='TensorFlow float32 scalar Tensor',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='distribution-loss-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'TensorFlow is a software dependency, not a registered atom dependency.',
            'The source-qualified loss pair does not qualify the complete OSIC workflow or empirical transfer.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/distribution_loss_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
