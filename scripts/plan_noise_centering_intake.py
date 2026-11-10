"""Prepare one non-serving provider identity with explicit amplitude contracts."""
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
EVIDENCE_COMMIT = '305854c6'
EVIDENCE = 'docs/reviews/lanl_noise_centering_validation.json'
MODULE = 'sciona.atoms.ml.noise_median_centering'
PROVIDER_REVISION = '52ee6d0'
CONSTRAINTS = {
    'windows': 'Nonempty real NumPy array with axes (windows,samples); no complex, boolean or object values. Values must remain finite after float64 conversion and arithmetic. Caller supplies window boundaries, sample-rate context and amplitude units. Integer precision beyond float64 may be lost.',
    'additive_noise': 'Real finite NumPy array with exactly the same shape, sample alignment, amplitude coordinate and scale as windows. No broadcasting. Caller supplies distribution, magnitude, seed and reuse policy; no randomness is generated here.',
    'amplitude_coordinate': 'Explicit nonblank string naming the common amplitude representation and scale of both arrays. No unit conversion or runtime inference of physical dimensions. Witness rejects unequal dimensions, including one known and one unknown.',
}
OUTPUT = 'Independent float64 array with input shape. Add supplied noise before subtracting each row median; input arrays unchanged. Baseline information is lost and noise is added. No denoising or predictive-benefit guarantee. Empty windows and nonfinite or overflowing arithmetic reject. Rows may be batched independently; splitting sample windows changes results. Witness preserves shape and matching dimension, without claiming numeric bounds.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed execution evidence changed')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('synthetic_only') is not True
            or len(report['cases']) != 4
            or {r['synthetic_family'] for r in report['cases']} != {'constant', 'quantized', 'oscillatory', 'sparse_impulses'}
            or not all(r['windows'] == 2 and r['samples_per_window'] == 150000
                and r['exact_prose_equation_equal'] is True and r['finite'] is True for r in report['cases'])):
        raise ValueError('Complete synthetic arithmetic evidence required')
    for relative, digest in report['software_sha256'].items():
        if sha(ROOT.parent/relative) != digest:
            raise ValueError('Qualified implementation changed')
    return raw


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    relative = 'src/sciona/atoms/ml/noise_median_centering.py'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+relative], cwd=repo)
    if committed != (repo/relative).read_bytes():
        raise ValueError('Immutable provider changed')
    definitions = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
        artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 1 or specs[0].source_symbol != 'add_noise_median_center':
        raise ValueError('Exactly one registered operation required')
    spec = specs[0]
    identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s', (spec.fqdn, identity)).fetchone():
                raise ValueError('Provider identity already exists')
        for table in ('atom_versions', 'artifact_versions'):
            if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                raise ValueError('Version identity collision')
    node = definitions[spec.source_symbol]
    ports = [dict(direction='input', ordinal=i, name=a.arg, type_desc=ast.unparse(a.annotation),
        required=True, default_value_repr='', constraints=CONSTRAINTS[a.arg]) for i, a in enumerate(node.args.args)]
    ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
        required=True, default_value_repr='', constraints=OUTPUT))
    fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
        'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
    record = dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity), version_id=str(spec.version_id),
        content_hash=spec.content_hash, provider_sha256=sha(repo/relative), semver=spec.semver,
        fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports)
    return dict(format='noise-centering-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=[record],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(Path(__file__)),
        limitations=['Non-serving draft only; metadata, dependencies, semantic review and transactional publication gates remain required.',
            'Prose-equation qualification, not recovered historical source-code execution.',
            'Independent arithmetic operation with explicit application adapters; no original LANL CDG approval.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/noise_centering_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=1, ports_per_catalog=len(result['new_providers'][0]['interfaces']), applied=False)))
