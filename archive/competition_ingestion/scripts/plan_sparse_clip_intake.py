"""Prepare exact non-serving sparse-clipping provider identities and port contracts."""
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
EVIDENCE_COMMIT = '46ae528a'
EVIDENCE = 'docs/reviews/sparse_clip_feature_integration.json'
MODULE = 'sciona.atoms.ml.fitted_sparse_clipping'
CONSTRAINTS = {
    'training': 'Canonical real numeric CSR matrix with at least one row and one feature. Float32 conversion precedes fitting. Maximum uses stored entries only, excludes implicit zeros and ignores NaNs. Empty or all-NaN stored data, infinities and negative fitted maxima are rejected. Caller supplies training-only data in comparable normalized numeric feature coordinates.',
    'values': 'Canonical real numeric CSR matrix matching fitted feature width, identity, ordering and normalized coordinate. Converted to float32 without densification; NaNs become zero and infinities clip to the frozen [0, upper] interval. Empty query rows supported. Duplicate entries and unsorted CSR indices must be handled explicitly upstream.',
    'feature_columns': 'Nonempty ordered list of unique nonblank feature identifiers matching matrix width. Identifiers bind feature semantics and order, not data rows. Caller must preserve fitted vocabulary and preprocessing semantics; equal width alone is insufficient.',
    'feature_coordinate': 'Nonblank identifier of common comparable normalized numeric feature coordinates. A global maximum across unlike physical units is not meaningful; caller must normalize upstream. Exact identifier must match fitted state.',
    'state': 'Portable JSON dictionary with exactly format, feature_columns, feature_coordinate and upper. Format sciona.fitted_sparse_clip.v1; finite nonnegative upper at most float32 maximum. No stored observations or executable deserialization.',
}



def plan():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed connected execution evidence changed')
    evidence = json.loads(raw)
    if (not evidence['passed'] or evidence['registered_clip_stages'] != 3 or len(evidence['cases']) != 9
            or not all(r['exact_canonical_csr_equal'] for r in evidence['cases'])):
        raise ValueError('Registered-provider execution qualification required')
    for relative, digest in evidence['software_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Execution implementation changed: '+relative)
    repo = ROOT.parent/'sciona-atoms-ml'
    provider_path = repo/'src/sciona/atoms/ml/fitted_sparse_clipping.py'
    provider = dict(revision='398cc29', sha256=evidence['provider_sha256'],
                    functions=['fit_sparse_clip', 'apply_sparse_clip'])
    committed = subprocess.check_output(['git', 'show', provider['revision']+':src/sciona/atoms/ml/fitted_sparse_clipping.py'], cwd=repo)
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
            output = CONSTRAINTS['state'] if name.startswith('fit_') else 'Independent canonical float32 CSR matrix with unchanged shape and stored-entry topology, clipped to [0, fitted upper]. Input matrix is not mutated. Explicit zeros are retained; NaNs become zero and infinities are clipped.'
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
                              required=True, default_value_repr='', constraints=output))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+name, atom_id=str(identity), version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=provider['sha256'], semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='sparse-clip-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=provider['revision'], planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Draft identities only; canonical draft and legacy flagged, both unpublished, until separate publication qualification.',
            'Witnesses establish shape and coordinate compatibility only; caller owns training-only fitting, feature vocabulary and semantic correctness of coordinate labels.',
            'Complete metadata, dependency qualification, transaction rollback tests and served-selection verification remain required.',
            'No original competition CDG approval or historical winning-ensemble completeness is implied.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/sparse_clip_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), ports_per_catalog=sum(len(p['interfaces']) for p in result['new_providers']), applied=False)))
