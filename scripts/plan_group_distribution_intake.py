"""Prepare group-subset scorer with an exact approved KS dependency."""
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
EVIDENCE_COMMIT = '40df3de1'
EVIDENCE = 'docs/reviews/lanl_group_distribution_validation.json'
MODULE = 'sciona.atoms.ml.group_distribution_matching'
PROVIDER_REVISION = '08c7fb3'
CONSTRAINTS = {
    'training': 'Nonempty finite real rows-by-features NumPy matrix. Caller provides common feature ordering, units and preprocessing with reference. Converted to float64; large integer precision may be lost.',
    'reference': 'Nonempty finite real rows-by-features NumPy matrix with training width and identical feature coordinates. Reference population must be available when selection occurs; query-driven selection is transductive.',
    'row_groups': 'One integer group identifier per training row. Exact observed labels must equal group_labels. Whole groups are included without balancing; larger groups have greater empirical mass.',
    'candidates': 'Nonempty boolean candidates-by-groups mask with columns in group_labels order. Every row selects at least one whole group. Duplicate candidate rows allowed and preserved. No subset generation occurs.',
    'group_labels': 'Nonempty unique Python integer list defining candidate-mask column order; must exactly cover observed row_groups values.',
    'feature_columns': 'Unique nonblank string list defining identical ordered features in training and reference; length equals both matrix widths. Caller must establish alignment.',
    'feature_coordinate': 'Nonblank shared coordinate identifier binding upstream feature semantics and preprocessing. No inference, conversion or feature selection occurs.',
}
OUTPUT = 'Independent float64 candidates-by-features matrix of two-sided marginal empirical KS distances in [0,1]. Uses registered compute_ks_agreement with equal row weights. No p-values, averaging, candidate selection or tie-breaking. No guarantee of joint agreement or predictive accuracy. Witness checks shape and dimension compatibility; runtime also validates finite values and complete group membership.'


def check_dependency(db, item):
    for prefix, key in [('artifact', 'artifact_id'), ('atom', 'atom_id')]:
        rows = db.execute(f"""SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
            FROM {prefix}s a JOIN {prefix}_versions v USING({key})
            WHERE a.{key}=%s AND v.is_latest AND a.status='approved' AND a.is_publishable
            AND {prefix}_is_publishable(a.{key})
            AND EXISTS(SELECT 1 FROM catalog_{prefix}s_served s WHERE s.{key}=a.{key})""",
            (item['artifact_id'],)).fetchall()
        if rows != [{k:item[k] for k in ('fqdn','version_id','content_hash','trust_tier')}]:
            raise ValueError('Exact approved served KS dependency required in both catalogs')



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed execution evidence changed')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('synthetic_only') is not True
            or len(report['cases']) != 2 or len(report['reused_catalog_dependencies']) != 1
            or report['cases'][0]['candidates'] != 10000 or report['cases'][1]['candidates'] != 64
            or sum(r['distances_checked'] for r in report['cases']) != 40256
            or not all(r['maximum_absolute_error'] < 1e-12 and r['selected_oracle_minimizer'] is True
                       and r['construction'] for r in report['cases'])):
        raise ValueError('Complete group-distribution evidence required')
    for relative, digest in report['software_sha256'].items():
        if sha(ROOT.parent/relative) != digest:
            raise ValueError('Qualified implementation changed')
    return raw


def plan():
    raw = evidence()
    dependencies = json.loads(raw)['reused_catalog_dependencies']
    repo = ROOT.parent/'sciona-atoms-ml'
    relative = 'src/sciona/atoms/ml/group_distribution_matching.py'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+relative], cwd=repo)
    if committed != (repo/relative).read_bytes():
        raise ValueError('Immutable provider changed')
    definitions = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
        artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 1 or specs[0].source_symbol != 'group_subset_ks':
        raise ValueError('Exactly one registered operation required')
    spec = specs[0]
    identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        check_dependency(db, dependencies[0])
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
    return dict(format='group-distribution-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=[record], registered_provider_dependencies=dependencies,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(Path(__file__)),
        limitations=['Non-serving draft only; metadata, dependencies, semantic review and transactional publication gates remain required.',
            'Independent empirical-CDF qualification; no historical feature or model reconstruction claim.',
            'Whole-group scoring with explicit application adapters; no original LANL CDG approval.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/group_distribution_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=1, ports_per_catalog=len(result['new_providers'][0]['interfaces']), applied=False)))
