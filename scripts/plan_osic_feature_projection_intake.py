"""Freeze the source-schema feature projection and its synthetic evidence."""
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
PROVIDER_REVISION = 'c4a796f'
EVIDENCE_COMMIT = '2c54a813'
MODULE = 'sciona.atoms.ml.domain_adapters.osic_feature_projection'
PROVIDER_PATH = 'src/sciona/atoms/ml/domain_adapters/osic_feature_projection.py'
GRAPH_COMMIT = 'c529089b'
EVIDENCE_FILES = ('docs/reviews/osic_feature_projection.json', 'docs/reviews/osic_quantile_replay_graph_unicode_correction.json')
CONSTRAINTS = {
    'prepared': ('dict', 'Plain dict with exactly features, feature_columns, observed_mask and observed_values. Upstream replay owns value validation; this adapter accepts arbitrary borrowed values. Caller owns availability, units, row alignment and downstream mutation.'),
}
OUTPUTS = {'unpack_quantile_features': 'New outer tuple in fixed order: features, feature_columns, observed_mask, observed_values. Preserve borrowed identity and shared aliases. No copying, numerical validation, casting, field renaming or model execution.'}
COMMON = ' Concrete in-process projection only; abstract witness rejects. Source-schema adapter, not a generic numerical algorithm. No empirical transfer, HTTP transport or whole-CDG approval claim.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    from scripts.validate_osic_feature_projection import validate
    reports = {}
    for commit, name in zip((EVIDENCE_COMMIT, GRAPH_COMMIT), EVIDENCE_FILES):
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        if raw != (ROOT/name).read_bytes():
            raise ValueError('Immutable qualification evidence changed')
        reports[name] = json.loads(raw)
    local, graph = [reports[name] for name in EVIDENCE_FILES]
    if validate() != local:
        raise ValueError('Fresh projection qualification differs')
    if not all(r.get('passed') is True and r.get('synthetic_only') is True for r in reports.values()):
        raise ValueError('Successful synthetic qualification required')
    execution = graph['graph_execution']
    replay = graph['raw_replay_qualification']
    if (execution['nodes'] != 6 or execution['edges'] != 8
            or execution['missing_edges_rejected'] != 8
            or execution['projection_borrows_values'] is not True
            or replay['fresh_process_replay']['features_exact'] is not True
            or replay['fresh_process_replay']['predictions_exact'] is not True
            or replay['private_artifacts_unchanged'] is not True):
        raise ValueError('Qualified graph execution required')
    for name, digest in graph['software_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Qualified graph implementation changed')
    import importlib.util
    for module, digest in execution['provider_sha256'].items():
        if sha(importlib.util.find_spec(module).origin) != digest:
            raise ValueError('Qualified graph provider changed')
    for name, digest in replay['software_sha256'].items():
        path = importlib.util.find_spec(name).origin if name.startswith('sciona.atoms.') else ROOT/name
        if sha(path) != digest:
            raise ValueError('Qualified raw replay implementation changed')
    if sha(ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH) != local['provider_sha256']:
        raise ValueError('Qualified projection changed')
    return json.dumps(reports, sort_keys=True, separators=(',', ':')).encode()


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 1 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly one frozen feature projection required')
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
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='tuple',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='osic-feature-projection-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, graph_evidence_commit=GRAPH_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'The provider uses Python container operations and has no registered atom dependencies. Quantile-graph evidence is a composed use, not a dependency of projection.',
            'Borrowed values are not validated or copied; callers retain responsibility for downstream mutation, alignment and units.',
            'The graph remains unapproved; original workflow execution and empirical transfer are separate.'])



if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_feature_projection_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
