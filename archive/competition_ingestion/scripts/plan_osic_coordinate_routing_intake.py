"""Freeze the source-schema coordinate routing and its synthetic evidence."""
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
PROVIDER_REVISION = 'bfe6e9b'
EVIDENCE_COMMIT = '65036201'
MODULE = 'sciona.atoms.ml.domain_adapters.osic_coordinate_routing'
PROVIDER_PATH = 'src/sciona/atoms/ml/domain_adapters/osic_coordinate_routing.py'
GRAPH_COMMIT = '5ed15bda'
EVIDENCE_FILES = ('docs/reviews/osic_combined_graph_structure.json', 'docs/reviews/osic_combined_inference_graph.json')
CONSTRAINTS = {
    'image_response_coordinate': ('str', 'Nonblank exact Python string matching the private image slope state. Preserve spelling without trimming. Caller establishes the relation between slope, time and response units.'),
    'quantile_response_coordinate': ('str', 'Nonblank exact Python string matching the private quantile response state. Preserve spelling without trimming. Caller establishes compatibility with reduced image predictions and uncertainty meanings.'),
}
OUTPUTS = {'route_response_coordinates': 'Tuple of the original image and quantile coordinate strings in that order. Keep declarations independent and preserve exact identity/spelling. Equal labels are permitted but do not prove compatibility. No unit inference, conversion, numerical calculation or state authentication.'}
COMMON = ' Concrete in-process declaration routing only; abstract witness rejects. Caller owns physical meanings and downstream compatibility. No empirical transfer, HTTP transport or whole-CDG approval claim.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    from scripts.validate_osic_combined_graph_structure import validate
    reports = {}
    for commit, name in zip((EVIDENCE_COMMIT, GRAPH_COMMIT), EVIDENCE_FILES):
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        if raw != (ROOT/name).read_bytes():
            raise ValueError('Immutable qualification evidence changed')
        reports[name] = json.loads(raw)
    local, graph = [reports[name] for name in EVIDENCE_FILES]
    if validate() != local:
        raise ValueError('Fresh coordinate routing qualification differs')
    if not all(r.get('passed') is True and r.get('synthetic_only') is True for r in reports.values()):
        raise ValueError('Successful synthetic qualification required')
    execution = graph['execution']
    if (execution['nodes'] != 15 or execution['edges'] != 26
            or local['missing_edges_rejected'] != 26 or local['coordinate_cases'] != 3
            or local['coordinate_rejections'] != 11
            or not all(execution[k] is True for k in ('exact_image_predictions','exact_quantile_features',
                'exact_quantile_predictions','exact_keyed_blend','independent_weighted_blend_exact',
                'distinct_response_coordinates_preserved','private_state_unchanged','numeric_bundle_unchanged'))):
        raise ValueError('Qualified combined execution and declaration routing required')
    for section in ('software_sha256','overlay_software_sha256'):
        for name, digest in graph[section].items():
            if sha(ROOT/name) != digest:raise ValueError('Qualified graph implementation changed')
    import importlib.util
    if execution['provider_sha256'] != local['provider_sha256']:
        raise ValueError('Structural and numerical provider implementations differ')
    for module, digest in execution['provider_sha256'].items():
        if sha(importlib.util.find_spec(module).origin) != digest:
            raise ValueError('Qualified graph provider changed')
    if sha(ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH) != local['provider_sha256'][MODULE]:
        raise ValueError('Qualified coordinate routing changed')
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
        raise ValueError('Exactly one coordinate routing required')
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
    return dict(format='osic-coordinate-routing-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, graph_evidence_commit=GRAPH_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'The provider uses only Python string validation and tuple construction, with no registered atom dependencies. Combined-graph evidence is composed use, not a dependency of routing.',
            'Coordinate strings are immutable borrowed values; callers retain responsibility for meaning, state agreement and unit compatibility.',
            'The graph remains unapproved; original workflow execution and empirical transfer are separate.'])



if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_coordinate_routing_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
