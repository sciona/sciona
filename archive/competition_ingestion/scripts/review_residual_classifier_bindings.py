"""Read-only publication preflight for the complete numerical residual graph."""
import hashlib
import inspect
import json
from pathlib import Path
import tomllib

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.visualizer.runner import _ensure_atoms_imported
from scripts.review_conditional_correction_environment import dependency_closure

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    graph = build_residual_classifier_graph()
    digest = encode_execution_graph(graph)[0]
    directory = ROOT/'docs/reviews'
    execution = json.loads((directory/'residual_classifier_graph_execution.json').read_text())
    witnesses = json.loads((directory/'residual_classifier_graph_witnesses.json').read_text())
    if not execution['passed'] or not witnesses['passed'] or execution['graph_sha256'] != digest or witnesses['graph_sha256'] != digest:
        raise ValueError('Matching passing runtime and symbolic evidence required')
    if execution['builder_sha256'] != sha(ROOT/'sciona/residual_classifier_graph.py') or execution['validator_sha256'] != sha(ROOT/'scripts/validate_residual_classifier_graph.py') or witnesses['validator_sha256'] != sha(ROOT/'scripts/validate_residual_graph_witnesses.py'):
        raise ValueError('Evidence implementation drift')
    if len(execution['scenarios']) != 10 or not all(s['final_int32_predictions_exact'] and s['conditional_offsets_exact'] and s['selected_populations_exact'] for s in execution['scenarios']):
        raise ValueError('Complete numerical execution evidence missing')
    repo = ROOT.parent/'sciona-atoms-ml'
    specs = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', repo), artifact_root=repo/'src/sciona/atoms')
    _ensure_atoms_imported()
    bindings = {}
    for node in graph.nodes:
        runtime = node.matched_primitive
        matches = [s for s in specs if s.import_module+'.'+s.source_symbol == runtime]
        if len(matches) != 1 or runtime not in REGISTRY:
            raise ValueError('Unique registered inventory binding required: '+runtime)
        spec = matches[0]
        function = REGISTRY[runtime]['impl']
        if Path(inspect.getsourcefile(function)).resolve() != spec.file_path.resolve():
            raise ValueError('Provider import shadowing')
        if list(inspect.signature(function).parameters) != [p.name for p in node.inputs]:
            raise ValueError('Graph/callable parameter mismatch: '+runtime)
        binding = bindings.setdefault(runtime, dict(runtime=runtime, fqdn=spec.fqdn,
            content_hash=spec.content_hash, version_id=str(spec.version_id),
            source_sha256=sha(spec.file_path), nodes=[]))
        binding['nodes'].append(node.node_id)
    source_projects = {name: tomllib.loads(path.read_text())['project'] for name,path in [
        ('sciona-atoms-ml', repo/'pyproject.toml'), ('sciona', ROOT/'pyproject.toml')]}
    closure = dependency_closure(['sciona-atoms-ml[xgboost]', 'sciona', 'psycopg', 'python-dotenv'], source_projects)
    if not closure['compatible']:
        raise ValueError('Provisioned in-process dependency closure differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for binding in bindings.values():
            rows = db.execute('SELECT a.artifact_id,a.status,a.is_publishable,v.version_id,v.content_hash '
                'FROM artifacts a LEFT JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND v.is_latest '
                'WHERE a.fqdn=%s', (binding['fqdn'],)).fetchall()
            if len(rows) > 1:
                raise ValueError('Ambiguous current canonical atom identity')
            binding['current_catalog'] = [{k:str(v) if k in ('artifact_id','version_id') and v is not None else v for k,v in row.items()} for row in rows]
        source = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',
            ('7d6f7b78-1dff-5757-9326-4edad55aaa68',)).fetchone()
        if source != dict(status='draft', is_publishable=False, content_hash='5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'):
            raise ValueError('Original source intake identity/status differs')
    return dict(format='residual-classifier-binding-review.v1', passed=True, approved=False,
        read_only=True, catalog_mutations=0, proposed_tier=3, graph_sha256=digest, nodes=len(graph.nodes),
        unique_runtime_bindings=len(bindings), bindings=list(bindings.values()), dependency_closure=closure,
        source_manifest_sha256={'provider':sha(repo/'pyproject.toml'),'matcher':sha(ROOT/'pyproject.toml')},
        evidence_sha256={name:sha(directory/name) for name in ['residual_classifier_graph_execution.json','residual_classifier_graph_witnesses.json']},
        reviewer_sha256=sha(Path(__file__)), remaining=['License notice closure and exact execution-dependency evidence pins.',
            'Transactional draft/version import, negative database gates and approval rollback/apply/idempotency/served execution.',
            'Catalog-facing domain adapter graph and complete original-intake qualification remain separate required work.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/residual_classifier_binding_review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('bindings','dependency_closure')}))
