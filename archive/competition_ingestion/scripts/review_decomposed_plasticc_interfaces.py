"""Read-only registration, exact interface, boundary and catalog identity audit."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import get_args, get_origin, get_type_hints
import numpy as np
from sciona.architect.models import EdgeKind
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_plasticc_graph import build_decomposed_plasticc_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import boundary_ports, provider_ports
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.audit_decomposed_plasticc_readiness import sha
from scripts.promote_matern_providers import check as check_matern

EVIDENCE = 'docs/reviews/decomposed_plasticc_graph_execution_v2.json'
KERNEL_PLAN = 'docs/reviews/matern_provider_publication_plan.json'
KERNEL = 'sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms.matern_kernel_matrix'


def pinned(path, commit):
    raw = (ROOT/path).read_bytes()
    if raw != subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT):
        raise ValueError('Committed evidence changed: '+path)
    return json.loads(raw)


def interfaces(fn, node):
    kinds = {object: 'object', dict: 'dict', np.ndarray: 'ndarray', list: 'list', int: 'int', float: 'float', bool: 'bool'}
    signature = inspect.signature(fn); hints = get_type_hints(fn)
    checked = provider_ports(node, fn)
    for port, parameter in zip(checked['input'], signature.parameters.values(), strict=True):
        if parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or parameter.default is not inspect.Parameter.empty:
            raise ValueError('New provider inputs must be explicit and required')
        if hints.get(port.name) not in kinds or kinds[hints[port.name]] != port.type_desc:
            raise ValueError('Input annotation differs')
    returned = get_args(hints['return']) if get_origin(hints['return']) is tuple else (hints['return'],)
    if any(t not in kinds for t in returned) or [kinds[t] for t in returned] != [p.type_desc for p in checked['output']]:
        raise ValueError('Output annotation differs')
    return [dict(direction=direction, name=p.name, ordinal=i, type_desc=p.type_desc,
                 required=p.required, default_value_repr=p.default_value_repr, constraints=p.constraints)
            for direction in ('input', 'output') for i, p in enumerate(checked[direction])]


def review():
    execution = pinned(EVIDENCE, 'b5a0a8be')
    if execution['passed'] is not True or len(execution['cases']) != 2:
        raise ValueError('Complete execution evidence required')
    if any(sha(ROOT/p) != h for p, h in execution['implementation_sha256'].items()):
        raise ValueError('Executed software changed')
    graph = build_decomposed_plasticc_graph()
    if encode_execution_graph(graph)[0] != execution['graph_sha256']:
        raise ValueError('Executed graph changed')
    directory = ROOT.parent/'sciona-atoms-ml'
    wanted = {n.matched_primitive for n in graph.nodes}; specs = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory), artifact_root=directory/'src/sciona/atoms'):
        name = spec.import_module+'.'+spec.source_symbol
        if name not in wanted: continue
        if name in specs: raise ValueError('Ambiguous registration')
        specs[name] = spec
    if set(specs) != wanted or len(specs) != 25: raise ValueError('Exact25-provider inventory required')
    kernel_plan = pinned(KERNEL_PLAN, 'd650497e')
    kernel = kernel_plan['records'][0]
    records = {}; ports = {}
    for node in graph.nodes:
        name = node.matched_primitive; spec = specs[name]
        module = importlib.import_module(spec.import_module); fn = getattr(module, spec.source_symbol)
        if Path(inspect.getfile(inspect.unwrap(fn))).resolve() != spec.file_path.resolve():
            raise ValueError('Runtime origin differs')
        if sha(spec.file_path) != execution['provider_sha256'][name]: raise ValueError('Executed provider changed')
        ports[node.node_id] = provider_ports(node, fn)
        if name == KERNEL:
            if spec.fqdn != kernel['fqdn'] or str(spec.version_id) != kernel['version_id'] or spec.content_hash != kernel['content_hash']:
                raise ValueError('Published kernel identity differs')
            for direction in ('input', 'output'):
                actual = [(p.name, p.type_desc, p.required, p.default_value_repr) for p in ports[node.node_id][direction]]
                expected = [(p['name'], p['type_desc'], p['required'], p['default_value_repr']) for p in kernel['interfaces'] if p['direction'] == direction]
                if actual != expected: raise ValueError('Published kernel ports differ')
            continue
        if not name.startswith(('sciona.atoms.ml.decomposed_plasticc_numeric.', 'sciona.atoms.ml.decomposed_plasticc_lifecycle.')):
            raise ValueError('Unexpected provider namespace')
        try: module.concrete_only(*[object() for _ in node.inputs])
        except ValueError: pass
        else: raise ValueError('Concrete witness must reject symbolic execution')
        record = dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
            content_hash=spec.content_hash, provider_sha256=sha(spec.file_path), interfaces=interfaces(fn, node))
        if name in records and records[name] != record: raise ValueError('Repeated provider interfaces differ')
        records[name] = record
    callable_nodes = {n.node_id for n in graph.nodes if any(e.source_id == n.node_id for e in graph.edges)
        and all(e.edge_kind == EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id == n.node_id)}
    if sorted(callable_nodes) != execution['callable_nodes']: raise ValueError('Callable boundary differs')
    concrete = graph.model_copy(update={'nodes': [n for n in graph.nodes if n.node_id not in callable_nodes]})
    boundary = boundary_ports(concrete, ports)
    if [p.name for p in boundary['input']] != ['payload'] or [p.name for p in boundary['output']] != ['result']:
        raise ValueError('Complete root boundary differs')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_matern(db, kernel_plan, published=True)
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (kernel['artifact_id'],)).fetchone():
            raise ValueError('Qualified kernel is not served')
        for record in records.values():
            module, symbol = record['runtime_fqdn'].rsplit('.', 1)
            if db.execute('SELECT 1 FROM atoms WHERE import_module=%s AND source_symbol=%s', (module, symbol)).fetchone():
                raise ValueError('Runtime already cataloged')
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                if db.execute(f'''SELECT 1 FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s OR v.version_id=%s OR v.content_hash=%s''',
                    (record['catalog_fqdn'], record['version_id'], record['content_hash'])).fetchone():
                    raise ValueError('Catalog identity already present')
    return dict(format='decomposed-plasticc-interfaces.v1', passed=True, approved=False, catalog_mutations=0,
        graph_digest=execution['graph_sha256'], new_providers=list(records.values()), existing_providers=[dict(runtime_fqdn=KERNEL, **kernel)],
        root_inputs=['payload'], root_outputs=['result'], callable_only_nodes=sorted(callable_nodes),
        execution_evidence_sha256=sha(ROOT/EVIDENCE), kernel_publication_plan_sha256=sha(ROOT/KERNEL_PLAN),
        code_sha256={p: sha(ROOT/p) for p in ['scripts/review_decomposed_plasticc_interfaces.py',
            'scripts/promote_connectomics_execution.py', 'scripts/promote_matern_providers.py']},
        limits=['Read-only exact interface and identity review; semantic contracts and qualification remain pending.',
                'Duplicate checks cover registration, version and content identities, not differently named semantic alternatives.',
                'Published kernel approval is checked and preserved; no new provider or original CDG is approved here.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_plasticc_interfaces.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, new_providers=len(result['new_providers']), reused_providers=len(result['existing_providers']), approved=False)))
