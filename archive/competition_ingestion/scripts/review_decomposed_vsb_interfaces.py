"""Read-only exact interface, duplicate identity and published filter review."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.architect.models import EdgeKind
from sciona.decomposed_vsb_graph import build_decomposed_vsb_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_future_sales_interfaces import interfaces
from scripts.decomposed_vsb_provider_contracts import CONTRACTS
from scripts.promote_connectomics_execution import boundary_ports, provider_ports
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.audit_decomposed_vsb_readiness import sha
from scripts.promote_lfilter_providers import check as check_filter

EVIDENCE = 'docs/reviews/decomposed_vsb_graph_execution.json'
FILTER_PLAN = 'docs/reviews/lfilter_provider_publication_plan.json'
PREFIX = 'sciona.atoms.ml.decomposed_vsb_execution.'
FILTER = 'sciona.atoms.scipy.signal.lfilter'


def pinned(path, commit):
    raw = (ROOT/path).read_bytes()
    if raw != subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT):
        raise ValueError('Committed evidence changed: '+path)
    return json.loads(raw)


def review():
    execution = pinned(EVIDENCE, '5dacb736')
    if execution['passed'] is not True or len(execution['cases']) != 2:
        raise ValueError('Complete registered execution required')
    if any(sha(ROOT/p) != h for p, h in execution['implementation_sha256'].items()):
        raise ValueError('Executed implementation changed')
    graph = build_decomposed_vsb_graph()
    if encode_execution_graph(graph)[0] != execution['graph_sha256']:
        raise ValueError('Executed graph changed')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            name = spec.import_module+'.'+spec.source_symbol
            if name not in wanted:
                continue
            if name in specs:
                raise ValueError('Ambiguous registration')
            specs[name] = spec
    if set(specs) != wanted or len(specs) != 23 or len(CONTRACTS) != 22:
        raise ValueError('Exact23-provider inventory required')
    records = {}; ports = {}
    for node in graph.nodes:
        name = node.matched_primitive; spec = specs[name]
        module = importlib.import_module(spec.import_module); fn = getattr(module, spec.source_symbol)
        if Path(inspect.getfile(inspect.unwrap(fn))).resolve() != spec.file_path.resolve():
            raise ValueError('Runtime origin differs')
        if sha(spec.file_path) != execution['provider_sha256'][name]:
            raise ValueError('Executed provider changed')
        ports[node.node_id] = provider_ports(node, fn)
        if name == FILTER:
            continue
        if not name.startswith(PREFIX):
            raise ValueError('Unexpected provider namespace')
        contract = CONTRACTS[spec.source_symbol]
        if (list(contract['parameters']) != list(inspect.signature(fn).parameters)
                or not all(contract.values()) or not all(contract['parameters'].values())):
            raise ValueError('Complete ordered semantic parameter inventory required')
        try:
            module.concrete_only(*[object() for _ in node.inputs])
        except ValueError:
            pass
        else:
            raise ValueError('Concrete-only witness must reject symbolic execution')
        record = dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
                      content_hash=spec.content_hash, provider_sha256=sha(spec.file_path),
                      interfaces=interfaces(fn, node), **contract)
        if name in records and records[name] != record:
            raise ValueError('Repeated provider node interfaces differ')
        records[name] = record
    if len(records) != 22:
        raise ValueError('Exactly22 distinct new providers required')
    callable_only = {n.node_id for n in graph.nodes if any(e.source_id == n.node_id for e in graph.edges)
        and all(e.edge_kind == EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id == n.node_id)}
    expected = {'filter', 'fit_fold', 'baseline_arguments', 'baseline_residual', 'window_maxima',
                'rank_peaks', 'amplitude_gradient', 'select_ranked_peaks', 'peak_descriptors', 'phase_quadrants'}
    if callable_only != expected:
        raise ValueError('Exact ten-callable boundary required')
    concrete = graph.model_copy(update={'nodes': [n for n in graph.nodes if n.node_id not in callable_only]})
    boundary = boundary_ports(concrete, ports)
    if [p.name for p in boundary['input']] != ['payload'] or [p.name for p in boundary['output']] != ['result']:
        raise ValueError('Complete root boundary differs')
    filter_plan = pinned(FILTER_PLAN, '922c32ca')
    if filter_plan['publisher_sha256'] != sha(ROOT/'scripts/promote_lfilter_providers.py'):
        raise ValueError('Published filter verifier changed')
    reused = filter_plan['records'][0]; spec = specs[FILTER]
    if reused['fqdn'] != spec.fqdn or reused['version_id'] != str(spec.version_id) or reused['content_hash'] != spec.content_hash:
        raise ValueError('Reused filter registration differs')
    filter_node = next(n for n in graph.nodes if n.node_id == 'filter')
    for direction, node_ports in [('input', filter_node.inputs), ('output', filter_node.outputs)]:
        qualified = [p for p in reused['interfaces'] if p['direction'] == direction]
        actual = [(p.name, p.type_desc, p.required, p.default_value_repr or '') for p in node_ports]
        expected_ports = [(p['name'], p['type_desc'], p['required'], p['default_value_repr'] or '') for p in qualified]
        if actual != expected_ports:
            raise ValueError('Published filter graph interface differs')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_filter(db, filter_plan, published=True)
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (reused['artifact_id'],)).fetchone():
            raise ValueError('Qualified filter is not served')
        for r in records.values():
            module, symbol = r['runtime_fqdn'].rsplit('.', 1)
            if db.execute('SELECT 1 FROM atoms WHERE import_module=%s AND source_symbol=%s', (module, symbol)).fetchone():
                raise ValueError('Runtime implementation already cataloged')
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                if db.execute(f'''SELECT 1 FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s OR v.version_id=%s OR v.content_hash=%s''',
                    (r['catalog_fqdn'], r['version_id'], r['content_hash'])).fetchone():
                    raise ValueError('Implementation or version identity already cataloged')
    paths = ['scripts/review_decomposed_vsb_interfaces.py', 'scripts/decomposed_vsb_provider_contracts.py',
             'scripts/review_decomposed_future_sales_interfaces.py', 'scripts/promote_connectomics_execution.py',
             'scripts/promote_lfilter_providers.py']
    return dict(format='decomposed-vsb-interfaces.v1', passed=True, approved=False, catalog_mutations=0,
        graph_digest=execution['graph_sha256'], new_providers=list(records.values()),
        existing_providers=[dict(runtime_fqdn=FILTER, **reused)], root_inputs=['payload'], root_outputs=['result'],
        callable_only_nodes=sorted(callable_only), execution_evidence_sha256=sha(ROOT/EVIDENCE),
        filter_publication_plan_sha256=sha(ROOT/FILTER_PLAN), code_sha256={p: sha(ROOT/p) for p in paths},
        limits=['Exact registration, interface and identity review only; semantic qualification and publication remain pending.',
                'Duplicate checks cover runtime, version and content identity; differently named semantic alternatives require bounded review.',
                'Ten callable-only nodes require exact qualified bindings; callability and array shapes do not prove semantics or provenance.',
                'Numeric operations are independently selectable; triple grouping, nine-feature order and five-way ensemble policy remain explicit.',
                'Existing filter canonical approval, versioned interfaces and prior legacy history are preserved.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_vsb_interfaces.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, new_providers=len(result['new_providers']),
                         reused_providers=len(result['existing_providers']), approved=False)))
