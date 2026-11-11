"""Check provider registration, callable interfaces and explicit recipe contracts."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_openvaccine_graph import build_decomposed_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_m5u_implementation import interfaces
from scripts.promote_connectomics_execution import boundary_ports, provider_ports
from scripts.decomposed_openvaccine_provider_contracts import CONTRACTS

ROOT = Path(__file__).resolve().parents[1]
GENERIC = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'
PREFIX = 'sciona.atoms.ml.decomposed_openvaccine_execution.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    graph = build_decomposed_openvaccine_graph()
    wanted = {n.matched_primitive for n in graph.nodes}
    if len(graph.nodes) != 6 or len(graph.edges) != 7 or len(wanted) != 6:
        raise ValueError('Complete six-provider graph required')
    if set(CONTRACTS) != {n.matched_primitive.removeprefix(PREFIX)
                          for n in graph.nodes if n.matched_primitive.startswith(PREFIX)}:
        raise ValueError('Exact new recipe contract inventory required')
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory),
                                            artifact_root=directory/'src/sciona/atoms'):
            name = spec.import_module+'.'+spec.source_symbol
            if name in wanted:
                if name in specs:
                    raise ValueError('Ambiguous provider registration')
                specs[name] = spec
    if set(specs) != wanted:
        raise ValueError('Missing registered provider')
    records, ports = [], {}
    for node in graph.nodes:
        name = node.matched_primitive
        spec = specs[name]
        module = importlib.import_module(spec.import_module)
        function = getattr(module, spec.source_symbol)
        if Path(inspect.getfile(function)).resolve() != spec.file_path.resolve():
            raise ValueError('Provider origin differs from registration')
        contract = interfaces(function, node)
        ports[node.node_id] = provider_ports(node, function)
        semantics = None
        if name.startswith(PREFIX):
            semantics = CONTRACTS[spec.source_symbol]
            if set(semantics) != {'operation', 'plain_limits', 'validity_bounds', 'parameters',
                                  'outputs', 'dependencies', 'reuse'}:
                raise ValueError('Complete semantic contract fields required')
            if list(semantics['parameters']) != list(inspect.signature(function).parameters):
                raise ValueError('Semantic parameters differ from callable')
            if (any(not isinstance(semantics[k], str) or not semantics[k].strip()
                    for k in ('operation', 'plain_limits', 'validity_bounds', 'outputs', 'reuse'))
                    or not semantics['dependencies'] or any(not v for v in semantics['parameters'].values())):
                raise ValueError('Explicit semantics and dependencies required')
            try:
                module.concrete_only(**{p.name: object() for p in node.inputs})
            except ValueError:
                pass
            else:
                raise ValueError('Recipe witness must reject symbolic execution')
        records.append(dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id), content_hash=spec.content_hash,
            provider_sha256=sha(spec.file_path), interfaces=contract, semantics=semantics,
            classification='generic_operation' if name == GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph, ports)
    if ([p.name for p in boundary['input']] != ['payload']
            or [p.name for p in boundary['output']] != ['result']):
        raise ValueError('Root invocation boundary differs')
    paths = ['scripts/review_decomposed_openvaccine_interfaces.py',
        'scripts/decomposed_openvaccine_provider_contracts.py',
        'scripts/review_decomposed_m5u_implementation.py', 'scripts/promote_connectomics_execution.py',
        'sciona/decomposed_openvaccine_graph.py', 'sciona/services/execution_graph_codec.py']
    return dict(format='decomposed-openvaccine-interfaces.v1', passed=True, approved=False,
        catalog_mutations=0, graph_nodes=6, graph_edges=7,
        graph_digest=encode_execution_graph(graph)[0], providers=records,
        root_inputs=['payload'], root_outputs=['result'],
        code_sha256={p: sha(ROOT/p) for p in paths},
        imported_module_sha256={name: sha(inspect.getfile(importlib.import_module(name)))
            for name in ('sciona.atoms.supabase_seed', 'sciona.atoms.provider_inventory')},
        limits=['Local registration, callable/graph interfaces and explicit semantic contract presence are verified.',
                'Semantic correctness, complete execution, installed dependency files and native catalog eligibility require separate qualification.',
                'Existing prepare and ordered-sum metadata are not modified or newly certified by this report.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/decomposed_openvaccine_interfaces.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], providers=len(report['providers']), approved=False)))
