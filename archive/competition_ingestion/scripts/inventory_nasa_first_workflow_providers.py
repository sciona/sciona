"""Inventory exact lifecycle dependencies, including lazily executed branches.

This is a read-only catalog snapshot and contract inventory, not approval.
No runtime inputs or model payloads are inspected or exported.
"""
import contextlib
import hashlib
import inspect
import io
import json
from collections import defaultdict
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY
from sciona.guarded_prediction_graphs import EMIT, validate_descriptor
from sciona.nasa_first_state_graphs import (
    build_nasa_first_training_state_graph, build_nasa_first_state_inference_graph,
)
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.visualizer.runner import _ensure_atoms_imported

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _ensure_atoms_imported()
    graphs = {
        'training': build_nasa_first_training_state_graph(),
        'inference': build_nasa_first_state_inference_graph(),
    }
    uses = defaultdict(list)
    graph_records = {}

    def visit(label, graph):
        digest, _, _ = encode_execution_graph(graph)
        graph_records[label] = dict(sha256=digest, nodes=len(graph.nodes), edges=len(graph.edges))
        for node in graph.nodes:
            runtime = node.matched_primitive
            if runtime not in REGISTRY:
                raise ValueError('Unregistered graph provider: ' + str(runtime))
            signature = inspect.signature(REGISTRY[runtime]['impl'])
            names = [p.name for p in node.inputs]
            signature.bind(**dict.fromkeys(names))
            uses[runtime].append(dict(graph=label, node=node.node_id,
                inputs=[p.model_dump(mode='json') for p in node.inputs],
                outputs=[p.model_dump(mode='json') for p in node.outputs]))
        for key, value in graph.metadata.items():
            if isinstance(value, dict) and {'hash', 'nodes', 'edges', 'output_node', 'output_port'} <= value.keys():
                nested, output = validate_descriptor(value)
                visit(label + '/' + key, nested)
                uses[EMIT].append(dict(graph=label + '/' + key, node='__emit',
                    controller_injected=True, output_type=output.type_desc,
                    input_names=['value', 'receive'], output_names=['result']))

    for label, graph in graphs.items():
        visit(label, graph)
    repo = ROOT.parent / 'sciona-atoms-ml'
    specs = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', repo), artifact_root=repo/'src/sciona/atoms')
    specs = {s.import_module + '.' + s.source_symbol: s for s in specs}
    records = []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for runtime, sites in sorted(uses.items()):
            spec = specs[runtime]
            implementation = REGISTRY[runtime]['impl']
            path = Path(inspect.getsourcefile(implementation)).resolve()
            if not path.is_relative_to(repo.resolve()):
                raise ValueError('Provider implementation outside expected repository')
            catalog = db.execute('SELECT artifact_id,status,is_publishable FROM artifacts WHERE fqdn=%s', (spec.fqdn,)).fetchall()
            served = db.execute('SELECT DISTINCT artifact_id,review_status,trust_readiness FROM catalog_artifacts_served WHERE fqdn=%s', (spec.fqdn,)).fetchall()
            versions = db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s', (str(spec.version_id),)).fetchall()
            records.append(dict(runtime_fqdn=runtime, fqdn=spec.fqdn, version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_source=str(path.relative_to(repo)), provider_sha256=sha(path),
                role='domain_adapter' if '.domain_adapters.' in runtime else 'reusable_operation',
                callable_inputs=list(inspect.signature(implementation).parameters),
                description=inspect.getdoc(implementation), graph_uses=sites,
                catalog=catalog, served=served, matching_version=versions))
    return dict(read_only=True, approved=False, catalog_mutations=0,
        graph_records=graph_records, provider_count=len(records),
        reusable_operations=sum(r['role']=='reusable_operation' for r in records),
        domain_adapters=sum(r['role']=='domain_adapter' for r in records),
        served_provider_count=sum(bool(r['served']) for r in records),
        providers=records, inventory_sha256=sha(Path(__file__)),
        limitations=['Served presence alone does not prove current version approval or semantic compatibility.',
                     'Graph port variants require a canonical provider contract before staging.',
                     'Runtime package/license review and version-bound publication gates remain required.'])


if __name__ == '__main__':
    report = inventory()
    (ROOT/'docs/reviews/competition_nasa_first_workflow_providers.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps({k:report[k] for k in ['provider_count','reusable_operations','domain_adapters','served_provider_count','catalog_mutations']}))
