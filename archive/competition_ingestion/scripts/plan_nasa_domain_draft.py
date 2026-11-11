"""Resolve a domain draft while preserving approved numerical provider identities."""
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.domain_adapters.airport_features as provider
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY
from sciona.nasa_domain_graph import build_nasa_domain_graph
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.plan_residual_classifier_draft import ROOT, sha, merge_ports, plan as numerical_plan
from scripts.validate_residual_classifier_database_gates import check_staged

FQDN = 'cdg.domain.airport.corrected_residual_classifier'


def audit():
    report_path = ROOT/'docs/reviews/competition_nasa_domain_graph.json'
    report = json.loads(report_path.read_text())
    graph = build_nasa_domain_graph()
    digest = encode_execution_graph(graph)[0]
    if not report['passed'] or report['graph_sha256'] != digest or report['output_rows'] != 640 or not report['materialized_numerical_preflight_before_fitting']:
        raise ValueError('Matching domain execution evidence required')
    paths = dict(domain_provider=Path(provider.__file__), domain_builder=ROOT/'sciona/nasa_domain_graph.py',
                 domain_validator=ROOT/'scripts/validate_nasa_domain_graph.py',
                 materialized_preflight=ROOT/'sciona/residual_metadata_preflight.py')
    import importlib
    for name, expected in report['execution_source_sha256'].items():
        path = paths.get(name)
        if path is None:
            path = Path(importlib.import_module(name).__file__) if name.startswith('sciona.atoms.ml.') else ROOT/name
        if sha(path) != expected:
            raise ValueError('Domain execution source drift: '+name)
    repo = ROOT.parent/'sciona-atoms-ml'
    specs = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),artifact_root=repo/'src/sciona/atoms')
    bindings = {}
    for node in graph.nodes:
        runtime = node.matched_primitive
        matches = [s for s in specs if s.import_module+'.'+s.source_symbol == runtime]
        if len(matches)!=1 or runtime not in REGISTRY:
            raise ValueError('Unique registered implementation required: '+runtime)
        spec = matches[0]
        function = REGISTRY[runtime]['impl']
        if Path(inspect.getsourcefile(function)).resolve()!=spec.file_path.resolve() or list(inspect.signature(function).parameters)!=[p.name for p in node.inputs]:
            raise ValueError('Provider shadowing or parameter mismatch')
        bindings.setdefault(runtime,dict(runtime=runtime,fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,source_sha256=sha(spec.file_path)))
    return dict(passed=True,approved=False,catalog_mutations=0,graph_sha256=digest,bindings=list(bindings.values()),
        scope='Draft decomposed corrected single-airport feature preparation and residual-classifier execution; materialized metadata preflight before fitting.',
        evidence_sha256={str(report_path.relative_to(ROOT)):sha(report_path)},
        source_sha256=report['execution_source_sha256'],planner_sha256=sha(Path(__file__)),
        limitations=report['limitations'])


def check_reused(db, core):
    check_staged(db,core,approved=True)


def plan():
    semantic, core = audit(), numerical_plan()
    graph = build_nasa_domain_graph()
    core_graph = build_residual_classifier_graph()
    original_nodes = {node.node_id:node for node in core_graph.nodes}
    if {node.node_id:node for node in graph.nodes if node.node_id in original_nodes} != original_nodes:
        raise ValueError('Approved numerical node definitions changed')
    if any(edge not in graph.edges for edge in core_graph.edges):
        raise ValueError('Approved numerical internal wiring changed')
    reused = {atom['runtime_fqdn']:atom for atom in core['atoms']}
    identity = uuid5(NAMESPACE_URL,'sciona-reusable-cdg:'+FQDN)
    atoms, bindings = [], []
    for spec in semantic['bindings']:
        selected = [node for node in graph.nodes if node.matched_primitive==spec['runtime']]
        if spec['runtime'] in reused:
            atom = reused[spec['runtime']]
            if atom['version_id']!=spec['version_id'] or atom['content_hash']!=spec['content_hash']:
                raise ValueError('Approved numerical implementation changed')
        else:
            atom = dict(fqdn=spec['fqdn'],artifact_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec['fqdn'])),
                version_id=spec['version_id'],content_hash=spec['content_hash'],runtime_fqdn=spec['runtime'],
                inputs=merge_ports(selected,'inputs'),outputs=merge_ports(selected,'outputs'))
            atoms.append(atom)
        for node in selected:
            bindings.append(dict(atom,node_id=node.node_id,inputs=[p.model_dump() for p in node.inputs],
                                 outputs=[p.model_dump() for p in node.outputs]))
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_reused(db,core)
        for item in atoms+[dict(fqdn=FQDN,artifact_id=str(identity))]:
            for table,key in [('artifacts','artifact_id')]+([('atoms','atom_id')] if item['fqdn']!=FQDN else []):
                rows=db.execute(f'SELECT {key} AS identity FROM {table} WHERE fqdn=%s',(item['fqdn'],)).fetchall()
                if len(rows)>1 or any(str(row['identity'])!=item['artifact_id'] for row in rows):
                    raise ValueError('Catalog identity conflict')
    connected={(edge.target_id,edge.input_name) for edge in graph.edges}
    boundary={}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) not in connected:
                value=port.model_dump()
                if port.name in boundary and boundary[port.name]!=value:
                    raise ValueError('Ambiguous domain graph input')
                boundary[port.name]=value
    return dict(fqdn=FQDN,artifact_id=str(identity),version_id=str(uuid5(identity,semantic['graph_sha256'])),
        graph_sha256=semantic['graph_sha256'],atoms=atoms,reused_atoms=core['atoms'],bindings=bindings,
        boundary_inputs=list(boundary.values()),boundary_outputs=[p.model_dump() for p in next(n for n in graph.nodes if n.node_id=='domain_output').outputs],
        mandatory_provenance=core['mandatory_provenance'],numerical_core=core)


if __name__=='__main__':
    proposed=plan()
    (ROOT/'docs/reviews/nasa_domain_draft_plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    print(json.dumps(dict(new_atoms=len(proposed['atoms']),reused_atoms=len(proposed['reused_atoms']),
        bindings=len(proposed['bindings']),boundary_inputs=len(proposed['boundary_inputs']),approved=False)))
