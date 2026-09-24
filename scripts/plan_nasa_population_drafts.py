"""Plan explicit population variants reusing the approved lifecycle atoms."""
import inspect
from functools import partial
import json
from pathlib import Path
from uuid import NAMESPACE_URL,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.domain_adapters.airport_state as provider
import sciona.atoms.ml.domain_adapters.airport_population as population
import sciona.atoms.ml.pipeline.keyed_routing as routing
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY
from sciona.nasa_population_graphs import build_population_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.plan_nasa_lifecycle_drafts import ROOT,sha,audit as lifecycle_audit,plan as lifecycle_plan
from scripts.plan_residual_classifier_draft import merge_ports
from scripts.validate_nasa_lifecycle_database_gates import check_staged as check_lifecycle

BUILDERS={'training_10':partial(build_population_graph,'training',10),
    **{f'inference_{count}':partial(build_population_graph,'inference',count) for count in range(1,11)}}
FQDNS={key:'cdg.domain.airport.population_residual_classifier_'+key for key in BUILDERS}


def audit():
    parent=lifecycle_audit()
    path=ROOT/'docs/reviews/competition_nasa_population_graphs.json'
    report=json.loads(path.read_text())
    if not report['passed'] or report['training_populations']!=10 or report['source_oracle_predictions']!=640 or report['predictions_compared']!=3520:
        raise ValueError('Complete population source execution evidence required')
    if not report['training_without_queries'] or not report['source_calibration_offsets_exact'] or report['fresh_processes']!=10:
        raise ValueError('Population lifecycle evidence incomplete')
    cases=report['inference_cases']
    if [case['active_populations'] for case in cases]!=list(range(1,11)):
        raise ValueError('All active count variants required')
    for case in cases:
        if not all(case[key] for key in ['passed','no_refit','identity_alignment_exact','source_predictions_exact']):
            raise ValueError('Population inference comparison failed')
        if case['branches']!=case['active_populations'] or case['unqueried_populations_skipped']!=10-case['branches']:
            raise ValueError('Sparse population coverage differs')
    if report['shared_source_sha256']!=parent['shared_source_sha256']:
        raise ValueError('Approved shared source pins differ')
    modules={'state_provider':provider,'population_provider':population,'routing_provider':routing}
    for name,digest in report['source_sha256'].items():
        selected=Path(modules[name].__file__) if name in modules else ROOT/name
        if sha(selected)!=digest:raise ValueError('Population source drift: '+name)
    hashes={'training_10':report['training_graph_sha256'],**{f"inference_{case['active_populations']}":case['graph_sha256'] for case in cases}}
    for key,builder in BUILDERS.items():
        if encode_execution_graph(builder())[0]!=hashes[key]:raise ValueError('Population graph drift: '+key)
    registration_path=ROOT/'docs/reviews/nasa_population_registration.json'
    registration=json.loads(registration_path.read_text())
    if not registration['passed'] or registration['registered_before']!=0 or registration['registered_after']!=39 or registration['provider_preimports']:
        raise ValueError('Cold standard provider discovery required')
    for name,digest in registration['source_sha256'].items():
        if sha(ROOT.parent/name)!=digest:raise ValueError('Provider registration code drift')
    return dict(passed=True,approved=False,catalog_mutations=0,
        scope='Explicit ten-population training and inference variants for one through ten queried populations; provisioned in-process runtime.',
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [path,registration_path]},
        population_source_sha256=report['source_sha256'],shared_source_sha256=report['shared_source_sha256'],
        registration_source_sha256=registration['source_sha256'],limitations=report['limitations'],planner_sha256=sha(Path(__file__)))


def check_parent(db,parent):
    check_lifecycle(db,parent,approved=True)


def plan():
    semantic,parent=audit(),lifecycle_plan()
    graphs={kind:builder() for kind,builder in BUILDERS.items()}
    repo=ROOT.parent/'sciona-atoms-ml'
    specs=_parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),artifact_root=repo/'src/sciona/atoms')
    existing={atom['runtime_fqdn']:atom for atom in parent['atoms']+parent['reused_atoms']}
    nodes_by_runtime={}
    for graph in graphs.values():
        for node in graph.nodes:nodes_by_runtime.setdefault(node.matched_primitive,[]).append(node)
    all_atoms,new_atoms,reused={},{},{}
    for runtime,nodes in nodes_by_runtime.items():
        matches=[spec for spec in specs if spec.import_module+'.'+spec.source_symbol==runtime]
        if len(matches)!=1 or runtime not in REGISTRY:raise ValueError('Unique lifecycle provider required')
        spec=matches[0];function=REGISTRY[runtime]['impl']
        if Path(inspect.getsourcefile(function)).resolve()!=spec.file_path.resolve():raise ValueError('Provider import shadowing')
        if any(list(inspect.signature(function).parameters)!=[port.name for port in node.inputs] for node in nodes):
            raise ValueError('Callable parameter mismatch')
        if runtime in existing:
            atom=existing[runtime]
            if atom['version_id']!=str(spec.version_id) or atom['content_hash']!=spec.content_hash:
                raise ValueError('Approved provider identity changed')
            reused[runtime]=atom
        else:
            atom=dict(fqdn=spec.fqdn,artifact_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),
                version_id=str(spec.version_id),content_hash=spec.content_hash,runtime_fqdn=runtime,
                inputs=merge_ports(nodes,'inputs'),outputs=merge_ports(nodes,'outputs'))
            new_atoms[runtime]=atom
        all_atoms[runtime]=atom
    proposed={}
    for kind,graph in graphs.items():
        digest=encode_execution_graph(graph)[0]
        identity=uuid5(NAMESPACE_URL,'sciona-reusable-cdg:'+FQDNS[kind])
        bindings=[dict(all_atoms[node.matched_primitive],node_id=node.node_id,
            inputs=[port.model_dump() for port in node.inputs],outputs=[port.model_dump() for port in node.outputs]) for node in graph.nodes]
        connected={(edge.target_id,edge.input_name) for edge in graph.edges};boundary={}
        for node in graph.nodes:
            for port in node.inputs:
                if (node.node_id,port.name) not in connected:
                    value=port.model_dump()
                    if port.name in boundary and boundary[port.name]!=value:raise ValueError('Ambiguous lifecycle input')
                    boundary[port.name]=value
        final='exhausted' if kind.startswith('training_') else 'population_output'
        proposed[kind]=dict(fqdn=FQDNS[kind],artifact_id=str(identity),version_id=str(uuid5(identity,digest)),
            graph_sha256=digest,bindings=bindings,boundary_inputs=list(boundary.values()),
            boundary_outputs=[port.model_dump() for port in next(node for node in graph.nodes if node.node_id==final).outputs],
            scope=graph.metadata['scope'])
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_parent(db,parent)
        for item in list(new_atoms.values())+list(proposed.values()):
            tables=[('artifacts','artifact_id')]+([('atoms','atom_id')] if item['fqdn'] not in FQDNS.values() else [])
            for table,key in tables:
                rows=db.execute(f'SELECT {key} AS identity FROM {table} WHERE fqdn=%s',(item['fqdn'],)).fetchall()
                if len(rows)>1 or any(str(row['identity'])!=item['artifact_id'] for row in rows):raise ValueError('Catalog identity collision')
    return dict(graphs=proposed,atoms=list(new_atoms.values()),reused_atoms=list(reused.values()),parent_lifecycle=parent,
                mandatory_provenance=parent['mandatory_provenance'],semantic_review=semantic)


if __name__=='__main__':
    proposed=plan()
    (ROOT/'docs/reviews/nasa_population_draft_plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    print(json.dumps(dict(new_atoms=len(proposed['atoms']),reused_atoms=len(proposed['reused_atoms']),
        graphs=len(proposed['graphs']),bindings=sum(len(graph['bindings']) for graph in proposed['graphs'].values()),approved=False)))
