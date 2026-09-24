"""Plan exact lifecycle drafts and reuse the approved domain/numerical atoms."""
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.domain_adapters.airport_state as provider
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY
from sciona.nasa_lifecycle_graphs import build_nasa_training_graph,build_nasa_inference_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.plan_nasa_domain_draft import ROOT,sha,audit as domain_audit,plan as domain_plan
from scripts.plan_residual_classifier_draft import merge_ports
from scripts.validate_nasa_domain_database_gates import check_staged as check_domain

BUILDERS={'training':build_nasa_training_graph,'inference':build_nasa_inference_graph}
FQDNS={'training':'cdg.domain.airport.residual_classifier_training',
       'inference':'cdg.domain.airport.residual_classifier_inference'}


def audit():
    parent=domain_audit()
    path=ROOT/'docs/reviews/competition_nasa_lifecycle_graphs.json'
    report=json.loads(path.read_text())
    if not report['passed'] or report['source_predictions']!=640 or len(report['scenarios'])!=10:
        raise ValueError('Complete lifecycle execution evidence required')
    if not all(case['training_without_queries'] and case['fresh_inference_without_refitting'] and case['identity_alignment_exact'] for case in report['scenarios']):
        raise ValueError('Lifecycle separation evidence missing')
    if not all(case['selected_populations_exact'] and case['conditional_offsets_exact'] and case['final_int32_predictions_exact'] for case in report['source_comparisons']):
        raise ValueError('Source comparison failed')
    if report['shared_execution_source_sha256']!=parent['source_sha256']:
        raise ValueError('Approved domain dependency pins differ')
    for name,digest in report['lifecycle_source_sha256'].items():
        selected=Path(provider.__file__) if name=='state_provider' else ROOT/name
        if sha(selected)!=digest:raise ValueError('Lifecycle source drift: '+name)
    for kind,builder in BUILDERS.items():
        if encode_execution_graph(builder())[0]!=report[kind+'_graph_sha256']:
            raise ValueError('Lifecycle graph drift: '+kind)
    return dict(passed=True,approved=False,catalog_mutations=0,scope='Separate corrected single-airport training and inference graphs with complete private runtime model state.',
        evidence_sha256={str(path.relative_to(ROOT)):sha(path)},lifecycle_source_sha256=report['lifecycle_source_sha256'],
        shared_source_sha256=parent['source_sha256'],limitations=report['limitations'],planner_sha256=sha(Path(__file__)))


def check_parent(db,parent):
    check_domain(db,parent,approved=True)


def plan():
    semantic,parent=audit(),domain_plan()
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
        final='learned_state' if kind=='training' else 'domain_output'
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
    return dict(graphs=proposed,atoms=list(new_atoms.values()),reused_atoms=list(reused.values()),parent_domain=parent,
                mandatory_provenance=parent['mandatory_provenance'],semantic_review=semantic)


if __name__=='__main__':
    proposed=plan()
    (ROOT/'docs/reviews/nasa_lifecycle_draft_plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    print(json.dumps(dict(new_atoms=len(proposed['atoms']),reused_atoms=len(proposed['reused_atoms']),
        graphs=len(proposed['graphs']),bindings=sum(len(graph['bindings']) for graph in proposed['graphs'].values()),approved=False)))
