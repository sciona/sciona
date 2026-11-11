"""Plan an executable revision of the original intake with immutable provenance."""
import hashlib
import inspect
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.competition_import import _canonical
from sciona.ghost.registry import REGISTRY
from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.nasa_population_graphs import build_population_graph
from sciona.visualizer.runner import _ensure_atoms_imported
from scripts.review_conditional_correction import ROOT,SOURCE_VERSION,SOURCE_HASH,require,sha

ARTIFACT_ID='60331cce-3af5-55d4-bce4-aa6217fcbe62'
FQDN='cdg.competition.solution.kaggle.nasa_airport_pushback_phase_1_3rd_place_1st'


def origin_rows(db):
    result={}
    for table in ['artifact_versions','artifact_io_specs','artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_dependencies']:
        key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
        rows=db.execute(f"SELECT to_jsonb(t)-'created_at'-'updated_at'-'is_latest' AS row FROM {table} t WHERE {key}=%s ORDER BY (to_jsonb(t)-'created_at'-'updated_at'-'is_latest')::text",(SOURCE_VERSION,)).fetchall()
        result[table]=hashlib.sha256(_canonical([row['row'] for row in rows]).encode()).hexdigest()
    return result


def check_source_anchor(db):
    row=db.execute('SELECT a.artifact_id,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
    require(row and str(row['artifact_id'])==ARTIFACT_ID and row['fqdn']==FQDN and row['content_hash']==SOURCE_HASH,'Immutable intake identity differs')
    audits=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(SOURCE_VERSION,)).fetchall()
    require(len(audits)==1,'Unique original intake snapshot required')
    snapshot=audits[0]['details']['snapshot']
    require(hashlib.sha256(_canonical(snapshot).encode()).hexdigest()==SOURCE_HASH,'Immutable intake snapshot differs')


def check_atoms(db,atoms):
    """Verify the reused atom versions directly, without mutable source-status assumptions."""
    for atom in atoms:
        for table,versions,ports,key in [('artifacts','artifact_versions','artifact_io_specs','artifact_id'),
                                       ('atoms','atom_versions','atom_io_specs','atom_id')]:
            extra=',import_module,source_symbol' if table=='atoms' else ''
            row=db.execute(f'SELECT fqdn,status,is_publishable{extra} FROM {table} WHERE {key}=%s',(atom['artifact_id'],)).fetchone()
            require(row and row['fqdn']==atom['fqdn'] and row['status']=='approved' and row['is_publishable'],'Approved provider identity differs')
            if table=='atoms':require(row['import_module']+'.'+row['source_symbol']==atom['runtime_fqdn'],'Provider runtime identity differs')
            row=db.execute(f'SELECT {key} AS identity,content_hash,is_latest,trust_tier FROM {versions} WHERE version_id=%s',(atom['version_id'],)).fetchone()
            require(row and str(row['identity'])==atom['artifact_id'] and row['content_hash']==atom['content_hash']
                and row['is_latest'] and row['trust_tier']==3,'Approved provider version differs')
            columns=['direction','name','ordinal','type_desc','constraints','required','default_value_repr']
            if key=='artifact_id':columns.append('dim_signature')
            actual=db.execute(f'SELECT {",".join(columns)} FROM {ports} WHERE version_id=%s ORDER BY direction,ordinal',(atom['version_id'],)).fetchall()
            expected=[]
            for direction,values in [('input',atom['inputs']),('output',atom['outputs'])]:
                for ordinal,port in enumerate(values):
                    expected.append({name:direction if name=='direction' else ordinal if name=='ordinal' else port[name] for name in columns})
            require(actual==expected,'Approved provider ports differ')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(atom['version_id'],)).fetchone(),'Unresolved provider audit failure')
        for view,key in [('catalog_artifacts_served','artifact_id'),('catalog_atoms_served','atom_id')]:
            # The unified view intentionally unions canonical and legacy atoms.
            expression='DISTINCT artifact_id' if view=='catalog_artifacts_served' else '*'
            require(db.execute(f'SELECT count({expression}) AS n FROM {view} WHERE {key}=%s',(atom['artifact_id'],)).fetchone()['n']==1,'Provider serving identity differs')


def check_parents(db,parents):
    for key,parent in parents.items():
        row=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s',
            (parent['artifact_id'],parent['version_id'])).fetchone()
        require(row==dict(fqdn=parent['fqdn'],status='approved',is_publishable=True,content_hash=parent['graph_sha256'],is_latest=True,trust_tier=3),'Approved parent version differs')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(parent['version_id'],)).fetchone(),'Unresolved parent audit failure')
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(parent['version_id'],)).fetchall()
        expected=[dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],bound_version_content_hash=b['content_hash'],status='active',
            evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'],provider_version_id=b['version_id'],output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
            for b in sorted(parent['bindings'],key=lambda b:b['node_id'])]
        require(bindings==expected,'Approved parent bindings differ')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        restored=_artifact_document_to_cdg(document,version_id=parent['version_id'],content_hash=parent['graph_sha256'],require_execution_envelope=True)
        require(restored==build_population_graph(key.split('_')[0],10),'Approved parent graph differs')


def plan():
    directory=ROOT/'docs/reviews'
    execution=json.loads((directory/'nasa_complete_workflow_execution.json').read_text())
    require(execution['passed'] and execution['predictions_compared']==768,'Complete workflow execution required')
    require([case['active_query_populations'] for case in execution['cases']]==[10,2],'Full and sparse source cases required')
    for case in execution['cases']:
        require(all(case[name] for name in ['source_predictions_exact','source_masks_and_offsets_exact','no_fitting_after_state_handoff','original_query_order_preserved']),'Complete workflow semantics failed')
        require(case['regressor_fits']==20 and case['classifier_fits']==10,'Complete source fit coverage required')
    population=json.loads((directory/'nasa_population_publication_review.json').read_text())
    require(execution['shared_source_sha256']==population['shared_source_sha256'],'Shared execution pins differ')
    paths={**execution['implementation_sha256'],**population['source_sha256'],**population['shared_source_sha256']}
    import importlib
    modules={'state_provider':'sciona.atoms.ml.domain_adapters.airport_state',
        'population_provider':'sciona.atoms.ml.domain_adapters.airport_population','routing_provider':'sciona.atoms.ml.pipeline.keyed_routing',
        'domain_provider':'sciona.atoms.ml.domain_adapters.airport_features'}
    for name,digest in paths.items():
        if name in modules or name.startswith('sciona.atoms.ml.'):
            path=Path(importlib.import_module(modules.get(name,name)).__file__)
        elif name=='domain_builder':path=ROOT/'sciona/nasa_domain_graph.py'
        elif name=='domain_validator':path=ROOT/'scripts/validate_nasa_domain_graph.py'
        elif name=='materialized_preflight':path=ROOT/'sciona/residual_metadata_preflight.py'
        else:path=ROOT/name
        require(sha(path)==digest,'Execution source drift: '+name)
    for name,digest in population['registration_source_sha256'].items():
        require(sha(ROOT.parent/name)==digest,'Registration source drift')
    frozen=json.loads((directory/'nasa_population_draft_plan.json').read_text())
    atoms=frozen['atoms']+frozen['reused_atoms']
    parents={key:frozen['graphs'][key] for key in ['training_10','inference_10']}
    graph=build_complete_workflow_graph(10);digest=encode_execution_graph(graph)[0]
    require(digest==execution['cases'][0]['graph_sha256'],'Complete workflow graph drift')
    _ensure_atoms_imported()
    repo=ROOT.parent/'sciona-atoms-ml'
    specs=_parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),artifact_root=repo/'src/sciona/atoms')
    by_runtime={atom['runtime_fqdn']:atom for atom in atoms}
    require(len(by_runtime)==39 and set(by_runtime)=={node.matched_primitive for node in graph.nodes},'Exact reused provider set required')
    for runtime,atom in by_runtime.items():
        matches=[spec for spec in specs if spec.import_module+'.'+spec.source_symbol==runtime]
        require(len(matches)==1 and runtime in REGISTRY,'Unique runtime provider required')
        spec=matches[0];impl=REGISTRY[runtime]['impl']
        require(spec.content_hash==atom['content_hash'] and str(spec.version_id)==atom['version_id']
            and spec.fqdn==atom['fqdn'] and Path(inspect.getsourcefile(impl)).resolve()==spec.file_path.resolve(),'Runtime provider drift')
        require(all(list(inspect.signature(impl).parameters)==[p.name for p in node.inputs]
            for node in graph.nodes if node.matched_primitive==runtime),'Callable contract differs')
    bindings=[dict(by_runtime[node.matched_primitive],node_id=node.node_id,
        inputs=[p.model_dump() for p in node.inputs],outputs=[p.model_dump() for p in node.outputs]) for node in graph.nodes]
    connected={(edge.target_id,edge.input_name) for edge in graph.edges};inputs={}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) not in connected:
                require(port.name not in inputs or inputs[port.name]==port.model_dump(),'Ambiguous root interface')
                inputs[port.name]=port.model_dump()
    inputs['airports']['constraints']+=' Exactly ten distinct training populations are required by this entrypoint.'
    inputs['prediction_records']['constraints']+=' Exactly ten modeled query populations are required by this entrypoint; use the qualified sparse inference variants for smaller active counts.'
    version=str(uuid5(UUID(ARTIFACT_ID),'corrected-complete-execution:'+digest))
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_source_anchor(db);check_atoms(db,atoms);check_parents(db,parents)
        original=origin_rows(db)
    prior_path=directory/'nasa_corrected_intake_plan.json'
    if prior_path.exists():
        prior=json.loads(prior_path.read_text())
        require(prior['original_version_id']==SOURCE_VERSION and prior['original_content_hash']==SOURCE_HASH
            and prior['original_rows_sha256']==original,'Previously frozen original records changed')
    names=['nasa_complete_workflow_execution.json','nasa_original_intake_scope_audit.json','nasa_population_publication_review.json',
        'nasa_population_served_verification.json','nasa_population_draft_plan.json']
    return dict(artifact_id=ARTIFACT_ID,fqdn=FQDN,version_id=version,graph_sha256=digest,trust_tier=3,
        original_version_id=SOURCE_VERSION,original_content_hash=SOURCE_HASH,original_rows_sha256=original,
        inputs=list(inputs.values()),outputs=[p.model_dump() for p in next(n for n in graph.nodes if n.node_id=='inference_population_output').outputs],
        atoms=atoms,bindings=bindings,scope=graph.metadata['scope'],
        provenance_policy='Original version identity, snapshot and graph records are immutable; canonical approval state may advance only through qualified version promotion.',
        parent_graphs=parents,
        evidence_sha256={name:sha(directory/name) for name in names},planner_sha256=sha(Path(__file__)),
        corrected_descriptions=execution['original_scope_audit']['corrections'])


if __name__=='__main__':
    proposed=plan()
    (ROOT/'docs/reviews/nasa_corrected_intake_plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    print(json.dumps(dict(planned=True,new_atoms=0,reused_atoms=len(proposed['atoms']),nodes=len(proposed['bindings']),
        original_identity_preserved=True,new_version_id=proposed['version_id'],catalog_mutations=0)))
