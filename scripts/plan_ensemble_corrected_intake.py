"""Plan a full corrected original ensemble without replacing source history."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from uuid import UUID, uuid5
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.competition_import import _canonical
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.approve_ensemble_community import provider_closure
from scripts.import_riemannian_execution import SOURCE_ID, SOURCE_VERSION, SOURCE_HASH
from scripts.audit_ensemble_original_intake_scope import PARENT_VERSION
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.promote_audio_windows_canonical import check as check_window, qualification as window_qualification

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = str(SOURCE_ID)
FQDN = 'cdg.competition.solution.kaggle.barachant_seizure_1st'
SCOPE = 'Train all eleven source model branches across three ordered populations with separate family training membership, explicit source configurations and identity-aligned full-vector rank blending.'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_graph():
    graph = build_ensemble_execution_graph().model_copy(deep=True)
    graph.metadata.update(artifact_source='corrected_original_competition_execution', publication_status='draft',
        original_source_version_id=str(SOURCE_VERSION), original_source_content_hash=SOURCE_HASH,
        qualified_parent_version_id=PARENT_VERSION, scope=SCOPE,
        limitations='Current runtime reconstruction, with all eleven branches retained. Caller establishes family training membership, physical prediction identity, axes, sampling and units. Coherence reference updates depend on the entire prediction batch. No historical-engine equivalence or predictive-quality claim. Original source history remains unchanged.')
    return graph


def check_source_anchor(db):
    row = db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
    require(row == dict(artifact_id=ARTIFACT_ID,fqdn=FQDN,content_hash=SOURCE_HASH),'Immutable intake identity differs')
    rows = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(SOURCE_VERSION,)).fetchall()
    require(len(rows)==1 and hashlib.sha256(_canonical(rows[0]['details']['snapshot']).encode()).hexdigest()==SOURCE_HASH,'Immutable intake snapshot differs')


def origin_rows(db):
    return history(db,SOURCE_VERSION)


def runtime_evidence(library):
    report = json.loads((ROOT/'docs/reviews/ensemble_restored_r_models.json').read_text())
    require(report['passed'] and not report['approved'] and report['catalog_mutations']==0 and set(report['models'])=={'xgb','svm','glm'},'Complete restored model qualification required')
    require(report['validator_sha256']==sha(ROOT/'scripts/validate_ensemble_restored_r_models.py'),'Restored model validator changed')
    for name,digest in report['historical_evidence_sha256'].items():
        require(sha(ROOT/'docs/reviews'/name)==digest,'Historical model qualification changed')
    api_path=ROOT/'docs/reviews/ensemble_restored_r_contract_audit.json'
    require(sha(api_path)==report['restored_api_audit_sha256'],'Restored API evidence changed')
    api=json.loads(api_path.read_text())
    for name,digest in api['compiled_library_sha256'].items():
        require(sha(library/name)==digest,'Qualified R library changed')
    for branch,item in report['models'].items():
        path=ROOT/f'docs/reviews/ensemble_restored_r_{branch}_parity.json'
        require(item['exact_historical_results_except_binary_hashes'] and sha(path)==item['report_sha256'] and sha(ROOT/f'scripts/validate_andriy_r_{branch}.py')==item['unchanged_oracle_sha256'],'Restored model oracle changed')
        result=json.loads(path.read_text())
        module=importlib.import_module({'xgb':'sciona.atoms.ml.xgboost.andriy_r_xgb','svm':'sciona.atoms.ml.liblinear.andriy_r_svm','glm':'sciona.atoms.ml.glmnet.andriy_r_glm'}[branch])
        require(sha(module.__file__)==result['provider_sha256'] and result['compiled_library_sha256']==api['compiled_library_sha256'],'Restored provider/runtime differs')
    return sha(ROOT/'docs/reviews/ensemble_restored_r_models.json')


def check_atoms(db,atoms):
    for atom in atoms:
        for table,versions,ports,key in [('artifacts','artifact_versions','artifact_io_specs','artifact_id'),('atoms','atom_versions','atom_io_specs','atom_id')]:
            row=db.execute(f'SELECT fqdn,status,is_publishable FROM {table} WHERE {key}=%s',(atom['artifact_id'],)).fetchone()
            require(row==dict(fqdn=atom['fqdn'],status='approved',is_publishable=True),'Approved provider identity differs')
            row=db.execute(f'SELECT {key}::text AS identity,content_hash,is_latest,trust_tier FROM {versions} WHERE version_id=%s',(atom['version_id'],)).fetchone()
            require(row==dict(identity=atom['artifact_id'],content_hash=atom['content_hash'],is_latest=True,trust_tier=3),'Approved provider version differs')
            actual=db.execute(f'SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM {ports} WHERE version_id=%s ORDER BY direction,ordinal',(atom['version_id'],)).fetchall()
            require(actual==atom['interfaces'],'Approved provider ports differ')
        row=db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(atom['artifact_id'],)).fetchone()
        require(row['import_module']+'.'+row['source_symbol']==atom['runtime_fqdn'],'Provider runtime identity differs')
        require(sha(importlib.import_module(row['import_module']).__file__)==atom['provider_sha256'],'Provider source changed')
        deps=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(atom['version_id'],)).fetchall()
        require(sorted(deps,key=lambda d:d['dependency_artifact_fqdn'])==atom['dependencies'],'Provider closure differs')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(atom['version_id'],)).fetchone(),'Unresolved provider audit failure')
        require(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(atom['artifact_id'],)).fetchone() and db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s',(atom['artifact_id'],)).fetchone(),'Provider not served')


def check_parents(db,parents):
    parent=parents['ensemble']
    graph=build_ensemble_execution_graph()
    digest,_,_=encode_execution_graph(graph)
    row=db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(PARENT_VERSION,)).fetchone()
    require(row==dict(artifact_id=parent['artifact_id'],fqdn=parent['fqdn'],status='approved',is_publishable=True,content_hash=digest,is_latest=True,trust_tier=3),'Approved parent differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
    require(_artifact_document_to_cdg(document,version_id=PARENT_VERSION,content_hash=digest,require_execution_envelope=True)==graph,'Approved parent graph differs')
    actual=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(PARENT_VERSION,)).fetchall()
    require(actual==parent['stored_bindings'],'Approved parent bindings differ')
    require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(PARENT_VERSION,)).fetchone(),'Unresolved parent audit failure')
    deps=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(PARENT_VERSION,)).fetchall()
    require(deps==[dict(dependency_artifact_fqdn=FQDN,dependency_content_hash=SOURCE_HASH,dependency_role='cdg',optional=False)],'Parent source provenance differs')
    rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='ensemble-execution-community.v1' AND passed AND source_kind='automated' AND status='completed'",(PARENT_VERSION,)).fetchall()
    require(len(rows)==1,'Parent automated approval missing')
    for name,digest in rows[0]['details']['evidence_sha256'].items():
        require(sha(ROOT/'docs/reviews'/name)==digest,'Parent approval evidence changed')


def plan():
    # Runtime locations are local execution configuration and never enter the plan.
    import os
    library=Path(os.environ.get('SCIONA_ENSEMBLE_R_LIBRARY',str(ROOT/'.venv/ensemble-r-runtime/library')))
    runtime_evidence(library)
    scoped=json.loads((ROOT/'docs/reviews/ensemble_post_window_dependency_scope.json').read_text())
    require(scoped['passed'] and not scoped['canonical_provider_gaps'] and scoped['auditor_sha256']==sha(ROOT/'scripts/audit_ensemble_original_intake_scope.py'),'Original scope reconciliation changed')
    graph=build_graph();digest,_,_=encode_execution_graph(graph)
    closure=provider_closure(ROOT,graph)
    require(hashlib.sha256(_canonical(closure).encode()).hexdigest()==scoped['provider_closure_sha256'],'Reconciled provider closure changed')
    atoms=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_source_anchor(db)
        require(origin_rows(db)==scoped['original_history_sha256'],'Reconciled original history changed')
        for runtime,record in sorted(closure.items()):
            row=db.execute('SELECT artifact_id::text FROM artifact_versions WHERE version_id=%s',(record['version_id'],)).fetchone()
            require(row is not None,'Canonical provider missing')
            if runtime=='sciona.atoms.audio_speech.atoms.audio_windows':
                q=window_qualification()
                e=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='audio-windows-canonical-community.v1'",(record['version_id'],)).fetchone()
                require(e is not None,'Window canonical approval missing')
                check_window(db,q,e['details']['legacy_identity_sha256']);interfaces=q['ports']
            else:
                es=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND passed AND runner_version=ANY(%s)",(record['version_id'],['riemannian-provider-intake.v1','relative-power-provider-intake.v1','combined-feature-provider-intake.v1','feng-knn-provider-intake.v1','feng-xgb-provider-intake.v1','feng-expanded-provider-intake.v1','andriy-provider-intake.v1','ensemble-adapter-intake.v1'])).fetchall()
                require(len(es)==1 and es[0]['details']['source_sha256']==record['provider_sha256'],'Exact provider intake changed')
                interfaces=sorted(es[0]['details']['interfaces'],key=lambda p:(p['direction'],p['ordinal']))
            deps=sorted([dict(dependency_artifact_fqdn=closure[d]['catalog_fqdn'],dependency_content_hash=closure[d]['content_hash'],dependency_role='logic_atom',optional=False) for d in record['dependencies']],key=lambda d:d['dependency_artifact_fqdn'])
            atoms.append(dict(artifact_id=row['artifact_id'],fqdn=record['catalog_fqdn'],runtime_fqdn=runtime,version_id=record['version_id'],content_hash=record['content_hash'],provider_sha256=record['provider_sha256'],interfaces=interfaces,dependencies=deps))
        check_atoms(db,atoms)
        parent=db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash AS graph_sha256,v.version_id::text FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(PARENT_VERSION,)).fetchone()
        parent['stored_bindings']=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(PARENT_VERSION,)).fetchall()
        by_runtime={a['runtime_fqdn']:a for a in atoms}
        expected=[]
        for n in graph.nodes:
            a=by_runtime[n.matched_primitive]
            require(list(inspect.signature(getattr(importlib.import_module(n.matched_primitive.rsplit('.',1)[0]),n.matched_primitive.rsplit('.',1)[1])).parameters)==[p.name for p in n.inputs],'Node callable inputs differ')
            expected.append(dict(node_id=n.node_id,bound_artifact_fqdn=a['fqdn'],bound_version_content_hash=a['content_hash'],status='active',evidence_summary=dict(runtime_fqdn=n.matched_primitive,provider_version_id=a['version_id'],output_aliases_by_ordinal=[p.name for p in n.outputs])))
        # Compare executable binding semantics, while retaining the complete
        # historical evidence below for exact subsequent integrity checks.
        semantic_bindings=[dict(
            node_id=b['node_id'],bound_artifact_fqdn=b['bound_artifact_fqdn'],
            bound_version_content_hash=b['bound_version_content_hash'],status=b['status'],
            evidence_summary={key:b['evidence_summary'].get(key) for key in
                ('runtime_fqdn','provider_version_id','output_aliases_by_ordinal')}
        ) for b in parent['stored_bindings']]
        require(sorted(expected,key=lambda b:b['node_id'])==semantic_bindings,'Parent exact binding semantics differ')
        check_parents(db,{'ensemble':parent})
        original=origin_rows(db)
    consumed={(e.target_id,e.input_name) for e in graph.edges};producers={e.source_id for e in graph.edges}
    inputs=[p.model_dump() for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in consumed]
    outputs=[p.model_dump() for n in graph.nodes if n.node_id not in producers for p in n.outputs]
    require(len(inputs)==18 and len({p['name'] for p in inputs})==18 and len(outputs)==1,'Complete entrypoint boundary differs')
    names=['ensemble_post_window_dependency_scope.json','ensemble_restored_r_models.json','ensemble_restored_r_contract_audit.json','ensemble_restored_r_xgb_parity.json','ensemble_restored_r_svm_parity.json','ensemble_restored_r_glm_parity.json','audio_windows_canonical_served.json']
    result=dict(artifact_id=ARTIFACT_ID,fqdn=FQDN,version_id=str(uuid5(SOURCE_ID,'corrected-complete-execution:'+digest)),graph_sha256=digest,trust_tier=3,
        original_version_id=str(SOURCE_VERSION),original_content_hash=SOURCE_HASH,original_rows_sha256=original,inputs=inputs,outputs=outputs,atoms=atoms,
        bindings=[dict(by_runtime[n.matched_primitive],node_id=n.node_id,outputs=[p.model_dump() for p in n.outputs]) for n in graph.nodes],parent_graphs={'ensemble':parent},scope=SCOPE,
        evidence_sha256={name:sha(ROOT/'docs/reviews'/name) for name in names},planner_sha256=sha(__file__),
        qualification_scope='Source/structure and current provider/runtime qualification for staging only. Fresh full stored execution and publication gates remain required.')
    prior=ROOT/'docs/reviews/ensemble_corrected_intake_plan.json'
    if prior.exists():
        require(json.loads(prior.read_text())==result,'Frozen corrected plan changed')
    return result


if __name__=='__main__':
    proposed=plan()
    (ROOT/'docs/reviews/ensemble_corrected_intake_plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    print(json.dumps(dict(planned=True,nodes=len(proposed['bindings']),providers=len(proposed['atoms']),catalog_mutations=0)))
