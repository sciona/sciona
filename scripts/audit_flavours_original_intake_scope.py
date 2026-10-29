"""Reconcile the original Flavours intake with its monolithic served derivative.

Read-only structural/source audit. Historical execution is preserved evidence,
not fresh runtime qualification or approval of a decomposed original version.
"""
import ast
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.competition_import import _canonical
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.build_flavours_execution_graph import build_graph
from scripts.stage_remaining_legacy_physics_revisions import history

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='6256d5b1-d742-53b1-9ff4-c2c7265ce981'
ORIGINAL_HASH='1f7003883cd8067a1fcaa170601336d99e2911e2024a0590eb41425757dc0539'
PARENT='238ca008-f9ec-5193-8a8b-74742ce42bc1'
STAGES={
    'feature_selection_safe':'Replace the placeholder with explicit physical feature construction, restricted-column selection and regression/classifier views. Caller establishes column semantics and units.',
    'flatness_constrained_gradient_boosting':'Absent from recovered winning method. Preserve repeated regression, five classifier branches and their full refits instead; do not certify an unimplemented flatness objective.',
    'noise_injection_decorrelation':'Absent from recovered winning method. Preserve the fifty-model neural ensemble and source nonlinear blend; do not introduce noise injection.',
    'compute_ks_agreement':'Expose the existing weighted two-population metric and its separate evaluation-only inputs.',
    'compute_cvm_mass_decorrelation':'Expose the existing protected-variable rank metric with the source window/step and threshold. Failed diagnostics remain reported failures.',
    'roc_auc_truncated_weighted':'Expose the existing metric after strict source quality selection; out-of-fold discrimination is diagnostic because upstream regression is not nested within classifier folds.',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def audit():
    retained_path=ROOT/'docs/reviews/competition_flavours_publication_verification.json'
    retained=json.loads(retained_path.read_text())
    runtime=retained['served_runtime_validation']
    require(runtime['passed'] and runtime['synthetic_only'] and retained['full_training_and_inference_budgets']
        and runtime['model_fits']==115 and runtime['tree_boosting_rounds']==111458
        and runtime['neural_epochs']==150000 and runtime['neural_optimizer_updates']==600000,'Historical full-scope evidence differs')
    functions={}
    for name,digest in runtime['source_sha256'].items():
        path=ROOT/name
        require(sha(path)==digest,'Historical implementation changed: '+name)
        functions[name]=[node.name for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)]
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/flavours_execution.py'
    require(sha(provider)==runtime['graph_execution']['provider_sha256'],'Historical wrapper changed')
    graph=build_graph();digest,nodes,edges=encode_execution_graph(graph)
    require(digest==retained['graph_sha256'] and len(nodes)==1 and not edges,'Served reference is no longer the reviewed wrapper graph')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(ORIGINAL,)).fetchone()
        require(original and original['content_hash']==ORIGINAL_HASH and original['status']=='draft'
            and not original['is_publishable'] and original['is_latest'],'Original intake state differs')
        intakes=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        require(len(intakes)==1,'Original source evidence ambiguous')
        snapshot=intakes[0]['details']['snapshot']
        require(hashlib.sha256(_canonical(snapshot).encode()).hexdigest()==ORIGINAL_HASH,'Original source snapshot changed')
        require({s['stage_id'] for s in snapshot['template']['stages']}==set(STAGES),'Original stage coverage changed')
        parent=db.execute('SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(PARENT,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest']
            and parent['trust_tier']==3 and parent['content_hash']==digest,'Parent approval differs')
        require(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(parent['artifact_id'],)).fetchone(),'Parent is not served')
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(original['artifact_id'],)).fetchone(),'Original unexpectedly served')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=PARENT,content_hash=digest,require_execution_envelope=True)==graph,'Stored parent differs')
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings WHERE version_id=%s',(PARENT,)).fetchall()
        require(len(bindings)==1 and bindings[0]['node_id']=='execute' and bindings[0]['status']=='active'
            and bindings[0]['bound_artifact_fqdn']==graph.nodes[0].matched_primitive,'Parent wrapper binding differs')
        provider_version=db.execute("SELECT v.version_id::text FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest AND v.trust_tier=3 AND a.status='approved' AND a.is_publishable",(bindings[0]['bound_artifact_fqdn'],bindings[0]['bound_version_content_hash'])).fetchall()
        require(len(provider_version)==1,'Exact approved wrapper missing')
        deps=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(PARENT,)).fetchall()
        require(any(d==dict(dependency_artifact_fqdn=original['fqdn'],dependency_content_hash=ORIGINAL_HASH,dependency_role='cdg',optional=False) for d in deps),'Original provenance missing')
        original_history=history(db,ORIGINAL)
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=False,
        original_version_id=ORIGINAL,original=original,original_history_sha256=original_history,
        parent_version_id=PARENT,parent=parent,parent_nodes=1,parent_edges=0,parent_bindings=bindings,
        wrapper_version_id=provider_version[0]['version_id'],original_stage_corrections=STAGES,
        current_source_functions=functions,current_source_sha256=runtime['source_sha256'],
        wrapper_sha256=sha(provider),retained_execution_evidence_sha256=sha(retained_path),
        full_scope_budgets=dict(regression_fits=35,classifier_fits=30,neural_fits=50,
            tree_boosting_rounds=111458,neural_epochs=150000,
            historical_fixture_neural_optimizer_updates=600000),
        decomposition_requirements=[
            'Retain physical feature adapters with explicit units and column-order contracts; expose generic numerical transforms separately.',
            'Expose repeated cross-fit regression, classifier fold fitting/full refits, neural fold/seed training and checkpoint replay as graph composition with explicit reusable contracts.',
            'Retain training/query/control identity and row alignment. Evaluation-only weights, quality and correlation targets must not enter fitting.',
            'Reuse existing registered agreement, rank-decorrelation and weighted-AUC operations after current identity/contract qualification; do not duplicate them.',
            'Keep nonlinear source blend, all model budgets, partition rules and reported diagnostic failures; do not substitute a reduced passing workflow.',
            'Qualify a new decomposed version under the original artifact identity while preserving the original snapshot and existing derivative evidence.'],
        fresh_runtime_qualified=False,original_promotion_eligible=False,
        limitations=['Source hash and catalog structure checks only; no model execution or cross-domain empirical-effectiveness claim.',
            'Neural optimizer-update count depends on input population sizes; the recorded 600000 updates describe the historical synthetic fixture.',
            'Historical full-budget execution belongs to the single-wrapper derivative and does not qualify an unbuilt decomposed graph.'],
        auditor_sha256=sha(__file__))


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/flavours_original_intake_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,parent_nodes=1,original_promotion_eligible=False,catalog_mutations=0)))
