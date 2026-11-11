"""Read-only original-scope and reusable-operation audit for video multilabel."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT, connect

ORIGINAL='13b48adf-8fb5-5ea2-a26c-44344bb8907e'
SOURCE_HASH='30d0444ec640598942c6221136e608d7b15449732ceb848939e1424321c496ad'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    report_path=ROOT/'docs/reviews/competition_video_multilabel_semantic_review.json'
    semantic=json.loads(report_path.read_text())
    graph_path=ROOT/'docs/reviews/competition_video_multilabel_graph_execution.json'
    execution=json.loads(graph_path.read_text())
    drift=[p for p,d in execution['code_sha256'].items() if sha(ROOT/p)!=d]
    versions={name:metadata.version(name) for name in semantic['dependencies']}
    runtime_drift={name:dict(qualified=old,current=versions[name]) for name,old in semantic['dependencies'].items() if old!=versions[name]}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute("""SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,e.details
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id)
            WHERE v.version_id=%s AND e.runner_version='competition-intake.v1'""",(ORIGINAL,)).fetchall()
        if len(rows)!=1 or rows[0]['content_hash']!=SOURCE_HASH:
            raise ValueError('Unique original source required')
        original=dict(rows[0]);snapshot=original.pop('details')['snapshot']
        if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Immutable intake drift')
        derivative=db.execute("""SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest""",
            (original['fqdn']+'.execution',)).fetchall()
        if len(derivative)!=1 or derivative[0]['content_hash']!=execution['serialized_graph_sha256']:
            raise ValueError('Reference derivative differs')
        served=bool(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(derivative[0]['artifact_id'],)).fetchone())
        threshold_candidates=db.execute("""SELECT fqdn,status FROM artifacts WHERE artifact_kind='atom'
            AND (fqdn ILIKE %s OR fqdn ILIKE %s) ORDER BY fqdn""",('%f1%threshold%','%attention%pool%')).fetchall()
    paths=['sciona/video_multilabel_contract.py','sciona/video_multilabel_network.py',
        'sciona/video_multilabel_training.py','sciona/video_multilabel_thresholds.py',
        '../sciona-atoms/src/sciona/atoms/scoring/atoms.py']
    return dict(format='decomposed-video-multilabel-readiness.v1',read_only=True,catalog_mutations=0,
        approved=False,original=original,source_version_id=ORIGINAL,
        source_stages=[dict(stage_id=s['stage_id'],description=s['description']) for s in snapshot['template']['stages']],
        reference=derivative[0],reference_served=served,implementation_drift=drift,runtime_drift=runtime_drift,
        semantic_report_sha256=sha(report_path),execution_report_sha256=sha(graph_path),
        inspected_implementation_sha256={p:sha(ROOT/p) for p in paths},
        reuse_candidates=threshold_candidates,
        findings=[
            'The approved two-node derivative runs attention/context/sparse-fusion/threshold operations inside a recipe; these are not independently selectable graph providers.',
            'Original feature_stream_loading explicitly requires sharded and memory-aware access. Existing prepare deep-copies a complete JSON population; only padded frame batches are bounded. Disk streaming is explicitly excluded by the derivative. Original promotion requires implementing and qualifying sharded access, not deleting that requirement.',
            'Masked learned attention and context gating are independent of video semantics and should have reusable tensor contracts preserving differentiation, masking and frame-order invariance.',
            'Sparse vocabulary fitting and per-label logistic training must stay training-only; group isolation and calibration population policy remain recipe responsibilities.',
            'Existing expected_f1_threshold maximizes an expected-score objective from probabilities and a positive-count estimate. The recipe maximizes empirical F1 against held-out binary labels with highest-threshold ties. They are not interchangeable.',
            'The original permits alternative pooling, head and calibration families; the attention/context-gate/sparse-linear/F1 branch is a declared realization, not a claim to implement every alternative.'],
        next_actions=[
            'Reexecute the existing derivative under the current bounded runtime and investigate dependency drift before reusing historical execution evidence.',
            'Implement sharded feature access with training-only vocabulary and group checks, preserving every configured epoch and complete supplied population.',
            'Inventory and qualify reusable masked pooling, context gating, sparse fit/predict, score fusion and empirical threshold operations; connect them explicitly in the original graph.',
            'Revalidate full stored execution and native catalog eligibility before original Tier 3 publication.'],
        limits=['Scope/reuse audit only; no new runtime correctness or promotion claim. Public software and generic algorithm-template metadata only.'],
        auditor_sha256=sha(__file__))


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_video_multilabel_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original=result['original']['status'],reference_served=result['reference_served'],
        implementation_drift=result['implementation_drift'],runtime_drift=result['runtime_drift'])))
