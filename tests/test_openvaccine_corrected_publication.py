"""Reject reduced computation, wrong identities and stale publication evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts import promote_openvaccine_corrected_intakes as publisher


@pytest.fixture(params=['covid','degradation'])
def review_evidence(monkeypatch,request):
    target=request.param
    root=publisher.ROOT
    intake=publisher.staging.Intake(target)
    plan=json.loads(intake.PLAN.read_text())
    providers=json.loads(publisher.staging.provider_publication.PLAN.read_text())
    semantic=json.loads(publisher.source_review.REPORT.read_text())
    execution_path=intake.report_path('execution')
    execution=json.loads(execution_path.read_text())
    monkeypatch.setattr(publisher.staging.Intake,'qualified_plan',lambda self:(plan,providers))
    monkeypatch.setattr(publisher.staging,'connect',MagicMock())
    monkeypatch.setattr(publisher.staging.provider_publication,'check',lambda *args:None)
    monkeypatch.setattr(publisher.source_review,'review',lambda:semantic)
    read_text=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**kw:
        json.dumps(execution) if self==execution_path else read_text(self,*a,**kw))
    return target,execution


def test_complete_execution_accepted(review_evidence):
    assert publisher.review(review_evidence[0])[2]['proposed_tier'] == 3


@pytest.mark.parametrize('field,value', [
    ('status','failed'),('approved',True),('catalog_mutations',1),('synthetic_only',False),
    ('stored_candidate',False),('all_nodes_uncached',False),('format','wrong'),
    ('target','wrong'),('actual_imports_verified',False),('active_dependency_constraints_satisfied',False),
    ('stored_candidate',1)])
def test_incomplete_execution_rejected(review_evidence,field,value):
    review_evidence[1][field]=value
    with pytest.raises(ValueError,match='Complete stored source'):publisher.review(review_evidence[0])


@pytest.mark.parametrize('field,value', [
    ('actual_runner_nodes',5),('actual_runner_edges',6),('models',19),
    ('refinement_rounds',0),('pretraining_steps',0),('initial_supervised_steps',19),
    ('refinement_supervised_steps',19),('pseudo_label_steps',19),('full_reference_retrained',False),
    ('exact_reconstruction_result',False),('prediction_and_teacher_bytes_equal',False),
    ('strict_json_output',False),('graph_codec_roundtrip',False),('models',20.0)
])


def test_incomplete_lifecycle_rejected(review_evidence,field,value):
    review_evidence[1]['checks'][field]=value
    with pytest.raises(ValueError,match='model lifecycle'):publisher.review(review_evidence[0])


@pytest.mark.parametrize('field,value', [
    ('native_threads',8),('tensorflow_intra_threads',8),('tensorflow_inter_threads',8),('serial_models',False)])
def test_resource_policy_rejected(review_evidence,field,value):
    review_evidence[1]['resource_policy'][field]=value
    with pytest.raises(ValueError,match='resource policy'):publisher.review(review_evidence[0])


@pytest.mark.parametrize('field,value', [
    ('serialized_graph_sha256','0'*64),('candidate_version_id','wrong'),('plan_sha256','0'*64),
    ('bound_provider_sha256',{})])
def test_wrong_stored_identity_rejected(review_evidence,field,value):
    review_evidence[1][field]=value
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review(review_evidence[0])


def test_environment_drift_rejected(review_evidence):
    review_evidence[1]['runtime']={}
    with pytest.raises(ValueError,match='environment'):publisher.review(review_evidence[0])


@pytest.mark.parametrize('fault', ['missing','changed','extra'])
def test_implementation_hash_fault_rejected(review_evidence,fault):
    hashes=review_evidence[1]['code_sha256']
    if fault=='missing':hashes.pop('sciona/visualizer/runner.py')
    elif fault=='changed':hashes['sciona/visualizer/runner.py']='0'*64
    else:hashes['unexpected']='0'*64
    with pytest.raises(ValueError,match='implementation'):publisher.review(review_evidence[0])


@pytest.mark.parametrize('symbol', ['openvaccine_prepare','features','fit_forecast','blend_inputs','ordered_weighted_sum','result'])
def test_each_bound_provider_required(review_evidence,symbol):
    hashes=review_evidence[1]['bound_provider_sha256']
    hashes.pop(next(key for key in hashes if key.endswith('.'+symbol)))
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review(review_evidence[0])


def test_unpinned_reference_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','pending')
    with pytest.raises(ValueError,match='Immutable'):publisher.publish('covid',apply=True)


def test_unavailable_commit_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','0'*40)
    def missing(*args,**kwargs):
        raise publisher.subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(publisher.subprocess,'check_output',missing)
    with pytest.raises(ValueError,match='commit is unavailable'):publisher.publish('covid',apply=True)


def test_different_committed_report_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','0'*40)
    monkeypatch.setattr(publisher.subprocess,'check_output',lambda *args,**kwargs:b'wrong report')
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'qualified report')
    with pytest.raises(ValueError,match='differs from qualified report'):publisher.publish('covid',apply=True)


def test_reviewed_software_drift_rejected(review_evidence):
    review_evidence[1]['provisioned_software']={}
    with pytest.raises(ValueError,match='environment'):publisher.review(review_evidence[0])


def test_other_original_plan_rejected(review_evidence):
    target,execution=review_evidence
    other=next(k for k in publisher.staging.TARGETS if k!=target)
    execution['candidate_version_id']=json.loads(publisher.staging.Intake(other).PLAN.read_text())['version_id']
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review(target)
