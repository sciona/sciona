"""Reject reduced computation, wrong identities and stale publication evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts import promote_porto_corrected_intake as publisher


@pytest.fixture
def review_evidence(monkeypatch):
    root = publisher.ROOT
    plan = json.loads(publisher.intake.PLAN.read_text())
    providers = json.loads(publisher.intake.provider_publication.PLAN.read_text())
    semantic = json.loads((root/'docs/reviews/porto_decomposed_provider_semantics.json').read_text())
    execution_path = root/'docs/reviews/porto_corrected_intake_execution.json'
    execution = json.loads(execution_path.read_text())
    monkeypatch.setattr(publisher.intake, 'qualified_plan', lambda: (plan, providers))
    monkeypatch.setattr(publisher.intake, 'connect', MagicMock())
    monkeypatch.setattr(publisher.intake.provider_publication, 'check', lambda *args: None)
    from scripts import review_porto_decomposed_providers as provider_review
    monkeypatch.setattr(provider_review, 'review', lambda: semantic)
    read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw:
        json.dumps(execution) if self == execution_path else read_text(self, *a, **kw))
    return execution


def test_complete_execution_accepted(review_evidence):
    assert publisher.review()[2]['proposed_tier'] == 3


@pytest.mark.parametrize('field,value', [
    ('status','failed'),('approved',True),('catalog_mutations',1),('synthetic_only',False),
    ('stored_candidate',False),('all_nodes_uncached',False),('format','wrong')])
def test_incomplete_execution_rejected(review_evidence,field,value):
    review_evidence[field]=value
    with pytest.raises(ValueError,match='Complete stored source'):publisher.review()


@pytest.mark.parametrize('field,value', [
    ('actual_runner_nodes',5),('actual_runner_edges',8),('models_fitted',5),('dae_models',4),
    ('synthetic_query_rows',1),('full_configured_schedules',False),('exact_reconstruction_result',False),
    ('strict_json_output',False),('graph_codec_roundtrip',False)])

def test_incomplete_lifecycle_rejected(review_evidence,field,value):
    review_evidence['checks'][field]=value
    with pytest.raises(ValueError,match='model lifecycle'):publisher.review()


@pytest.mark.parametrize('field,value', [
    ('native_threads',8),('lightgbm_threads',8),('torch_threads',8),('torch_interop_threads',8),('serial_models',False)])
def test_resource_policy_rejected(review_evidence,field,value):
    review_evidence['resource_policy'][field]=value
    with pytest.raises(ValueError,match='resource policy'):publisher.review()


@pytest.mark.parametrize('field,value', [
    ('serialized_graph_sha256','0'*64),('candidate_version_id','wrong'),('plan_sha256','0'*64),
    ('bound_provider_sha256',{})])
def test_wrong_stored_identity_rejected(review_evidence,field,value):
    review_evidence[field]=value
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review()


def test_environment_drift_rejected(review_evidence):
    review_evidence['runtime']={}
    with pytest.raises(ValueError,match='environment'):publisher.review()


@pytest.mark.parametrize('fault', ['missing','changed','extra'])
def test_implementation_hash_fault_rejected(review_evidence,fault):
    hashes=review_evidence['code_sha256']
    if fault=='missing':hashes.pop('sciona/visualizer/runner.py')
    elif fault=='changed':hashes['sciona/visualizer/runner.py']='0'*64
    else:hashes['unexpected']='0'*64
    with pytest.raises(ValueError,match='implementation'):publisher.review()


@pytest.mark.parametrize('symbol', ['porto_prepare','features','fit','blend_inputs','ordered_weighted_sum','result'])
def test_each_bound_provider_required(review_evidence,symbol):
    hashes=review_evidence['bound_provider_sha256']
    hashes.pop(next(key for key in hashes if key.endswith('.'+symbol)))
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review()


def test_unpinned_reference_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','pending')
    with pytest.raises(ValueError,match='Immutable'):publisher.publish(apply=True)


def test_unavailable_commit_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','0'*40)
    def missing(*args,**kwargs):
        raise publisher.subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(publisher.subprocess,'check_output',missing)
    with pytest.raises(ValueError,match='commit is unavailable'):publisher.publish(apply=True)


def test_different_committed_report_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','0'*40)
    monkeypatch.setattr(publisher.subprocess,'check_output',lambda *args,**kwargs:b'wrong report')
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'qualified report')
    with pytest.raises(ValueError,match='differs from qualified report'):publisher.publish(apply=True)
