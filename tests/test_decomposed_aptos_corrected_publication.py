"""Reject reduced computation, wrong identities and stale publication evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts import promote_aptos_corrected_intake as publisher


@pytest.fixture
def review_evidence(monkeypatch):
    root = publisher.ROOT
    plan = json.loads(publisher.intake.PLAN.read_text())
    providers = json.loads(publisher.intake.provider_publication.PLAN.read_text())
    semantic = json.loads((root/'docs/reviews/aptos_decomposed_provider_semantics.json').read_text())
    execution_path = root/'docs/reviews/aptos_corrected_intake_execution.json'
    execution = json.loads(execution_path.read_text())
    monkeypatch.setattr(publisher.intake, 'qualified_plan', lambda: (plan, providers))
    monkeypatch.setattr(publisher.intake, 'connect', MagicMock())
    monkeypatch.setattr(publisher.intake.provider_publication, 'check', lambda *args: None)
    from scripts import review_aptos_decomposed_providers as provider_review
    monkeypatch.setattr(provider_review, 'review', lambda: semantic)
    read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw:
        json.dumps(execution) if self == execution_path else read_text(self, *a, **kw))
    return execution


def test_complete_execution_accepted(review_evidence):
    assert publisher.review()[2]['proposed_tier'] == 3


@pytest.mark.parametrize('field,value', [
    ('passed',False),('approved',True),('catalog_mutations',1),('synthetic_only',False),
    ('model_count',7),('training_stage_fits',15),('first_stage_epochs_per_model',1),
    ('second_stage_epochs_per_model',1),('optimizer_updates',279),('native_threads',8),
    ('torch_interop_threads',8),('reference_augmentation_used',False),
    ('checkpoint_replays_exact',False),('temporary_checkpoints_removed',False),
    ('target_role_oracle_passed',False),('final_ensemble_and_threshold_oracle_passed',False),
    ('format','wrong'),
])
def test_incomplete_execution_rejected(review_evidence,field,value):
    review_evidence[field]=value
    with pytest.raises(ValueError):publisher.review()


@pytest.mark.parametrize('field,value', [
    ('serialized_graph_sha256','0'*64),('provider_sha256','0'*64),
    ('actual_runner_nodes',4),('edges',6),('all_nodes_uncached',False),('codec_roundtrip',False),
    ('population_provider_sha256','0'*64),('generic_provider_sha256','0'*64),
    ('stored_candidate',False),('candidate_version_id','wrong'),('plan_sha256','0'*64),
])
def test_wrong_stored_graph_rejected(review_evidence,field,value):
    review_evidence['graph_execution'][field]=value
    with pytest.raises(ValueError,match='Stored execution identity'):publisher.review()


@pytest.mark.parametrize('field',['installed_package_versions','artifact_sha256'])
def test_environment_drift_rejected(review_evidence,field):
    review_evidence[field]={}
    with pytest.raises(ValueError,match='environment or pretrained'):publisher.review()


def test_missing_implementation_hash_rejected(review_evidence):
    review_evidence['sha256'].pop('sciona/visualizer/runner.py')
    with pytest.raises(ValueError,match='incomplete'):publisher.review()


def test_changed_implementation_rejected(review_evidence):
    review_evidence['sha256']['sciona/visualizer/runner.py']='0'*64
    with pytest.raises(ValueError,match='implementation changed'):publisher.review()


def test_extra_implementation_rejected(review_evidence):
    review_evidence['sha256']['unexpected']='0'*64
    with pytest.raises(ValueError,match='incomplete'):publisher.review()


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
