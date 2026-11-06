"""Publication must reject reduced schedules, replaced callbacks and stale evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts import promote_adversarial_corrected_intake as publisher


@pytest.fixture
def review_evidence(monkeypatch):
    root=publisher.ROOT
    plan=json.loads(publisher.intake.PLAN.read_text())
    providers=json.loads(publisher.intake.provider_publication.PLAN.read_text())
    semantic=json.loads((root/'docs/reviews/adversarial_normalizer_provider_semantics.json').read_text())
    execution_path=root/'docs/reviews/adversarial_corrected_intake_execution.json'
    execution=json.loads(execution_path.read_text())
    monkeypatch.setattr(publisher.intake,'qualified_plan',lambda:(plan,providers))
    monkeypatch.setattr(publisher.intake,'connect',MagicMock())
    monkeypatch.setattr(publisher.intake.provider_publication,'check',lambda *args:None)
    from scripts import review_adversarial_normalizer_providers as provider_review
    monkeypatch.setattr(provider_review,'review',lambda:semantic)
    read_text=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**kw:json.dumps(execution) if self==execution_path else read_text(self,*a,**kw))
    return execution


def test_complete_frozen_execution_accepted(review_evidence):
    assert publisher.review()[2]['proposed_tier']==3


@pytest.mark.parametrize('branch',['non_targeted','targeted_large','targeted_small'])
@pytest.mark.parametrize('field,value',[
    ('full_models',1),('complete_iterations',1),('exact_reference_update_checks',0),
    ('registered_normalization_calls',1),('registered_callable_identity',False),
    ('neural_batch_size',1),('padded_entries',0),('actual_runner_nodes',2),
    ('only_real_entry_encoded',False),('decoded_png_matches_source_scaling',False),
    ('fixed_projection_bounds',False),('input_and_rng_unchanged',False),
])
def test_incomplete_or_substituted_computation_rejected(review_evidence,branch,field,value):
    review_evidence['checks'][branch][field]=value
    with pytest.raises(ValueError,match='complete neural schedules'):
        publisher.review()


@pytest.mark.parametrize('field,value',[
    ('result','failed'),('stored_candidate',False),('candidate_version_id','wrong'),
    ('original_artifact_id','wrong'),('serialized_graph_sha256','0'*64),
    ('plan_sha256','0'*64),('execution_provider_sha256','0'*64),('generic_provider_sha256','0'*64),
])
def test_stale_or_wrong_identity_rejected(review_evidence,field,value):
    review_evidence[field]=value
    with pytest.raises(ValueError):publisher.review()


def test_missing_schedule_rejected(review_evidence):
    del review_evidence['checks']['targeted_small']
    with pytest.raises(ValueError):publisher.review()


def test_missing_implementation_hash_rejected(review_evidence):
    review_evidence['sha256'].pop('sciona/visualizer/runner.py')
    with pytest.raises(ValueError,match='incomplete'):publisher.review()


def test_changed_implementation_rejected(review_evidence):
    review_evidence['sha256']['sciona/visualizer/runner.py']='0'*64
    with pytest.raises(ValueError,match='implementation changed'):publisher.review()


def test_unpinned_reference_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher,'EVIDENCE_COMMIT','pending')
    with pytest.raises(ValueError,match='Immutable'):publisher.publish(apply=True)
