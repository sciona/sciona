"""Reject reduced computation, wrong identities and stale publication evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts import promote_amex_corrected_intake as publisher


@pytest.fixture
def review_evidence(monkeypatch):
    root = publisher.ROOT
    plan = json.loads(publisher.intake.PLAN.read_text())
    providers = json.loads(publisher.intake.provider_publication.PLAN.read_text())
    semantic = json.loads((root/'docs/reviews/amex_decomposed_provider_semantics.json').read_text())
    execution_path = root/'docs/reviews/amex_corrected_intake_execution.json'
    execution = json.loads(execution_path.read_text())
    monkeypatch.setattr(publisher.intake, 'qualified_plan', lambda: (plan, providers))
    monkeypatch.setattr(publisher.intake, 'connect', MagicMock())
    monkeypatch.setattr(publisher.intake.provider_publication, 'check', lambda *args: None)
    from scripts import review_amex_decomposed_providers as provider_review
    monkeypatch.setattr(provider_review, 'review', lambda: semantic)
    read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw:
        json.dumps(execution) if self == execution_path else read_text(self, *a, **kw))
    return execution


def test_complete_execution_accepted(review_evidence):
    assert publisher.review()[2]['proposed_tier'] == 3


@pytest.mark.parametrize('field,value', [
    ('passed', False), ('approved', True), ('catalog_mutations', 1),
    ('stored_candidate', False), ('actual_runner_nodes', 2), ('edges', 1),
    ('all_nodes_uncached', False), ('models', 1), ('dart_models', 1),
    ('rounds_per_dart_model', 10), ('neural_models', 1), ('epochs_per_neural_model', 1),
    ('exact_reference_output', False), ('exact_source_blend', False), ('strict_json', False),
    ('input_and_torch_rng_preserved', False), ('format', 'wrong'),
    ('serialized_graph_sha256', '0'*64), ('candidate_version_id', 'wrong'),
    ('original_artifact_id', 'wrong'), ('plan_sha256', '0'*64),
    ('provider_sha256', '0'*64), ('generic_provider_sha256', '0'*64),
])
def test_incomplete_or_stale_evidence_rejected(review_evidence, field, value):
    review_evidence[field] = value
    with pytest.raises(ValueError):
        publisher.review()


@pytest.mark.parametrize('phase', ['reference', 'graph'])
def test_missing_model_rejected(review_evidence, phase):
    review_evidence['training_records'][phase].pop()
    with pytest.raises(ValueError):
        publisher.review()


@pytest.mark.parametrize('field,value', [
    ('evaluated_rounds', 1), ('retained_iterations', 0), ('retained_iterations', 4501),
])
def test_matching_but_invalid_training_traces_rejected(review_evidence, field, value):
    for phase in ('reference', 'graph'):
        review_evidence['training_records'][phase][0][field] = value
    with pytest.raises(ValueError):
        publisher.review()


def test_training_disagreement_rejected(review_evidence):
    review_evidence['training_records']['graph'][0]['retained_iterations'] = 1
    with pytest.raises(ValueError):
        publisher.review()


def test_missing_implementation_hash_rejected(review_evidence):
    review_evidence['implementation_sha256'].pop('sciona/visualizer/runner.py')
    with pytest.raises(ValueError, match='incomplete'):
        publisher.review()


def test_changed_implementation_rejected(review_evidence):
    review_evidence['implementation_sha256']['sciona/visualizer/runner.py'] = '0'*64
    with pytest.raises(ValueError, match='implementation changed'):
        publisher.review()


def test_unpinned_reference_cannot_publish(monkeypatch):
    monkeypatch.setattr(publisher, 'EVIDENCE_COMMIT', 'pending')
    with pytest.raises(ValueError, match='Immutable'):
        publisher.publish(apply=True)
