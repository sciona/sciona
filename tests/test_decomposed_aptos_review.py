"""Reject incomplete full-training evidence and changed computation boundaries."""
import json
from pathlib import Path

import pytest

from scripts.review_decomposed_aptos import ROOT, check_training_body, review


@pytest.fixture
def evidence(monkeypatch):
    path = ROOT/'docs/reviews/decomposed_aptos_graph.json'
    document = json.loads(path.read_text())
    read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw:
        json.dumps(document) if self == path else read(self, *a, **kw))
    return document


@pytest.mark.parametrize('field,value', [
    ('passed', False), ('approved', True), ('catalog_mutations', 1),
    ('synthetic_only', False), ('model_count', 7), ('training_stage_fits', 15),
    ('first_stage_epochs_per_model', 1), ('second_stage_epochs_per_model', 1),
    ('optimizer_updates', 279), ('reference_augmentation_used', False),
    ('checkpoint_replays_exact', False), ('temporary_checkpoints_removed', False),
    ('target_role_oracle_passed', False), ('final_ensemble_and_threshold_oracle_passed', False),
    ('native_threads', 8), ('torch_interop_threads', 8),
])
def test_incomplete_training_rejected(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError, match='Complete decomposed training evidence'):
        review()


@pytest.mark.parametrize('field,value', [
    ('serialized_graph_sha256', '0'*64), ('actual_runner_nodes', 4),
    ('edges', 6), ('all_nodes_uncached', False), ('codec_roundtrip', False),
])
def test_changed_graph_execution_rejected(evidence, field, value):
    evidence['graph_execution'][field] = value
    with pytest.raises(ValueError, match='Registered graph execution differs'):
        review()


def test_missing_source_hash_rejected(evidence):
    evidence['sha256'].pop('sciona/decomposed_aptos_pipeline.py')
    with pytest.raises(ValueError, match='Complete implementation evidence'):
        review()


def test_changed_training_dependency_rejected(monkeypatch):
    from sciona import decomposed_aptos_pipeline as candidate
    monkeypatch.setattr(candidate, 'train_epochs', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='Training dependency changed: train_epochs'):
        check_training_body()
