"""Registered count scorer contract and exact primitive execution checks."""
import pytest
from sciona.atoms.ml.text_nlp.bm25 import score_bm25
from sciona.ghost.registry import REGISTRY, get_witness
from tests import test_text_pair_bm25 as baseline
from tests.test_text_pair_bm25 import (
    test_numeric_hand_calculation_and_empty_query,
    test_length_normalization_and_frequency_saturation,
    test_invalid_counts_controls_and_overflow,
)


@pytest.fixture(autouse=True)
def registered_numeric_calls(monkeypatch):
    monkeypatch.setattr(baseline, 'bm25_score', score_bm25)


def test_exact_registration_and_abstract_rejection():
    name = score_bm25.__module__ + '.' + score_bm25.__name__
    assert REGISTRY[name]['impl'] is score_bm25
    with pytest.raises(ValueError, match='Concrete aligned counts'):
        get_witness(name)()


def test_caller_count_sequences_are_unchanged():
    tf, df = [2, 1], [3, 5]
    assert score_bm25(tf, df, 4, 10, 3., 1.2, .75) > 0
    assert tf == [2, 1] and df == [3, 5]
