"""Synthetic source checks for reservation, mutation and candidate isolation."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.atoms.particle_tracking.track_matching.atoms import greedy_track_commit
from sciona.decomposed_trackml_commit_lifecycle import commit_operation, candidate_class
from sciona.trackml_candidates import Candidates
from scripts.review_trackml_commit_reuse import source_function


@pytest.mark.parametrize('reserve', [False, True])
@pytest.mark.parametrize('limit', [None, 0, 1, 4])
@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_source_mutation_defaults_and_reservation(reserve, limit, dtype):
    source, _ = source_function()
    matrix = np.array([[1, 2, 0], [1, 3, 4], [4, 5, 6], [7, 7, 0]], dtype=dtype)
    used = np.zeros(8, dtype=bool)
    expected, mask = matrix.copy(), used.copy()
    kwargs = dict(min_nhits=3, max_nrows=limit, reserve_skipped=reserve)
    source(expected, mask, **kwargs)
    assert commit_operation(greedy_track_commit)(matrix, used, **kwargs) is None
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(used, mask)
    assert matrix.dtype == dtype


def test_candidate_copy_preserves_binding_without_global_mutation():
    original = Candidates.submit
    first, second = candidate_class(greedy_track_commit), candidate_class(greedy_track_commit)
    assert first is not second
    assert first.submit.__globals__['zeroUsedHits'] is not second.submit.__globals__['zeroUsedHits']
    candidate = first(SimpleNamespace())
    assert type(candidate.copy()) is first
    assert Candidates.submit is original


@pytest.mark.parametrize('matrix', [np.array([[0, 0]]), np.array([[-1, 1]]), np.array([[1, 9]])])
def test_invalid_rows_rejected_before_provider(matrix):
    def forbidden(*a, **k):
        pytest.fail('Invalid domain reached provider')
    used = np.zeros(3, dtype=bool)
    saved = matrix.copy()
    with pytest.raises(ValueError):
        commit_operation(forbidden)(matrix, used)
    np.testing.assert_array_equal(matrix, saved)
    assert not used.any()
