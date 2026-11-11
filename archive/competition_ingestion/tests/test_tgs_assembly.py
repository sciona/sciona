import numpy as np
import pytest

from sciona.tgs_assembly import candidate_edges, assemble_edges, construct_mosaics


def test_candidates_match_exhaustive_join_oracle():
    reference = np.array([[10., 0], [0, 0], [.2, 0], [20, 0]])
    query = np.array([[.01, 0], [10.1, 0], [20.1, 0], [30, 0]])
    def exhaustive(a, b):
        rows = []
        for origin, vector in enumerate(b):
            candidates = sorted((float(np.linalg.norm(vector - item)), i) for i, item in enumerate(a))
            (d1, target), (d2, _) = candidates[:2]
            rows.append((origin, target, d1, 1 - d1 / d2))
        return rows
    forward, reverse = exhaustive(reference, query), exhaustive(query, reference)
    expected = [(i, j) for i, j, distance, confidence in forward
                for _, winner, _, reverse_confidence in reverse
                if i == winner and i != j and distance < 10 and confidence > .25 and reverse_confidence > .25]
    assert expected.count((0, 1)) == 2
    assert candidate_edges(reference, query) == expected


def test_ties_and_zero_distances_are_not_arbitrarily_selected():
    assert candidate_edges(np.zeros((4, 3)), np.zeros((4, 3))) == []


def test_complete_square_preserves_origin_and_orientation():
    result = assemble_edges(4, [(1, 0), (3, 2)], [(0, 2), (1, 3)])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], [[0, 1], [2, 3]])


def test_holes_and_component_order():
    grids = assemble_edges(5, [(1, 0), (4, 3)], [(0, 2)])
    np.testing.assert_array_equal(grids[0], [[0, 1], [2, -1]])
    np.testing.assert_array_equal(grids[1], [[3, 4]])


def test_contradictory_cycle_rejected():
    with pytest.raises(ValueError, match='contradicts'):
        assemble_edges(3, [(0, 1), (1, 2), (2, 0)], [])


def test_constant_images_have_no_mosaics():
    assert construct_mosaics(np.ones((3, 5, 6))) == []
