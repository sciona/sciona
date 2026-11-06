"""Synthetic numerical oracles for independently reusable array operations."""
import itertools
import numpy as np
import pytest
from sciona.constrained_array_ops import project_nonnegative_fixed_sum, smooth_periodic_rows
from sciona.physical_operator_projection import project_and_smooth


def exhaustive_projection(row, total):
    if total == 0:
        return np.zeros_like(row)
    for mask in itertools.product((False, True), repeat=len(row)):
        active = np.asarray(mask)
        if not active.any():
            continue
        multiplier = (row[active].sum() - total) / active.sum()
        if (row[active] > multiplier).all() and (row[~active] <= multiplier).all():
            return np.maximum(row - multiplier, 0)
    raise AssertionError('No feasible active set')


def test_projection_exhaustive_active_sets_and_equivariance():
    rng = np.random.default_rng(902)
    rows = rng.normal(size=(30, 5))
    totals = rng.uniform(.01, 8, size=30)
    expected = np.stack([exhaustive_projection(r, t) for r, t in zip(rows, totals)])
    actual = project_nonnegative_fixed_sum(rows, totals)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual.sum(1), totals, atol=1e-12)
    np.testing.assert_allclose(project_nonnegative_fixed_sum(actual, totals), actual, atol=1e-12)
    np.testing.assert_allclose(project_nonnegative_fixed_sum(rows + 17, totals), actual, atol=1e-12)
    np.testing.assert_allclose(project_nonnegative_fixed_sum(rows[:, ::-1], totals), actual[:, ::-1], atol=1e-12)


def test_zero_and_singleton_projection():
    np.testing.assert_array_equal(project_nonnegative_fixed_sum([[9, -3]], [0]), [[0, 0]])
    np.testing.assert_array_equal(project_nonnegative_fixed_sum([[-9], [50]], [3, 7]), [[3], [7]])


@pytest.mark.parametrize('width', [1, 2, 7])
@pytest.mark.parametrize('weight', [0., .2, .5])
def test_smoothing_against_explicit_circular_matrix(width, weight):
    rows = np.arange(3 * width, dtype=float).reshape(3, width) - 4
    original = rows.copy()
    matrix = np.zeros((width, width))
    for i in range(width):
        matrix[i, i] += 1 - 2 * weight
        matrix[i, (i - 1) % width] += weight
        matrix[i, (i + 1) % width] += weight
    actual = smooth_periodic_rows(rows, weight)
    np.testing.assert_allclose(actual, rows @ matrix.T)
    np.testing.assert_allclose(actual.sum(1), rows.sum(1))
    np.testing.assert_array_equal(rows, original)
    assert not np.shares_memory(rows, actual)


def test_composition_matches_retained_physical_recipe_without_mutation():
    rng = np.random.default_rng(31)
    rows = rng.normal(size=(4, 13))
    initial = rng.uniform(size=rows.shape)
    original = rows.copy()
    for weight in (0, .2, .5):
        projected = project_nonnegative_fixed_sum(rows, initial.sum(1))
        result = smooth_periodic_rows(projected, weight)
        np.testing.assert_allclose(result, project_and_smooth(rows, initial, smoothing=weight), atol=1e-12)
        assert (result >= 0).all()
    np.testing.assert_array_equal(rows, original)


@pytest.mark.parametrize('values,totals', [([], []), ([[1]], [-1]), ([[1]], [float('inf')]),
    ([[float('nan')]], [1]), ([[1, 2]], [[3]]), ([[True]], [1]), ([[1j]], [1]), ([['1']], [1])])
def test_invalid_projection(values, totals):
    with pytest.raises(ValueError):
        project_nonnegative_fixed_sum(values, totals)


@pytest.mark.parametrize('weight', [True, -.1, .6, float('nan'), float('inf')])
def test_invalid_smoothing(weight):
    with pytest.raises(ValueError):
        smooth_periodic_rows([[1, 2]], weight)


def test_extreme_projection_rejected_and_large_smoothing_finite():
    with pytest.raises(ValueError):
        project_nonnegative_fixed_sum([[-1e308, 1e308]], [1])
    np.testing.assert_allclose(smooth_periodic_rows([[1e308, 1e308]], .5), [[1e308, 1e308]])
