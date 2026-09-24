import numpy as np
import pytest

from sciona.atoms.ml.model_selection.population_masks import grouped_holdout_masks, residual_threshold_mask


def test_group_partition_is_exhaustive_disjoint_and_repeatable():
    groups = np.repeat(np.arange(10, dtype=np.int64), np.arange(1, 11))
    train, held = grouped_holdout_masks(groups, .4, 42)
    assert np.all(train ^ held)
    assert not set(groups[train]) & set(groups[held])
    assert len(np.unique(groups[train])) == 4
    assert len(np.unique(groups[held])) == 6
    np.testing.assert_array_equal(train, grouped_holdout_masks(groups, .4, 42)[0])
    np.testing.assert_array_equal(groups, np.repeat(np.arange(10, dtype=np.int64), np.arange(1, 11)))


@pytest.mark.parametrize('groups,fraction,seed', [
    (np.array([1, 1]), .4, 42), (np.array([1., 2.]), .4, 42),
    (np.arange(10), 0., 42), (np.arange(10), 1., 42),
    (np.arange(10), float('nan'), 42), (np.arange(10), .4, True),
    (np.arange(10), .4, -1), (np.arange(10), .4, 2**32),
    (np.arange(2), .01, 42)])
def test_group_invalid_inputs_rejected(groups, fraction, seed):
    with pytest.raises(ValueError):
        grouped_holdout_masks(groups, fraction, seed)


def test_residual_boundary_no_rounding_and_empty_selection():
    observed = np.array([20., -20., 20.001, 0.])
    predicted = np.zeros(4)
    np.testing.assert_array_equal(residual_threshold_mask(observed, predicted, 20.), [True, True, False, True])
    assert not residual_threshold_mask(np.array([1.]), np.array([0.]), 0.).any()
    assert residual_threshold_mask(np.array([], dtype=float), np.array([], dtype=float), 0.).size == 0
    np.testing.assert_array_equal(observed, [20., -20., 20.001, 0.])


@pytest.mark.parametrize('target,predicted,threshold', [
    (np.array([1]), np.array([0.]), 1.),
    (np.array([np.nan]), np.array([0.]), 1.),
    (np.array([1.]), np.array([0., 1.]), 1.),
    (np.array([1.]), np.array([0.]), -1.),
    (np.array([1.]), np.array([0.]), True),
    (np.array([1.]), np.array([0.]), float('inf')),
    (np.array([1e308]), np.array([-1e308]), 1.)])
def test_residual_invalid_inputs_rejected(target, predicted, threshold):
    with pytest.raises(ValueError):
        residual_threshold_mask(target, predicted, threshold)


@pytest.mark.parametrize('domain,scale', [('manufacturing_dimensions', .01), ('energy_forecasts', 100.)])
def test_synthetic_domain_transfer(domain, scale):
    groups = np.repeat(np.arange(10, dtype=np.int64), 3)
    train, held = grouped_holdout_masks(groups, .4, 42)
    assert not np.any(train & held) and domain
    observed = np.array([10., 15., 30.])*scale
    predictions = np.array([10., 10., 10.])*scale
    np.testing.assert_array_equal(residual_threshold_mask(observed, predictions, 5.*scale), [True, True, False])


def test_residual_witness_rejects_incompatible_threshold_units():
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    from sciona.ghost.dimensions import DimensionalSignature
    from sciona.atoms.ml.model_selection.population_masks import witness_residual_threshold_mask
    value = AbstractArray(shape=(3,), dtype='float64', dim=DimensionalSignature(L=1))
    with pytest.raises(ValueError, match='share units'):
        witness_residual_threshold_mask(value, value, AbstractScalar(dtype='float64', dim=DimensionalSignature(T=1)))


def test_group_witness_rejects_dimensioned_fraction():
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    from sciona.ghost.dimensions import DimensionalSignature
    from sciona.atoms.ml.model_selection.population_masks import witness_grouped_holdout_masks
    with pytest.raises(ValueError, match='dimensionless'):
        witness_grouped_holdout_masks(AbstractArray(shape=(3,), dtype='int64'),
            AbstractScalar(dtype='float64', dim=DimensionalSignature(T=1)), AbstractScalar(dtype='int64'))
