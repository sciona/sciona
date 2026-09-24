import numpy as np
import pytest
from sciona.atoms.ml.calibration.adaptive_history import adaptive_history_statistics


def stats(times, values, minimum=3):
    return adaptive_history_statistics(np.array(times, dtype=np.int64), np.array(values, dtype=np.float64),
                                       0, np.array([60, 120, 180], dtype=np.int64), minimum)


def test_empty_initial_window_uses_longest_window_directly():
    assert stats([-70, -130], [10, 30]) == (2, 20., 10.)


def test_sparse_initial_window_does_not_repeatedly_expand():
    assert stats([-10, -130], [10, 30]) == (1, 10., 0.)


def test_dense_window_excludes_old_values_and_both_boundaries():
    count, mean, std = stats([-1, -2, -3, -60, 0, -70], [1, 2, 3, 100, 200, 300])
    assert count == 3 and mean == 2
    assert std == pytest.approx(np.sqrt(2/3))


def test_empty_final_window_preserves_undefined_statistics():
    count, mean, std = stats([], [])
    assert count == 0 and np.isnan(mean) and np.isnan(std)


def test_arbitrary_order_does_not_change_membership():
    assert stats([-130, -10], [30, 10]) == stats([-10, -130], [10, 30])


def test_integer_time_boundaries_do_not_overflow():
    lower = np.iinfo(np.int64).min
    assert adaptive_history_statistics(np.array([lower], dtype=np.int64), np.array([4.]),
        lower+1, np.array([60, 120, 180], dtype=np.int64), 3) == (1, 4., 0.)


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')]])
def test_invalid_values_rejected(values):
    with pytest.raises(ValueError, match='Finite'):
        stats([-1], values)


@pytest.mark.parametrize('minimum', [True, 0, 1.5])
def test_invalid_count_control_rejected(minimum):
    with pytest.raises(ValueError):
        stats([-1], [1], minimum)


def test_witness_rejects_incompatible_time_units():
    from sciona.atoms.ml.calibration.adaptive_history import witness_adaptive_history_statistics
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    from sciona.ghost.dimensions import DimensionalSignature
    with pytest.raises(ValueError, match='Consistent time units'):
        witness_adaptive_history_statistics(AbstractArray(shape=(2,),dtype='int64',dim=DimensionalSignature(T=1)),
            AbstractArray(shape=(2,)), AbstractScalar(dtype='int64',dim=DimensionalSignature(L=1)),
            AbstractArray(shape=(3,),dtype='int64'), AbstractScalar(dtype='int64'))


def test_witness_rejects_dimensioned_sample_count():
    from sciona.atoms.ml.calibration.adaptive_history import witness_adaptive_history_statistics
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    from sciona.ghost.dimensions import DimensionalSignature
    with pytest.raises(ValueError, match='Count must be dimensionless'):
        witness_adaptive_history_statistics(AbstractArray(shape=(2,),dtype='int64'), AbstractArray(shape=(2,)),
            AbstractScalar(dtype='int64'), AbstractArray(shape=(3,),dtype='int64'),
            AbstractScalar(dtype='int64',dim=DimensionalSignature(T=1)))


@pytest.mark.parametrize('widths', [[60,60,180],[0,120,180],[180,120,60],[60,120]])
def test_invalid_window_policy_rejected(widths):
    with pytest.raises(ValueError,match='lookbacks'):
        adaptive_history_statistics(np.array([-1],dtype=np.int64),np.array([1.]),0,
                                    np.array(widths,dtype=np.int64),3)


def test_nonempty_overflow_is_not_reported_as_missing_history():
    with pytest.raises(ValueError,match='Statistics outside'):
        stats([-1,-2],[1e308,1e308])


def test_witness_checks_known_units_when_history_units_are_unknown():
    from sciona.atoms.ml.calibration.adaptive_history import witness_adaptive_history_statistics
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    from sciona.ghost.dimensions import DimensionalSignature
    with pytest.raises(ValueError,match='Consistent time units'):
        witness_adaptive_history_statistics(AbstractArray(shape=(2,),dtype='int64'),AbstractArray(shape=(2,)),
            AbstractScalar(dtype='int64',dim=DimensionalSignature(L=1)),
            AbstractArray(shape=(3,),dtype='int64',dim=DimensionalSignature(T=1)),AbstractScalar(dtype='int64'))


def test_witness_rejects_fractional_query_metadata():
    from sciona.atoms.ml.calibration.adaptive_history import witness_adaptive_history_statistics
    from sciona.ghost.abstract import AbstractArray, AbstractScalar
    with pytest.raises(ValueError,match='Integer query'):
        witness_adaptive_history_statistics(AbstractArray(shape=(2,),dtype='int64'),AbstractArray(shape=(2,)),
            AbstractScalar(dtype='float64'),AbstractArray(shape=(3,),dtype='int64'),AbstractScalar(dtype='int64'))
