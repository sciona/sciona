import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.serial_feature_union import SerialFeatureUnion


class NumericBranch(BaseEstimator, TransformerMixin):
    def __init__(self, mutate=False, bad_rows=False):
        self.mutate = mutate
        self.bad_rows = bad_rows

    def fit(self, frame, targets=None):
        if self.mutate:
            frame.iloc[:, 0] = 99.
            if targets is not None:
                targets[:] = 99.
        self.mean_ = frame.to_numpy().mean(axis=0)
        return self

    def transform(self, frame):
        result = frame.to_numpy(dtype=float) - self.mean_
        return result[:-1] if self.bad_rows else result


def test_order_independent_inputs_and_training_only_state():
    frame = pd.DataFrame({'a': [1., 3.], 'b': [10., 20.]})
    targets = np.array([2., 4.])
    original = frame.copy(deep=True)
    template = NumericBranch(mutate=True)
    union = SerialFeatureUnion([('mutator', ['a'], template), ('ordinary', ['b', 'a'], NumericBranch())])
    actual = union.fit_transform(frame, targets).toarray()
    np.testing.assert_array_equal(actual, [[0., -5., -1.], [0., 5., 1.]])
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_array_equal(targets, [2., 4.])
    assert not hasattr(template, 'mean_')
    query = pd.DataFrame({'b': [30.], 'a': [5.]})
    np.testing.assert_array_equal(union.transform(query).toarray(), [[-94., 15., 3.]])
    assert union.transform(query.iloc[:0]).shape == (0, 3)


def test_failed_refit_invalidates_state():
    frame = pd.DataFrame({'a': [1., 3.]})
    union = SerialFeatureUnion([('one', ['a'], NumericBranch())])
    union.fit_transform(frame)
    union.branches = [('one', ['missing'], NumericBranch())]
    with pytest.raises(ValueError): union.fit_transform(frame)
    with pytest.raises(ValueError): union.transform(frame)


@pytest.mark.parametrize('branches', [[], [('x', [], NumericBranch())],
    [('x', ['a'], NumericBranch()), ('x', ['a'], NumericBranch())],
    [('x', ['a', 'a'], NumericBranch())], [('x', ['a'], object())]])
def test_invalid_branch_contract(branches):
    with pytest.raises(ValueError):
        SerialFeatureUnion(branches).fit_transform(pd.DataFrame({'a': [1.]}))


def test_row_count_and_mutable_cells_rejected():
    with pytest.raises(ValueError):
        SerialFeatureUnion([('x', ['a'], NumericBranch(bad_rows=True))]).fit_transform(pd.DataFrame({'a': [1., 2.]}))
    with pytest.raises(ValueError):
        SerialFeatureUnion([('x', ['a'], NumericBranch())]).fit_transform(pd.DataFrame({'a': [[1], [2]]}))


def test_inference_width_drift_and_unaligned_targets_rejected():
    frame = pd.DataFrame({'a': [1., 3.]})
    union = SerialFeatureUnion([('x', ['a'], NumericBranch())])
    with pytest.raises(ValueError):
        union.fit_transform(frame, np.array([1.]))
    union.fit_transform(frame)
    union._fitted[0][2].mean_ = np.array([0., 1.])
    with pytest.raises(ValueError):
        union.transform(frame)
