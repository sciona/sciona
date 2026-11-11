"""Synthetic mathematical oracles for the complete-workflow blend component."""
import numpy as np
import pytest

from sciona.tgs_ensemble import blend_predictions


def test_source_snapshot_weight_order_and_constant_branch_policy():
    keras = np.broadcast_to(np.array([0.1, 0.2, 0.3, 0.8])[:, None, None, None, None], (4, 2, 2, 2, 2))
    raw = np.arange(8.).reshape(2, 2, 2)
    result = blend_predictions(keras, raw, [False, True], 2)
    # Hand arithmetic: final snapshot gets triple weight; only Keras is zeroed.
    expected = np.stack(((0.5 + raw[0] / 7) / 2, raw[1] / 14))
    np.testing.assert_allclose(result['scores'], expected)
    np.testing.assert_array_equal(result['masks'], expected > 0.5)
    first = blend_predictions(keras, raw, [False, False], 1)
    np.testing.assert_allclose(first['scores'], (0.35 + raw / 7) / 2)


def test_strict_thresholds_and_population_dependence():
    keras = np.zeros((4, 1, 1, 1, 3))
    result = blend_predictions(keras, [[[0., 0.4, 1.]]], [False], 3)
    np.testing.assert_array_equal(result['masks'], [[[False, False, False]]])
    np.testing.assert_allclose(result['confidence'], [1 / 3])
    expanded = blend_predictions(np.zeros((4, 1, 2, 1, 3)),
                                 [[[0., 0.4, 1.]], [[0., 1., 2.]]], [False, False], 3)
    assert expanded['scores'][0, 0, 2] == 0.25
    assert result['scores'][0, 0, 2] == 0.5


@pytest.mark.parametrize('bad', [np.zeros((1, 2, 2)), np.full((1, 2, 2), np.nan)])
def test_undefined_population_rejected(bad):
    with pytest.raises(ValueError):
        blend_predictions(np.zeros((4, 1, 1, 2, 2)), bad, [False], 1)


def test_inputs_unchanged_and_bad_alignment_rejected():
    keras = np.full((4, 2, 1, 2, 2), 0.5)
    original = keras.copy()
    blend_predictions(keras, np.arange(4.).reshape(1, 2, 2), [True], 1)
    np.testing.assert_array_equal(keras, original)
    with pytest.raises(ValueError):
        blend_predictions(keras, np.arange(4.).reshape(1, 2, 2), [0], 1)
