"""Synthetic parity for distinct preparation policies and coordinate frames."""
from unittest.mock import patch
import numpy as np
import pytest
from sciona import dsb_decomposed_preparation as candidate
from sciona import dsb_training_preprocessing as training
from sciona import dsb_world_preprocessing as world
from sciona import dsb_components as inference


def fixture():
    shape = (12, 14, 16)
    volume = np.linspace(-1200, 600, np.prod(shape)).reshape(shape)
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    left[2:10, 3:11, 2:7] = True
    right[2:10, 3:11, 9:14] = True
    return volume, left, right, np.array([1.2, .8, 1.5])


@pytest.mark.parametrize('policy', ['training', 'inference'])
def test_masks_preserve_crop_spacing_and_inputs(policy):
    args = fixture()
    saved = [a.copy() for a in args]
    actual_fn = candidate.prepare_training_masks if policy == 'training' else candidate.preprocess_from_masks
    expected_fn = training.prepare_training_masks if policy == 'training' else inference.preprocess_from_masks
    with patch.object(candidate, 'resample_regular_volume', wraps=candidate.resample_regular_volume) as spy:
        actual = actual_fn(*args)
    assert spy.call_count == 1
    expected = expected_fn(*args)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    for a, e in zip(args, saved):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize('flip', [False, True])
def test_world_annotation_frame(flip):
    args = fixture()
    kwargs = dict(origin_zyx=[10., 20., 30.], annotations_xyz_diameter=np.array([[34., 24., 10., 3.]]), flip=flip)
    actual = candidate.prepare_world_training_volume(*args, **kwargs)
    expected = world.prepare_world_training_volume(*args, **kwargs)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    assert actual[1].shape == (1, 4)  # World origin-plane annotation remains valid.


def test_voxel_annotation_adapter_keeps_sentinel():
    volume, left, right, spacing = fixture()
    rows = np.array([[3., 4., 0., 2.]])
    with patch.object(candidate, 'segment_volume', return_value=(left, right)):
        actual = candidate.prepare_training_volume(volume, spacing, rows)
    with patch.object(training, 'segment_volume', return_value=(left, right)):
        expected = training.prepare_training_volume(volume, spacing, rows)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    assert actual[1].shape == (0, 4)
