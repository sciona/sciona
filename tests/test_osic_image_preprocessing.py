import cv2
import numpy as np
import pytest

from sciona.osic_image_preprocessing import inference_plane, training_plane


@pytest.fixture(autouse=True)
def serial_opencv():
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1


def test_constant_planes_expose_training_inference_difference():
    pixels = np.full((3, 7), 1024, dtype=np.int16)
    original = pixels.copy()
    np.testing.assert_array_equal(training_plane(pixels, -100., 2.), np.full((512, 512), .562))
    np.testing.assert_array_equal(inference_plane(pixels), np.full((512, 512), .5))
    np.testing.assert_array_equal(pixels, original)


def test_identity_size_preserves_signed_values_without_clipping():
    pixels = np.arange(512*512, dtype=np.int32).reshape(512, 512) - 131072
    trained = training_plane(pixels, 0., -2.)
    inferred = inference_plane(pixels)
    np.testing.assert_array_equal(trained, pixels / -2000.)
    np.testing.assert_array_equal(inferred, pixels / 2048.)
    assert not np.shares_memory(trained, pixels)
    assert not np.shares_memory(inferred, pixels)


@pytest.mark.parametrize('pixels', [np.empty((0, 3), dtype=np.int16),
    np.zeros(3, dtype=np.int16), np.zeros((2, 3), dtype=np.float64),
    np.zeros((2, 3), dtype=bool)])
def test_reject_invalid_decoded_plane(pixels):
    with pytest.raises(ValueError):
        inference_plane(pixels)


@pytest.mark.parametrize('intercept,slope', [(0., 0.), (np.nan, 1.), (0., np.inf)])
def test_reject_invalid_metadata(intercept, slope):
    with pytest.raises(ValueError):
        training_plane(np.ones((2, 2), dtype=np.uint16), intercept, slope)
