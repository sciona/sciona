import numpy as np
import pytest

from sciona.wheat_prediction_boundary import prepare_prediction


def test_fasterrcnn_scales_before_strict_threshold_and_preserves_inputs():
    boxes = np.array([[0., 0., 20., 20.]] * 3)
    scores = np.array([.25, .2501, .8])
    old_boxes, old_scores = boxes.copy(), scores.copy()
    result, retained, labels = prepare_prediction(boxes, scores, detector='fasterrcnn', view=7, image_size=20)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(retained, old_scores[1:] * .8)
    np.testing.assert_array_equal(labels, [0., 0.])
    np.testing.assert_array_equal(boxes, old_boxes)
    np.testing.assert_array_equal(scores, old_scores)


def test_effdet_converts_xywh_before_normalization_and_clipping():
    b, s, _ = prepare_prediction(np.array([[5., 10., 10., 20.]]), np.array([.9]),
                                detector='effdet', view=7, image_size=20)
    np.testing.assert_array_equal(b, [[.25, .5, .75, 1.]])
    np.testing.assert_array_equal(s, [.9])


@pytest.mark.parametrize('detector', ['effdet', 'fasterrcnn'])
def test_empty_detector_output_retains_empty_view(detector):
    b, s, labels = prepare_prediction(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                                      detector=detector, view=0, image_size=512)
    assert b.shape == (0, 4) and s.shape == labels.shape == (0,)
