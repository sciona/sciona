import numpy as np
import pytest

from sciona.tgs_round_outputs import complete_round


def test_pseudo_barrier_retains_every_mask_and_separate_confident_selection():
    images = np.broadcast_to(np.linspace(0, 1, 101), (4, 101, 101)).copy()
    images[0] = 0
    scores = np.zeros_like(images)
    scores[0] = .9
    scores[1] = .5
    scores[2, :, :1] = .9
    scores[3] = .8
    result = complete_round(2, scores, images)
    assert result['masks'].shape == (4, 101, 101)
    assert result['confidence'].tolist() == [1., 0., 1., 0.]
    assert result['area'].tolist() == [10201, 0, 101, 10201]
    assert result['keras_indices'].tolist() == [2]
    assert result['torch_folds'][0].tolist() == [2]
    assert result['masks'][0].all()  # Constant/low-confidence masks remain available.
    assert not result['masks'][1].any()


def test_final_mosaic_requirement_cannot_be_omitted():
    values = np.ones((1, 101, 101))
    with pytest.raises(ValueError, match='requires labeled'):
        complete_round(3, values, values)


def test_query_boundary_rejects_invalid_probabilities():
    values = np.ones((1, 101, 101))
    with pytest.raises(ValueError, match='aligned finite'):
        complete_round(1, values * np.nan, values)
