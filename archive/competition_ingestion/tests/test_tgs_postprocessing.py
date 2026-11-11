"""Known-layout synthetic scene; actual feature, search and assembly code runs."""
import numpy as np
import pytest

from sciona.tgs_assembly import construct_mosaics
from sciona.tgs_postprocessing import postprocess
from sciona.tgs_round_outputs import complete_round


def scene():
    rng = np.random.default_rng(283)
    tiles = rng.uniform(0, 1, (16, 101, 101))
    for row in range(4):
        for column in range(4):
            index = row * 4 + column
            if column < 3:
                seam = rng.uniform(0, 1, 97)
                tiles[index, 2:-2, -2:] = seam[:, None]
                tiles[index + 1, 2:-2, :2] = seam[:, None]
            if row < 3:
                seam = rng.uniform(0, 1, 97)
                tiles[index, -2:, 2:-2] = seam[None, :]
                tiles[index + 4, :2, 2:-2] = seam[None, :]
    for x in (slice(0, 2), slice(-2, None)):
        for y in (slice(0, 2), slice(-2, None)):
            tiles[:, x, y] = 0
    return tiles


def test_known_scene_layout_survives_population_permutation():
    images = scene()
    order = np.random.default_rng(48).permutation(16)
    grids = construct_mosaics(images[order].transpose(0, 2, 1))
    assert len(grids) == 1
    np.testing.assert_array_equal(order[grids[0]], np.arange(16).reshape(4, 4))


def test_joint_assembly_and_mask_propagation():
    images = scene()
    masks = np.zeros((4, 101, 101), dtype=bool)
    for column in range(4):
        masks[column, :, 10 + column] = True
    result = postprocess(images[:4], masks, images[4:], np.zeros_like(images[4:]))
    assert result['mosaic_count'] == 1 and result['mosaic_tile_count'] == 16
    assert result['replaced_query_indices'] == list(range(12))
    np.testing.assert_array_equal(result['masks'], np.tile(masks, (3, 1, 1)))


def test_final_round_connects_ensemble_scores_to_real_mosaic_propagation():
    images = scene()
    masks = np.zeros((4, 101, 101), dtype=bool)
    masks[:, :, 20] = True
    result = complete_round(3, np.full_like(images[4:], .3), images[4:],
                            training_images=images[:4], training_masks=masks)
    assert result['stage'] == 3
    assert result['mosaic_count'] == 1
    assert result['replaced_query_indices'] == list(range(12))
    np.testing.assert_array_equal(result['masks'], np.tile(masks, (3, 1, 1)))


def test_no_neighbors_preserves_strict_score_threshold():
    train = np.ones((1, 101, 101))
    query = np.ones((2, 101, 101))
    scores = query * 0.5
    scores[1] = 0.6
    result = postprocess(train, train, query, scores)
    assert result['mosaic_count'] == 0
    np.testing.assert_array_equal(result['masks'], scores > 0.5)


def test_bad_scores_rejected():
    images = np.ones((1, 101, 101))
    with pytest.raises(ValueError, match='scores'):
        postprocess(images, images, images, images * np.nan)
