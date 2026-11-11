import numpy as np
import pytest
from sciona.tgs_pseudo_selection import select_pseudo_labels


def test_confidence_area_and_constant_image_boundaries():
    area = np.array([0,0,19,20,500,501,2550,2551,5100,5101,7650,7651,9180,9181])
    confidence = np.ones(len(area))
    confidence[3] = .98
    eligible = np.ones(len(area), bool)
    eligible[1] = False
    result = select_pseudo_labels(confidence,area,eligible)
    selected = set(np.concatenate(result['torch_folds']))
    assert selected == {0,4,5,6,7,8,9,10,11,12}
    assert set(result['keras_indices']) == set(range(len(area))) - {1}


def test_empty_downsampling_and_uneven_source_fold_slices():
    count = 1001
    result = select_pseudo_labels(np.ones(count),np.full(count,3000),np.ones(count,bool))
    assert [len(fold) for fold in result['torch_folds']] == [240,240,260,240,21]
    assert len(set(np.concatenate(result['torch_folds']))) == count
    empty = select_pseudo_labels(np.ones(count),np.zeros(count,dtype=int),np.ones(count,bool))
    assert len(np.concatenate(empty['torch_folds'])) == 501
    np.testing.assert_array_equal(empty['torch_folds'][0],empty['metadata_order'][::2])


def test_selection_does_not_change_global_rng():
    np.random.seed(712)
    expected = np.random.rand(3)
    np.random.seed(712)
    select_pseudo_labels(np.array([.9,.899,.97]),np.array([0,0,501]),np.ones(3,bool))
    np.testing.assert_array_equal(np.random.rand(3),expected)


def test_fractional_area_rejected():
    with pytest.raises(ValueError):
        select_pseudo_labels([1.],[2.5],[True])
