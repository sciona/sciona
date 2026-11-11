import numpy as np
import pytest

from sciona.tgs_checkpoint_selection import average_torch_folds, ensemble_selection


def test_stage_three_preserves_fold_weights_despite_unequal_snapshots():
    selected = ensemble_selection(3)
    counts = [len(fold['indices']) for fold in selected['pytorch']]
    assert counts == [4, 4, 4, 3, 3]
    values = [np.full((1, 2, 2), value) for value in [0., 0., 0., 1., 1.]]
    np.testing.assert_array_equal(average_torch_folds(values), np.full((1, 2, 2), .4))
    assert .4 != np.average([0., 0., 0., 1., 1.], weights=counts)


def test_rounds_select_validation_best_keras_and_exclude_first_torch_cycle():
    first, second, third = [ensemble_selection(stage) for stage in (1, 2, 3)]
    assert all(fold['indices'] == [1, 2, 3, 4, 5] for fold in first['pytorch'])
    assert [group[0]['fit'] for group in first['keras']] == [f'keras.r1.p{p}.f0' for p in [4, 3, 2, 1]]
    assert second['keras'] == third['keras']
    assert all(fold['selection'] == 'best_validation' for group in third['keras'] for fold in group)
    assert third['keras_weights'] == [1, 1, 1, 3]
    assert not first['mosaic_postprocessing'] and third['mosaic_postprocessing']


def test_rejects_invalid_stage_and_fold_population():
    for stage in (True, 0, 4, 1.0):
        with pytest.raises(ValueError):
            ensemble_selection(stage)
    with pytest.raises(ValueError):
        average_torch_folds([np.zeros((1, 2, 2))] * 4)
    with pytest.raises(ValueError):
        average_torch_folds([np.zeros((1, 2, 2))] * 4 + [np.full((1, 2, 2), np.nan)])
