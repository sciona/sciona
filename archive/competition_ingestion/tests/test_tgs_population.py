import numpy as np
import pytest

from sciona.tgs_population import select_fit_population


def fixture():
    return np.tile(np.arange(5), 2), np.array([False] + [True] * 9), np.array([False, True, True, True])


def test_keras_excludes_constant_labeled_images_both_sides():
    folds, labeled, query = fixture()
    result = select_fit_population('keras.r1.p0.f1', folds, labeled, query)
    assert result['training_labeled'].tolist() == [2, 3, 4, 5, 7, 8, 9]
    assert result['validation_labeled'].tolist() == [1, 6]
    assert len(result['training_query']) == 0


def test_pseudo_warmups_have_different_training_and_validation_populations():
    folds, labeled, query = fixture()
    keras = select_fit_population('keras.r2.p0.f0', folds, labeled, query,
                                  confidence=np.array([1., .89, .9, 1.]), area=np.zeros(4, dtype=int))
    assert sorted(keras['training_query'].tolist()) == [2, 3]
    assert keras['validation_labeled'].tolist() == [5]
    torch = select_fit_population('torch.p2.f0', folds, labeled, query,
                                  pseudo_validation=np.array([0, 3, 8]))
    assert torch['training_query'].tolist() == [0, 1, 2, 3]
    assert torch['validation_labeled'].tolist() == [0, 3, 8]
    assert len(torch['training_labeled']) == len(keras['training_labeled']) == 0


def test_same_index_pseudo_bucket_does_not_replace_labeled_validation():
    folds, labeled, _ = fixture()
    flags = np.ones(501, dtype=bool)
    kw = dict(confidence=np.ones(501), area=np.full(501, 100, dtype=int))
    a = select_fit_population('torch.p1.f0', folds, labeled, flags, **kw)
    b = select_fit_population('torch.p1.f1', folds, labeled, flags, **kw)
    assert len(a['training_query']) == len(b['training_query']) == 110
    assert not np.intersect1d(a['training_query'], b['training_query']).size
    assert a['validation_labeled'].tolist() == [0, 5]
    assert 0 in b['training_labeled']  # PyTorch retains constant images.
    assert not np.intersect1d(a['training_labeled'], a['validation_labeled']).size


@pytest.mark.parametrize('validation', [None, [0, 0], [-1], [10], [1.5]])
def test_pseudo_validation_cannot_be_inferred_or_invalid(validation):
    with pytest.raises(ValueError, match='validation positions'):
        select_fit_population('torch.p2.f0', *fixture(), pseudo_validation=validation)


@pytest.mark.parametrize('key', ['keras.r1.p5.f0', 'keras.r2.p0.f1', 'torch.p2.f1', 'unknown'])
def test_rejects_nonexistent_fit(key):
    with pytest.raises(ValueError):
        select_fit_population(key, *fixture())
