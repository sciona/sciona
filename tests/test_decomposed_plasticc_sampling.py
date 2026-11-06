"""Synthetic cross-domain parity and leakage rejection for sampling atoms."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sciona.decomposed_plasticc_sampling import balanced_weights, reference_stratified_folds
from sciona.plasticc_weighting import evaluate_weights_flat, label_folds


def population(seed):
    labels = pd.Series(np.repeat(['category-a', 'category-b', 'category-c'], 6),
                       index=[f'synthetic-{i}' for i in range(18)])
    refs = pd.Series(None, index=labels.index, dtype=object)
    for i in (0, 3, 8, 12, 17):
        key = f'derived-{i}'
        labels.loc[key] = labels.iloc[i]
        refs.loc[key] = f'synthetic-{i}'
    order = np.random.RandomState(seed).permutation(len(labels))
    return labels.iloc[order], refs.iloc[order]


@pytest.mark.parametrize('seed', [0, 19, 102])
@pytest.mark.parametrize('folds', [2, 3, 6])
def test_reference_folds_exact_source_parity(seed, folds):
    labels, refs = population(seed)
    dataset = SimpleNamespace(metadata=pd.DataFrame({'class': labels, 'reference_object_id': refs}))
    expected = label_folds(dataset, folds, seed, settings={})
    actual = reference_stratified_folds(labels, refs, folds, seed)
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    for key, reference in refs.dropna().items():
        assert actual[key] == actual[reference]
    assert set(actual) == set(range(folds))


@pytest.mark.parametrize('weighted', [False, True])
def test_balancing_exact_source_parity(weighted):
    labels, _ = population(19)
    weights = dict(zip(sorted(labels.unique()), [0.5, 2, 3])) if weighted else None
    expected = evaluate_weights_flat(SimpleNamespace(metadata=pd.DataFrame({'class': labels})), weights)
    pd.testing.assert_series_equal(balanced_weights(labels, weights), expected, check_names=False)


@pytest.mark.parametrize('fault', ['missing', 'chain', 'class', 'alignment', 'count', 'seed'])
def test_invalid_reference_contract(fault):
    labels, refs = population(0)
    seed = 0; folds = 3
    if fault == 'missing': refs.loc['derived-0'] = 'absent'
    if fault == 'chain': refs.loc['derived-0'] = 'derived-3'
    if fault == 'class': labels.loc['derived-0'] = 'category-c'
    if fault == 'alignment': refs = refs.iloc[::-1]
    if fault == 'count': folds = 7
    if fault == 'seed': seed = -1
    with pytest.raises(ValueError):
        reference_stratified_folds(labels, refs, folds, seed)
