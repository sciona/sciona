"""Reusable class balancing and reference-preserving stratified folds.

Adapted from Avocado, Copyright (c) 2019 Kyle Boone.
MIT: docs/licenses/Avocado-MIT.txt; source commit
cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Inputs are aligned Series; a null reference denotes an original observation.
No astronomy fields, dataset container, or implicit random seed is required.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _labels(labels):
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError('labels must be a nonempty Series')
    if not labels.index.is_unique or labels.index.hasnans or labels.isna().any():
        raise ValueError('labels require unique nonnull indices and nonnull classes')


def balanced_weights(labels, class_weights):
    """Return N/count(class) times the explicit positive class weight.

    Pass None for equal class weights. Preserve source arithmetic and index.
    """
    _labels(labels)
    counts = labels.value_counts()
    weights = {}
    for name, count in counts.items():
        value = 1 if class_weights is None else class_weights[name]
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError('class weights must be finite and positive')
        weights[name] = value * len(labels) / count
    return labels.map(weights)


def reference_stratified_folds(labels, references, num_folds, seed):
    """Stratify originals and assign each derivative its original's fold.

    References must point directly to originals of the same class. Chains,
    missing references and conflicting labels are rejected before splitting.
    Ordering and sklearn's shuffled StratifiedKFold semantics match Avocado.
    """
    _labels(labels)
    if not isinstance(references, pd.Series) or not references.index.equals(labels.index):
        raise ValueError('references must be an aligned Series')
    if type(num_folds) is not int or num_folds < 2:
        raise ValueError('num_folds must be an integer of at least two')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    original = references.isna()
    original_labels = labels[original]
    if original_labels.empty or (original_labels.value_counts() < num_folds).any():
        raise ValueError('each original class must cover every fold')
    derived = references[~original]
    if not derived.isin(original_labels.index).all():
        raise ValueError('references must point directly to originals')
    if not np.array_equal(derived.map(original_labels).to_numpy(), labels[~original].to_numpy()):
        raise ValueError('derived labels must match their original')
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    mapping = {}
    for fold, (_, validation) in enumerate(splitter.split(original_labels, original_labels)):
        for key in original_labels.index[validation]:
            mapping[key] = fold
    result = references.map(mapping)
    result[original] = labels.index.to_series().map(mapping)
    return result.astype(int)
