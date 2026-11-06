"""Reusable ensemble inference and explicit PLAsTiCC output policy.

Derived from Avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
The source's unshifted exponential normalization is intentional. Its qualified
domain excludes overflow/underflow that yields nonfinite probabilities.
"""
import numpy as np
import pandas as pd
from .plasticc_predictions import create_kaggle_predictions
from .decomposed_plasticc_training import extract_features


def normalize_scores(scores):
    """Normalize each row with the exact retained exp/sum arithmetic."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not all(scores.shape) or scores.dtype.kind not in 'iuf' or not np.isfinite(scores).all():
        raise ValueError('nonempty finite real score matrix required')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        exponentials = np.exp(scores)
        result = exponentials / np.sum(exponentials, axis=1)[:, None]
    if not np.isfinite(result).all():
        raise ValueError('scores exceed the retained normalization domain')
    return result


def predict_scores(model, features, classes):
    """Predict one model at its selected iteration with explicit class order."""
    if not np.array_equal(model.classes_, classes):
        raise ValueError('model class ordering must match output columns')
    result = np.asarray(model.predict_proba(features, raw_score=True, num_iteration=model.best_iteration_))
    if result.shape != (len(features), len(classes)):
        raise ValueError('model returned an incompatible score matrix')
    return result


def average_folds(models, features, classes, score_operation, normalize_operation):
    """Average every fold in source order without modifying model outputs."""
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError('at least one model required')
    if not isinstance(features, pd.DataFrame) or features.empty or not features.index.is_unique:
        raise ValueError('nonempty uniquely indexed features required')
    classes = pd.Index(classes)
    if classes.empty or not classes.is_unique or classes.hasnans:
        raise ValueError('unique nonnull class labels required')
    total = 0
    for model in models:
        scores = score_operation(model, features, classes)
        probabilities = np.asarray(normalize_operation(scores))
        if probabilities.shape != (len(features), len(classes)) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError('fold returned invalid probability mass')
        if isinstance(total, int):
            total = probabilities.copy()
        else:
            total += probabilities
    total /= len(models)
    return pd.DataFrame(total, index=features.index, columns=classes)


def task_policy(population, probabilities):
    """Apply retained galactic masks and task-tuned unknown-class formula.

    This remains astronomy policy, not a reusable novelty detector.
    """
    if not probabilities.index.equals(population.metadata.index):
        raise ValueError('probabilities and population must align exactly')
    return create_kaggle_predictions(population, probabilities)


def encode(final):
    values = final.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError('prediction produced invalid probability mass')
    return {'version': 1, 'object_ids': list(final.index), 'classes': [int(c) for c in final.columns],
            'probabilities': values.tolist()}


def predict(completed, score_operation, normalize_operation):
    classifier = completed.classifier
    population = completed.prediction
    features = extract_features(population, classifier.featurizer)
    probabilities = average_folds(classifier.classifiers, features, classifier.train_predictions.columns,
                                  score_operation, normalize_operation)
    population.predictions = probabilities
    population.classifier = classifier
    return encode(task_policy(population, probabilities))
