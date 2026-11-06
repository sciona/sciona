"""Explicit PLAsTiCC lifecycle policy and reusable weighted fold execution.

Derived from Avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
All configured folds execute serially. Supplied functions are invocation-local
bindings; catalog qualification must authenticate them before publication.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .plasticc_defaults import DEFAULTS
from .plasticc_features import PlasticcFeaturizer
from .plasticc_runtime import CLASSES, Completed
from .plasticc_training import LightGBMClassifier
from .plasticc_weighting import evaluate_weights_redshift, weighted_multi_logloss


@dataclass(repr=False)
class FoldResult:
    models: list
    predictions: pd.DataFrame
    importances: pd.DataFrame


def fit_folds(features, labels, weights, folds, parameters, fit):
    """Fit every supplied fold and retain out-of-fold scores and importances.

    Generic indexed tabular inputs; fit is an explicit six-input weighted
    train/validation callable returning a sklearn-style classifier. Parameters
    must explicitly select one worker, preserving the shared CPU budget.
    """
    if not isinstance(features, pd.DataFrame) or features.empty or not features.index.is_unique:
        raise ValueError('nonempty uniquely indexed features required')
    for value in (labels, weights, folds):
        if not isinstance(value, pd.Series) or not value.index.equals(features.index):
            raise ValueError('labels, weights and folds must align exactly')
    if labels.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('nonnull labels and finite positive weights required')
    if folds.dtype.kind not in 'iu' or set(folds) != set(range(int(folds.max()) + 1)) or folds.max() < 1:
        raise ValueError('at least two contiguous nonempty folds required')
    if type(parameters) is not dict or type(parameters.get('n_jobs')) is not int or parameters['n_jobs'] != 1:
        raise ValueError('serial fold execution requires explicit n_jobs=1')
    classes = np.unique(labels)
    for fold in range(int(folds.max()) + 1):
        if not np.array_equal(np.unique(labels[folds != fold]), classes):
            raise ValueError('each training partition must contain every class')
    predictions = pd.DataFrame(-np.ones((len(labels), len(classes))), index=labels.index, columns=classes)
    models = []; importances = pd.DataFrame()
    for fold in range(int(folds.max()) + 1):
        train = folds != fold; validation = folds == fold
        model = fit(features[train], labels[train], weights[train],
                    features[validation], labels[validation], weights[validation], **parameters)
        if not np.array_equal(model.classes_, classes):
            raise ValueError('fitted class ordering differs from output columns')
        predictions[validation] = model.predict_proba(features[validation], num_iteration=model.best_iteration_)
        importance = pd.DataFrame({'feature': features.columns, 'gain': model.feature_importances_, 'fold': fold + 1})
        importances = pd.concat([importances, importance], axis=0, sort=False)
        models.append(model)
    return FoldResult(models, predictions, importances)


def augment(prepared):
    return prepared.augmentor.augment_dataset('runtime-augmented', prepared.training, prepared.num_augments)


def extract_features(population, featurizer):
    population.extract_raw_features(featurizer)
    return population.select_features(featurizer)


def assign_folds(population, prepared, operation):
    metadata = population.metadata
    references = metadata.get('reference_object_id', pd.Series(None, index=metadata.index, dtype=object))
    return operation(metadata['class'], references, prepared.num_folds, prepared.fold_seed)


def training_weights(population, prepared, balance):
    class_weights = None if prepared.class_weighting == 'flat' else {c: 2 if c in (15, 64) else 1 for c in CLASSES}
    if prepared.weighting == 'flat':
        weights = balance(population.metadata['class'], class_weights)
    else:
        weights = evaluate_weights_redshift(population, class_weights, settings=DEFAULTS)
    return weights, class_weights


def complete(prepared, population, featurizer, result, weights, class_weights):
    """Retain all source loss diagnostics in memory rather than printing them."""
    labels = population.metadata['class']
    diagnostics = {}
    masks = {'all': np.ones(len(labels), dtype=bool)}
    if 'reference_object_id' in population.metadata:
        masks['original'] = population.metadata['reference_object_id'].isna()
    for name, mask in masks.items():
        diagnostics[name] = (
            weighted_multi_logloss(labels[mask], result.predictions[mask], weights[mask], class_weights),
            weighted_multi_logloss(labels[mask], result.predictions[mask], class_weights=class_weights))
    classifier = LightGBMClassifier('runtime-classifier', featurizer, class_weights=class_weights)
    classifier.classifiers = result.models
    classifier.train_predictions = result.predictions
    classifier.importances = result.importances
    classifier.train_classes = labels
    classifier.loss_diagnostics = diagnostics
    return Completed(classifier, prepared.prediction)


def train(prepared, fold_operation, balance_operation, fit_operation):
    """Compose explicit stages; default training remains serial without truncation."""
    population = augment(prepared)
    featurizer = PlasticcFeaturizer()
    features = extract_features(population, featurizer)
    folds = assign_folds(population, prepared, fold_operation)
    weights, class_weights = training_weights(population, prepared, balance_operation)
    parameters = dict(prepared.training_parameters)
    parameters.setdefault('n_jobs', 1)
    result = fit_folds(features, population.metadata['class'], weights, folds, parameters, fit_operation)
    return complete(prepared, population, featurizer, result, weights, class_weights)
