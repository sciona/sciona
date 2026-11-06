"""Pending full text-pair recipe; no catalog approval is implied by this module."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sciona.atoms.ml.calibration.empirical_f1 import (
    apply_column_thresholds, fit_empirical_f1_thresholds,
)
from sciona.atoms.ml.model_selection.materialized_prediction import average
from sciona.atoms.ml.sklearn.isotonic.atoms import (
    isotonic_regression_fit, isotonic_regression_transform,
)
from sciona.atoms.ml.sklearn.decomposition.sparse_projection import (
    fit_sparse_projection, transform_sparse_projection,
)
from sciona.text_pair_bm25 import PairBM25, _pairs
from sciona.text_pair_features import PairFeatures
from sciona.text_pair_normalization_policy import NormalizationPolicy
from sciona.text_pair_training import _labels


@dataclass(frozen=True)
class Projection:
    components_: np.ndarray
    singular_values_: np.ndarray

    def transform(self, matrix):
        return transform_sparse_projection(matrix, self.components_)


class CompletePairFeatures(PairFeatures):
    """Retain twelve lexical/LSA/context features and append symmetric BM25.

    Inputs must already have the caller's spelling/domain/expansion policy
    applied. Training-only unique documents fit TF-IDF, LSA and BM25. Existing
    current-edge exclusion is retained for training context features. The
    inherited numerical feature assembly is source-pinned at publication time.
    """
    def fit(self, pairs, *, components, seed, iterations, oversamples, k1, b):
        normalized = [tuple(pair) for pair in _pairs(pairs)]
        keys = [tuple(sorted(pair)) for pair in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate normalized training pair')
        docs = sorted({text for pair in normalized for text in pair})
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', lowercase=False)
        matrix = vectorizer.fit_transform(docs)
        fitted_components, singular = fit_sparse_projection(matrix, components, iterations, oversamples, seed)
        bm25 = PairBM25(normalized, k1=k1, b=b)
        self.vectorizer, self.svd, self.bm25 = vectorizer, Projection(fitted_components, singular), bm25
        self.pairs = Counter(keys)
        self.adjacency = defaultdict(Counter)
        for a, b in normalized:
            self.adjacency[a][b] += 1
            if a != b:
                self.adjacency[b][a] += 1
        self.fitted_pairs = normalized
        return self

    def _features(self, normalized, exclude_self):
        base = super()._features(normalized, exclude_self)
        return np.column_stack((base, self.bm25.transform(normalized)))

    def transform(self, pairs):
        # Deliberately avoid applying normalization a second time.
        return self._features(_pairs(pairs), False)


@dataclass(frozen=True)
class CompleteFit:
    features: CompletePairFeatures
    linear: object
    forest: object

    def scores(self, normalized_pairs):
        with threadpool_limits(limits=1):
            x = self.features.transform(normalized_pairs)
            values = average([self.linear.predict_proba(x)[:, 1], self.forest.predict_proba(x)[:, 1]])
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid ensemble scores')
        return values


def fit_normalized(training, labels, *, components, seed, iterations, oversamples,
                   k1, b, trees, max_depth, min_leaf):
    """Train only on normalized training rows; caller owns population separation."""
    normalized = _pairs(training)
    targets = _labels(labels, len(normalized))
    with threadpool_limits(limits=1):
        features = CompletePairFeatures().fit(normalized, components=components, seed=seed,
                    iterations=iterations, oversamples=oversamples, k1=k1, b=b)
    return fit_feature_models(features, targets, seed=seed, trees=trees, max_depth=max_depth, min_leaf=min_leaf)


def fit_feature_models(features, targets, *, seed, trees, max_depth, min_leaf):
    """Fit the declared ensemble on a fitted feature recipe's training rows."""
    if any(type(value) is not int or value < 1 for value in (trees, max_depth, min_leaf)):
        raise ValueError('Positive integer forest controls required')
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        x = features.transform_training()
        targets = _labels(targets, len(x))
        linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        forest = RandomForestClassifier(n_estimators=trees, max_depth=max_depth,
                    min_samples_leaf=min_leaf, random_state=seed, n_jobs=1)
        linear.fit(x, targets)
        forest.fit(x, targets)
    return CompleteFit(features, linear, forest)


def execute(training, training_labels, calibration, calibration_labels, query, *, policy, **controls):
    """Normalize, fit, scale held-out relevance scores and select an F1 cutoff.

    Returned scores retain raw ensemble values. Scaled scores use an increasing
    isotonic mapping fitted only on calibration labels, bounded to [0,1] with
    endpoint clipping. This is a calibration procedure, not an empirical
    probability-calibration or accuracy guarantee. Decision calibration on the
    scaled scores maximizes held-out empirical binary F1 (highest cutoff
    on a tie); calibration_f1 is an in-sample selection metric, not unbiased test
    accuracy. Output row order matches query input, including repeated rows.
    This explicit binary task realization requires source-scope review before
    promotion; it is not a generic competition submission-schema claim.
    """
    if not isinstance(policy, NormalizationPolicy):
        raise ValueError('Explicit normalization policy required')
    train, cal, target = policy.normalize_populations(training, calibration, query)
    cy = _labels(calibration_labels, len(cal))
    fitted = fit_normalized(train, training_labels, **controls)
    return calibrate_and_predict(fitted, cal, cy, target)


def calibrate_and_predict(fitted, cal, labels, target):
    """Shared held-out score-scaling/decision policy; caller enforces splits."""
    cy = _labels(labels, len(cal))
    raw_cal_scores = fitted.scores(cal)
    mapping = isotonic_regression_fit(raw_cal_scores, cy.astype(np.float64),
                increasing=True, y_min=0., y_max=1., out_of_bounds='clip')
    cal_scores = isotonic_regression_transform(raw_cal_scores, mapping).reshape(-1, 1)
    thresholds = fit_empirical_f1_thresholds(cal_scores, cy.reshape(-1, 1))
    selected = apply_column_thresholds(cal_scores, thresholds)[:, 0]
    tp = int(np.count_nonzero((selected == 1) & (cy == 1)))
    empirical_f1 = 2 * tp / (int(selected.sum()) + int(cy.sum()))
    query_scores = fitted.scores(target)
    scaled = isotonic_regression_transform(query_scores, mapping).reshape(-1, 1)
    matches = apply_column_thresholds(scaled, thresholds)
    return dict(scores=query_scores.tolist(), scaled_scores=scaled[:, 0].tolist(), matches=matches[:, 0].tolist(),
                threshold=float(thresholds[0]), calibration_f1=empirical_f1,
                metric='binary_f1', score_scaling='increasing_isotonic_clip', feature_count=13, query_rows=len(target))
