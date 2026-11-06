"""Separate title/description realization of the original generic pair topology."""
from collections import Counter, defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from sciona.atoms.ml.text_nlp.atoms import levenshtein
from sciona.text_pair_complete_recipe import CompletePairFeatures, fit_feature_models, calibrate_and_predict
from sciona.text_pair_normalization_policy import NormalizationPolicy
from sciona.text_pair_training import _labels


def normalize_records(rows, policy):
    """Each row pairs two {title, description} records; neither field is merged."""
    if not isinstance(policy, NormalizationPolicy) or type(rows) is not list or not rows:
        raise ValueError('Explicit policy and nonempty record-pair list required')
    for row in rows:
        if type(row) not in (list, tuple) or len(row) != 2:
            raise ValueError('Two records per pair required')
        for record in row:
            if type(record) is not dict or set(record) != {'title', 'description'}:
                raise ValueError('Exact title and description fields required')
    titles = policy.normalize_pairs([[a['title'], b['title']] for a, b in rows])
    descriptions = policy.normalize_pairs([[a['description'], b['description']] for a, b in rows])
    return [((ta, da), (tb, db)) for (ta, tb), (da, db) in zip(titles, descriptions, strict=True)]


def normalize_populations(training, calibration, query, policy):
    populations = [normalize_records(rows, policy) for rows in (training, calibration, query)]
    keys = [{tuple(sorted(row)) for row in rows} for rows in populations]
    if any(len(keys[i]) != len(populations[i]) for i in (0, 1)):
        raise ValueError('Duplicate normalized record pair')
    if any(keys[a] & keys[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Overlapping normalized record populations')
    return tuple(populations)


class FieldPairFeatures:
    """Description lexical/LSA/BM25 features, title edit, full-record context.

    Preserve the thirteen-feature order. Column 4 is title edit similarity;
    columns 8:12 use (title,description) entity identities. Other columns use
    descriptions. Repeated descriptions with distinct titles remain distinct
    examples. Description vocabulary/projection/corpus statistics fit unique
    training descriptions, with no held-out fitting.
    """
    def fit(self, normalized, **controls):
        self.rows = list(normalized)
        keys = [tuple(sorted(row)) for row in normalized]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('Nonempty distinct normalized record pairs required')
        description_pairs = sorted({tuple(sorted((a[1], b[1]))) for a, b in normalized})
        self.descriptions = CompletePairFeatures().fit(description_pairs, **controls)
        self.pairs = Counter(keys)
        self.adjacency = defaultdict(Counter)
        for a, b in normalized:
            self.adjacency[a][b] += 1
            if a != b:
                self.adjacency[b][a] += 1
        return self

    def _features(self, rows, exclude_self):
        values = self.descriptions.transform([(a[1], b[1]) for a, b in rows])
        for i, (a, b) in enumerate(rows):
            values[i, 4] = 1 - levenshtein(a[0], b[0]) / max(len(a[0]), len(b[0]))
            left, right = self.adjacency.get(a, Counter()).copy(), self.adjacency.get(b, Counter()).copy()
            count = self.pairs[tuple(sorted((a, b)))]
            if exclude_self:
                left[b] -= 1
                right[a] -= 1
                count -= 1
            na = {n for n, c in left.items() if c > 0}
            nb = {n for n, c in right.items() if c > 0}
            da, db = sum(max(0, c) for c in left.values()), sum(max(0, c) for c in right.values())
            values[i, 8:12] = [min(da, db), max(da, db), len(na & nb) / len(na | nb) if na | nb else 1., count]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite field-specific features')
        return values

    def transform_training(self):
        return self._features(self.rows, True)

    def transform(self, rows):
        return self._features(rows, False)


def execute(training, training_labels, calibration, calibration_labels, query, *, policy,
            components, seed, iterations, oversamples, k1, b, trees, max_depth, min_leaf):
    """Full separate-field recipe with the shared ensemble and calibration policy."""
    train, cal, target = normalize_populations(training, calibration, query, policy)
    y, cy = _labels(training_labels, len(train)), _labels(calibration_labels, len(cal))
    with threadpool_limits(limits=1):
        features = FieldPairFeatures().fit(train, components=components, seed=seed,
                    iterations=iterations, oversamples=oversamples, k1=k1, b=b)
        fitted = fit_feature_models(features, y, seed=seed, trees=trees, max_depth=max_depth, min_leaf=min_leaf)
        return calibrate_and_predict(fitted, cal, cy, target)
