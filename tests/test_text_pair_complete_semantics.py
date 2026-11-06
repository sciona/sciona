"""Synthetic population isolation and independent calibration-policy checks."""
import copy

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from sciona.text_pair_complete_stages import (
    prepare_fields, normalize_fields, fit_features, fit_models, predict, format_rows,
)
from tests.test_text_pair_complete_graph import field_payload


def models(payload):
    return fit_models(fit_features(normalize_fields(prepare_fields(payload))))


@pytest.mark.parametrize('change', ['calibration_labels', 'query_text'])
def test_heldout_changes_cannot_change_fitted_training_features_or_models(change):
    p = field_payload()
    baseline = models(p)
    altered = copy.deepcopy(p)
    if change == 'calibration_labels':
        altered['calibration_labels'] = [1 - x for x in altered['calibration_labels']]
    else:
        altered['query_pairs'].append([
            {'title': 'unseen nebula', 'description': 'unknown azure quasar'},
            {'title': 'remote comet', 'description': 'distant lunar cloud'}])
    other = models(altered)
    a, b = baseline.fitted, other.fitted
    assert a.features.descriptions.vectorizer.vocabulary_ == b.features.descriptions.vectorizer.vocabulary_
    np.testing.assert_array_equal(a.features.descriptions.svd.components_, b.features.descriptions.svd.components_)
    np.testing.assert_array_equal(a.features.transform_training(), b.features.transform_training())
    np.testing.assert_array_equal(a.scores(baseline.populations.training), b.scores(baseline.populations.training))


def test_calibration_matches_independent_isotonic_and_exhaustive_f1_selection():
    p = field_payload()
    p['query_pairs'].append(copy.deepcopy(p['query_pairs'][0]))
    state = models(p)
    populations = state.populations
    raw_cal = state.fitted.scores(populations.calibration)
    y = np.asarray(p['calibration_labels'])
    reference = IsotonicRegression(increasing=True, y_min=0., y_max=1., out_of_bounds='clip')
    scaled_cal = reference.fit_transform(raw_cal, y)
    candidates = []
    for threshold in np.unique(scaled_cal):
        selected = scaled_cal >= threshold
        f1 = 2 * np.count_nonzero(selected & (y == 1)) / (selected.sum() + y.sum())
        candidates.append((float(f1), float(threshold)))
    f1, threshold = max(candidates)
    raw_query = state.fitted.scores(populations.query)
    expected_query = reference.predict(raw_query)
    before = state.fitted.features.transform_training().copy()
    result = format_rows(predict(state))
    np.testing.assert_array_equal(state.fitted.features.transform_training(), before)
    np.testing.assert_allclose([r['scaled_score'] for r in result['rows']], expected_query, rtol=0, atol=1e-15)
    assert result['threshold'] == threshold
    assert result['calibration_selection_f1'] == f1
    assert [r['relevant'] for r in result['rows']] == (expected_query >= threshold).astype(int).tolist()
    assert result['rows'][0]['scaled_score'] == result['rows'][-1]['scaled_score']
    assert [r['row_index'] for r in result['rows']] == list(range(len(populations.query)))
