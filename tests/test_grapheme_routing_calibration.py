import numpy as np
import pytest

from sciona.bengali_joint_labels import encode_components
from sciona.bengali_routing import route_predictions
from sciona.component_recall import component_recall_utilities
from sciona.grapheme_routing_calibration import calibrate_grapheme_router, submission_components


def scalar_score(truth, prediction, weights, importance):
    scores = []
    for column in range(truth.shape[1]):
        labels = sorted({int(truth[row, column]) for row in range(len(truth)) if weights[row] > 0})
        recalls = []
        for label in labels:
            denominator = sum(weights[row] for row in range(len(truth)) if truth[row, column] == label)
            numerator = sum(weights[row] for row in range(len(truth))
                            if truth[row, column] == label and prediction[row, column] == label)
            recalls.append(numerator/denominator)
        scores.append(sum(recalls)/len(recalls))
    return sum(score*weight for score, weight in zip(scores, importance))/sum(importance)


def test_general_component_utilities_match_scalar_weighted_macro_recall():
    rng = np.random.default_rng(317)
    for components in (1, 3, 5):
        truth = rng.integers(0, 4, (21, components))
        prediction = rng.integers(0, 6, (21, components))
        weights = rng.integers(0, 9, 21)
        importance = np.arange(1, components+1)
        utility = component_recall_utilities(truth, prediction, sample_weight=weights,
                                            component_weight=importance)
        assert utility.mean() == pytest.approx(scalar_score(truth, prediction, weights, importance))
        assert (utility[weights == 0] == 0).all()


def test_grapheme_grid_matches_full_routing_then_independent_metric():
    rng = np.random.default_rng(318)
    truth = rng.integers(0, 3, (25, 3))
    seen = encode_components(rng.integers(0, 3, (25, 3)))
    novel_components = rng.integers(0, 3, (25, 3))
    novel_components[::4, 2] = 7
    unseen = encode_components(novel_components)
    membership = np.zeros(14784, dtype=bool)
    membership[seen[::3]] = True
    confidence = np.linspace(0, 1, 25)
    weights = rng.integers(0, 7, 25)
    grid = [0., .2, .5, .8, 1.]
    result = calibrate_grapheme_router(confidence, seen, unseen, truth,
        seen_class_mask=membership, sample_weight=weights, thresholds=grid)
    scores = []
    for threshold in grid:
        prediction = route_predictions(confidence, seen, unseen, seen_class_mask=membership, threshold=threshold)
        # Independent source submission correction, after routing joint classes.
        decoded = np.array([[value//88, (value//8)%11, 2 if value%8 == 7 else value%8] for value in prediction])
        scores.append(scalar_score(truth, decoded, weights, [2, 1, 1]))
    np.testing.assert_allclose(result['candidate_utilities'], scores, rtol=1e-14, atol=1e-14)
    assert result['threshold'] == grid[int(np.argmax(scores))]
    np.testing.assert_array_equal(submission_components([7]), [[0, 0, 2]])


@pytest.mark.parametrize('fault', ['negative_weight', 'zero_weight', 'nonfinite_weight', 'misalignment', 'float_labels', 'component_weight'])
def test_bad_metric_contracts_rejected(fault):
    truth = np.array([[0, 0], [1, 1]])
    prediction = truth.copy()
    options = dict(sample_weight=[1., 1.], component_weight=[1., 1.])
    if fault == 'negative_weight': options['sample_weight'] = [-1., 1.]
    elif fault == 'zero_weight': options['sample_weight'] = [0., 0.]
    elif fault == 'nonfinite_weight': options['sample_weight'] = [np.inf, 1.]
    elif fault == 'misalignment': prediction = prediction[:1]
    elif fault == 'float_labels': truth = truth.astype(float)
    else: options['component_weight'] = [1.]
    with pytest.raises(ValueError): component_recall_utilities(truth, prediction, **options)


def test_internal_consonant_state_is_not_accepted_as_canonical_truth():
    with pytest.raises(ValueError, match='canonical'):
        calibrate_grapheme_router([.5], [0], [7], [[0, 0, 7]],
            seen_class_mask=np.zeros(14784, dtype=bool), sample_weight=[1.], thresholds=[0., 1.])


def test_small_positive_weight_class_retains_its_macro_share():
    utilities = component_recall_utilities([[0], [1]], [[0], [0]],
        sample_weight=[1e308, 1e-308], component_weight=[1.])
    assert utilities.mean() == .5
