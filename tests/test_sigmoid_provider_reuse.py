"""Existing sigmoid providers: weighted fits, stable prediction and input contracts."""
import numpy as np
import pytest
from icontract.errors import ViolationError
from scipy.special import expit
from sklearn.calibration import _SigmoidCalibration

from sciona.atoms.ml.sklearn.calibration.atoms import (
    sigmoid_calibration_fit, sigmoid_calibration_predict,
)
from sciona.atoms.ml.sklearn.calibration.state_models import SigmoidCalibrationState


@pytest.mark.parametrize('kind', ['ordinary', 'constant', 'large', 'imbalanced'])
@pytest.mark.parametrize('weighted', [False, True])
def test_existing_fit_matches_estimator_and_preserves_inputs(kind, weighted):
    rng = np.random.default_rng(105)
    scores = rng.normal(size=40)
    labels = (np.arange(40) % 2).astype(np.float64)
    if kind == 'constant': scores[:] = 0
    if kind == 'large': scores *= 1e6
    if kind == 'imbalanced': labels[:] = 0; labels[-1] = 1
    weight = rng.uniform(.1, 3, size=40) if weighted else None
    saved = [x.copy() if x is not None else None for x in (scores, labels, weight)]
    state = sigmoid_calibration_fit(scores, labels, sample_weight=weight)
    oracle = _SigmoidCalibration().fit(scores, labels, sample_weight=weight)
    assert (state.a, state.b) == (oracle.a_, oracle.b_)
    query = np.array([-1e6, -1., 0., 1., 1e6])
    actual = sigmoid_calibration_predict(query, state)
    assert actual.dtype == np.float64
    assert actual.tobytes() == oracle.predict(query).tobytes()
    for actual_input, before in zip((scores, labels, weight), saved):
        if before is not None: assert np.array_equal(actual_input, before)


@pytest.mark.parametrize('a,b', [(1., 0.), (-1., 3.), (0., -2.)])
def test_prediction_has_explicit_stable_sigmoid_meaning(a, b):
    scores = np.array([-1e200, -10., 0., 10., 1e200])
    expected = expit(-(a*scores+b))
    actual = sigmoid_calibration_predict(scores, SigmoidCalibrationState(a, b))
    assert actual.tobytes() == expected.tobytes()
    assert np.isfinite(actual).all()


@pytest.mark.parametrize('fault', ['rank_scores', 'rank_labels', 'length'])
def test_fit_rejects_invalid_shapes(fault):
    scores = np.array([-2., -1., 1., 2.]); labels = np.array([0., 0., 1., 1.])
    if fault == 'rank_scores': scores = scores[:, None]
    if fault == 'rank_labels': labels = labels[:, None]
    if fault == 'length': labels = labels[:-1]
    with pytest.raises(ViolationError): sigmoid_calibration_fit(scores, labels)


@pytest.mark.parametrize('fault', ['rank', 'slope', 'intercept', 'nan_score'])
def test_predict_rejects_invalid_shapes_or_nonfinite_result(fault):
    scores = np.array([-1., 1.]); state = SigmoidCalibrationState(1., 0.)
    if fault == 'rank': scores = scores[:, None]
    if fault == 'slope': state = SigmoidCalibrationState(np.inf, 0.)
    if fault == 'intercept': state = SigmoidCalibrationState(1., np.nan)
    if fault == 'nan_score': scores[0] = np.nan
    with pytest.raises(ViolationError): sigmoid_calibration_predict(scores, state)
