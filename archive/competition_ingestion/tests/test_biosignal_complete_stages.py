from copy import deepcopy
import json
import numpy as np
import pytest
from scripts.biosignal_sequence_synthetic import payload
from sciona import biosignal_complete_stages as stages
from sciona import biosignal_sequence_contract as reference


def case(change):
    p = payload()
    p['controls']['epochs'] = 3
    if change == 'no_mask':
        p['controls'].update(max_frequency=0, max_time=0)
    if change == 'single_batch':
        p['controls']['batch_size'] = 100
    return p


@pytest.mark.parametrize('change', ['baseline', 'no_mask', 'single_batch'])
def test_full_lifecycle_parity(change):
    p = case(change)
    original = deepcopy(p)
    result = stages.predict(stages.calibrate(stages.fit(stages.prepare(p))))
    expected = reference.execute(reference.prepare(p))
    assert result.keys() == expected.keys()
    for key in result:
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)
    assert p == original


def test_generic_threshold_matches_independent_reference():
    rng = np.random.default_rng(17)
    for _ in range(40):
        scores = rng.integers(0, 6, size=12).astype(np.float64) / 5
        labels = rng.integers(0, 2, size=12, dtype=np.int64)
        labels[:2] = [0, 1]
        threshold = stages.fit_empirical_f1_thresholds(scores[:, None], labels[:, None])[0]
        assert threshold == reference.threshold(scores, labels)
        result = stages.apply_column_thresholds(scores[:, None], np.asarray([threshold]))[:, 0]
        np.testing.assert_array_equal(result, scores >= reference.threshold(scores, labels))


def test_calibration_and_prediction_do_not_refit(monkeypatch):
    fitted = stages.fit(stages.prepare(case('baseline')))
    weights = {name: value.clone() for name, value in fitted.model.network.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Refitting forbidden')
    monkeypatch.setattr(stages, 'train', forbidden)
    first = stages.predict(stages.calibrate(fitted))
    second = stages.predict(stages.calibrate(fitted))
    assert first == second
    for name, value in fitted.model.network.state_dict().items():
        assert value.equal(weights[name])


def test_detached_preparation_and_subject_revalidation():
    p = case('baseline')
    prepared = stages.prepare(p)
    p['training']['labels'][0] = 1
    assert json.loads(prepared.configuration)['training']['labels'][0] == 0
    invalid = json.loads(prepared.configuration)
    invalid['query']['subjects'][0] = invalid['training']['subjects'][0]
    with pytest.raises(ValueError, match='overlap'):
        stages.fit(stages.Prepared(json.dumps(invalid)))
