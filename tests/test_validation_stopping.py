import pytest
from sciona.validation_stopping import ValidationStopping


@pytest.mark.parametrize('mode,sign', [('min', 1), ('max', -1)])
def test_threshold_patience_reset_and_earliest_tie(mode, sign):
    policy = ValidationStopping(patience=2, min_delta=.25, mode=mode)
    original = policy
    for value, best, stale in [(4., 0, 0), (3.75, 0, 1), (3.5, 2, 0), (3.5, 2, 1), (3.6, 2, 2)]:
        policy = policy.observe(sign * value)
        assert (policy.best_index, policy.stale) == (best, stale)
    assert original.evaluations == 0 and original.best is None
    assert policy.stopped and policy.evaluations == 5
    with pytest.raises(ValueError):
        policy.observe(sign * 1.)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -float('inf'), '1'])
def test_invalid_metric_does_not_advance(value):
    policy = ValidationStopping(1)
    with pytest.raises(ValueError):
        policy.observe(value)
    assert policy.evaluations == 0


@pytest.mark.parametrize('kwargs', [dict(patience=0), dict(patience=True), dict(patience=1, min_delta=-1),
    dict(patience=1, min_delta=float('nan')), dict(patience=1, mode='other')])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        ValidationStopping(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(best=1.), dict(evaluations=True), dict(stale=1),
    dict(evaluations=1, best=1., best_index=2), dict(evaluations=1, best=float('nan'), best_index=0),
    dict(min_delta=10**1000)])
def test_inconsistent_state_and_unrepresentable_threshold(kwargs):
    with pytest.raises(ValueError):
        ValidationStopping(patience=2, **kwargs)
