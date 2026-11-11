import pytest

from sciona.tgs_callbacks import ValidationControl


def control(**kwargs):
    return ValidationControl(**dict(learning_rate=.01, stop_patience=4, reduce_patience=2,
                                   factor=.5, minimum=.001, **kwargs))


def test_small_improvements_save_best_but_do_not_reset_plateau():
    state = control()
    assert state.observe(.5)['save_best']
    assert state.observe(.50004)['save_best']
    decision = state.observe(.50008)
    assert decision['save_best'] and decision['reduced']
    assert decision['learning_rate'] == .005
    assert not decision['stop']


def test_final_epoch_still_reduces_rate_and_stops():
    state = control()
    state.observe(.5)
    for _ in range(3):
        assert not state.observe(.4)['stop']
    decision = state.observe(.4)
    assert decision['stop'] and decision['reduced']
    assert decision['learning_rate'] == .0025
    with pytest.raises(ValueError):
        state.observe(.9)


def test_invalid_metric_does_not_change_state():
    state = control()
    before = vars(state).copy()
    with pytest.raises(ValueError):
        state.observe(float('nan'))
    assert vars(state) == before
