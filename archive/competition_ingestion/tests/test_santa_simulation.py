import copy
import numpy as np
import pytest
from sciona.santa_simulation import BanditEnvironment, simulate


class FixedAgent:
    def __init__(self): self.seen=[]
    def predict(self, history, *, seed):
        assert not hasattr(history, 'thresholds')
        self.seen.append((history.step, int(history.own.sum()), int(history.opponent.sum())))
        return dict(action=4)


def test_shared_action_rewards_before_double_decay():
    env=BanditEnvironment([100]*100)
    rewards, expected=env.step([4,4],[99,100])
    assert rewards==[1,0] and expected==[100/101]*2
    assert env.thresholds[4]==pytest.approx(100*.97**2)
    assert env.thresholds[3]==100
    rewards,_=env.step([4,4],[94,95])
    assert rewards==[1,0]


def test_fractional_threshold_probability_by_exhaustive_draws():
    rewards=[]
    for draw in range(101):
        env=BanditEnvironment([51]*100)
        env.step([0,1],[100,100])
        values, expected=env.step([0,1],[draw,draw])
        rewards.append(values[0])
        assert expected[0]==50/101
    assert sum(rewards)==50


def test_full_horizon_and_deterministic_hidden_state():
    thresholds=list(range(1,101));before=copy.deepcopy(thresholds)
    agent=FixedAgent()
    a=simulate(agent,thresholds,seed=13)
    b=simulate(FixedAgent(),thresholds,seed=13)
    assert a==b and a['rounds']==a['agent_observations']==2000
    assert a['total_selections']==4000
    assert agent.seen==[(i,i,i) for i in range(2000)]
    assert thresholds==before
    assert all(0 <= v <= 2000 for v in a['rewards']+a['expected_rewards'])


def test_fitted_models_execute_full_game_without_refit():
    from tests.test_santa_boosting import fit
    models=fit()
    before=[m.model_to_string() for m in (models.normal,models.transformed)]
    result=simulate(models,[50]*100,seed=19)
    assert result['agent_observations']==2000 and result['total_selections']==4000
    assert all(np.isfinite(result['expected_rewards']))
    assert [m.model_to_string() for m in (models.normal,models.transformed)]==before


@pytest.mark.parametrize('actions,draws',[([True,0],[0,0]),([100,0],[0,0]),([0,0],[-1,0]),([0,0],[0,101])])
def test_invalid_round_is_atomic(actions,draws):
    env=BanditEnvironment([50]*100)
    before=env.thresholds.copy()
    with pytest.raises(ValueError):env.step(actions,draws)
    np.testing.assert_array_equal(env.thresholds,before)
    assert env.rounds==0
