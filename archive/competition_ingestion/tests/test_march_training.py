import copy
import numpy as np
import pytest
from sciona.march_training import fit_predict
from scripts.march_synthetic import game,CONTROLS


def inputs():
    games=[]
    for season in range(1,5):
        games += [game(0,0,1,80,70,season),game(1,1,2,75,70,season),game(2,2,0,60,80,season)]
    def row(s,a,b,y):return dict(season=s,order=10+a,a=a,b=b,outcome=y)
    training=[row(1,0,1,1),row(2,1,2,0)]
    calibration=[row(3,0,1,1),row(3,1,2,1)]
    prediction=[dict(season=4,order=10,a=0,b=1),dict(season=4,order=10,a=1,b=0)]
    return games,training,calibration,prediction


def run(args):return fit_predict(*args,feature_controls=CONTROLS,regularization=1.,max_iterations=1000,tolerance=1e-8,seed=12)


def test_complete_training_calibration_and_swap_symmetry():
    r=run(inputs())
    assert r['training_rows']==r['calibration_rows']==4 and r['predicted_matchups']==2
    assert np.isfinite(r['probabilities']).all()
    assert sum(r['probabilities'])==pytest.approx(1.,abs=1e-15)
    assert np.array_equal(r['probabilities'],run(inputs())['probabilities'])


@pytest.mark.parametrize('case',['overlapping_seasons','future_regular_game','duplicate','unseen_team'])
def test_leakage_and_identity_boundaries(case):
    args=list(copy.deepcopy(inputs()))
    if case=='overlapping_seasons':args[2][0]['season']=2
    elif case=='future_regular_game':args[0].append(game(11,0,1,70,80,1))
    elif case=='duplicate':args[1][1]=copy.deepcopy(args[1][0])
    else:args[3][0]['b']=99
    with pytest.raises(ValueError):run(args)


def test_calibration_does_not_refit_base(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    original=LogisticRegression.fit;calls=[]
    def fit(self,x,y,*args,**kwargs):
        calls.append(len(x));return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(LogisticRegression,'fit',fit)
    run(inputs())
    assert calls==[4]
