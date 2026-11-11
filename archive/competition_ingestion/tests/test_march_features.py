import numpy as np
import pytest
from sciona.march_features import season_features,matchup_features

from scripts.march_synthetic import game,CONTROLS


def test_efficiency_elo_and_matchup_orientation():
    r=season_features([game(0,0,1,80,70)],**CONTROLS)
    a,b=r[1]['teams'][0],r[1]['teams'][1]
    assert a[0]==80/65 and a[1]==70/65 and a[2]==10/65
    assert a[4]==1516 and b[4]==1484
    forward=matchup_features(r,1,0,1);reverse=matchup_features(r,1,1,0)
    assert np.array_equal(forward[:5],reverse[5:10])
    assert np.array_equal(forward[10:],-reverse[10:])


def test_pagerank_against_linear_system_and_order_invariance():
    games=[game(0,0,1,80,70),game(1,1,2,80,70),game(2,0,2,70,80),game(3,0,1,80,70)]
    result=season_features(games,**CONTROLS)
    reversed_result=season_features(games[::-1],**CONTROLS)
    # Directed loser->winner transition; repeated games retain their weights.
    adjacency=np.array([[0,0,1],[2,0,0],[0,1,0]],dtype=float)
    transition=adjacency/adjacency.sum(axis=1)[:,None]
    reference=np.linalg.solve(np.eye(3)-.85*transition.T,np.full(3,.15/3))
    assert np.allclose([result[1]['teams'][i][3] for i in range(3)],reference,atol=1e-12)
    for i in range(3):assert np.array_equal(result[1]['teams'][i],reversed_result[1]['teams'][i])
    assert sum(result[1]['teams'][i][4] for i in range(3))==pytest.approx(4500.)


def test_future_season_cannot_change_previous_features():
    first=[game(0,0,1,80,70)]
    before=season_features(first,**CONTROLS)
    after=season_features(first+[game(0,0,1,10,100,season=2)],**CONTROLS)
    for i in (0,1):assert np.array_equal(before[1]['teams'][i],after[1]['teams'][i])


@pytest.mark.parametrize('case',['duplicate','tie','negative','possessions','same_team','unknown_field'])
def test_invalid_games(case):
    g=game(0,0,1,80,70);games=[g]
    if case=='duplicate':games.append(g.copy())
    elif case=='tie':g['a_box']['points']=70
    elif case=='negative':g['b_box']['turnovers']=-1
    elif case=='possessions':g['b_box']['offensive_rebounds']=1000
    elif case=='same_team':g['b']=0
    else:g['extra']=1
    with pytest.raises(ValueError):season_features(games,**CONTROLS)
