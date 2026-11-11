import numpy as np
import pytest
from sciona.atoms.causal_inference.feature_primitives.selected_features import FEATURE_ORDER,extract_causal_selected_features


def test_selected_features_pair_orientation_variable_lengths_and_no_mutation():
    rng=np.random.default_rng(8)
    pairs=[(rng.normal(size=n),rng.normal(size=n)) for n in [40,52]]
    before=[tuple(a.copy() for a in pair) for pair in pairs]
    types=[('Numerical','Numerical')]*2
    values=extract_causal_selected_features(pairs,types)
    reversed_values=extract_causal_selected_features([(y,x) for x,y in pairs],types)
    assert values.shape==(4,43) and len(set(FEATURE_ORDER))==43
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[::2],reversed_values[1::2])
    np.testing.assert_allclose(values[1::2],reversed_values[::2])
    for pair,copy in zip(pairs,before):
        for a,b in zip(pair,copy):np.testing.assert_array_equal(a,b)


def test_binary_entropy_and_moments_against_hand_calculation():
    x=np.tile([0.,0.,1.,1.],10);y=np.tile([0.,1.,0.,1.],10)
    values=extract_causal_selected_features([(x,y)],[('Binary','Binary')])[0]
    row=dict(zip(FEATURE_ORDER,values))
    entropy=np.log(2)+.7/80
    joint=np.log(4)+.7*3/80
    assert row['discrete_entropy_a']==pytest.approx(entropy)
    assert row['discrete_conditional_entropy_ab']==pytest.approx(joint-entropy)
    assert row['discrete_mi']==0
    assert row['moment21_ab']==pytest.approx(0)
    assert row['moment31_ab']==pytest.approx(0)
    assert row['log_sample_count']==pytest.approx(np.log(40))


@pytest.mark.parametrize('pair,types',[
    (([1,1,1],[1,2,3]),('Numerical','Numerical')),
    (([1,2,3],[1,2]),('Numerical','Numerical')),
    (([1,2,3],[1,2,3]),('Binary','Numerical')),
    (([1,2,3],[1,2,3]),('unknown','Numerical')),
])
def test_undefined_or_invalid_observations_rejected(pair,types):
    with pytest.raises(ValueError):extract_causal_selected_features([pair],[types])


@pytest.mark.asyncio
async def test_raw_pair_graph_trains_and_predicts_end_to_end(tmp_path,monkeypatch):
    from sciona.causal_prediction_execution import build_raw_pair_causal_training_prediction_graph
    from sciona.visualizer.runner import CDGExecutionSession
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    rng=np.random.default_rng(10)
    train=[(rng.normal(size=32).tolist(),rng.normal(size=32).tolist()) for _ in range(6)]
    test=[(rng.normal(size=36).tolist(),rng.normal(size=36).tolist()) for _ in range(2)]
    result=await CDGExecutionSession(None,'synthetic-raw-pairs','raw').execute({'training_pairs':train,'prediction_pairs':test,'training_types':[['Numerical','Numerical']]*6,'prediction_types':[['Numerical','Numerical']]*2,'pair_labels':[-1,0,1,-1,0,1],'weights':[.383,.370,.247],'n_estimators':3,'max_depth':2},cdg=build_raw_pair_causal_training_prediction_graph())
    assert result['status']=='completed' and len(result['trace'])==8
    scores=np.load(tmp_path/'raw'/'ensemble'/'out_causal_scores.npy')
    assert scores.shape==(4,) and np.isfinite(scores).all() and np.all(np.abs(scores)<=1)
    np.testing.assert_array_equal(scores[::2],-scores[1::2])
    np.testing.assert_array_equal(np.load(tmp_path/'raw'/'features'/'out_training_labels.npy'),[-1,1,0,0,1,-1,-1,1,0,0,1,-1])


def test_conditional_entropy_uses_discretized_response():
    from sciona.atoms.causal_inference.conditional_statistics.source_entropy import discretized_conditional_entropy_variance
    from sciona.atoms.causal_inference.feature_primitives.atoms import discretize_and_bin
    from collections import Counter
    x=np.repeat([0.,1.],40)
    y=np.concatenate([np.linspace(-.1,.1,40),np.linspace(-4.,4.,40)])
    yd=discretize_and_bin(y,'Numerical')
    entropies=[]
    for group in [yd[:40],yd[40:]]:
        counts=list(Counter(group).values())
        entropies.append(-sum((v/40)*np.log(v/40) for v in counts)+.7*(len(counts)-1)/80)
    expected=np.std(entropies)/np.log(len(set(yd)))
    assert expected>0
    assert discretized_conditional_entropy_variance(x,'Binary',y,'Numerical')==pytest.approx(expected)
