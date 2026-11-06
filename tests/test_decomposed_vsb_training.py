"""Full native repeated-fold parity and rejection before partial training."""
from unittest.mock import Mock
import numpy as np
import pytest
from sciona.vsb_training import fit
from sciona.vsb_threshold import repeated_folds,threshold
from sciona.decomposed_vsb_training import recipe_inputs,fit_fold,ensemble,threshold_inputs,result


def fixture():
    rng=np.random.default_rng(98);x=rng.normal(size=(200,9));labels=np.zeros((200,3),int)
    labels[x[:,0]>0,0]=1;x[0,2]=np.nan
    return x,labels,rng.normal(size=(8,9))


@pytest.mark.parametrize('repetitions,rounds,patience',[(25,3,2),(2,30,5),pytest.param(25,10000,100,id='full_source_limits')])
def test_complete_native_training_and_threshold_parity(repetitions,rounds,patience):
    x,labels,q=fixture()
    if rounds==10000:
        # Independent synthetic labels exercise the actual default ceiling and
        # stopping patience without claiming predictive skill.
        labels[:]=0;labels[100:,0]=1
    saved=[v.copy() for v in (x,labels,q)]
    expected=fit(x,labels,q,repetitions=repetitions,rounds=rounds,patience=patience)
    ordered,y,query,signal_labels=recipe_inputs(x,labels,q)
    plans=repeated_folds(y,seed=123948,repetitions=repetitions);calls=[]
    def tracked(*args):
        calls.append(tuple(a.copy() for a in args[3:6]));return fit_fold(*args)
    trained=ensemble(ordered,y,query,plans,repetitions,23974,rounds,patience,tracked)
    actual=result(trained,threshold(*threshold_inputs(signal_labels,trained['training_probabilities'])))
    assert len(calls)==5*repetitions
    for called,planned in zip(calls,plans):
        for a,b in zip(called,planned):np.testing.assert_array_equal(a,b)
    assert actual.keys()==expected.keys()
    for key in expected:
        if isinstance(expected[key],np.ndarray):np.testing.assert_array_equal(actual[key],expected[key])
        else:assert actual[key]==expected[key]
    for a,b in zip((x,labels,q),saved):np.testing.assert_array_equal(a,b)


def test_fold_accepts_domain_independent_feature_width():
    x,labels,q=fixture();y=labels.any(axis=1).astype(int)
    train,val,test=repeated_folds(y,seed=123948,repetitions=1)[0]
    outputs=fit_fold(x[:,:4],y,q[:,:4],train,val,test,3,4,2)
    for scores,size in zip(outputs[:4],(len(train),len(val),len(test),len(q))):
        assert scores.shape==(size,) and np.isfinite(scores).all()
    assert 1<=outputs[4]<=4


@pytest.mark.parametrize('fault',['missing_fold','duplicate_index','missing_population','coverage','seed','rounds','patience','callable'])
def test_bad_plan_or_controls_fail_before_any_fit(fault):
    x,labels,q=fixture();x,y,q,_=recipe_inputs(x,labels,q)
    plans=repeated_folds(y,seed=123948,repetitions=2);callback=Mock(side_effect=AssertionError('Must not fit'))
    seed,rounds,patience=23974,3,2
    if fault=='missing_fold':plans.pop()
    elif fault=='duplicate_index':plans[0][0][0]=plans[0][0][1]
    elif fault=='missing_population':plans[0]=(plans[0][0][1:],plans[0][1],plans[0][2])
    elif fault=='coverage':plans[-1]=plans[0]
    elif fault=='seed':seed=True
    elif fault=='rounds':rounds=0
    elif fault=='patience':patience=False
    function=None if fault=='callable' else callback
    with pytest.raises(ValueError):ensemble(x,y,q,plans,2,seed,rounds,patience,function)
    callback.assert_not_called()


def test_threshold_fit_and_final_decision_keep_distinct_comparisons():
    labels=np.array([[0,1,0],[1,0,0]],dtype=int);scores=np.array([.2,.8])
    y,p=threshold_inputs(labels,scores)
    np.testing.assert_array_equal(y,labels.ravel());np.testing.assert_array_equal(p,[.2,.2,.2,.8,.8,.8])
    finalized=result({'probabilities':np.array([.2,.5,.8])},{'threshold':.5,'mcc':0.})
    np.testing.assert_array_equal(finalized['signal_decisions'],[[0,0,0],[0,0,0],[1,1,1]])
