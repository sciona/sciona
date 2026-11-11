"""Complete reference parity, calibration isolation and reusable matrix inputs."""
import ast
import copy
import inspect
from unittest.mock import patch
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sciona import decomposed_march_runtime as d
from sciona.march_contract import prepare,execute
from sciona.march_training import fit_predict
from scripts.march_synthetic import payload
from sciona.atoms.ml.sklearn.calibration.atoms import sigmoid_calibration_fit,sigmoid_calibration_predict
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum


def complete(arrays):
    calibration,y,weight,forward,reverse,counts=d.fit_scores(arrays)
    state=sigmoid_calibration_fit(calibration,y,sample_weight=weight)
    rows,coefficients=d.projection_inputs(sigmoid_calibration_predict(forward,state),sigmoid_calibration_predict(reverse,state))
    return d.result(ordered_weighted_sum(rows,coefficients),counts)


def varied(seed):
    p=payload();rng=np.random.default_rng(seed)
    for game in p['games']:
        a,b=map(int,rng.integers(45,120,size=2))
        if a==b:b+=1
        game['a_box']['points']=a;game['b_box']['points']=b
        for box in ('a_box','b_box'):
            game[box]['field_goal_attempts']=int(rng.integers(45,80))
            game[box]['free_throw_attempts']=int(rng.integers(5,35))
            game[box]['turnovers']=int(rng.integers(5,15))
    p['seed']=seed;p['regularization']=[.01,.7,3.,20.][seed%4]
    p['feature_controls']['elo_scale']=[200.,400.,800.][seed%3]
    return p


@pytest.mark.parametrize('seed',range(12))
def test_complete_reference_byte_parity(seed):
    p=varied(seed);before=copy.deepcopy(p)
    actual=complete(d.pairings(d.features(prepare(p))))
    expected=execute(prepare(p))
    assert actual==expected and p==before
    assert np.asarray(actual['probabilities']).tobytes()==np.asarray(expected['probabilities']).tobytes()


def test_matchup_building_preserves_reference_statements():
    def block(fn):
        statements=ast.parse(inspect.getsource(fn)).body[0].body
        start=next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets))
        end=next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [t.id for t in n.targets[0].elts]==['cx','cy'])
        return ast.dump(ast.Module(body=statements[start:end+1],type_ignores=[]),include_attributes=False)
    assert block(d.pairings)==block(fit_predict)


@pytest.mark.parametrize('width',[1,3,8,23])
def test_matrix_only_reuse_and_frozen_calibration(width):
    rng=np.random.default_rng(901+width)
    a=dict(train_x=rng.normal(size=(32,width)),train_y=np.arange(32,dtype=np.int64)%2,
        calibration_x=rng.normal(size=(24,width)),calibration_y=np.arange(24,dtype=np.int64)%2,
        forward_x=rng.normal(size=(7,width)),reverse_x=rng.normal(size=(7,width)),
        regularization=.7,max_iterations=1500,tolerance=1e-9,seed=17)
    saved=copy.deepcopy(a)
    actual=complete(a)
    base=make_pipeline(StandardScaler(),LogisticRegression(C=.7,max_iter=1500,tol=1e-9,solver='lbfgs',random_state=17)).fit(a['train_x'],a['train_y'])
    indices=np.arange(24)
    reference=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid',ensemble=False,n_jobs=1,cv=[(indices,indices)]).fit(a['calibration_x'],a['calibration_y'])
    expected=.5*(reference.predict_proba(a['forward_x'])[:,1]+1-reference.predict_proba(a['reverse_x'])[:,1])
    assert np.asarray(actual['probabilities']).tobytes()==expected.tobytes()
    for k,value in saved.items():
        if isinstance(value,np.ndarray):assert np.array_equal(a[k],value)
        else:assert a[k]==value


def test_calibration_never_refits_base_and_coefficients_match():
    a=d.pairings(d.features(prepare(varied(2))))
    fitted=[]
    original=LogisticRegression.fit
    def capture(model,x,y,*args,**kwargs):
        result=original(model,x,y,*args,**kwargs)
        fitted.append(model)
        return result
    with patch.object(LogisticRegression,'fit',capture):
        calibration,y,weight,forward,reverse,counts=d.fit_scores(a)
        assert len(fitted)==1
        coef=fitted[0].coef_.copy();intercept=fitted[0].intercept_.copy()
        state=sigmoid_calibration_fit(calibration,y,sample_weight=weight)
        sigmoid_calibration_predict(forward,state);sigmoid_calibration_predict(reverse,state)
        assert len(fitted)==1
        assert np.array_equal(fitted[0].coef_,coef) and np.array_equal(fitted[0].intercept_,intercept)
    captured=[];original_cal=CalibratedClassifierCV.fit
    def retain(model,x,y,*args,**kwargs):
        value=original_cal(model,x,y,*args,**kwargs);captured.append(model);return value
    with patch.object(CalibratedClassifierCV,'fit',retain):execute(prepare(varied(2)))
    oracle=captured[0].calibrated_classifiers_[0].calibrators[0]
    assert state.a==oracle.a_ and state.b==oracle.b_


def test_optimizer_nonconvergence_remains_an_error():
    from sklearn.exceptions import ConvergenceWarning
    p=varied(3);p['max_iterations']=1
    with pytest.raises(ConvergenceWarning):execute(prepare(p))
    with pytest.raises(ConvergenceWarning):complete(d.pairings(d.features(prepare(p))))


def test_pagerank_exhaustion_remains_an_error():
    import networkx as nx
    p=payload();p['feature_controls']['pagerank_iterations']=1
    with pytest.raises(nx.PowerIterationFailedConvergence):execute(prepare(p))
    with pytest.raises(nx.PowerIterationFailedConvergence):d.features(prepare(p))


@pytest.mark.parametrize('fault',['missing','nan_prediction','wrong_width','bad_labels','one_class','unaligned_labels','different_prediction_count','float32','invalid_seed','invalid_iterations'])
def test_invalid_matrix_packet_rejected_before_model_fit(fault):
    a=d.pairings(d.features(prepare(payload())))
    if fault=='missing':a.pop('reverse_x')
    elif fault=='nan_prediction':a['reverse_x'][0,0]=np.nan
    elif fault=='wrong_width':a['calibration_x']=a['calibration_x'][:,:-1]
    elif fault=='bad_labels':a['train_y']=a['train_y'].astype(float)
    elif fault=='one_class':a['calibration_y'][:]=0
    elif fault=='unaligned_labels':a['calibration_y']=a['calibration_y'][:-1]
    elif fault=='different_prediction_count':a['reverse_x']=a['reverse_x'][:-1]
    elif fault=='float32':a['train_x']=a['train_x'].astype(np.float32)
    elif fault=='invalid_seed':a['seed']=True
    else:a['max_iterations']=0
    with patch.object(d,'make_pipeline',side_effect=AssertionError('Fit must not begin')):
        with pytest.raises(ValueError):d.fit_scores(a)


@pytest.mark.parametrize('fault',['chronology','overlapping_seasons','duplicate_labels','unknown_team','extra_fields','missing_feature','nonfinite_feature'])
def test_feature_and_pairing_guards(fault):
    f=d.features(prepare(payload()))
    if fault=='chronology':f['prediction'][0]['order']=0
    elif fault=='overlapping_seasons':f['prediction'][0]['season']=3
    elif fault=='duplicate_labels':f['training'].append(copy.deepcopy(f['training'][0]))
    elif fault=='unknown_team':f['calibration'][0]['a']=100
    elif fault=='extra_fields':f['training'][0]['extra']=0
    elif fault=='missing_feature':f['seasons'][1]['teams'][0]=np.ones(4)
    else:f['seasons'][1]['teams'][0][0]=np.inf
    with pytest.raises(ValueError):d.pairings(f)


@pytest.mark.parametrize('fault',['float32','nonfinite','out_of_range','mismatch','empty'])
def test_invalid_projection_inputs(fault):
    a=np.array([.2,.8]);b=a.copy()
    if fault=='float32':a=a.astype(np.float32)
    elif fault=='nonfinite':a[0]=np.nan
    elif fault=='out_of_range':a[0]=1.1
    elif fault=='mismatch':b=b[:1]
    else:a=b=np.empty(0)
    with pytest.raises(ValueError):d.projection_inputs(a,b)


@pytest.mark.parametrize('fault',['range','nonfinite','dtype','count','missing','bool'])
def test_invalid_result(fault):
    total=np.array([.2,1.8]);counts=dict(training_rows=4,calibration_rows=4,predicted_matchups=2)
    if fault=='range':total[0]=2.1
    elif fault=='nonfinite':total[0]=np.inf
    elif fault=='dtype':total=total.astype(np.float32)
    elif fault=='count':counts['predicted_matchups']=1
    elif fault=='missing':counts.pop('training_rows')
    else:counts['training_rows']=True
    with pytest.raises(ValueError):d.result(total,counts)
