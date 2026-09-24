import numpy as np
import pytest
import xgboost

from sciona.atoms.ml.domain_adapters.airport_features import numerical_handoff
from sciona.ghost.dimensions import DimensionalSignature
from sciona.residual_metadata_preflight import validate_materialized_inputs


def payload():
    return dict(features=np.ones((8,3),dtype=np.float64),targets=np.ones(8,dtype=np.float64),
        groups=np.repeat(np.arange(4,dtype=np.int64),2),feature_names=['a','b','c'],
        prediction_features=np.ones((5,3),dtype=np.float64),train_fraction=.4,seed=42,
        maximum_error=20.,threshold=.5,prediction_feature_name='prediction')


def test_observed_shapes_propagate_without_fitting_and_values_pass_through(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Metadata preflight must not fit a model')
    monkeypatch.setattr(xgboost.XGBRegressor,'fit',forbidden)
    monkeypatch.setattr(xgboost.XGBClassifier,'fit',forbidden)
    p=payload()
    result=validate_materialized_inputs(p,DimensionalSignature(T=1))
    assert result['nodes_visited']==25
    assert result['final'].shape==(5,) and result['final'].dtype=='int32'
    assert result['final'].dim==DimensionalSignature(T=1)
    output=numerical_handoff(**p)
    assert all(actual is expected for actual,expected in zip(output,p.values()))


@pytest.mark.parametrize('fault',['target_rows','query_width','one_group','bad_fraction','bad_seed',
    'negative_error','bad_probability','name_collision','nan_target','infinite_feature','wrong_dtype'])
def test_materialized_boundary_rejects_invalid_inputs(fault):
    p=payload()
    if fault=='target_rows':p['targets']=p['targets'][:-1]
    elif fault=='query_width':p['prediction_features']=np.ones((5,4),dtype=np.float64)
    elif fault=='one_group':p['groups'][:]=0
    elif fault=='bad_fraction':p['train_fraction']=1.
    elif fault=='bad_seed':p['seed']=-1
    elif fault=='negative_error':p['maximum_error']=-1.
    elif fault=='bad_probability':p['threshold']=1.1
    elif fault=='name_collision':p['prediction_feature_name']='a'
    elif fault=='nan_target':p['targets'][0]=np.nan
    elif fault=='infinite_feature':p['features'][0,0]=np.inf
    else:p['prediction_features']=p['prediction_features'].astype(np.float32)
    with pytest.raises(ValueError):numerical_handoff(**p)


def test_query_missing_values_remain_runtime_values():
    p=payload()
    p['prediction_features'][0,0]=np.nan
    numerical_handoff(**p)
    assert np.isnan(p['prediction_features'][0,0])
