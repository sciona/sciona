"""Synthetic checks for full forecasting extraction and numeric reuse."""
import dataclasses
import numpy as np
import pytest
from sciona import decomposed_m5_runtime as runtime
from sciona.m5_execution import prepare
from sciona.m5_forecast import predict, predict_families
from sciona.m5_pooled_training import train
from sciona.decomposed_m5_graph import build_decomposed_m5_graph
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from tests.test_m5_execution import payload


@pytest.mark.parametrize('count',[1,2,70,101])
def test_exact_family_reduction(count):
    families=np.random.default_rng(7).random((6,count,28))
    original=families.copy()
    values,weights=runtime.blend_inputs(dict(families=families))
    assert np.array_equal(ordered_weighted_sum(values,weights).reshape(count,28)/6,np.sum(families,axis=0)/6)
    assert np.array_equal(families,original)


@pytest.mark.parametrize('fault',['family','horizon','nan','negative','dtype'])
def test_invalid_forecasts(fault):
    values=np.ones((6,2,28))
    if fault=='family':values=values[:-1]
    elif fault=='horizon':values=values[:,:,:-1]
    elif fault=='nan':values[0,0,0]=np.nan
    elif fault=='negative':values[0,0,0]=-1
    else:values=values.astype(np.float32)
    with pytest.raises(ValueError):runtime.blend_inputs(dict(families=values))


def test_prepared_integrity():
    prepared=prepare(payload())
    with pytest.raises(ValueError):runtime.features(dataclasses.replace(prepared,fingerprint='0'*64))


def test_full_native_lifecycle_matches_reference():
    prepared=prepare(payload())
    features=runtime.features(prepared)
    fitted=runtime.fit(prepared,features)
    forecasted=runtime.forecast(features,fitted)
    values,weights=runtime.blend_inputs(forecasted)
    result=runtime.result(prepared,forecasted,ordered_weighted_sum(values,weights))
    reference=predict(features,fitted['models'],fitted['cutoff'])
    assert result==dict(forecast=reference['forecast'].tolist(),models=reference['models'],horizon=28,
                        model_families=6,scope=reference['scope'])
    assert all(m.report['evaluated_rounds']==3000 for m in fitted['models'])
    assert runtime.train is train and runtime.predict_families is predict_families


def test_complete_graph():
    graph=build_decomposed_m5_graph()
    assert len(graph.nodes)==7 and len(graph.edges)==11
    assert next(n for n in graph.nodes if n.node_id=='blend').matched_primitive.endswith('ordered_sum.ordered_weighted_sum')
