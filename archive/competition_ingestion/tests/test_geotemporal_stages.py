"""Full retained-recipe parity, fold isolation and separated prediction stages."""
from copy import deepcopy
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from scripts.geotemporal_synthetic import payload
from sciona import geotemporal_stages as stages
from sciona.geotemporal_contract import prepare,execute


def changed_payload(change):
    p=payload()
    if change=='no_smoothing':p['controls']['smoothing']=0.
    if change=='separate_query_times':p['query'][1]['time']=11.
    return p


@pytest.mark.parametrize('change',['baseline','no_smoothing','separate_query_times'])
def test_complete_parity(change):
    p=changed_payload(change);original=deepcopy(p)
    result=stages.finish(stages.predict(stages.fit(stages.features(stages.prepare(p)))))
    expected=execute(prepare(p))
    assert result.keys()==expected.keys()
    for key in result:np.testing.assert_allclose(result[key],expected[key],rtol=1e-12,atol=1e-12)
    assert p==original


def test_heldout_targets_excluded_from_own_features_and_model():
    p=payload();first=stages.fit(stages.features(stages.prepare(p)))
    for row in p['observations'][3:6]:row['target']+=1000
    second=stages.fit(stages.features(stages.prepare(p)))
    for key in ('training_features','validation_features','predictions'):
        np.testing.assert_array_equal(first.fold_audits[0][key],second.fold_audits[0][key])
    assert not np.array_equal(first.fold_audits[1]['training_features'],second.fold_audits[1]['training_features'])
    assert first.validation_predictions[:3]==(None,None,None)
    assert all(tree.tree_.n_node_samples[0]==9 for tree in first.estimator.estimators_)


def test_future_context_excluded_from_earlier_fold():
    p=payload();first=stages.features(stages.prepare(p))
    for row in p['context']:
        if row['time']>5:row['value']+=1000
    second=stages.features(stages.prepare(p))
    for key in ('training_features','validation_features'):
        np.testing.assert_array_equal(first.folds[0][key],second.folds[0][key])


def test_prediction_and_finish_do_not_refit(monkeypatch):
    fitted=stages.fit(stages.features(stages.prepare(payload())))
    before=[tree.tree_.value.copy() for tree in fitted.estimator.estimators_]
    def forbidden(*args,**kwargs):raise AssertionError('Prediction must not refit')
    monkeypatch.setattr(ExtraTreesRegressor,'fit',forbidden)
    first=stages.finish(stages.predict(fitted));second=stages.finish(stages.predict(fitted))
    assert first==second
    for tree,values in zip(fitted.estimator.estimators_,before,strict=True):np.testing.assert_array_equal(tree.tree_.value,values)


def test_detached_configuration_and_revalidation():
    p=payload();prepared=stages.prepare(p);p['query'][0]['time']=0
    assert stages.features(prepared).query_features.shape==(2,11)
    with pytest.raises(ValueError):stages.features(stages.Prepared(prepared.configuration.replace('"planar"','"latitude_longitude"')))


def test_registered_numeric_provider_used_for_folds_and_finish(monkeypatch):
    original=stages.blend_neighbor_means;calls=[]
    def observe(values,neighbors,strength):
        calls.append((len(values),neighbors.dtype,strength))
        return original(values,neighbors,strength)
    monkeypatch.setattr(stages,'blend_neighbor_means',observe)
    stages.finish(stages.predict(stages.fit(stages.features(stages.prepare(payload())))))
    assert calls==[(3,np.dtype(bool),.2),(3,np.dtype(bool),.2),(2,np.dtype(bool),.2)]
