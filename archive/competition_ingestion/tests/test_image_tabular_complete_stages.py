"""Synthetic full-budget parity and population isolation for explicit stages."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from scripts.image_tabular_synthetic import payload
from sciona import image_tabular_complete_stages as stages
from sciona.image_tabular_contract import execute,prepare


@pytest.mark.parametrize('change',['baseline','no_dropout','no_smoothing'])
def test_full_lifecycle_parity(change):
    p=payload()
    if change=='no_dropout':p['controls']['dropout']=0.
    if change=='no_smoothing':p['controls']['smoothing']=0.
    before=deepcopy(p)
    result=stages.predict(stages.calibrate(stages.fit(stages.prepare(p))))
    expected=execute(prepare(p))
    assert result.keys()==expected.keys()
    for name in result:
        np.testing.assert_allclose(result[name],expected[name],rtol=1e-12,atol=1e-12)
    assert p==before


def test_calibration_prediction_never_refit(monkeypatch):
    fitted=stages.fit(stages.prepare(payload()))
    weights=[{k:v.clone() for k,v in m.network.state_dict().items()} for m in fitted.ensemble.models]
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected refit')
    monkeypatch.setattr(stages,'train',forbidden)
    result=stages.predict(stages.calibrate(fitted))
    assert result['folds']==3 and result['epochs']==40 and result['oof_rows']==12
    for model,expected in zip(fitted.ensemble.models,weights,strict=True):
        for k,v in model.network.state_dict().items():assert torch.equal(v,expected[k])


def test_prepared_detached_and_group_boundaries_revalidated():
    p=payload();prepared=stages.prepare(p);before=prepared.configuration
    p['query']['images'][0][0][0][0]=.7
    assert prepared.configuration==before
    forged=json.loads(before)
    forged['query']['groups'][0]=forged['training']['groups'][0]
    with pytest.raises(ValueError,match='overlap'):
        stages.configuration(stages.Prepared(json.dumps(forged)))
