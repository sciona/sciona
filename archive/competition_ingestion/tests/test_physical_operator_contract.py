import json
import math
import numpy as np
import pytest
from scripts.physical_operator_synthetic import payload
from sciona import physical_operator_contract as contract


def test_complete_mixed_mode_lifecycle():
    p=payload();expected=[]
    # Add a second analytic diffusion mode to every population. Query targets
    # remain solely in this test oracle and never enter the training boundary.
    for name in ('training','validation','query'):
        part=p[name]
        for i,row in enumerate(part['states']):
            tau=part['diffusivity'][i]*part['elapsed'][i]
            second=[.07*math.sin(4*math.pi*x+.2*i) for x in p['grid']]
            part['states'][i]=[v+d for v,d in zip(row,second)]
            if name!='query':
                part['targets'][i]=[v+d*math.exp(-16*math.pi**2*tau) for v,d in zip(part['targets'][i],second)]
            else:
                # Exact spectral evolution of this synthetic trigonometric field.
                frequencies=np.fft.rfftfreq(len(row),d=1/len(row))
                expected.append(np.fft.irfft(np.fft.rfft(part['states'][i])*np.exp(-4*np.pi**2*frequencies**2*tau),n=len(row)))
    result=contract.execute(contract.prepare(p));json.dumps(result,allow_nan=False)
    predictions=np.asarray(result['predictions']);expected=np.asarray(expected)
    assert predictions.shape==(2,16) and (predictions>=0).all()
    np.testing.assert_allclose(predictions.mean(1),expected.mean(1),atol=1e-12)
    # Accuracy requirement is limited to this synthetic fixture.
    mean_baseline=np.mean(expected,axis=1,keepdims=True)
    assert np.mean((predictions-expected)**2)<np.mean((mean_baseline-expected)**2)
    assert result['max_mean_conservation_error']<1e-12


def test_prepared_copy_and_mutation_rejection(monkeypatch):
    p=payload();prepared=contract.prepare(p)
    p['query']['groups'][0]='changed'
    assert prepared.payload['query']['groups'][0]!='changed'
    prepared.payload['query']['groups'][0]='synthetic0'
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit',forbidden)
    with pytest.raises(ValueError):contract.execute(prepared)


def test_query_labels_rejected():
    p=payload();p['query']['targets']=p['query']['states']
    with pytest.raises(ValueError):contract.prepare(p)


def test_nonprepared_rejected():
    with pytest.raises(ValueError):contract.execute(payload())
