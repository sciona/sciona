"""Exact synthetic restoration, graph boundaries and malformed-input rejection."""
import hashlib
import json
from unittest.mock import patch
import numpy as np
import pytest
from sciona import decomposed_m5u_runtime as runtime
from sciona.m5u_harmonization import restore as reference_restore
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.decomposed_m5u_graph import build_decomposed_m5u_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph


def forecasts(multiplicity=2,signed=False):
    rng=np.random.default_rng(2913)
    levels=np.repeat(list(runtime.PARAMETERS),multiplicity)
    values=rng.normal(size=(9,28,len(levels))) if signed else rng.uniform(.1,5.,(9,28,len(levels)))
    return dict(unadjusted=values,levels=levels,
        factors=np.repeat(rng.uniform(.1,4.,14),multiplicity),adjustment=.7,
        quantiles=runtime.QUANTILES,hierarchy_indices=np.arange(len(levels)),
        reports={int(level):{'models':9} for level in runtime.PARAMETERS},
        normalization='retained_base_reference',scope='synthetic')


@pytest.mark.parametrize('multiplicity',[1,2,5])
@pytest.mark.parametrize('signed',[False,True])
def test_exact_reference_restoration_and_complete_output(multiplicity,signed):
    raw=forecasts(multiplicity,signed)
    original={key:value.copy() for key,value in raw.items() if isinstance(value,np.ndarray)}
    state,values,coefficients=runtime.restoration_inputs(raw)
    native_before=state['native'].copy()
    actual=runtime.restore(state,ordered_weighted_sum(values,coefficients))
    expected=reference_restore(raw['unadjusted'],raw['levels'],raw['factors'],raw['quantiles'],adjustment=.7)
    assert actual.tobytes()==expected.tobytes()
    assert state['native'].tobytes()==native_before.tobytes()
    for key,saved in original.items():assert raw[key].tobytes()==saved.tobytes()
    output=runtime.result(raw,actual)
    assert output['forecast']==expected.tolist()
    assert output['levels']==raw['levels'].tolist()
    assert output['hierarchy_indices']==list(range(len(raw['levels'])))
    assert output['quantiles']==list(runtime.QUANTILES) and output['horizon']==28
    assert output['reports']==raw['reports']
    json.dumps(output,allow_nan=False)


@pytest.mark.parametrize('fault', ['horizon','quantiles','dtype','infinity','levels','factor','adjustment','zero_median'])
def test_invalid_restoration_inputs_rejected(fault):
    raw=forecasts()
    if fault=='horizon':raw['unadjusted']=raw['unadjusted'][:,:27]
    elif fault=='quantiles':raw['quantiles']=(.5,)
    elif fault=='dtype':raw['unadjusted']=raw['unadjusted'].astype(np.float32)
    elif fault=='infinity':raw['unadjusted'][0,0,0]=np.inf
    elif fault=='levels':raw['levels'][raw['levels']==1]=2
    elif fault=='factor':raw['factors'][0]=0
    elif fault=='adjustment':raw['adjustment']=1.
    else:raw['unadjusted'][4,:,raw['levels']==1]=0
    with pytest.raises(ValueError):runtime.restoration_inputs(raw)


@pytest.mark.parametrize('fault', ['shape','dtype','infinity','missing_level'])
def test_invalid_generic_sum_or_state_rejected(fault):
    state,values,weights=runtime.restoration_inputs(forecasts())
    total=ordered_weighted_sum(values,weights)
    if fault=='shape':total=total[:27]
    elif fault=='dtype':total=total.astype(np.float32)
    elif fault=='infinity':total[0]=np.inf
    else:state['selected']=list(range(1,9))
    with pytest.raises(ValueError):runtime.restore(state,total)


def test_noncanonical_prepared_input_rejected_before_training():
    raw='{}'
    prepared=runtime.Prepared(raw,hashlib.sha256(raw.encode()).hexdigest())
    with patch.object(runtime,'run_unrestored') as run:
        with pytest.raises(ValueError):runtime.fit_forecast(prepared)
        run.assert_not_called()


def test_fit_forecast_uses_full_default_schedule_without_speed_overrides():
    prepared=object();payload=object();config={'units':object()}
    with patch.object(runtime,'checked_payload',return_value=payload), \
         patch.object(runtime,'unpack',return_value=config), \
         patch.object(runtime,'run_unrestored',return_value={'complete':True}) as run:
        assert runtime.fit_forecast(prepared)=={'complete':True}
    run.assert_called_once_with(**config)


def test_graph_codec_and_registered_provider_boundaries():
    import importlib
    import inspect
    graph=build_decomposed_m5u_graph()
    assert len(graph.nodes)==6 and len(graph.edges)==8
    digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph
    assert next(n for n in graph.nodes if n.node_id=='blend').matched_primitive.endswith('.ordered_sum.ordered_weighted_sum')
    for node in graph.nodes:
        module,symbol=node.matched_primitive.rsplit('.',1)
        function=getattr(importlib.import_module(module),symbol)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
