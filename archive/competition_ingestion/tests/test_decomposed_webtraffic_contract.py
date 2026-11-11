"""Synthetic complete recipe boundary and final-record integration."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from scripts.validate_webtraffic_graph import synthetic_payload
from sciona.decomposed_webtraffic_contract import unpack,forecast_axes,result
from sciona import decomposed_webtraffic_features as f
from sciona import decomposed_webtraffic_lifecycle as lifecycle
from sciona.webtraffic_lifecycle import advance,save_checkpoint,predict_checkpoints
from sciona.webtraffic_payload import execute_payload
from sciona.panel_output import aligned_suffix_records


def complete_payload():
    p=synthetic_payload();p['version']=2
    days=[d.date().isoformat() for d in pd.date_range(pd.Timestamp(p['start_day'])+pd.Timedelta(days=800),periods=63)]
    p['output']=dict(rows=p['pages'].copy(),columns=days[-3:],identifiers=np.arange(len(p['pages'])*3).reshape(-1,3).tolist())
    return p


def test_full_pipeline_and_explicit_output_records():
    torch.set_num_threads(1)
    payload=complete_payload();saved=deepcopy(payload)
    frame,config,valid,added,offset,lags,thresholds,rows,cols,ids=unpack(payload)
    logged,missing,starts,ends=f.prepare(frame,valid,added,offset)
    corr=f.correlations(logged.to_numpy(),starts,ends,lags,thresholds,offset)
    page_map,encoded=f.page_metadata(logged.index.tolist())
    dow,lagged,last=f.calendar(logged.columns[0],logged.columns[-1],added)
    tensors,plain=f.assemble(logged,missing,corr,page_map,encoded,f.popularity(logged),dow,lagged,last)
    prepared=lifecycle.configure(tensors,plain,config)
    trained=lifecycle.train(prepared,lifecycle.initialize(prepared),advance,save_checkpoint)
    predictions,_=predict_checkpoints(trained['checkpoints'],lifecycle.prediction_batches(prepared),pd.Index(tensors['page_ix']))
    dated=lifecycle.date_forecast(prepared,predictions)
    values,r,c=forecast_axes(dated)
    output=result(*aligned_suffix_records(values,r,c,rows,cols,ids))
    reference=deepcopy(payload);reference.pop('output');reference['version']=1
    expected=execute_payload(reference)
    assert expected['pages']==r and expected['days']==c and expected['forecasts']==values.tolist()
    assert output==dict(version=2,identifiers=ids.ravel().tolist(),values=[v for row in expected['forecasts'] for v in row[-3:]])
    assert payload==saved


def test_unpack_does_not_alias_or_prepare_features():
    payload=complete_payload();parts=unpack(payload)
    payload['counts'][2][2]=999;payload['output']['rows'][0]='changed';payload['output']['identifiers'][0][0]=999
    assert parts[0].iloc[2,2]!=999 and parts[7][0]!='changed' and parts[9][0,0]==0
    assert parts[0].shape==(16,800) and parts[2:7]==(0.,63,0,[365,91],[1.5,2.])


@pytest.mark.parametrize('fault',['version','row_order','column_order','non_suffix','empty_columns','ids_bool','ids_overflow','ids_shape','nan','unknown_output','unknown_config'])
def test_invalid_root_rejected_before_training(fault):
    p=complete_payload()
    if fault=='version':p['version']=1
    elif fault=='row_order':p['output']['rows'].reverse()
    elif fault=='column_order':p['output']['columns'].reverse()
    elif fault=='non_suffix':p['output']['columns'].pop()
    elif fault=='empty_columns':p['output']['columns']=[]
    elif fault=='ids_bool':p['output']['identifiers'][0][0]=True
    elif fault=='ids_overflow':p['output']['identifiers'][0][0]=2**63
    elif fault=='ids_shape':p['output']['identifiers'].pop()
    elif fault=='nan':p['counts'][1][1]=float('nan')
    elif fault=='unknown_output':p['output']['extra']=1
    else:p['config']['extra']=1
    with pytest.raises(ValueError):unpack(p)
