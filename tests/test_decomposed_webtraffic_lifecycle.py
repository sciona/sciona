"""Complete synthetic training/checkpoint parity, including explicit EMA phases."""
import numpy as np
import pandas as pd
import pytest
import torch
from sciona import decomposed_webtraffic_features as features
from sciona import decomposed_webtraffic_lifecycle as stages
from sciona.webtraffic_runtime import RuntimeConfig,prepare_runtime,train_runtime,forecast_runtime
from sciona.webtraffic_lifecycle import advance,save_checkpoint,predict_checkpoints
from scripts.validate_webtraffic_graph import synthetic_payload


def equal_state(actual,expected):
    assert type(actual) is type(expected)
    if isinstance(actual,torch.Tensor):assert torch.equal(actual,expected)
    elif isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:equal_state(actual[key],expected[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equal_state(a,b)
    else:assert actual==expected


@pytest.mark.parametrize('seed,phase,steps',[(2,'after',1),(7,'before',2)])
def test_complete_training_and_forecast_parity(seed,phase,steps):
    torch.set_num_threads(1)
    payload=synthetic_payload()
    frame=pd.DataFrame(np.array(payload['counts'],dtype=np.float32),index=payload['pages'],
        columns=pd.date_range(payload['start_day'],periods=800))
    config=RuntimeConfig(seed=seed,batch_size=2,max_steps=steps,save_from_step=1,
        prediction_batch_size=7,ema_parameter_phase=phase,ema_step_phase=phase)
    expected_prepared=prepare_runtime(frame,config)
    expected=train_runtime(expected_prepared)
    expected_forecast=forecast_runtime(expected_prepared,expected)
    logged,mask,starts,ends=features.prepare(frame,0.,63,0)
    correlations=features.correlations(logged.to_numpy(),starts,ends,[365,91],[1.5,2.],0)
    mapping,encoded=features.page_metadata(logged.index.tolist())
    dow,lagged,last=features.calendar(logged.columns[0],logged.columns[-1],63)
    tensors,plain=features.assemble(logged,mask,correlations,mapping,encoded,features.popularity(logged),dow,lagged,last)
    prepared=stages.configure(tensors,plain,config)
    assert prepared.schedule==expected_prepared.schedule
    initialized=stages.initialize(prepared);updates=[];saves=[]
    def tracked_update(state,batches,observations):
        updates.append(state['shared_step'])
        return advance(state,batches,observations)
    def tracked_save(checkpoints,state):
        saves.append(state['shared_step'])
        return save_checkpoint(checkpoints,state)
    actual=stages.train(prepared,initialized,tracked_update,tracked_save)
    assert updates==list(range(steps+1))
    assert saves==[step for step,_,save in prepared.schedule if save]
    equal_state(actual,expected)
    forecast,_=predict_checkpoints(actual['checkpoints'],stages.prediction_batches(prepared),pd.Index(tensors['page_ix']))
    actual_forecast=stages.date_forecast(prepared,forecast)
    pd.testing.assert_frame_equal(actual_forecast,expected_forecast,check_exact=True)
    assert actual['state']['shared_step']==steps+1 and len(actual['checkpoints'])==len(saves)
    with pytest.raises(ValueError,match='single-use'):stages.train(prepared,initialized,advance,save_checkpoint)
