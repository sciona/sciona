"""Explicit source lifecycle stages with separately bound update/checkpoint operations.

Preserve all configured three-model updates and source RNG consumption. Mutable
stream bundles are single-use invocation state, never serialized or published.
"""
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
import torch
from .webtraffic_runtime import PreparedRuntime
from .webtraffic_batches import window_batches
from .webtraffic_initialization import uniform_shapes,initialize_from_uniforms
from .webtraffic_lifecycle import batch_inputs,initial_lifecycle
from .webtraffic_schedule import training_schedule


def configure(tensors,plain,config):
    config.validate()
    if plain['data_days']<=409:raise ValueError('More than 409 observed days required')
    # Source can hang when an infinite repeated population rejects every window.
    # Require at least one potentially nonzero training series before creating streams.
    values=np.asarray(tensors['hits'])
    zero=(~np.isfinite(values))|(values==0)
    prefix=np.pad(np.cumsum(zero,axis=1),((0,0),(1,0)))
    high=plain['data_days']-346
    empties=prefix[:,283:283+high]-prefix[:,:high]
    if not np.any(empties<=round(283*(1-config.train_completeness))):
        raise ValueError('No possible source training window passes completeness')
    schedule=list(training_schedule(plain['n_pages'],batch_size=config.batch_size,max_epoch=config.max_epoch,
                                   max_steps=config.max_steps,save_from_step=config.save_from_step))
    if not any(save for _,_,save in schedule):raise ValueError('Schedule produces no checkpoints for inference')
    return PreparedRuntime(tensors,plain,config,schedule)


@dataclass
class TrainingStreams:
    prepared: PreparedRuntime
    streams: list
    generators: list
    first: list
    state: dict
    consumed: bool = False


def initialize(prepared):
    cfg=prepared.config;tensors=prepared.tensors;days=prepared.plain['data_days'];n=prepared.plain['n_pages']
    streams=[];generators=[];first=[];parameters=[]
    for model_index in range(3):
        seed=cfg.seed+model_index
        rng=np.random.RandomState(seed)
        order=rng.permutation(n)
        def offsets(random=rng):
            for _ in itertools.count():yield int(random.randint(0,days-346))
        stream=window_batches(tensors,order,offsets(),data_days=days,batch_size=cfg.batch_size,
                              epochs=None,completeness=cfg.train_completeness)
        record=next(stream);first.append(record);streams.append(stream)
        generator=torch.Generator(device='cpu').manual_seed(seed);generators.append(generator)
        xdepth,ydepth=record[1].shape[-1],record[5].shape[-1]
        draws={name:torch.rand(shape,generator=generator) for name,shape in uniform_shapes(xdepth,ydepth).items()}
        parameters.append(initialize_from_uniforms(draws,xdepth,ydepth))
    return TrainingStreams(prepared,streams,generators,first,initial_lifecycle(parameters))


def train(prepared,initialized,advance_function,checkpoint_function):
    """Run every scheduled update using externally supplied lifecycle operations.

    The graph must bind the qualified operations. Callability alone cannot prove
    that arbitrary injected functions preserve optimization or checkpoint policy.
    Consumes the invocation's streams even if an update fails; retry with fresh
    initialization to preserve deterministic RNG semantics.
    """
    if not isinstance(initialized,TrainingStreams) or initialized.consumed:
        raise ValueError('Fresh single-use training streams required')
    if initialized.prepared is not prepared:raise ValueError('Training streams belong to a different invocation')
    if not callable(advance_function) or not callable(checkpoint_function):
        raise ValueError('Explicit update and checkpoint operations required')
    if initialized.state['shared_step']!=0:raise ValueError('Fresh model state required')
    initialized.consumed=True
    cfg=prepared.config
    streams,generators,first=initialized.streams,initialized.generators,initialized.first
    state=initialized.state;checkpoints=[]
    for step,_,save in prepared.schedule:
        records=first if step==1 else [next(stream) for stream in streams]
        batches=[]
        for record,generator in zip(records,generators):
            batch=len(record[0]);gate=torch.rand(batch,267,generator=generator)<.9967589439360334
            masks=[dict(state=torch.rand(63,batch,267,generator=generator)<.99,
                        output=torch.rand(63,batch,267,generator=generator)<.975)]
            batches.append(batch_inputs(record,gate,masks))
        observed=step if cfg.ema_step_phase=='after' else step-1
        observations=[dict(parameter_phase=cfg.ema_parameter_phase,global_step=observed) for _ in range(3)]
        state,_=advance_function(state,batches,observations)
        if save:checkpoints=checkpoint_function(checkpoints,state)
    return dict(state=state,checkpoints=checkpoints)


def prediction_batches(prepared):
    tensors=prepared.tensors;plain=prepared.plain
    records=list(window_batches(tensors,np.arange(plain['n_pages']),[],data_days=plain['data_days'],
                                epochs=1,training=False,batch_size=prepared.config.prediction_batch_size,completeness=.01))
    return records


def date_forecast(prepared,frame):
    """Attach the source final-horizon axis without changing integer forecasts."""
    if frame.shape[1]!=63:raise ValueError('Complete source horizon required')
    result=frame.copy()
    result.columns=pd.date_range(prepared.plain['data_end']+pd.Timedelta(days=1),periods=63)
    return result
