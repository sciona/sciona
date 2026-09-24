import asyncio
import copy
import numpy as np
import pytest

import sciona.guarded_prediction_graphs as guarded
from sciona.model_slot_graph import build_model_slot_graph


def descriptor():
    return guarded.graph_descriptor(build_model_slot_graph(), 'select', 'slots')


@pytest.mark.parametrize('first,second,route', [('valid','unused','primary'), ('error','valid','baseline'),
    ('invalid','error','constant'), ('error','invalid','constant')])
def test_lazy_order_and_output_validation(monkeypatch,first,second,route):
    calls=[]
    async def execute(graph, inputs):
        outcome=[first,second][len(calls)];calls.append(outcome)
        if outcome=='error':raise ValueError('synthetic prediction failure')
        if outcome=='invalid':return np.zeros((2,1))
        assert outcome=='valid'
        return np.array([3.,4.])
    monkeypatch.setattr(guarded,'execute_graph_value',execute)
    values,actual=asyncio.run(guarded.guarded_prediction(descriptor(),descriptor(),{},2,30.))
    assert actual==route and len(calls)==(1 if route=='primary' else 2)
    np.testing.assert_array_equal(values,[30.,30.] if route=='constant' else [3.,4.])


def test_empty_and_corrupt_definition_do_not_execute(monkeypatch):
    async def forbidden(*args):raise AssertionError('Branch invoked')
    monkeypatch.setattr(guarded,'execute_graph_value',forbidden)
    values,route=asyncio.run(guarded.guarded_prediction(descriptor(),descriptor(),{},0,30.))
    assert route=='empty' and values.shape==(0,)
    broken=copy.deepcopy(descriptor());broken['hash']='0'*64
    with pytest.raises(ValueError):
        asyncio.run(guarded.guarded_prediction(broken,descriptor(),{},2,30.))


@pytest.mark.parametrize('error',[KeyboardInterrupt,SystemExit,asyncio.CancelledError])
def test_control_exceptions_propagate(monkeypatch,error):
    async def execute(*args):raise error('synthetic interrupt')
    monkeypatch.setattr(guarded,'execute_graph_value',execute)
    with pytest.raises(error):
        asyncio.run(guarded.guarded_prediction(descriptor(),descriptor(),{},2,30.))
