"""Synthetic evaluation parity, repeat policies and random stream isolation."""
import random
from unittest.mock import patch
import numpy as np
import pytest
from sciona import dsb_decomposed_evaluation_samples as samples
from sciona.dsb_components import prepare_classifier_batch as reference_crops
from sciona.dsb_validation_batches import ValidationBatchFactory as Original
from sciona.dsb_decomposed_validation_batches import ValidationBatchFactory as Candidate


def same_rng(a,b):
    first,second=a.get_state(),b.get_state()
    assert first[0]==second[0] and first[2:]==second[2:]
    np.testing.assert_array_equal(first[1],second[1])


@pytest.mark.parametrize('empty',[False,True])
def test_inference_crops_and_rng_isolation(empty):
    volume=np.arange(32**3,dtype=np.float32).reshape(1,32,32,32)
    proposals=np.empty((0,5)) if empty else np.array([[2.,0.,2.,3.,10.],[2.,31.,29.,30.,12.]])
    global_state=np.random.get_state()
    states=[]
    original=samples.crop_classifier_proposal
    def spy(*args,**kwargs):
        rng=kwargs['rng'];before=np.random.RandomState();before.set_state(rng.get_state())
        result=original(*args,**kwargs)
        same_rng(rng,before);states.append(True)
        return result
    with patch.object(samples,'crop_classifier_proposal',spy):
        actual=samples.prepare_classifier_batch(volume,proposals,3,16)
    expected=reference_crops(volume,proposals,3,16)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    assert len(states)==len(proposals)
    saved=np.random.RandomState();saved.set_state(global_state)
    current=np.random.RandomState();current.set_state(np.random.get_state())
    same_rng(saved,current)


def test_validation_repeated_passes_and_both_random_streams():
    volume=np.full((1,64,64,64),100,dtype=np.uint8)
    boxes=np.array([[24.,24,24,10],[32.,32,32,31]])
    proposals=np.column_stack(([1.,3.],boxes))
    heldout=dict(volumes=[volume]*3,proposals=[proposals]*3,boxes=[boxes]*3,labels=[0,1,0])
    populations=dict(detector=dict(volumes=[volume],boxes=[boxes]),classifier_validation=heldout,
                     classifier_training_validation={**heldout,'labels':[1,0,1]})
    def factory(cls):
        return cls(populations,detector_batch_size=3,classifier_batch_size=2,topk=3,
                   detector_crop_size=32,classifier_crop_size=16,
                   numpy_rng=np.random.RandomState(9),label_rng=random.Random(9))
    actual,expected=factory(Candidate),factory(Original)
    for task in ['detector','classifier_validation','classifier_training_validation','detector']:
        a,b=list(actual(task)),list(expected(task))
        assert len(a)==len(b)
        assert [len(batch[0]) for batch in a]==([3,1] if task=='detector' else [2,1])
        for left,right in zip(a,b):
            for x,y in zip(left,right):np.testing.assert_array_equal(x,y)
        same_rng(actual.rng,expected.rng)
        assert actual.label_rng.getstate()==expected.label_rng.getstate()
