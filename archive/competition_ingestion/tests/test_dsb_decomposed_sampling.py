"""Synthetic integration checks against the retained source-qualified factory."""
import random
import numpy as np
import pytest
from sciona.dsb_batches import TrainingBatchFactory as ReferenceFactory
from sciona.dsb_decomposed_batches import TrainingBatchFactory
from sciona.dsb_decomposed_sampling import detector_sampling_table, classifier_training_sample
from sciona.dsb_sampling import detector_sampling_table as reference_table
from sciona.dsb_training import classifier_training_sample as reference_sample


def factory(cls, seed):
    volume=np.arange(48**3,dtype=np.uint32).reshape(1,48,48,48).astype(np.uint8)
    boxes=np.array([[24.,24.,24.,10.],[18.,18.,18.,12.],[30.,30.,30.,14.]])
    proposals=np.column_stack(([3.,2.,1.],boxes))
    return cls([volume],[boxes],[0],[volume]*3,[proposals]*3,[np.array([1,0,1])]*3,[1,0,1],
        detector_order=[3,2,0,1],classifier_order=[2,0,1],detector_batch_size=3,classifier_batch_size=2,
        topk=2,detector_crop_size=32,classifier_crop_size=16,
        numpy_rng=np.random.RandomState(seed),label_rng=random.Random(seed+1))


def same_rng(a,b):
    first,second=a.get_state(),b.get_state()
    assert first[0]==second[0] and first[2:]==second[2:]
    np.testing.assert_array_equal(first[1],second[1])


@pytest.mark.parametrize('seed',[0,7,31])
@pytest.mark.parametrize('profile',[
    dict(flip=False,rotate=False,swap=False,scale=False),
    dict(flip=True,rotate=False,swap=True,scale=True),
    dict(flip=True,rotate=True,swap=False,scale=False),
])
def test_complete_batches_preserve_repeated_passes_and_both_rng_streams(seed,profile):
    candidate,reference=factory(TrainingBatchFactory,seed),factory(ReferenceFactory,seed)
    np.testing.assert_array_equal(candidate.table,reference.table)
    for task in ['classifier','detector','classifier','detector']:
        actual,expected=list(candidate(task,profile)),list(reference(task,profile))
        assert [len(b[0]) for b in actual]==([3,1] if task=='detector' else [2,1])
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        same_rng(candidate.rng,reference.rng)
        assert candidate.label_rng.getstate()==reference.label_rng.getstate()


@pytest.mark.parametrize('resolution',[.5,1.,2.])
def test_table_keeps_image_associations_and_threshold_units(resolution):
    boxes=[np.array([[1.,2.,3.,d] for d in [6,7,30,31,40,41]]),np.empty((0,4)),
           np.array([[4.,5.,6.,80.]])]
    actual=detector_sampling_table(boxes,resolution)
    np.testing.assert_array_equal(actual,reference_table(boxes,resolution))
    assert actual[-1,0]==2
    assert detector_sampling_table([np.empty((0,4))],resolution).shape==(0,5)


@pytest.mark.parametrize('count',[0,1,2,4])
def test_proposal_count_boundaries_keep_zero_slots_and_rng(count):
    volume=np.arange(32**3,dtype=np.float32).reshape(1,32,32,32)
    proposals=np.array([[float(i),16.,16.,16.,6.] for i in range(count)]).reshape(count,5)
    a,b=np.random.RandomState(19),np.random.RandomState(19)
    args=dict(topk=2,crop_size=16,scale=True,flip=True,rotate=True,swap=False)
    actual=classifier_training_sample(volume,proposals,np.ones(count),rng=a,**args)
    expected=reference_sample(volume,proposals,np.ones(count),rng=b,**args)
    for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
    same_rng(a,b)
    assert actual[0].shape==(2,1,16,16,16)
    if count<2:assert not actual[0][count:].any()


def test_runtime_path_calls_reusable_operations(monkeypatch):
    from sciona import dsb_decomposed_sampling as adapter
    calls={'repeat':0,'sample':0}
    repeat,sample=adapter.repeat_rows_by_threshold,adapter.sequential_softmax_indices
    def repeat_spy(*args,**kwargs):
        calls['repeat']+=1
        return repeat(*args,**kwargs)
    def sample_spy(*args,**kwargs):
        calls['sample']+=1
        return sample(*args,**kwargs)
    monkeypatch.setattr(adapter,'repeat_rows_by_threshold',repeat_spy)
    monkeypatch.setattr(adapter,'sequential_softmax_indices',sample_spy)
    candidate=factory(TrainingBatchFactory,4)
    assert calls=={'repeat':1,'sample':0}
    list(candidate('classifier',dict(flip=False,rotate=False,swap=False,scale=False)))
    assert calls=={'repeat':1,'sample':3}


def test_draft_graph_binds_new_factory_and_roundtrips():
    from sciona.dsb_decomposed_execution import build_dsb_sampling_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    from sciona.atoms.dl.detection.dsb_decomposed_training import dsb_decomposed_training_batches
    graph=build_dsb_sampling_graph()
    node=next(n for n in graph.nodes if n.node_id=='training_batches')
    assert node.matched_primitive.endswith('dsb_decomposed_training.dsb_decomposed_training_batches')
    assert graph.metadata['publication_status']=='draft'
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'synthetic'} for n in nodes],
        'cdg_edges':[{**e,'version_id':'synthetic'} for e in edges]},version_id='synthetic',
        content_hash=digest,require_execution_envelope=True)
    assert restored==graph
    ref=factory(ReferenceFactory,7)
    arrays=dict(detector_volumes=ref.detector_volumes,boxes_by_image=ref.boxes_by_image,
        eligible_image_indices=ref.eligible,classifier_volumes=ref.classifier_volumes,
        proposals_by_case=ref.proposals,known_by_case=ref.known,case_labels=ref.labels)
    options=dict(detector_order=ref.orders['detector'],classifier_order=ref.orders['classifier'],
        detector_batch_size=3,classifier_batch_size=2,topk=2,detector_crop_size=32,classifier_crop_size=16,
        numpy_rng=np.random.RandomState(7),label_rng=random.Random(8))
    actual=dsb_decomposed_training_batches(arrays,options)
    assert isinstance(actual,TrainingBatchFactory)
    for a,b in zip(actual('classifier',{}),ref('classifier',{})):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    same_rng(actual.rng,ref.rng)
