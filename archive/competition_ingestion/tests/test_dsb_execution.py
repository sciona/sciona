import asyncio
import numpy as np
import pytest
import torch

from sciona.dsb_checkpoint import checkpoint_bytes
from sciona.dsb_execution import build_dsb_inference_graph
from sciona.dsb_network import DetectorNet,CaseNet
from sciona.dsb_pipeline import infer_volume
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def test_serialized_dsb_inference_uses_discovered_providers(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(29)
    graph=build_dsb_inference_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    assert restored==graph
    detector,classifier=DetectorNet().eval(),CaseNet(2).eval()
    with torch.no_grad():
        detector.output[-1].weight.zero_();detector.output[-1].bias.zero_()
        detector.output[-1].bias[::5]=2.
    z,y,x=np.indices((24,64,64))
    volume=np.full(z.shape,50.)
    volume[((z-12)/9)**2+((y-32)/17)**2+((x-20)/9)**2<1]=-900.
    options=dict(requested_spacing=(6.,6.,6.),tile_side=32,margin=0,crop_size=16,topk=2)
    expected=infer_volume(volume,(6.,6.,6.),detector,classifier,**options)
    assert len(expected['proposals'])>=2
    inputs=dict(volume=volume,spacing=(6.,6.,6.),**options,
        detector_checkpoint=checkpoint_bytes(detector,30),
        classifier_checkpoint=checkpoint_bytes(classifier,121))
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    # No explicit provider imports or manual registry injection: use discovery.
    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-dsb','case').execute(inputs,cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==3
    for name in ['case_probability','proposal_probabilities']:
        np.testing.assert_array_equal(np.load(tmp_path/'case'/'prediction'/('out_'+name+'.npy')),expected[name])


def test_dsb_witness_rejects_mismatched_topk_and_training_mode():
    runner._ensure_atoms_imported()
    from sciona.atoms.dl.detection.dsb_execution import witness_restore_dsb_models,witness_dsb_predict
    from sciona.ghost.abstract import AbstractArray
    detector,classifier=witness_restore_dsb_models({}, {},topk=2)
    volume=AbstractArray(shape=(1,32,32,32),dtype='uint8')
    result=witness_dsb_predict(volume,detector,classifier,topk=2,crop_size=16,tile_side=32,margin=0)
    assert result[1].shape==(1,2)
    with pytest.raises(ValueError,match='classifier'):
        witness_dsb_predict(volume,detector,classifier,topk=3)
    with pytest.raises(ValueError,match='detector'):
        witness_dsb_predict(volume,{**detector,'mode':'train'},classifier,topk=2)


@pytest.mark.parametrize('multiple_epochs',[False,True,'arrays','detector_init','raw'])
def test_serialized_training_checkpoint_reaches_inference(tmp_path,monkeypatch,multiple_epochs):
    import random
    from sciona.dsb_execution import build_dsb_training_inference_graph,build_dsb_training_range_inference_graph,build_dsb_array_training_inference_graph
    from sciona.dsb_execution import build_dsb_detector_initialized_graph
    from sciona.dsb_execution import build_dsb_raw_training_graph
    from sciona.dsb_batches import TrainingBatchFactory
    torch.set_num_threads(1);torch.manual_seed(37)
    cube=np.full((1,48,48,48),100,dtype=np.uint8)
    boxes=np.array([[24.,24.,24.,10.],[18.,18.,18.,12.],[30.,30.,30.,14.]])
    arrays=dict(detector_volumes=[cube],boxes_by_image=[boxes],eligible_image_indices=[0],
        classifier_volumes=[cube]*3,proposals_by_case=[np.column_stack(([3.,2.,1.],boxes))]*3,
        known_by_case=[np.array([1,0,1])]*3,case_labels=[1,0,1])
    options=dict(detector_order=[3,2,0,1],classifier_order=[2,0,1],detector_batch_size=3,
        classifier_batch_size=2,topk=2,detector_crop_size=32,classifier_crop_size=16)
    validation=TrainingBatchFactory(**arrays,**options,numpy_rng=np.random.RandomState(3),label_rng=random.Random(3))
    profile=dict(flip=False,rotate=False,swap=False,scale=False)
    det=list(validation('detector',profile));cases=list(validation('classifier',profile))
    detector,classifier=DetectorNet().eval(),CaseNet(2).eval()
    with torch.no_grad():
        detector.output[-1].weight.zero_();detector.output[-1].bias.zero_()
        detector.output[-1].bias[::5]=-4.
    z,y,x=np.indices((24,64,64));volume=np.full(z.shape,50.)
    volume[((z-12)/9)**2+((y-32)/17)**2+((x-20)/9)**2<1]=-900.
    graph=(build_dsb_raw_training_graph() if multiple_epochs=='raw' else
           build_dsb_detector_initialized_graph() if multiple_epochs=='detector_init' else
           build_dsb_array_training_inference_graph() if multiple_epochs=='arrays' else
           build_dsb_training_range_inference_graph() if multiple_epochs else build_dsb_training_inference_graph())
    validation_cases=dict(volumes=arrays['classifier_volumes'],proposals=arrays['proposals_by_case'],
                          boxes=[boxes]*3,labels=arrays['case_labels'])
    raw_detector=dict(kind='world',preparation=dict(volume=cube.astype(float)[0],
        left_mask=np.ones(cube.shape[1:],dtype=bool),right_mask=np.zeros(cube.shape[1:],dtype=bool),
        spacing=[1,1,1],origin_zyx=[0,0,0],annotations_xyz_diameter=boxes))
    raw_cases=[dict(**raw_detector,proposals=np.column_stack(([3.,2.,1.],boxes)),case_label=label)
               for label in [1,0,1]]
    digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-dsb','training').execute(dict(
        raw_collections=dict(training_detector=[raw_detector],training_classifier=raw_cases,
            validation_detector=[raw_detector],validation_classifier=raw_cases,training_validation_classifier=raw_cases),
        eligible_image_indices=[0],
        training_arrays=None if multiple_epochs=='raw' else arrays,
        sampling_options={**options,'numpy_rng':np.random.RandomState(7),'label_rng':random.Random(9)},
        validation_factory=None if multiple_epochs in ('arrays','detector_init','raw') else lambda task:det if task=='detector' else cases,
        validation_arrays=dict(detector=dict(volumes=[cube],boxes=[boxes]),
            classifier_validation=validation_cases,classifier_training_validation=validation_cases),
        validation_options=dict(detector_batch_size=3,classifier_batch_size=2,topk=2,
            detector_crop_size=32,classifier_crop_size=16,numpy_rng=np.random.RandomState(4),label_rng=random.Random(4)),
        initial_classifier_checkpoint=None if multiple_epochs in ('detector_init','raw') else checkpoint_bytes(classifier,120),
        start_epoch=121,classifier_variant=4,
        **({'end_epoch':122} if multiple_epochs else {'epoch':121}),
        freeze_batchnorm=True,detector_checkpoint=checkpoint_bytes(detector,30),
        volume=volume,spacing=(6.,6.,6.),requested_spacing=(6.,6.,6.),topk=2,crop_size=16,tile_side=32,margin=0,
    ),cdg=restored))
    assert result['status']=='completed'
    order=[entry['node_id'] for entry in result['trace']]
    if multiple_epochs=='raw':
        assert order.index('raw_preparation')<order.index('training_batches')
        assert order.index('raw_preparation')<order.index('validation_batches')
    assert order.index('training_batches')<order.index('training')<order.index('models')<order.index('prediction')
    predicted=np.load(tmp_path/'training'/'prediction'/'out_case_probability.npy')
    assert predicted.shape==(1,) and np.isfinite(predicted).all()
