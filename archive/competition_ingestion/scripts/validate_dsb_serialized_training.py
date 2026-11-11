"""Independent raw arrays -> serialized training checkpoint source comparison."""
import argparse
import asyncio
import hashlib
import io
import json
import random
import signal
import tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.dsb_source_preparation import SourcePreparation
from scripts.dsb_source_batches import SourceTrainingBatches,SourceValidationBatches
from scripts.dsb_source_lifecycle import SourceLifecycle
from scripts.dsb_source_inference import SourceInference
from scripts.validate_dsb_epoch import assert_state
from sciona.dsb_checkpoint import checkpoint_bytes
from sciona.dsb_network import DetectorNet
from sciona.dsb_execution import build_dsb_raw_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def source_collections(prepare,raw):
    result={}
    for key,records in raw.items():
        values=[prepare(record) for record in records]
        result[key]=dict(volumes=[v for v,_ in values],boxes=[b for _,b in values])
        if 'classifier' in key:
            result[key].update(proposals=[r['proposals'].copy() for r in records],labels=[r['case_label'] for r in records])
    return result


async def validate(root,source_root,include_inference=False):
    torch.set_num_threads(1);runner._ensure_atoms_imported()
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    prepare=SourcePreparation(source_root,pins)
    cube=np.full((48,48,48),100.,dtype=float)
    boxes=np.array([[24.,24.,24.,10.],[18.,18.,18.,12.],[30.,30.,30.,14.]])
    detector_record=dict(kind='world',preparation=dict(volume=cube,left_mask=np.ones(cube.shape,dtype=bool),
        right_mask=np.zeros(cube.shape,dtype=bool),spacing=[1.,1.,1.],origin_zyx=[0.,0.,0.],annotations_xyz_diameter=boxes))
    records=[dict(**detector_record,proposals=np.column_stack(([3.,2.,1.],boxes)),case_label=label) for label in [1,0,1]]
    raw=dict(training_detector=[detector_record],training_classifier=records,validation_detector=[detector_record],
        validation_classifier=records,training_validation_classifier=records)
    opts=dict(detector_order=[3,2,0,1],classifier_order=[2,0,1],detector_batch_size=3,classifier_batch_size=2,
        topk=2,detector_crop_size=32,classifier_crop_size=16)
    val_opts={k:v for k,v in opts.items() if not k.endswith('_order')}
    graph=build_dsb_raw_training_graph();digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    inference=SourceInference(source_root,pins) if include_inference else None
    z,y,x=np.indices((24,64,64));inference_volume=np.full(z.shape,50.)
    inference_volume[((z-12)/9)**2+((y-32)/17)**2+((x-20)/9)**2<1]=-900.
    inference_options=dict(topk=2,crop_size=16,tile_side=32,margin=0,tile_batch_size=1)
    reports=[]
    for variant,start,end,freeze in [(3,49,51,False),(4,159,161,True)]:
        source=SourceLifecycle(source_root,pins,variant)
        torch.manual_seed(83);detector=DetectorNet()
        if include_inference:
            with torch.no_grad():
                detector.output[-1].weight.zero_();detector.output[-1].bias.zero_()
                detector.output[-1].bias[::5]=2. if variant==3 else -4.
        payload=checkpoint_bytes(detector,100)
        torch.manual_seed(89);initial_rng=torch.get_rng_state()
        capture={};save=runner.save_intermediate_value;previous_dir=runner.RUNS_DIR
        def save_value(directory,node,name,value):
            if node=='training' and name in ['out_classifier_checkpoint','out_metrics']:capture[name]=value
            if node=='prediction' and name in ['out_case_probability','out_proposal_probabilities']:capture[name]=value.copy()
            return save(directory,node,name,value)
        with tempfile.TemporaryDirectory(prefix='sciona-dsb-source-training-') as temporary:
            runner.RUNS_DIR=Path(temporary);runner.save_intermediate_value=save_value
            try:
                inputs=dict(raw_collections=raw,eligible_image_indices=[0],detector_checkpoint=payload,
                    start_epoch=start,end_epoch=end,classifier_variant=variant,freeze_batchnorm=freeze,topk=2,
                    sampling_options={**opts,'numpy_rng':np.random.RandomState(7),'label_rng':random.Random(9)},
                    validation_options={**val_opts,'numpy_rng':np.random.RandomState(4),'label_rng':random.Random(4)})
                if include_inference:
                    inputs.update(volume=inference_volume,spacing=[6.,6.,6.],requested_spacing=[6.,6.,6.],**inference_options)
                result=await runner.CDGExecutionSession(None,'synthetic-source-training','case').execute(
                    inputs,target_node_id=None if include_inference else 'training',cdg=graph)
            finally:runner.RUNS_DIR=previous_dir;runner.save_intermediate_value=save
        assert result['status']=='completed'
        actual_rng=torch.get_rng_state()
        actual=torch.load(io.BytesIO(capture['out_classifier_checkpoint']),weights_only=True)
        prepared=source_collections(prepare,raw)
        validation_arrays=dict(detector=prepared['validation_detector'],classifier_validation=prepared['validation_classifier'],
            classifier_training_validation=prepared['training_validation_classifier'])
        source_val=SourceValidationBatches(source_root,pins,validation_arrays,
            {**val_opts,'numpy_rng':np.random.RandomState(4),'label_rng':random.Random(4)})
        annotation_loader=SourceValidationBatches(source_root,pins,dict(detector=prepared['training_detector'],
            classifier_validation=prepared['training_classifier'],classifier_training_validation=prepared['training_classifier']),
            {**val_opts,'numpy_rng':np.random.RandomState(0),'label_rng':random.Random(0)})
        training_arrays=annotation_loader.loaders['classifier_validation'].arrays
        source_train=SourceTrainingBatches(source_root,pins,training_arrays,
            {**opts,'numpy_rng':np.random.RandomState(7),'label_rng':random.Random(9)})
        torch.set_rng_state(initial_rng)
        initialized=source.adapt(detector.state_dict(),2)
        model,optimizers=source.restore(initialized.state_dict(),2)
        validations=source.run(model,optimizers,source_train,source_val,start=start,end=end,freeze=freeze)
        assert actual['epoch']==end;assert_state(actual['state_dict'],model.state_dict())
        proposal_count=None
        if include_inference:
            processed=inference.preprocess(inference_volume,[6.,6.,6.],[6.,6.,6.])
            predicted=inference.predict(processed,detector.state_dict(),model.state_dict(),**inference_options)
            for name in ['case_probability','proposal_probabilities']:
                np.testing.assert_array_equal(capture['out_'+name],predicted[name])
            proposal_count=len(predicted['proposals'])
            assert (proposal_count>0)==(variant==3)
        assert_state(actual_rng,torch.get_rng_state())
        actual_validations=[v for epoch in capture['out_metrics'] for v in epoch['validation']]
        assert len(actual_validations)==len(validations)
        for value,(task,expected) in zip(actual_validations,validations):
            assert value['task']==task;metrics=value['metrics']
            if task=='detector':
                np.testing.assert_array_equal(metrics['mean_losses'],expected[:,:6].mean(axis=0))
                for name,index in [('positive_correct',6),('positive_count',7),('negative_correct',8),('negative_count',9)]:
                    assert metrics[name]==expected[:,index].sum()
            else:
                for name,key in [('classification_loss','mean_loss2'),('miss_loss','mean_missloss'),('accuracy','mean_acc'),
                    ('true_positives','tpn'),('false_positives','fpn'),('false_negatives','fnn')]:assert metrics[name]==expected[key]
        reports.append(dict(variant=variant,start_epoch=start,end_epoch=end,exact_checkpoint_rng_validation=True,
                            validation_passes=len(validations),executed_nodes=len(result['trace']),
                            exact_final_inference=include_inference,proposal_count=proposal_count))
    paths=sorted(root.glob('sciona/dsb_*.py'))+sorted(root.glob('scripts/dsb_source_*.py'))+[Path(__file__)]+[
        root/p for p in ['scripts/audit_competition_dsb_semantics.py','scripts/validate_dsb_validation_samples.py',
            'scripts/validate_dsb_epoch.py','scripts/validate_dsb_validation.py','scripts/validate_dsb_network.py',
            'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py','docs/reviews/competition_dsb_source_pins.json']]
    providers=sorted((root.parent/'sciona-atoms-dl/src/sciona/atoms/dl/detection').glob('dsb_*.py'))
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],execution_graph_sha256=digest,cases=reports,
        limitations=([] if include_inference else ['Comparison stops at serialized training output; final inference comparison remains.'])+[
            'World-coordinate raw preparation exercised; voxel preparation has separate source comparisons.',
            'Explicit RNG, sample orders and synthetic geometry; no historical runtime or accuracy claim.'],
        implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        provider_sha256={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers})


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--include-inference',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=asyncio.run(validate(root,args.source_root,args.include_inference))
    name='competition_dsb_full_source_execution.json' if args.include_inference else 'competition_dsb_serialized_training.json'
    (root/'docs/reviews'/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':result['cases'],'approved':False},indent=2))
