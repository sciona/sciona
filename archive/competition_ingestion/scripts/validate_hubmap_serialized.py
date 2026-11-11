"""Full synthetic lifecycle execution over source-validated numerical stages."""
import argparse
import asyncio
import tempfile
import io
import gc
import hashlib
import json
from pathlib import Path
import random
import signal
from unittest.mock import patch
import numpy as np
import torch
from sciona.hubmap_encoder import build_encoder
from sciona.hubmap_lifecycle import FoldStage,train_fold
from sciona.hubmap_execution import build_hubmap_lifecycle_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from scripts.hubmap_lifecycle_reference import prepare,train,infer
from scripts.validate_hubmap_network import exact_state
from sciona.hubmap_raw_inference import predict_slide


def validate(root,source,library,notebook):
    torch.set_num_threads(1);torch.manual_seed(367)
    encoder=build_encoder();state=encoder.state_dict();del encoder
    slides=[]
    for i in range(3):
        image=np.random.RandomState(373+i).randint(0,256,(35,37,3)).astype(np.uint8)
        mask=np.zeros((35,37),dtype=np.uint8)
        if i!=1:mask[::2]=1
        slides.append((image,mask))
    def stage(seed):
        options=dict(start_epoch=18,end_epoch=18,maximum_bin=4,training_batch_size=2,
                     validation_batch_size=2,input_side=32,python_rng=random.Random(seed),
                     numpy_rng=np.random.RandomState(seed+1),torch_generator=torch.Generator().manual_seed(seed+2))
        return FoldStage(state,seed,['synthetic-valid'],options)
    initial=stage(0);retraining=stage(1)
    pseudo=np.random.RandomState(379).randint(0,256,(35,37,3)).astype(np.uint8)
    final=np.random.RandomState(383).randint(0,256,(35,37,3)).astype(np.uint8)
    calls=[];masks=[];checkpoints=[]
    def observed_train(*args,**kwargs):
        result=train_fold(*args,**kwargs)
        checkpoints.append(result['checkpoints']['best_loss'])
        return result
    def observed(models,image,**options):
        result=predict_slide(models,image,**options)
        masks.append(result.copy())
        calls.append(dict(tta=options['tta'],models=len(models),foreground_pixels=int(result.sum())))
        return result
    pseudo_rng=dict(python_rng=random.Random(389),torch_generator=torch.Generator().manual_seed(397))
    final_rng=dict(python_rng=random.Random(401),torch_generator=torch.Generator().manual_seed(409))
    runner._ensure_atoms_imported()
    graph=build_hubmap_lifecycle_graph();digest,node_rows,edge_rows=encode_execution_graph(graph)
    graph=decode_execution_graph(node_rows,edge_rows,digest)
    capture={}
    def capture_value(directory,node,name,value):
        # Runner computation is unchanged; capture synthetic outputs in memory
        # instead of dumping full checkpoints and model representations to disk.
        if name.startswith('out_'):capture[(node,name[4:])]=value
    def execute_graph(slides,groups,pseudo_sources,final_images,**options):
        inputs=dict(slides=slides,groups=groups,pseudo_sources=pseudo_sources,final_images=final_images,**options)
        inputs['input_resolution']=options['inference_geometry']['input_resolution']
        with tempfile.TemporaryDirectory(prefix='sciona-synthetic-hubmap-graph-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture_value):
                output=asyncio.run(runner.CDGExecutionSession(None,'synthetic-hubmap','case').execute(inputs,cdg=graph))
        assert output['status']=='completed',output.get('error','Graph did not complete')
        checkpoints.extend(capture[('initial_training','initial_checkpoints')])
        checkpoints.extend(capture[('retraining','final_checkpoints')])
        for collection in capture[('pseudo_labels','pseudo_labeled_sources')]:
            for image,mask in collection:
                masks.append(mask)
                calls.append(dict(tta=4,models=len(capture[('initial_models','models')]),foreground_pixels=int(mask.sum())))
        for mask in capture[('final_prediction','predictions')]:
            masks.append(mask)
            calls.append(dict(tta=3,models=len(capture[('final_models','models')]),foreground_pixels=int(mask.sum())))
        result=dict(initial_epochs=capture[('initial_training','initial_metrics')],
                    retraining_epochs=capture[('retraining','retraining_metrics')],
                    checkpoints=capture[('retraining','final_checkpoints')],predictions=capture[('final_prediction','predictions')])
        capture.clear();gc.collect()
        return result
    result=execute_graph(slides,['synthetic-train-a','synthetic-train-b','synthetic-valid'],[[pseudo]],[final],
            initial_stages=[initial],retraining_stages=[retraining],initial_selectors=[(0,'best_loss')],
            final_selectors=[(0,'best_loss')],tile_size=32,
            pseudo_rng=pseudo_rng,
            final_rng=final_rng,
            inference_geometry=dict(resolution=32,input_resolution=32,pad_size=8,batch_size=4))
    assert [call['tta'] for call in calls]==[4,3]
    assert len(result['checkpoints'])==len(result['predictions'])==1
    assert result['predictions'][0].shape==final.shape[:2]
    assert result['predictions'][0].dtype==np.uint8
    assert np.isin(result['predictions'][0],[0,1]).all()
    assert len(result['initial_epochs'])==len(result['retraining_epochs'])==1
    # The reference begins from raw arrays and encoder state, not candidate
    # prepared populations or checkpoints. Every phase consumes its own output.
    reference_frame=prepare(root,source,slides,['synthetic-train-a','synthetic-train-b','synthetic-valid'])
    ref_initial=train(root,source,library,reference_frame,state,0)
    assert result['initial_epochs'][0]==ref_initial['epochs']
    def compare_stage(reference,stage,payload):
        exact_state(torch.load(io.BytesIO(payload),weights_only=True),torch.load(io.BytesIO(reference['checkpoint']),weights_only=True))
        assert reference['python_state']==stage.options['python_rng'].getstate()
        a=reference['numpy_state'];b=stage.options['numpy_rng'].get_state()
        assert a[0]==b[0] and a[2:]==b[2:];np.testing.assert_array_equal(a[1],b[1])
        assert torch.equal(reference['torch_state'],stage.options['torch_generator'].get_state())
    compare_stage(ref_initial,initial,checkpoints[0])
    ref_pseudo=infer(root,source,library,notebook,ref_initial['checkpoint'],pseudo,389,397)
    np.testing.assert_array_equal(ref_pseudo['mask'],masks[0])
    assert ref_pseudo['mask'].any()
    assert ref_pseudo['python_state']==pseudo_rng['python_rng'].getstate()
    assert torch.equal(ref_pseudo['torch_state'],pseudo_rng['torch_generator'].get_state())
    del ref_initial;gc.collect()
    pseudo_frame=prepare(root,source,[(pseudo,ref_pseudo['mask'])],['synthetic-pseudo'])
    ref_retrained=train(root,source,library,reference_frame,state,1,pseudo_frame)
    assert result['retraining_epochs'][0]==ref_retrained['epochs']
    compare_stage(ref_retrained,retraining,checkpoints[1])
    ref_final=infer(root,source,library,notebook,ref_retrained['checkpoint'],final,401,409,True)
    np.testing.assert_array_equal(ref_final['mask'],result['predictions'][0])
    assert ref_final['python_state']==final_rng['python_rng'].getstate()
    assert torch.equal(ref_final['torch_state'],final_rng['torch_generator'].get_state())
    paths=sorted(root.glob('sciona/hubmap_*.py'))+sorted(root.glob('scripts/validate_hubmap_*.py'))+[
        root/'scripts/hubmap_lifecycle_reference.py']+sorted(root.glob('docs/reviews/competition_hubmap_*pins.json'))
    return dict(approved=False,synthetic_only=True,independent_raw_lifecycle_reference=True,serialized_graph_sha256=digest,graph_nodes=len(graph.nodes),graph_edges=len(graph.edges),
                exact_both_stage_metrics_checkpoints_rng=True,exact_generated_pseudo_and_final_masks=True,
                inference_calls=calls,retained_checkpoints=len(result['checkpoints']),
                implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                provider_sha256=hashlib.sha256((root.parent/'sciona-atoms-dl/src/sciona/atoms/dl/segmentation/hubmap_execution.py').read_bytes()).hexdigest(),
                limitations=['One fold and one epoch per training stage, synthetic raw arrays and encoder state; no learned quality claim.',
                             'Reference executes original raw tiler/JPEG writer/selection/fold/sampling/augmentation/training and original pseudo/final inference.',
                             'Decoded raster array boundary and explicit zero-worker CPU generators; historical codecs/GPU execution excluded.',
                             'Ordered multi-fold routing has separate witnesses; Provider registration/lineage publication gates remain; graph ran through normal runner with intermediate persistence replaced by capture.'])


if __name__=='__main__':
    signal.alarm(180)
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args()
    report=validate(root,args.source_root,args.library_root,args.notebook)
    (root/'docs/reviews/competition_hubmap_serialized.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
