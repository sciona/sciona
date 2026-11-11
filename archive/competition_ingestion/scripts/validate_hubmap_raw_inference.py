"""Compare checkpoint-backed raw inference with original dataset/model/loop."""
import argparse
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import random
import signal
import cv2
import numpy as np
import torch
from sciona.hubmap_raw_inference import load_inference_model,predict_slide
from scripts.validate_hubmap_network import source_model,exact_state
from scripts.validate_hubmap_inference_tiles import source_tiles


def validate(root,source,library):
    torch.set_num_threads(1)
    original=source_model(root,source)
    torch.manual_seed(311)
    training_model=original((32,32),True,True,None,load_weights=False)
    # Synthetic state guarantees active segmentation, preventing a vacuous
    # all-shortcut end-to-end comparison with random classifier initialization.
    with torch.no_grad():
        training_model.clf[-1].weight.zero_();training_model.clf[-1].bias.fill_(1.)
    stream=io.BytesIO();torch.save(training_model.state_dict(),stream)
    payload=stream.getvalue();del training_model,stream;gc.collect()
    candidate=load_inference_model(payload,input_resolution=32)
    reference=original((32,32),False,False,.5,load_weights=False).eval()
    reference.load_state_dict(torch.load(io.BytesIO(payload),weights_only=True));del payload;gc.collect()
    exact_state(candidate.state_dict(),reference.state_dict())
    pins=json.loads((root/'docs/reviews/competition_hubmap_pseudo_pins.json').read_text())
    name='src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/utils_inference.py'
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    nodes=[n for n in ast.parse((source/name).read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in {'my_collate_fn','get_pred_mask'}]
    assert len(nodes)==2
    saved=random.getstate();cases=[]
    try:
        for h,w,res,inp,pad,batch,bias in [(35,37,32,32,8,4,1.),(17,19,1024,320,256,12,1.),(35,37,32,32,8,4,0.)]:
            with torch.no_grad():
                candidate.clf[-1].bias.fill_(bias);reference.clf[-1].bias.fill_(bias)
            image=np.random.RandomState(313).randint(0,256,(h,w,3)).astype(np.uint8)
            ds=source_tiles(root,source,library,image,resolution=res,input_resolution=inp,pad_size=pad)
            pr=random.Random(317);tg=torch.Generator().manual_seed(331)
            refg=torch.Generator().set_state(tg.get_state())
            logits=[[],[]];handles=[]
            for index,model in enumerate([candidate,reference]):
                def record(module,inputs,output,index=index):logits[index].append(output.detach().clone())
                handles.append(model.register_forward_hook(record))
            try:
                actual=predict_slide([candidate],image,python_rng=pr,torch_generator=tg,resolution=res,
                                     input_resolution=inp,pad_size=pad,batch_size=batch,tta=4,threshold=.5)
                random.seed(317)
                def loader(*args,**kwargs):
                    # Explicit generator is the declared current zero-worker
                    # execution boundary; all original batching remains.
                    return torch.utils.data.DataLoader(*args,**kwargs,generator=refg)
                ns=dict(np=np,cv2=cv2,torch=torch,DataLoader=loader,HuBMAPDataset=lambda *args:ds,
                        tqdm=lambda x:x,seed=0,device=torch.device('cpu'),
                        config=dict(test_batch_size=batch,tta=4,mask_threshold=.5))
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-raw-inference>','exec'),ns)
                with contextlib.redirect_stdout(io.StringIO()):expected,eh,ew=ns['get_pred_mask'](0,None,[[reference]])
                np.testing.assert_array_equal(actual,expected)
                assert actual.shape==(eh,ew)
                assert random.getstate()==pr.getstate()
                assert torch.equal(tg.get_state(),refg.get_state())
                assert len(logits[0])==len(logits[1])
                for a,b in zip(*logits):torch.testing.assert_close(a,b,rtol=0,atol=0)
                if bias==0:assert not actual.any()
                cases.append(dict(resolution=res,input_resolution=inp,tiles=len(ds),batch_size=batch,
                                  classifier_logit=bias,model_calls=len(logits[0]),exact_mask_logits_rng=True))
            finally:
                for handle in handles:handle.remove()
        exact_state(candidate.state_dict(),reference.state_dict())
    finally:random.setstate(saved)
    paths=['sciona/hubmap_raw_inference.py','sciona/hubmap_inference.py','sciona/hubmap_inference_tiles.py',
           'sciona/hubmap_inference_network.py','sciona/hubmap_checkpoint.py','sciona/hubmap_network.py','sciona/hubmap_encoder.py',
           'scripts/validate_hubmap_raw_inference.py','scripts/validate_hubmap_network.py',
           'scripts/validate_hubmap_inference_tiles.py','scripts/validate_hubmap_augmentation.py',
           'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_pseudo_pins.json',
           'docs/reviews/competition_hubmap_augmentation_pins.json']
    return dict(approved=False,synthetic_only=True,exact_training_checkpoint_to_inference_state=True,cases=cases,
                exact_final_buffers=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Decoded synthetic RGB arrays and synthetic full-network checkpoint; no learned accuracy claim.',
                             'Original dataset, transforms, full model and inference loop; only raster IO and explicit loader generator adapted.',
                             'Current CPU float32 execution; historical GPU/codec behavior excluded.',
                             'Single full-network checkpoint here; multi-model flip ordering independently checked with asymmetric witnesses.',
                             'Pseudo-label preparation/retraining, final notebook provenance and full publication gates remain.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_raw_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
