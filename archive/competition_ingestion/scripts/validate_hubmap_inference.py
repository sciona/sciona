"""Execute pinned pseudo-label inference with synthetic normalized tile inputs."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from sciona.hubmap_inference import predict_tiles


class SyntheticModel(torch.nn.Module):
    def __init__(self,offset):
        super().__init__()
        self.offset=offset
        self.calls=[]

    def forward(self,x):
        assert not torch.is_grad_enabled()
        self.calls.append(x.clone())
        # Asymmetric, spatially varying output makes flipped-input and reversed
        # prediction mistakes visible; these are synthetic witnesses, not models
        # offered as substitutes for the original full network.
        ramp=torch.linspace(-.8,.7,x.shape[-1])[None,None,None,:]
        return x[:,0:1].square()*.3-x[:,1:2]*.4+ramp+self.offset


def validate(root,source):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_hubmap_pseudo_pins.json').read_text())
    name='src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/utils_inference.py'
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    nodes=[n for n in ast.parse((source/name).read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in {'my_collate_fn','get_pred_mask'}]
    assert len(nodes)==2
    cases=0
    for height,width,resolution,pad in [(12,12,8,2),(11,13,8,2),(7,9,8,0),(513,519,1024,256)]:
        side=resolution-2*pad;nh=height//side+1;nw=width//side+1
        images=torch.from_numpy(np.random.RandomState(277).normal(size=(nh*nw,3,16,16)).astype(np.float32))
        interiors=[];windows=[]
        for i in range(nh*nw):
            y=i//nw*side;x=i%nw*side
            interiors.append([y,min(y+side,height),x,min(x+side,width)])
            windows.append([max(0,y-pad),min(y+side+pad,height),max(0,x-pad),min(x+side+pad,width)])
        class Tiles(torch.utils.data.Dataset):
            def __init__(self,*args):
                self.pred_sz=side;self.sz=resolution;self.num_h=nh;self.num_w=nw;self.h=height;self.w=width
            def __len__(self):return len(images)
            def __getitem__(self,i):return dict(img=images[i],p=interiors[i],q=windows[i])
        for tta in [1,2,3,4,6]:
            for threshold in [.3,.5,.7]:
                actual_models=[SyntheticModel(-.2).eval(),SyntheticModel(.15).eval()]
                reference_models=[SyntheticModel(-.2).eval(),SyntheticModel(.15).eval()]
                ns=dict(np=np,cv2=cv2,torch=torch,DataLoader=torch.utils.data.DataLoader,HuBMAPDataset=Tiles,
                        tqdm=lambda x:x,seed=0,device=torch.device('cpu'),
                        config=dict(test_batch_size=3,tta=tta,mask_threshold=threshold))
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-pseudo-inference>','exec'),ns)
                with contextlib.redirect_stdout(io.StringIO()):
                    expected,h,w=ns['get_pred_mask'](0,None,[reference_models])
                actual=predict_tiles(actual_models,images,interiors,windows,height=height,width=width,
                                     resolution=resolution,pad_size=pad,batch_size=3,tta=tta,threshold=threshold)
                assert (h,w)==actual.shape
                np.testing.assert_array_equal(actual,expected)
                for a,b in zip(actual_models,reference_models):
                    assert len(a.calls)==len(b.calls)
                    for x,y in zip(a.calls,b.calls):assert torch.equal(x,y)
                cases+=1
    paths=['sciona/hubmap_inference.py','scripts/validate_hubmap_inference.py',
           'docs/reviews/competition_hubmap_pseudo_pins.json']
    return dict(approved=False,synthetic_only=True,exact_stitched_mask_and_model_call_cases=cases,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Original inference loop executed; synthetic tile dataset and asymmetric synthetic models isolate TTA/resize/threshold/stitching.',
                             'Raw inference dataset/normalization and full-network classifier shortcut need independent checks.',
                             'No trained predictive accuracy, historical GPU fidelity or full CDG approval claimed.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
