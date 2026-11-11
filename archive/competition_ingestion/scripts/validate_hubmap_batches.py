"""Consecutive source balancing/shuffle/augmentation comparisons on synthetic IO."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_batches import training_batches
from scripts.validate_hubmap_loader import source_dataset,encode_synthetic,assert_rng
from scripts.validate_hubmap_sampling_schedule import source_functions


def validate(root,source,library):
    sample,_=source_functions(root,source)
    images=[np.random.RandomState(i+199).randint(0,256,(64,64,3)).astype(np.uint8) for i in range(8)]
    present=np.array([False]*3+[True]*5);bins=np.array([0,0,0,1,2,3,4,5])
    rles=[encode_synthetic(np.ones((64,64),dtype=np.uint8) if p else np.zeros((64,64),dtype=np.uint8)) for p in present]
    reports=[];saved_py=random.getstate();saved_np=np.random.get_state()
    try:
        for seed in range(8):
            pr=random.Random(seed);nr=np.random.RandomState(seed+7);tg=torch.Generator().manual_seed(seed+11)
            frame=pd.DataFrame(dict(is_masked=present,binned=bins,_position=np.arange(8)))
            for epoch in range(2):
                random.setstate(pr.getstate());np.random.set_state(nr.get_state());refg=torch.Generator().set_state(tg.get_state())
                order=sample(frame,dict(binned_max=4))
                dataset=source_dataset(root,source,library,[images[i] for i in order],[rles[i] for i in order],32,True)
                loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=0,generator=refg)
                expected=list(loader);py=random.getstate();ns=np.random.get_state()
                actual=training_batches(images,rles,present,bins,maximum_bin=4,batch_size=2,input_side=32,
                    python_rng=pr,numpy_rng=nr,torch_generator=tg)
                assert actual['population_size']==len(order) and len(actual['batches'])==len(expected)
                for a,b in zip(actual['batches'],expected):
                    for v,k in zip(a,['img','mask','label']):torch.testing.assert_close(v,b[k],rtol=0,atol=0)
                assert_rng(py,ns,pr,nr);assert torch.equal(tg.get_state(),refg.get_state())
                reports.append(dict(seed=seed,epoch=epoch+1,population=len(order),batches=len(expected),exact_tensors_all_rng=True))
    finally:
        random.setstate(saved_py);np.random.set_state(saved_np)
    paths=['sciona/hubmap_batches.py','sciona/hubmap_loader.py','sciona/hubmap_sampling.py','sciona/hubmap_augmentation.py',
        'scripts/validate_hubmap_batches.py','scripts/validate_hubmap_loader.py','scripts/validate_hubmap_sampling_schedule.py',
        'scripts/validate_hubmap_augmentation.py','docs/reviews/competition_hubmap_loader_pins.json',
        'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_augmentation_pins.json']
    return dict(approved=False,synthetic_only=True,cases=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Current PyTorch DataLoader with explicit CPU generator and zero workers; historical multiworker seeding excluded.',
            'Decoded synthetic BGR images and runtime RLE inputs only; no data/codec fidelity or CDG approval claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_batches.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=len(report['cases']),exact_tensors_all_rng=True)))
