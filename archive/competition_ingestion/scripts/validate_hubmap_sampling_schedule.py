"""Exact pinned-source sampling order/RNG and scheduler transition comparisons."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_sampling import balanced_epoch_indices
from sciona.hubmap_schedule import CosineLR


def source_functions(root, source_root):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    for name in ['src/02_train/run.py','src/05_train_with_pseudo_labels/run.py','src/scheduler.py']:
        assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==pins['files'][name]
    assert (source_root/'src/02_train/run.py').read_bytes()==(source_root/'src/05_train_with_pseudo_labels/run.py').read_bytes()
    tree=ast.parse((source_root/'src/02_train/run.py').read_text())
    bodies=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.For) and ast.unparse(node.target)=='epoch':
            start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n).startswith("trn_df['binned'] ="))
            end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('train_dataset ='))
            bodies.append(node.body[start:end])
    assert len(bodies)==1
    fn=ast.parse('def source_sample(trn_df,config):\n    pass').body[0]
    fn.body=copy.deepcopy(bodies[0])+ast.parse("return trn_df_balanced['_position'].to_numpy()").body
    ns=dict(pd=pd)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])), '<pinned-sampling>', 'exec'),ns)
    nodes=[n for n in ast.parse((source_root/'src/scheduler.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CosineLR']
    assert len(nodes)==1
    schedule_ns=dict(np=np,_LRScheduler=torch.optim.lr_scheduler._LRScheduler)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-scheduler>','exec'),schedule_ns)
    return ns['source_sample'],schedule_ns['CosineLR']


def same_rng(a,b):
    assert a[0]==b[0] and a[2:]==b[2:]
    np.testing.assert_array_equal(a[1],b[1])


def validate(root,source_root):
    sample,scheduler=source_functions(root,source_root)
    sample_cases=0;unequal=0;schedule_steps=0
    old_rng=np.random.get_state()
    try:
        for negatives,positives in [(1,3),(5,10),(7,2),(11,12)]:
            present=np.array([False]*negatives+[True]*positives)
            bins=np.arange(len(present))%7
            for seed in range(12):
                rng=np.random.RandomState(seed)
                frame=pd.DataFrame(dict(is_masked=present,binned=bins,_position=np.arange(len(present))))
                for epoch in range(3):
                    np.random.set_state(rng.get_state())
                    expected=sample(frame,dict(binned_max=4))
                    expected_rng=np.random.get_state()
                    actual=balanced_epoch_indices(present,bins,maximum_bin=4,rng=rng)
                    np.testing.assert_array_equal(actual,expected)
                    same_rng(rng.get_state(),expected_rng)
                    sample_cases+=1
                    unequal+=int(present[actual].sum()!=len(actual)-present[actual].sum())
    finally:
        np.random.set_state(old_rng)
    for period in [3,19]:
        for multiplier in [1,2]:
            for initial in [-1,2]:
                optimizers=[torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))],lr=1e-4,betas=(.9,.999),weight_decay=1e-5) for _ in range(2)]
                schedulers=[cls(opt,step_size_min=1e-6,t0=period,tmult=multiplier,curr_epoch=initial) for cls,opt in zip([CosineLR,scheduler],optimizers)]
                for step in range(60):
                    assert schedulers[0].state_dict()==schedulers[1].state_dict()
                    assert optimizers[0].param_groups[0]['lr']==optimizers[1].param_groups[0]['lr']
                    for opt,s in zip(optimizers,schedulers):opt.step();s.step()
                    schedule_steps+=1
                assert schedulers[0].state_dict()==schedulers[1].state_dict()
    paths=['sciona/hubmap_sampling.py','sciona/hubmap_schedule.py','scripts/validate_hubmap_sampling_schedule.py',
           'tests/test_hubmap_sampling.py','docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,exact_sampling_order_and_rng_cases=sample_cases,
        observed_unequal_class_count_cases=unequal,exact_scheduler_transitions=schedule_steps,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Explicit integer bins and both populations required; degenerate populations reject before RNG changes.',
            'Sampling precedes separate Torch DataLoader shuffling and drop_last; neither is claimed by this comparison.',
            'Current PyTorch scheduler lifecycle, including constructor initial step; no historical environment claim.',
            'No whole training or CDG publication approval.'])


if __name__=='__main__':
    signal.alarm(90)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_sampling_schedule.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
