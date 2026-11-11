"""Pinned sampler decisions with synthetic integer image identities only."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_sampling import detector_sampling_table,select_detector_sample


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree=ast.parse(checked_source(source_root,pins,'training/classifier/data_detector.py').expandtabs(8))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Detector')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    # Select source table construction without file loading or metadata filters.
    begin=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
               and ast.unparse(n)=='self.bboxes = []')
    table_code=compile(ast.Module(body=init.body[begin:begin+3],type_ignores=[]),'<pinned-sampling-table>','exec')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    begin=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
               and ast.unparse(n)=='isRandomImg = False')
    flags=copy.deepcopy(method.body[begin:begin+2])
    selection=copy.deepcopy(method.body[begin+2].body[0])
    # Preserve source image/box lookup; stop immediately before crop computation.
    selection.body=selection.body[:4]
    selection.orelse=selection.orelse[:4]
    decision_code=compile(ast.Module(body=flags+[selection],type_ignores=[]),'<pinned-sampler-decision>','exec')
    labels=[np.array([[2.,3.,4.,6.],[4.,5.,6.,7.]]),np.empty((0,4)),np.array([[8.,9.,10.,41.]])]
    subject=SimpleNamespace()
    scope=dict(np=np,self=subject,labels=labels,sizelim=6.,sizelim2=30.,sizelim3=40.)
    exec(table_code,scope)
    table=detector_sampling_table(labels)
    np.testing.assert_array_equal(table,subject.bboxes)
    cases=0;branches=set();corrected_cases=0
    for eligible in [[0,1,2],[2]]:
        for seed in range(12):
            for index in [0,len(table)-1,len(table),len(table)+1,3*len(table)+2]:
                original_rng=np.random.RandomState(seed);candidate_rng=np.random.RandomState(seed)
                proxy=SimpleNamespace(random=original_rng,load=lambda identity:identity)
                subject=SimpleNamespace(phase='train',bboxes=table,filenames=[0,1,2],
                                        kagglenames=eligible,sample_bboxes=labels)
                scope=dict(np=proxy,self=subject,idx=index)
                exec(decision_code,scope)
                actual=select_detector_sample(index,table,eligible,image_count=3,rng=candidate_rng)
                assert actual['image_index']==scope['filename']
                assert actual['random_crop']==scope['isRandom']
                assert actual['independent_image']==bool(scope['isRandomImg'])
                if actual['independent_image']:
                    assert actual['target'] is None
                    if eligible==[2]:
                        assert not np.array_equal(scope['bboxes'],labels[actual['image_index']])
                        corrected_cases+=1
                    else:np.testing.assert_array_equal(scope['bboxes'],labels[actual['image_index']])
                else:np.testing.assert_array_equal(actual['target'],scope['bbox'][1:])
                np.testing.assert_array_equal(original_rng.rand(10),candidate_rng.rand(10))
                branches.add((actual['random_crop'],actual['independent_image']))
                cases+=1
    assert branches=={(False,False),(True,False),(True,True)} and corrected_cases>0
    paths=['sciona/dsb_sampling.py','sciona/dsb_training.py','tests/test_dsb_sampling.py','scripts/validate_dsb_sampling.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],
                source_table_exact_parity=True,exact_selection_and_rng_cases=cases,
                all_three_branches_exercised=True,corrected_source_box_association_cases=corrected_cases,
                corrections=['Explicit global image index fixes source filtered-index box lookup.',
                             'Integer epoch length floors source float expression; not a literal __len__ equivalence claim.'],
                limitations=['Source file/clock operations replaced with integer identity stubs and explicit RNG.',
                             'This validates selection and table construction, not trained model quality.'],
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_sampling_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
