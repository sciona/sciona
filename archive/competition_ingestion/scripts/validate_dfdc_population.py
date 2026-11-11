"""Source population-selection oracle using entirely synthetic records."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_population import select_population


def assert_state(a,b):
    assert a[0]==b[0] and a[2:]==b[2:]
    np.testing.assert_array_equal(a[1],b[1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']=='training/datasets/classifier_dataset.py')
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_data','_oversample')]
    oracle=ast.ClassDef(name='Oracle',bases=[],keywords=[],body=methods,decorator_list=[])
    ast.fix_missing_locations(oracle)
    namespace={'np':np,'pd':pd}
    exec(compile(ast.Module(body=[oracle],type_ignores=[]),'<source-population>','exec'),namespace)
    labels=np.tile([0,1,1],8)
    folds=np.repeat([0,1,2],8)
    frame_numbers=np.arange(24)*10
    # Runtime-created synthetic placeholders only. No original membership module.
    df=pd.DataFrame({'position':np.arange(24),'video':[f'synthetic-{i}' for i in range(24)],
                     'label':labels,'fold':folds,'frame':frame_numbers})
    cases=0
    initial=np.random.get_state()
    try:
        for mode in ('train','val'):
            for rebalance in (False,True):
                for reduce_val in (False,True):
                    for epoch,seed in ((0,111),(1,777),(3,999)):
                        original=namespace['Oracle']()
                        original.df=df;original.mode=mode;original.fold=0
                        original.oversample_real=rebalance;original.reduce_val=reduce_val
                        with redirect_stdout(io.StringIO()):expected=original._prepare_data(epoch,seed)
                        expected_state=np.random.get_state()
                        np.random.set_state(initial)
                        actual=select_population(labels,folds,frame_numbers,mode=mode,epoch=epoch,
                                                 seed=seed,rebalance=rebalance,reduce_val=reduce_val)
                        np.testing.assert_array_equal(actual.indices,expected[:,0].astype(int))
                        assert_state(actual.numpy_state_after_shuffle,expected_state)
                        assert_state(np.random.get_state(),initial)
                        assert np.all(folds[actual.indices]!=0 if mode=='train' else folds[actual.indices]==0)
                        if mode=='train' and rebalance:
                            assert np.sum(labels[actual.indices]==0)==np.sum(labels[actual.indices]==1)
                        if mode=='val' and reduce_val:assert np.all(frame_numbers[actual.indices]%20==0)
                        # Compare continuation into augmentation's NumPy stream.
                        a,b=np.random.RandomState(),np.random.RandomState()
                        a.set_state(actual.numpy_state_after_shuffle);b.set_state(expected_state)
                        np.testing.assert_array_equal(a.uniform(size=12),b.uniform(size=12))
                        cases+=1
    finally:np.random.set_state(initial)
    # Independent seed-0 permutation for a small non-rebalanced population.
    got=select_population([0,1,0,1],[1,1,1,1],[0,1,2,3],mode='train',epoch=0,seed=0,rebalance=False)
    assert got.indices.tolist()==[2,3,1,0]
    empty=select_population([0,1],[0,0],[0,10],mode='train',epoch=0,seed=1)
    assert empty.indices.size==0
    rejections=0
    for kwargs in [dict(labels=[0,0,1],folds=[1,1,1],frame_numbers=[0,1,2],mode='train',epoch=0,seed=1),
                   dict(labels=[0,1],folds=[1,1],frame_numbers=[0,1],mode='train',epoch=1,seed=2**31),
                   dict(labels=[0,2],folds=[1,1],frame_numbers=[0,1],mode='train',epoch=0,seed=1)]:
        try:select_population(**kwargs)
        except ValueError:rejections+=1
        else:raise AssertionError('invalid population accepted')
    files=['sciona/dfdc_population.py','scripts/validate_dfdc_population.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-population-validation.v1','result':'passed','source_commit':pins['commit'],
            'checks':{'source_selection_and_rng_comparisons':cases,'independent_permutation':True,
                      'empty_selection_retained':True,'invalid_population_rejections':rejections},
            'semantics':['Training selects other folds and downsamples fake examples without replacement to match the real count.',
                         'Validation retains its fold; optional frame-number modulo20 thinning follows class concatenation.',
                         'Effective seed is (epoch+1)*seed; pandas downsampling and final shuffle use separate identically seeded streams.',
                         'Source post-shuffle RNG state returned for augmentation continuation without mutating caller RNG.'],
            'limits':'Synthetic positional arrays only; caller identities must be valid/nonmissing. Does not generate source grouped folds, prepare images, or execute full training. Original validation-membership module excluded.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_population.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
