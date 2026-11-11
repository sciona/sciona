"""Source population oracle and DataLoader assembly on synthetic prepared samples."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_dataset import PreparedDataset,make_loader
from sciona.dfdc_preparation import prepare_sample
from sciona.dfdc_transforms import create_train_transforms,create_val_transforms


class OracleDataset(Dataset):
    def __len__(self):return len(self.data)
    def __getitem__(self,index):
        row=self.records[int(self.data[index,0])]
        return prepare_sample(row['image'],row['label'],mode=self.mode,transforms=self.transforms,
            detector=self.detector,predictor=self.predictor,mask=row['mask'],landmarks=row['landmarks'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    p=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']=='training/datasets/classifier_dataset.py')
    cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef))
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_data','_oversample')]
    namespace={'np':np,'pd':pd}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-dataset-population>','exec'),namespace)
    for name in ('_prepare_data','_oversample'):setattr(OracleDataset,name,namespace[name])
    rng=np.random.default_rng(737)
    records=[{'image':rng.integers(1,256,(41,53,3),dtype=np.uint8),
              'label':int(i%3!=0),'fold':int(i>=30),'frame':i*10,
              'mask':np.full((41,53),255,dtype=np.uint8),'landmarks':None} for i in range(60)]
    df=pd.DataFrame({'position':range(60),'video':[f'synthetic-{i}' for i in range(60)],
                     'label':[r['label'] for r in records],'fold':[r['fold'] for r in records],
                     'frame':[r['frame'] for r in records]})
    detector=lambda image:[];predictor=lambda *args:None
    initial_py=random.getstate();initial_np=np.random.get_state();initial_torch=torch.random.get_rng_state()
    cases=0;total_batches=0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            for mode in ('train','val'):
                for batch_size in (6,12):
                    for epoch in (0,1):
                        original=OracleDataset();original.records=records;original.df=df
                        original.fold=0;original.mode=mode;original.oversample_real=True;original.reduce_val=True
                        original.detector=detector;original.predictor=predictor
                        original.transforms=(create_train_transforms if mode=='train' else create_val_transforms)(380)
                        adapted=PreparedDataset(records,mode=mode,detector=detector,predictor=predictor)
                        random.seed(738);np.random.seed(738);torch.manual_seed(738)
                        with redirect_stdout(io.StringIO()):original.data=original._prepare_data(epoch,777)
                        loader=DataLoader(original,batch_size=batch_size if mode=='train' else batch_size*2,
                            shuffle=mode=='train',drop_last=mode=='train',num_workers=0,pin_memory=False)
                        expected=list(loader);py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.random.get_rng_state()
                        random.seed(738);np.random.seed(738);torch.manual_seed(738)
                        adapted.reset(epoch,777)
                        np.testing.assert_array_equal(adapted.indices,original.data[:,0].astype(int))
                        actual=list(make_loader(adapted,batch_size=batch_size))
                        assert len(actual)==len(expected)
                        for a,b in zip(actual,expected):
                            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                        assert random.getstate()==py_state
                        state=np.random.get_state();assert state[0]==np_state[0] and state[2:]==np_state[2:]
                        np.testing.assert_array_equal(state[1],np_state[1]);assert torch.equal(torch.random.get_rng_state(),torch_state)
                        sizes=[len(batch['image']) for batch in actual]
                        if mode=='train':assert sizes==([6,6,6] if batch_size==6 else [12])
                        else:assert sizes==([12,3] if batch_size==6 else [15])
                        cases+=1;total_batches+=len(actual)
            # Preparation occurs on every access; repeated access consumes new augmentation.
            dataset=PreparedDataset(records,mode='train',detector=detector,predictor=predictor)
            dataset.reset(0,777);random.seed(740)
            a=dataset[0]['image'];b=dataset[0]['image']
            assert not torch.equal(a,b)
    finally:
        random.setstate(initial_py);np.random.set_state(initial_np);torch.random.set_rng_state(initial_torch)
    files=['sciona/dfdc_dataset.py','sciona/dfdc_population.py','sciona/dfdc_preparation.py','sciona/dfdc_transforms.py',
           'scripts/validate_dfdc_dataset.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-dataset-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_population_loader_cases':cases,'exact_prepared_batches':total_batches,
                 'python_numpy_torch_rng_continuation_exact':True,'independent_batch_sizes_and_tails':True,
                 'per_access_augmentation':True},
       'adaptations':['Default workers0 CPU execution; sourceworkers6/distributedsampler not covered.',
                      'Dataset.reset activates source postshuffleNumPy state; outer lifecycle must isolate all RNG streams.',
                      'Source populator AST plus independentlyvalidated preparation replaces source file-backedDataset.'],
       'limits':'Synthetic records and no-face hull stand-in; source population and loader batching compared with shared previouslyvalidated sample preparation. Actual dlib and B7 covered by separatecomponent reports. No multiworker, outer lifecycle or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
