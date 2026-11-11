"""Synthetic source sampler equivalence and private population boundary checks."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_population import prepare_records,balanced_indices,batches,check_split


def main():
    cache=Path('/private/tmp/sciona_cornell_source');path='src/dataloaders/imbalanced_dataset_sampler.py'
    raw=(cache/path).read_bytes();pins=json.loads((cache/'manifest.json').read_text())['pins']
    assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in pins if p['software_path']==path)
    ns=dict(torch=torch);exec(compile(ast.Module(body=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)],type_ignores=[]),'<source-sampler>','exec'),ns)
    raw_records=[dict(key=f'synthetic-{i}',waveform=np.linspace(-.2,.2,3200),sample_rate=32000,primary=c,secondary=[3]) for i,c in enumerate([0,0,0,1,2])]
    records=prepare_records(raw_records,training=True)
    class Dataset:
        def __len__(self):return len(records)
        def get_label(self,dataset,index):return records[index].primary
    for seed in range(16):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);expected=list(ns['ImbalancedDatasetSampler'](Dataset(),None))
        assert balanced_indices(records,seed)==expected
    train=list(batches(records,batch_size=2,training=True,seed=2,augmentation=lambda x,training:x.copy()))
    valid=list(batches(records,batch_size=2,training=False,seed=2))
    assert [len(b[0]) for b in train]==[2,2] and [len(b[0]) for b in valid]==[2,2,1]
    assert train[0][0].shape==(2,1,960000) and valid[0][0].shape==(2,2,960000)
    assert all((b[1][:,:,3]==1).all() and (b[2][:,:,3]==1).all() for b in train+valid)
    for record in raw_records:record['waveform'][:]=0
    assert any(records[0].waveform!=0)
    rejected=0
    for mutate in [lambda r:r[0].update(sample_rate=16000),lambda r:r[0].update(primary=None),lambda r:r[0].update(secondary=[0]),lambda r:r[1].update(key=r[0]['key'])]:
        values=[dict(r) for r in raw_records];mutate(values)
        try:prepare_records(values,training=True)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid records accepted')
    try:check_split(records,records)
    except ValueError:rejected+=1
    else:raise AssertionError('Overlapping split accepted')
    report=dict(status='passed',source_sampler_seeds=16,exact_source_indices=True,train_drop_last=True,validation_keeps_last=True,labels_and_secondary_preserved=True,private_inputs_copied=True,invalid_cases_rejected=rejected,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_population.py').read_bytes()).hexdigest(),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Synthetic populations and exact source weighted sampling; caller supplied fold membership and already-resampled audio. Full ensemble orchestration remains separate.')
    (ROOT/'docs/reviews/competition_cornell_population_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
