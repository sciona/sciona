"""Compare sequence packing with pinned private source on synthetic inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_sequence import encode_series,prediction_slots,pack_neural
PINS={'S4_feature_combined.py':'aca38743b8264103736089292abca7962176bc61d848c7c1af6bb2f2b8084377','utils.py':'38291db3eca21f9e6eb8f87de166accbf8d178e68e2cdeaa8f0be3c158caaf70'}


def main(directory):
    if not __debug__:raise RuntimeError('Assertions required')
    nodes=[]
    for name,digest in PINS.items():
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source pin differs')
        nodes.extend(n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('pad_target','one_hot_encoding','TaskDataset'))
    assert len(nodes)==3
    ns={'__name__':'pinned_sequence_source','np':np,'pd':pd,'torch':torch,'__builtins__':{'__build_class__':__build_class__,'len':len,'list':list,'enumerate':enumerate,'print':lambda *a:None}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned sequence source>','exec'),ns)
    rng=np.random.default_rng(106);lengths=[1,3,13];xs=[rng.integers(-100,100,(n,2)).astype(float) for n in lengths];cs=[rng.integers(0,3,(n,1)).astype(float) for n in lengths];xs[1][0,0]=np.nan
    ours=encode_series(xs,cs)['sequences']
    df=pd.DataFrame(np.vstack(xs),columns=['a','b']);df['c']=np.vstack(cs)[:,0]
    source=ns['one_hot_encoding'](df,['c'],True)/100.;source=source.fillna(0)
    np.testing.assert_array_equal(np.vstack(ours),source.to_numpy())
    probs=[rng.uniform(size=n) for n in lengths]
    np.testing.assert_array_equal(prediction_slots(probs),np.asarray([ns['pad_target'](p) for p in probs]))
    features=rng.uniform(size=(3,5));features[0,0]=0
    source.insert(0,'id',np.repeat(np.arange(3),lengths));f=pd.DataFrame(features);f.insert(0,'id',np.arange(3))
    ends=np.cumsum(lengths);starts=np.r_[0,ends[:-1]]
    dataset=ns['TaskDataset'](source,f,list(zip(starts,ends-1,np.arange(3))))
    reference=dataset.collate_fn([dataset[i] for i in range(3)]);actual=pack_neural(ours,features)
    for ours_key,theirs in [('series','batch_series'),('mask','batch_mask'),('features','batch_feature')]:
        np.testing.assert_array_equal(actual[ours_key],reference[theirs].numpy())
    files=[ROOT/'sciona/amex_sequence.py',ROOT/'tests/test_amex_sequence.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_software_pins=PINS,
        checks=dict(joint_encoded_channels=True,prediction_left_padding=True,neural_right_padding=True,tabular_complement_only=True,sequence_lengths=[1,3,13]),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Synthetic encoding and packing comparison only; upstream row-level cross-fit probabilities and downstream neural training remain pending.',
            'Original code read privately by runtime argument; no source implementation or data copied into repository.'])
    (ROOT/'docs/reviews/competition_amex_sequence_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-directory',type=Path,required=True);main(p.parse_args().source_directory)
