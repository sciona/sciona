"""Synthetic feeder, repeat/filter/batch and source-method window assembly checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sciona.webtraffic_assembly import assemble_features
from sciona.webtraffic_batches import feeder_arrays,fake_split_indices,window_batches
from scripts.validate_webtraffic_windows import Operations


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name in ['feeder.py','input_pipe.py']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    tree=ast.parse((source/'input_pipe.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InputPipe')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'cut','make_features','reject_filter'}]
    ns=dict(tf=Operations(),np=SimpleNamespace(NaN=np.nan))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-window-methods>','exec'),ns)
    counts=dict(batch_cases=0,source_record_comparisons=0,prefix_permutations=0,rejections=0)
    rng=np.random.RandomState(418)
    pages=sorted(f'SyntheticBatch{i}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org'])
        for agent in ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents'])
    values=rng.poisson(np.arange(2,18)[:,None],size=(16,800)).astype(np.float64)
    values[0]=0;values[1,::7]=np.nan
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range('2000-01-01',periods=800))
    tensors,plain=assemble_features(frame,add_days=63);arrays=feeder_arrays(tensors)
    assert arrays['hits'].dtype==np.float32 and arrays['lagged_ix'].dtype==np.int16
    indices=rng.permutation(16)
    for sampling in [0.,.25,.5,1.]:
        n=round(16*sampling);perm=rng.permutation(n)
        result,sampled,reported=fake_split_indices(16,perm,sampling)
        np.testing.assert_array_equal(result,perm);assert sampled==n and reported==16
        counts['prefix_permutations']+=1
    for training in [False,True]:
        for completeness in [0.,.9,1.]:
            starts=rng.randint(0,800-283-63,size=32).tolist()
            with np.errstate(invalid='ignore',divide='ignore'):
                batches=list(window_batches(tensors,indices,starts,data_days=800,epochs=2,batch_size=7,
                                            training=training,completeness=completeness))
                obj=SimpleNamespace(train_window=283,predict_window=63,verbose=False,
                                    max_train_empty=round(283*(1-completeness)),
                                    inp=SimpleNamespace(dow=arrays['dow'],lagged_ix=arrays['lagged_ix']))
                expected=[]
                for i,index in enumerate(np.tile(indices,2)):
                    start=starts[i] if training else 800-283
                    cut=ns['cut'](obj,arrays['hits'][index],start,start+346)
                    if not ns['reject_filter'](obj,*cut):continue
                    record=ns['make_features'](obj,*cut,*[arrays[k][index] for k in
                        ['pf_agent','pf_country','pf_site','page_ix','page_popularity','year_autocorr','quarter_autocorr']])
                    expected.append(record)
            actual=[tuple(column[row] for column in batch) for batch in batches for row in range(len(batch[0]))]
            assert len(actual)==len(expected)
            for a,b in zip(actual,expected):
                for av,bv in zip(a,b):np.testing.assert_equal(av,bv)
                counts['source_record_comparisons']+=1
            assert all(len(b[0])==7 for b in batches[:-1])
            if batches:assert len(batches[-1][0])==len(expected)%7 or len(batches[-1][0])==7
            counts['batch_cases']+=1
    for starts in [[],[-1]*16,[800]*16]:
        try:list(window_batches(tensors,indices,starts,data_days=800,epochs=1))
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Bad offset stream accepted')
    paths=['sciona/webtraffic_batches.py','scripts/validate_webtraffic_batches.py','sciona/webtraffic_windows.py',
           'sciona/webtraffic_assembly.py','sciona/webtraffic_features.py','sciona/webtraffic_calendar.py',
           'sciona/webtraffic_pages.py','scripts/validate_webtraffic_windows.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Explicit permutations and offsets; no TensorFlow random sequence or tf.data prefetch parity.',
                             'Original numerical methods executed with NumPy tensor-operation adapter.',
                             'FakeSplitter reports full train size even for sampled prefix; preserved and not used to claim sampled-route completion.',
                             'Full population-to-training/checkpoint lifecycle integration remains.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_batches.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
