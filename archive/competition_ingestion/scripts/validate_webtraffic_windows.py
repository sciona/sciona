"""Synthetic source-method checks; NumPy TF-operation adapter, not a TF runtime test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sciona import webtraffic_windows as candidate


class Operations:
    """Only the primitive tensor operations used by the source methods below."""
    int32 = np.int32
    concat = staticmethod(np.concatenate)
    fill = staticmethod(lambda shape, value: np.full(shape,value,dtype=np.float32))
    cast = staticmethod(lambda x,dtype: x.astype(dtype))
    maximum = staticmethod(np.maximum)
    gather = staticmethod(lambda x,index: x[index])
    zeros_like = staticmethod(np.zeros_like)
    where = staticmethod(np.where)
    is_nan = staticmethod(np.isnan)
    reduce_mean = staticmethod(np.mean)
    reduce_sum = staticmethod(np.sum)
    to_int32 = staticmethod(lambda x: x.astype(np.int32))
    equal = staticmethod(np.equal)
    sqrt = staticmethod(np.sqrt)
    squared_difference = staticmethod(lambda x,y: np.square(x-y))
    stack = staticmethod(np.stack)
    expand_dims = staticmethod(np.expand_dims)
    tile = staticmethod(np.tile)
    @staticmethod
    def split(values,sizes,axis=0):
        assert sum(sizes) == values.shape[axis]
        return np.split(values,np.cumsum(sizes)[:-1],axis=axis)
    def random_uniform(self, shape, low, high, dtype, seed):
        self.bounds = (low,high)
        assert low < high
        return low


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    path=source/'input_pipe.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['input_pipe.py']
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InputPipe')
    names={'cut','cut_train','cut_eval','reject_filter','make_features'}
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ops=Operations()
    class LegacyNumpy:
        NaN=np.nan
    ns=dict(tf=ops,np=LegacyNumpy())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-input-methods>','exec'),ns)
    counts=dict(cuts=0,feature_outputs=0,rejection_checks=0,training_bounds=0,evaluation_cuts=0)
    for seed in range(8):
        rng=np.random.RandomState(seed+915)
        days=800; train=283; predict=60; length=days+predict
        hits=rng.uniform(0,10,days).astype(np.float32)
        hits[::17]=np.nan; hits[1::19]=0
        dow=rng.uniform(-1,1,(length,2)).astype(np.float32)
        lagged=np.arange(length)[:,None]-np.array([91,182,273,365])[None,:]
        lagged=np.maximum(lagged,-1).astype(np.int16)
        obj=SimpleNamespace(train_window=train,predict_window=predict,verbose=False,
                            inp=SimpleNamespace(dow=dow,lagged_ix=lagged,data_days=days),
                            start_offset=13,back_offset=60,rand_seed=seed,max_predict_empty=0)
        obj.cut=lambda h,start,end:ns['cut'](obj,h,start,end)
        for start in [0,95,days-train-predict,days-train]:
            a=candidate.cut_window(hits,dow,lagged,start,train,predict)
            b=ns['cut'](obj,hits,start,start+train+predict)
            for av,bv in zip(a,b): np.testing.assert_equal(av,bv)
            counts['cuts']+=1
            assert not np.isnan(a[0]).any() and not np.isnan(a[3]).any()
            if start==days-train: assert np.isnan(a[1]).all()
            pf=[rng.normal(size=n).astype(np.float32) for n in [4,2,3]]
            args=(*a,*pf,'synthetic-page',np.float32(.7),np.float32(.2),np.float32(.4))
            actual=candidate.make_window_features(*args)
            expected=ns['make_features'](obj,*args)
            for av,bv in zip(actual,expected):
                np.testing.assert_equal(av,bv); counts['feature_outputs']+=1
            np.testing.assert_allclose(actual[2].mean(),0,atol=1e-6)
            np.testing.assert_allclose(np.mean(actual[2]**2),1,atol=1e-6)
            np.testing.assert_array_equal(actual[9][-3:],np.array([.7,.4,.2],dtype=np.float32))
            assert actual[1].shape==(train,19) and actual[5].shape==(predict,18)
            zero_count=int(np.sum(a[0]==0))
            for allowed in [zero_count-1,zero_count,zero_count+1]:
                obj.max_train_empty=allowed
                assert candidate.keep_window(a[0],allowed)==ns['reject_filter'](obj,*a)
                assert candidate.keep_window(a[0],allowed)==(allowed>=zero_count)
                counts['rejection_checks']+=1
        ns['cut_train'](obj,hits,'passthrough')
        assert ops.bounds==candidate.training_offset_bounds(days,train,predict,60,13)
        assert ops.bounds==(13,384)
        counts['training_bounds']+=1
        a=ns['cut_eval'](obj,hits,'passthrough')
        b=candidate.cut_window(hits,dow,lagged,13,train,predict)+('passthrough',)
        for av,bv in zip(a,b): np.testing.assert_equal(av,bv)
        counts['evaluation_cuts']+=1
    # The original has no standard-deviation floor; retain that behavior explicitly.
    args=(np.ones(train,dtype=np.float32),np.ones(predict,dtype=np.float32),
          dow[:train+predict],np.ones((train+predict,4),dtype=np.float32),
          *pf,'synthetic-constant',np.float32(0),np.float32(0),np.float32(0))
    with np.errstate(invalid='ignore',divide='ignore'):
        a=candidate.make_window_features(*args);b=ns['make_features'](obj,*args)
    for av,bv in zip(a,b):np.testing.assert_equal(av,bv)
    assert a[8]==0 and np.isnan(a[2]).all()
    paths=['sciona/webtraffic_windows.py','scripts/validate_webtraffic_windows.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Original source methods executed with an explicit NumPy tensor-operation adapter; TensorFlow is not installed.',
                             'No TensorFlow RNG sequence, tf.data scheduling, GRU training or model inference parity claim.',
                             'Float32 windows only; constant windows preserve source division-by-zero NaNs.',
                             'Source training offset upper bound subtracts start_offset and is exclusive; no correction applied.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_windows.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
