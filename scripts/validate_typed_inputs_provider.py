"""Qualify explicit numeric casting and collision-free structured label encoding."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.ml.pipeline import typed_inputs as provider

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    cases=[]
    # Values and labels are constructed fixtures for two independent uses.
    for application,base in [('motion',[[.1,-2.5],[7.25,1e-40]]),('power',[[230.1,.3],[12.5,-.75]])]:
        for source in ('float32','float64'):
            values=np.array(base,dtype=source)[:,::-1]
            original=values.copy()
            for target in ('float32','float64'):
                actual=provider.cast_finite_array(values,target)
                np.testing.assert_array_equal(actual,values.astype(target))
                assert actual.dtype==np.dtype(target) and actual.shape==values.shape
                assert not np.shares_memory(actual,values)
                np.testing.assert_array_equal(values,original)
                cases.append(dict(application=application,source_dtype=source,target_dtype=target,
                    exact_native_cast=True,shape_and_order_preserved=True,independent_storage=True,input_unchanged=True))
    descriptors=[['sensor','x,y'],['sensor,x','y'],['phase','α'],[' label '],['quote','"\\']]
    original=[item.copy() for item in descriptors]
    encoded=provider.encode_feature_descriptors(descriptors)
    assert [json.loads(item) for item in encoded]==descriptors and len(set(encoded))==len(descriptors)
    assert encoded[0]=='["sensor","x,y"]' and encoded[1]=='["sensor,x","y"]'
    assert '\\u03b1' in encoded[2] and descriptors==original
    encoded.append('synthetic-change');assert descriptors==original
    rejected=[]
    def reject(label,fn,*args):
        try:fn(*args)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid conversion accepted: '+label)
    for label,value,dtype in [('array_list',[[1.]],'float32'),('integer_array',np.array([1]),'float32'),
        ('empty_array',np.array([]),'float32'),('nan',np.array([np.nan]),'float32'),
        ('infinity',np.array([np.inf]),'float32'),('complex',np.array([1j]),'float32'),
        ('implicit_dtype',np.array([1.]),np.float32),('unknown_dtype',np.array([1.]),'float16'),
        ('overflow',np.array([np.finfo(np.float64).max]),'float32')]:
        reject(label,provider.cast_finite_array,value,dtype)
    for label,value in [('unordered_descriptors',{'x'}),('empty_descriptors',[]),
        ('empty_descriptor',[[]]),('flat_labels',['x']),('blank_label',[[' ']]),
        ('nonstring_label',[[1]]),('duplicate_descriptors',[['x'],['x']]),('tuple_descriptor',[('x',)])]:
        reject(label,provider.encode_feature_descriptors,value)
    reject('abstract_execution',provider.witness_typed_inputs)
    assert len(rejected)==18
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,cast_cases=cases,
        descriptor_roundtrip_exact=True,delimiter_collisions_avoided=True,unicode_escaped=True,
        input_descriptors_unchanged=True,rejected_contracts=rejected,
        provider_sha256=sha(inspect.getfile(provider)),validator_sha256=sha(__file__),numpy=np.__version__,
        limitations=['Native float narrowing can round and underflow; caller controls acceptable precision loss and coordinate meanings.',
            'Encoding preserves structured labels and order, not feature semantics or array alignment; real labels remain private runtime metadata.',
            'Synthetic motion and power examples demonstrate constructed conversion reuse, not empirical transfer.',
            'Provider qualification only; catalog publication and quantile graph integration remain pending.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/typed_inputs_provider.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cast_cases=len(result['cast_cases']),rejected=len(result['rejected_contracts']),approved=False)))
