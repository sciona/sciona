"""Compare explicit population routing with the frozen source-routing implementation."""
import hashlib
import inspect
import json
from pathlib import Path
import weakref
from unittest.mock import patch

import numpy as np
import pandas as pd
from sciona import osic_image_inference as original
from sciona.atoms.ml.domain_adapters import osic_image_routing as provider
from sciona.atoms.ml.domain_adapters.osic_image import reduce_image_predictions
from sciona.atoms.ml.pipeline.ordered_fields import select_values

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    baseline=pd.DataFrame(dict(entity=['b','a'],time=[3.,-1.],target=[8.,2.],numeric=[47.,53.],
        category_a=['Male','female'],category_b=['other','other'],uncertainty=[1.,2.]))
    query=pd.DataFrame(dict(key=['q0','q1','q2','q3'],entity=['a','b','a','b'],time=[8,-3,0,3]))
    planes={name:[('0.dcm',b'synthetic'),('1.dcm',b'synthetic')] for name in ('a','b')}
    model=object()
    cases=[]
    for dtype in (np.float32,np.float64):
        def prepare(records,numeric,category,status,*,decode_selected=None):
            assert decode_selected is None
            return dict(images=np.full((2,3),numeric,dtype=np.float64),
                metadata=np.full((2,4),.3,dtype=np.float64),selected_positions=np.array([0,1],np.int64))
        def predict(inputs):
            return (inputs[0][:,:1]/137+inputs[1][:,:1]).astype(dtype)
        with patch.object(original,'prepare_slice_inputs',side_effect=prepare):
            expected=original.predict(baseline,query,planes,predict)
        previous=[];calls=dict(prepare=0,select=0,predict=0)
        def tracked_prepare(*args):
            assert all(ref() is None for ref in previous), 'Prepared arrays retained across entities'
            value=prepare(*args)
            previous[:]=[weakref.ref(value['images']),weakref.ref(value['metadata'])]
            calls['prepare']+=1
            return value
        def selected(mapping,fields):
            calls['select']+=1
            assert fields==['images','metadata']
            return select_values(mapping,fields)
        def predicted(actual_model,inputs,batch_size):
            calls['predict']+=1
            assert actual_model is model and batch_size==32
            return predict(inputs)
        routed=provider.route_and_predict(baseline,query,planes,model,32,tracked_prepare,selected,predicted)
        assert all(ref() is None for ref in previous)
        actual=reduce_image_predictions(*routed[:6])
        np.testing.assert_array_equal(actual,expected['values'])
        assert routed[6]==expected['keys']
        for a,b in zip(routed[7],expected['selected_positions']):np.testing.assert_array_equal(a,b)
        assert calls==dict(prepare=2,select=2,predict=2)
        cases.append(dict(dtype=dtype.__name__,entities=2,queries=4,exact_source_routing=True,
            query_order_preserved=True,prepared_arrays_released=True,calls=calls))
    rejected=[]
    def forbidden(*a,**kw):raise AssertionError('Invalid routing reached computation')
    duplicate=baseline.copy();duplicate['entity']=['a','a']
    unknown=query.copy();unknown.loc[0,'entity']='unknown'
    duplicate_keys=query.copy();duplicate_keys.loc[0,'key']='q1'
    fractional=query.copy();fractional['time']=fractional['time'].astype(float)
    nonfinite=baseline.copy();nonfinite.loc[0,'time']=np.inf
    for name,b,q,p,batch in [('duplicate_entity',duplicate,query,planes,32),
        ('unknown_entity',baseline,unknown,planes,32),('duplicate_keys',baseline,duplicate_keys,planes,32),
        ('fractional_time_dtype',baseline,fractional,planes,32),('nonfinite',nonfinite,query,planes,32),
        ('missing_planes',baseline,query,{'a':planes['a']},32),('wrong_batch',baseline,query,planes,16),
        ('boolean_batch',baseline,query,planes,True)]:
        try:provider.route_and_predict(b,q,p,model,batch,forbidden,forbidden,forbidden)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid population accepted')
    try:provider.witness_image_routing()
    except ValueError:rejected.append('abstract_execution')
    else:raise AssertionError('Abstract execution accepted')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,cases=cases,
        rejected_contracts=rejected,provider_sha256=sha(inspect.getfile(provider)),
        software_sha256={name:sha(ROOT/name) for name in ('sciona/osic_image_routing.py',
            'sciona/osic_image_inference.py','sciona/osic_predictions.py','scripts/validate_osic_image_routing.py')},
        limitations=['Synthetic callbacks qualify population routing and lifetime behavior, not image decoding or model execution.',
            'Prepared-array release is local to the dispatcher; arbitrary callbacks may retain references.',
            'Source-specific mapped tables, query order and batch size remain explicit policy; dispatcher is not catalog-approved.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_image_routing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(result['cases']),rejected=len(result['rejected_contracts']))))
