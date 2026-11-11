"""Synthetic exact-byte, JSON roundtrip and malformed transport checks."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_codec import encode_array,decode_array,encode_state,decode_state


def main():
    cases=0
    for dtype in ('uint8','int16','int64','float16','float32'):
        for shape in ((),(0,3),(2,3),(2,3,4)):
            a=np.arange(int(np.prod(shape)),dtype=dtype).reshape(shape)
            for x in (a, np.asfortranarray(a).reshape(shape)):
                encoded=encode_array(x);decoded=decode_array(json.loads(json.dumps(encoded)))
                assert decoded.dtype.name==x.dtype.name and decoded.shape==x.shape
                assert decoded.tobytes()==np.ascontiguousarray(x).tobytes()
                assert decoded.flags.writeable
                cases+=1
    # Independent little-endian wire oracle, including big-endian noncontiguous input.
    values=np.array([-3,257,1000,-32768],dtype='>i2')[::2]
    wire=encode_array(values)
    assert base64.b64decode(wire['data'])==struct.pack('<hh',-3,1000)
    assert np.array_equal(decode_array(wire),values)
    scalar=encode_array(np.array(-0.,dtype=np.float32))
    assert base64.b64decode(scalar['data'])==struct.pack('<f',-0.)
    state={'weight':torch.arange(12,dtype=torch.float32).reshape(3,4).T,
           'counter':torch.tensor(2**63-1,dtype=torch.int64),'empty':torch.empty(0)}
    encoded=encode_state(state);decoded=decode_state(json.loads(json.dumps(encoded)))
    assert all(torch.equal(state[k],decoded[k]) and state[k].shape==decoded[k].shape for k in state)
    decoded['weight'].zero_();assert state['weight'].count_nonzero()>0
    rejected=0
    def reject(fn):
        nonlocal rejected
        try:fn()
        except ValueError:rejected+=1
        else:raise AssertionError('malformed numerical transport accepted')
    original=encode_array(np.ones(2,dtype=np.float32))
    for field,value in [('version',True),('dtype','object'),('shape',[True]),('shape',[-1]),
                        ('shape',[2**100]),('shape',[2**40,2**40]),('shape',[1]*9),
                        ('data','invalid!'),('data','é'*12)]:
        bad=copy.deepcopy(original);bad[field]=value;reject(lambda:decode_array(bad))
    reject(lambda:decode_array(dict(original,extra=1)))
    nan=dict(original,data=base64.b64encode(struct.pack('<ff',float('nan'),1.)).decode())
    reject(lambda:decode_array(nan))
    reject(lambda:decode_array(original,max_bytes=7))
    reject(lambda:encode_array(np.array([float('inf')],dtype=np.float32)))
    reject(lambda:encode_array(np.array([object()],dtype=object)))
    reject(lambda:encode_state({'x':torch.ones(3)},max_bytes=8))
    reject(lambda:decode_state({'x':original,'y':original},max_bytes=12))
    reject(lambda:decode_state({'x':dict(original,dtype='uint8')}))
    reject(lambda:encode_state({'x':torch.ones(1,dtype=torch.float16)}))
    reject(lambda:decode_array(original,max_bytes=True))
    files=['sciona/dfdc_codec.py','scripts/validate_dfdc_codec.py']
    report={'format':'dfdc-codec-validation.v1','result':'passed',
            'checks':{'array_roundtrips':cases,'independent_little_endian_bytes':True,
                      'scalar_signed_zero_preserved':True,'state_json_roundtrip_and_isolation':True,
                      'malformed_input_rejections':rejected},
            'limits':'Synthetic arrays and small state tensors only. This validates numerical JSON transport, not the full DFDC runtime schema or end-to-end serialized graph.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_codec.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
