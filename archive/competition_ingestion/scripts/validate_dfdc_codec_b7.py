"""Full synthetic B7 model state JSON roundtrip, without publishing tensors."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_codec import encode_state,decode_state


def main():
    torch.set_num_threads(2)
    with warnings.catch_warnings(),patch('torch.load',side_effect=AssertionError('implicit pickle load')):
        warnings.simplefilter('ignore',UserWarning)
        original=build_classifier(initialization='random',seed=978).eval()
        wire=json.dumps(encode_state(original.state_dict()),allow_nan=False)
        decoded=decode_state(json.loads(wire))
        del wire
        expected=original.state_dict()
        assert decoded.keys()==expected.keys()
        assert all(decoded[k].shape==v.shape and decoded[k].dtype==v.dtype and torch.equal(decoded[k],v)
                   for k,v in expected.items())
        restored=build_classifier(initialization='state',seed=979,state=decoded).eval()
        del decoded
        with torch.no_grad():
            image=torch.linspace(0,1,3*380*380).reshape(1,3,380,380)
            left=original(image);right=restored(image)
        assert torch.equal(left,right) and torch.isfinite(right).all()
        tensor_count=len(expected)
    files=['sciona/dfdc_codec.py','sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py',
           'scripts/validate_dfdc_codec_b7.py']
    report={'format':'dfdc-codec-b7-validation.v1','result':'passed',
            'checks':{'full_B7_json_state_tensors_exact':tensor_count,
                      'restored_380px_model_output_exact':True,'implicit_pickle_load_forbidden':True},
            'limits':'Random synthetic full B7 state, in-memory JSON, current CPU float32. No real dataset or pretrained state used. Full runtime payload/graph serialization remains separate.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_codec_b7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
