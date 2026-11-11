"""Actual full non-targeted padded attack through JSON input/output transport."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_runtime import run
from sciona.adversarial_ensemble import ORDER
from sciona.dfdc_codec import encode_array,decode_array
from sciona.adversarial_image_boundary import source_rgb


def main():
    torch.set_num_threads(2)
    rgb=(np.arange(299*299*3,dtype=np.uint32)%256).astype(np.uint8).reshape(1,299,299,3)
    payload={'version':1,'rgb':encode_array(rgb),'targets':None,
             'config':{'mode':'non_targeted','epsilon':16,'momentum':1.,'non_targeted_iterations':10},
             'initializations':{s:{'kind':'random','seed':1500+i} for i,s in enumerate(ORDER['non_targeted'])}}
    wire=json.dumps(payload,allow_nan=False);rng=torch.get_rng_state().clone()
    result=json.loads(json.dumps(run(json.loads(wire)),allow_nan=False))
    images=decode_array(result['normalized_images']);labels=decode_array(result['labels'])
    assert images.shape==rgb.shape and labels.shape==(1,) and 0<=labels[0]<1001
    assert len(result['batch_losses'])==1 and len(result['batch_losses'][0])==10
    assert len(result['pngs'])==1 and result['saved_pixel_bound_guaranteed'] is False
    normalized=(rgb.astype(np.float64)/255.*2.-1.).astype(np.float32)
    assert np.max(np.abs(images-normalized))<=32/255+1e-7 and np.any(images!=normalized)
    decoded=np.array(Image.open(BytesIO(base64.b64decode(result['pngs'][0],validate=True))))
    np.testing.assert_array_equal(decoded,source_rgb(images[0]))
    assert result['initialization_kinds']=={s:'random' for s in ORDER['non_targeted']}
    assert json.dumps(payload,allow_nan=False)==wire and torch.equal(rng,torch.get_rng_state())
    paths=['sciona/adversarial_runtime.py','scripts/validate_adversarial_json_execution.py',
           'sciona/adversarial_rgb_attack.py','sciona/adversarial_lifecycle.py',
           'sciona/adversarial_ensemble.py','sciona/adversarial_image_boundary.py','sciona/dfdc_codec.py']
    report={'format':'adversarial-json-execution-validation.v1','result':'passed',
            'checks':{'actual_full_models':8,'full_iterations':10,'neural_batch_size':10,
                      'real_entries':1,'padded_entries':9,'JSON_input_output_roundtrip':True,
                      'normalized_projection_bound':True,'source_scaled_PNG':True,'input_rng_unchanged':True},
            'limits':'Synthetic random model states, complete non-targeted JSON runtime with padded batch. No pretrained attack effectiveness or TF binary parity; five-model padded targeted branch and provider graph remain unvalidated.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_json_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
