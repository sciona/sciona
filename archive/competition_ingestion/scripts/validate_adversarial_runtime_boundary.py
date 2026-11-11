"""Synthetic JSON transport contracts; this script does not claim neural runtime execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_runtime import prepare
from sciona.adversarial_ensemble import ORDER
from sciona.dfdc_codec import encode_array,encode_state


def main():
    rgb=(np.arange(299*299*3,dtype=np.uint32)%256).astype(np.uint8).reshape(1,299,299,3)
    payload={'version':1,'rgb':encode_array(rgb),'targets':encode_array(np.array([17],np.int64)),
             'config':{'mode':'targeted','epsilon':4,'momentum':1.,'non_targeted_iterations':10},
             'initializations':{s:{'kind':'random','seed':1} for s in ORDER['targeted_small']}}
    wire=json.dumps(payload,allow_nan=False);decoded=prepare(json.loads(wire))
    np.testing.assert_array_equal(decoded['rgb'],rgb)
    assert decoded['targets'].tolist()==[17]
    decoded['rgb'].fill(0);assert json.dumps(payload,allow_nan=False)==wire
    # Transport accepts a synthetic state fragment, while execution must reject
    # incomplete model variables. Do not claim fragment passes model validation.
    fragment=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
    supplied=copy.deepcopy(payload)
    supplied['initializations']['InceptionV3']={'kind':'slim_tensors','seed':2,'tensors':encode_state({'synthetic/weights':fragment})}
    restored=prepare(json.loads(json.dumps(supplied)))
    assert torch.equal(restored['initializations']['InceptionV3']['tensors']['synthetic/weights'],fragment)
    invalid=[]
    for field,value in [('version',True),('targets',None),('extra',1)]:
        item=copy.deepcopy(payload);item[field]=value;invalid.append(item)
    for field,value in [('epsilon',float('nan')),('momentum',-1),('non_targeted_iterations',20)]:
        item=copy.deepcopy(payload);item['config'][field]=value;invalid.append(item)
    item=copy.deepcopy(payload);item['initializations'].pop('InceptionV3');invalid.append(item)
    item=copy.deepcopy(payload);item['initializations']['InceptionV3']['seed']=True;invalid.append(item)
    item=copy.deepcopy(payload);item['rgb']['data']='!'+item['rgb']['data'][1:];invalid.append(item)
    item=copy.deepcopy(payload);item['targets']=encode_array(np.array([1001],np.int64));invalid.append(item)
    for item in invalid:
        try:prepare(item)
        except ValueError:pass
        else:raise AssertionError('invalid runtime boundary accepted')
    paths=['sciona/adversarial_runtime.py','scripts/validate_adversarial_runtime_boundary.py',
           'sciona/dfdc_codec.py','sciona/adversarial_rgb_attack.py','sciona/adversarial_updates.py']
    report={'format':'adversarial-runtime-boundary-validation.v1','result':'passed',
            'checks':{'JSON_RGB_targets_roundtrip':True,'owned_decoded_arrays':True,
                      'explicit_state_fragment_transport':True,'invalid_cases_rejected':len(invalid)},
            'limits':'Transport-only synthetic checks; incomplete state fragment deliberately used only for decoding. Exact full-model state validation happens at execution. Full JSON neural runtime/provider graph not yet validated.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_runtime_boundary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
