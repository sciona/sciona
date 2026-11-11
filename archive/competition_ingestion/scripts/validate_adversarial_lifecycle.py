"""Full10/20/40 schedules with real ensembles and independent update observation."""
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import adversarial_lifecycle as lifecycle
from sciona.adversarial_ensemble import ORDER
from sciona.adversarial_updates import update


def main():
    torch.set_num_threads(2);checks={};rng=torch.get_rng_state().clone()
    for mode,epsilon,branch,count in [('non_targeted',16,'non_targeted',10),
                                     ('targeted',8,'targeted_large',20),
                                     ('targeted',4,'targeted_small',40)]:
        images=np.linspace(-1,1,299*299*3,dtype=np.float32).reshape(1,299,299,3)
        original=images.copy();targets=np.array([17],dtype=np.int64) if mode=='targeted' else None
        specs={scope:{'kind':'random','seed':1200+i} for i,scope in enumerate(ORDER[branch])}
        observed=0;last_x=images.copy();last_noise=np.zeros_like(images)
        lower=np.maximum(images-2*epsilon/255,-1);upper=np.minimum(images+2*epsilon/255,1)
        def observe(x,g,noise,lo,hi,**kwargs):
            nonlocal observed,last_x,last_noise
            np.testing.assert_array_equal(x,last_x);np.testing.assert_array_equal(noise,last_noise)
            np.testing.assert_array_equal(lo,lower);np.testing.assert_array_equal(hi,upper)
            # Independent float64 scalar-form reduction oracle with float32 tolerance.
            d=g.astype(np.float64)
            d=d/(np.mean(np.abs(d),axis=(1,2,3),keepdims=True) if mode=='non_targeted'
                 else np.std(d,axis=(1,2,3),keepdims=True))
            expected_noise=noise.astype(np.float64)+d
            if mode=='targeted':expected_noise/=np.std(expected_noise,axis=(1,2,3),keepdims=True)
            a,b=update(x,g,noise,lo,hi,**kwargs)
            np.testing.assert_allclose(b,expected_noise,rtol=3e-4,atol=2e-5)
            assert np.all(a>=lower) and np.all(a<=upper)
            assert np.isfinite(b).all()
            last_x=a.copy();last_noise=b.copy();observed+=1
            return a,b
        with patch.object(lifecycle,'update',side_effect=observe):
            result=lifecycle.attack(images,mode=mode,epsilon=epsilon,initializations=specs,targets=targets)
        assert observed==count==len(result['losses'])==result['configuration']['iterations']
        assert result['configuration']['branch']==branch
        np.testing.assert_array_equal(images,original)
        np.testing.assert_array_equal(result['images'],last_x)
        assert np.any(result['images']!=images)
        if targets is not None:np.testing.assert_array_equal(result['labels'],targets)
        assert torch.equal(rng,torch.get_rng_state())
        checks[branch]={'complete_iterations':count,'full_models':len(specs),'independent_momentum_checks':observed,
                        'fixed_original_projection_bounds':True,'input_unchanged':True}
        print(json.dumps({branch:checks[branch]}),flush=True)
        del result
        gc.collect()
    paths=['sciona/adversarial_lifecycle.py','scripts/validate_adversarial_lifecycle.py',
           'sciona/adversarial_ensemble.py','sciona/adversarial_updates.py',
           'docs/reviews/competition_adversarial_ensemble.json','docs/reviews/competition_adversarial_updates.json']
    report={'format':'adversarial-lifecycle-validation.v1','result':'passed','checks':checks,
            'limits':'Full actual randomly initialized ensembles on one synthetic normalized299RGB example. No pretrained attack-quality or historical TensorFlow numerical equivalence claim. Batch10 padded-tail/image-file quantization and serialized provider execution remain unvalidated.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
