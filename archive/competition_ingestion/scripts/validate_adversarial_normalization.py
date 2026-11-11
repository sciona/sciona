"""Pinned TF primitive AST with Torch operations and independent affine derivatives."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_normalization import FrozenBatchNorm


def main():
    pins_path=ROOT/'docs/reviews/competition_adversarial_tensorflow_reference.json'
    pins=json.loads(pins_path.read_text());base=Path('/private/tmp/sciona_adversarial_tensorflow_source')
    for row in pins['files']:assert hashlib.sha256((base/row['path']).read_bytes()).hexdigest()==row['sha256']
    tree=ast.parse((base/'tensorflow/python/ops/nn_impl.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batch_normalization')
    ns={'ops':SimpleNamespace(name_scope=lambda *a:nullcontext()),'math_ops':SimpleNamespace(rsqrt=torch.rsqrt)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<tf14-normalization>','exec'),ns)
    layer=next(n for n in ast.parse((base/'tensorflow/contrib/layers/python/layers/layers.py').read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name=='batch_norm')
    defaults=dict(zip([a.arg for a in layer.args.args][-len(layer.args.defaults):],layer.args.defaults))
    assert ast.literal_eval(defaults['scale']) is False and ast.literal_eval(defaults['center']) is True
    cases=0
    for dtype in (torch.float32,torch.float64):
        for scale in (False,True):
            for epsilon in (.001,1e-5):
                model=FrozenBatchNorm(3,epsilon=epsilon,scale=scale).to(dtype=dtype)
                model.moving_mean.copy_(torch.tensor([1.,-2.,.5]));model.moving_variance.copy_(torch.tensor([0.,2.,4.]))
                model.beta.copy_(torch.tensor([.2,-.3,.4]))
                if scale:model.gamma.copy_(torch.tensor([2.,-1.,0.]))
                state={k:v.clone() for k,v in model.state_dict().items()}
                x=torch.linspace(-2,2,72,dtype=dtype).reshape(2,3,3,4).requires_grad_()
                gamma=None if not scale else model.gamma.reshape(1,3,1,1)
                expected=ns['batch_normalization'](x,model.moving_mean.reshape(1,3,1,1),
                    model.moving_variance.reshape(1,3,1,1),model.beta.reshape(1,3,1,1),gamma,epsilon)
                model.train();actual=model(x)
                assert torch.equal(actual,expected)
                model.eval();assert torch.equal(actual,model(x))
                actual.sum().backward()
                multiplier=(torch.ones_like(model.beta) if not scale else model.gamma)/torch.sqrt(model.moving_variance+epsilon)
                torch.testing.assert_close(x.grad,multiplier.reshape(1,3,1,1).expand_as(x),rtol=1e-6,atol=1e-6)
                for k,v in model.state_dict().items():assert torch.equal(v,state[k])
                restored=FrozenBatchNorm(3,epsilon=epsilon,scale=scale).to(dtype=dtype)
                restored.load_state_dict(state,strict=True);assert torch.equal(actual,restored(x))
                assert ('gamma' in state)==scale and not list(model.parameters())
                cases+=1
    rejected=0
    for value in (-1.,float('nan')):
        model=FrozenBatchNorm(3);model.moving_variance[0]=value
        try:model(torch.zeros(1,3,2,2))
        except ValueError:rejected+=1
        else:raise AssertionError('invalid frozen variance accepted')
    paths=['sciona/adversarial_normalization.py','scripts/validate_adversarial_normalization.py',
           'docs/reviews/competition_adversarial_tensorflow_reference.json',
           'docs/licenses/Adversarial-TensorFlow-reference-Apache-2.0.txt']
    report={'format':'adversarial-normalization-validation.v1','result':'passed',
            'checks':{'source_primitive_cases':cases,'independent_input_derivatives':True,
                      'train_eval_and_state_immutability':True,'strict_roundtrip_and_optional_gamma':True,
                      'reference_slim_center_true_scale_false':True,'invalid_state_rejections':rejected},
            'limits':'Explicit TF1.4 primitive arithmetic with Torch-op oracle; no historical binary/fused-kernel equivalence. CPU synthetic tensors, normalization only. Full backbone/attack validation remains.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_normalization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
