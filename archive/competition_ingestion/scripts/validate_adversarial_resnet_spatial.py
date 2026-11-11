"""Original Slim helper execution and independent scalar convolution derivatives."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_resnet_spatial import conv2d_same


def main():
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    source=next(s for s in pins['sources'] if s['branch']=='non_targeted')
    p=Path('/private/tmp/sciona_adversarial_source/non_targeted/nets/resnet_utils.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in source['files'] if f['path']=='nets/resnet_utils.py')
    fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='conv2d_same')
    torch.set_num_threads(2);cases=0
    for height,width in ((5,7),(6,8)):
        for kernel in (3,4):
            for stride in (1,2):
                for rate in (1,2):
                    rng=np.random.default_rng(925)
                    image=rng.normal(size=(1,1,height,width));weights=rng.normal(size=(1,1,kernel,kernel))
                    x=torch.tensor(image,requires_grad=True);w=torch.tensor(weights,requires_grad=True)
                    def conv(inputs,num_outputs,kernel_size,*,stride,rate,padding,scope):
                        assert num_outputs==1 and kernel_size==kernel
                        if padding=='SAME':
                            assert stride==1
                            total=(kernel-1)*rate;before=total//2
                            inputs=F.pad(inputs,(before,total-before,before,total-before))
                        else:assert padding=='VALID'
                        return F.conv2d(inputs,w,stride=stride,dilation=rate)
                    ns={'slim':SimpleNamespace(conv2d=conv),'tf':SimpleNamespace(pad=lambda a,p:F.pad(a,(p[2][0],p[2][1],p[1][0],p[1][1])))}
                    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-resnet-padding>','exec'),ns)
                    expected=ns['conv2d_same'](x,1,kernel,stride,rate=rate)
                    actual=conv2d_same(x,w,stride=stride,rate=rate)
                    assert torch.equal(actual,expected)
                    actual.sum().backward()
                    # Scalar mathematical oracle: correlate valid indices only.
                    out=np.zeros((1,1,(height+stride-1)//stride,(width+stride-1)//stride))
                    dx=np.zeros_like(image);dw=np.zeros_like(weights);pad=((kernel-1)*rate)//2
                    for i in range(out.shape[2]):
                        for j in range(out.shape[3]):
                            for a in range(kernel):
                                for b in range(kernel):
                                    y=i*stride+a*rate-pad;z=j*stride+b*rate-pad
                                    if 0<=y<height and 0<=z<width:
                                        out[0,0,i,j]+=image[0,0,y,z]*weights[0,0,a,b]
                                        dx[0,0,y,z]+=weights[0,0,a,b];dw[0,0,a,b]+=image[0,0,y,z]
                    np.testing.assert_allclose(actual.detach().numpy(),out,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(x.grad.numpy(),dx,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(w.grad.numpy(),dw,rtol=1e-12,atol=1e-12)
                    assert torch.equal(actual,conv2d_same(x,w,rate=rate)[:,:,::stride,::stride])
                    cases+=1
    paths=['sciona/adversarial_resnet_spatial.py','scripts/validate_adversarial_resnet_spatial.py','docs/reviews/competition_adversarial_source_pins.json']
    report={'format':'adversarial-resnet-spatial-validation.v1','result':'passed',
            'checks':{'source_helper_cases':cases,'independent_scalar_values_input_and_weight_gradients':True,
                      'stride_one_then_subsample_equivalence':True},
            'limits':'CPUfloat64 synthetic single-channel convolution; pinned source helper with Torch conv/pad adapter. No TensorFlow binary, batchnorm, full ResNet or full attack claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_resnet_spatial.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
