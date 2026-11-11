"""Pinned bottleneck/topology checks and full299 ResNet input differentiation."""
import ast
from contextlib import contextmanager
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
from sciona.adversarial_resnet import Bottleneck,build_resnet101,same_max_pool
from sciona.adversarial_resnet_spatial import conv2d_same


class Shape(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='get_shape':
            return ast.copy_location(ast.Attribute(value=node.func.value,attr='shape',ctx=ast.Load()),node)
        return node


@contextmanager
def scope(*args,**kwargs):yield SimpleNamespace(original_name_scope='synthetic')


def main():
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    source=next(s for s in pins['sources'] if s['branch']=='non_targeted')
    p=Path('/private/tmp/sciona_adversarial_source/non_targeted/nets/resnet_v2.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in source['files'] if f['path']=='nets/resnet_v2.py')
    tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bottleneck')
    fn.decorator_list=[];fn=Shape().visit(fn);ast.fix_missing_locations(fn)
    torch.set_num_threads(2);cases=0
    for channels,depth,base,stride in ((4,4,2,1),(4,4,2,2),(3,8,2,2)):
        torch.manual_seed(933);unit=Bottleneck(channels,depth,base,stride).double()
        def conv(x,out,kernel,*,stride,scope,**kwargs):
            if scope=='shortcut':return unit.shortcut(x)
            if scope=='conv1':return F.relu(unit.bn1(unit.conv1(x)))
            if scope=='conv3':return unit.conv3(x)
            raise AssertionError('unexpected source convolution')
        def spatial(x,out,kernel,stride,*,rate,scope):
            assert scope=='conv2' and rate==1
            return F.relu(unit.bn2(conv2d_same(x,unit.conv2.weight,stride=stride)))
        ns={'tf':SimpleNamespace(variable_scope=scope,nn=SimpleNamespace(relu=F.relu)),
            'slim':SimpleNamespace(batch_norm=lambda x,**kw:F.relu(unit.preact(x)),conv2d=conv,
                utils=SimpleNamespace(last_dimension=lambda shape,**kw:shape[1],collect_named_outputs=lambda a,b,x:x)),
            'resnet_utils':SimpleNamespace(subsample=lambda x,stride,*a:x[:,:,::stride,::stride],conv2d_same=spatial)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-bottleneck>','exec'),ns)
        x=torch.randn(2,channels,6,8,dtype=torch.float64,requires_grad=True)
        expected=ns['bottleneck'](x,depth,base,stride);actual=unit(x)
        assert torch.equal(actual,expected)
        a=torch.autograd.grad(actual.sum(),x)[0];b=torch.autograd.grad(expected.sum(),x)[0]
        assert torch.equal(a,b);cases+=1
    # Independent SAME max-pool windows, especially even dimensions.
    for h,w in ((4,6),(5,7)):
        x=-torch.arange(1,h*w+1,dtype=torch.float64).reshape(1,1,h,w)
        actual=same_max_pool(x).numpy();expected=np.empty_like(actual)
        ph=max((actual.shape[2]-1)*2+3-h,0)//2;pw=max((actual.shape[3]-1)*2+3-w,0)//2
        for i in range(actual.shape[2]):
            for j in range(actual.shape[3]):
                expected[0,0,i,j]=x[0,0,max(0,2*i-ph):min(h,2*i-ph+3),max(0,2*j-pw):min(w,2*j-pw+3)].max()
        np.testing.assert_array_equal(actual,expected)
    rootfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resnet_v2_101')
    ns={'resnet_v2_block':lambda scope,**kw:kw,'resnet_v2':lambda inputs,blocks,*a,**kw:blocks}
    exec(compile(ast.Module(body=[rootfn],type_ignores=[]),'<source-resnet101-topology>','exec'),ns)
    topology=ns['resnet_v2_101'](None)
    state=torch.get_rng_state();model=build_resnet101(seed=934);assert torch.equal(state,torch.get_rng_state())
    assert [len(b) for b in model.blocks]==[r['num_units'] for r in topology]==[3,4,23,3]
    assert [[u.stride for u in b] for b in model.blocks]==[[1]*(r['num_units']-1)+[r['stride']] for r in topology]
    image=torch.linspace(-1,1,3*299*299).reshape(1,3,299,299).requires_grad_()
    buffers={k:v.clone() for k,v in model.named_buffers()}
    logits,endpoints=model(image);assert logits.shape==(1,1001)
    loss=F.cross_entropy(logits,torch.tensor([17]));gradient=torch.autograd.grad(loss,image)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0
    assert set(endpoints)=={'predictions'} and torch.allclose(endpoints['predictions'].sum(1),torch.ones(1))
    for k,v in model.named_buffers():assert torch.equal(v,buffers[k])
    files=['sciona/adversarial_resnet.py','sciona/adversarial_resnet_spatial.py','sciona/adversarial_normalization.py',
           'scripts/validate_adversarial_resnet.py','docs/reviews/competition_adversarial_source_pins.json']
    report={'format':'adversarial-resnet-validation.v1','result':'passed',
            'checks':{'source_bottleneck_value_and_input_gradient_cases':cases,'independent_pool_cases':2,
                      'source101_unit_topology':True,'actual299_full101_input_gradient':True,'classes':1001,
                      'frozen_buffers_unchanged':True,'initialization_rng_preserved':True},
            'limits':'Source bottleneck AST with explicit layout/component adapters; full CPUfloat32 random ResNet101 input differentiation. No whole-network TensorFlow binary numerical comparison or pretrained checkpoint mapping; other Inception families and ensemble iteration remain required.',
            'sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (ROOT/'docs/reviews/competition_adversarial_resnet.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
