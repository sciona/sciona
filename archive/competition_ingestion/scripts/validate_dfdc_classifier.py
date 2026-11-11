"""Actual source-resolution B7 execution with pinned classifier-head oracle."""
import argparse
import ast
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import torch
from torch import nn
import timm
from timm.models.efficientnet import tf_efficientnet_b7_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_classifier import DeepFakeClassifier, build_classifier
from sciona.dfdc_stochastic_depth import restore_stochastic_depth

def historical_encoder(**kwargs):
    model=tf_efficientnet_b7_ns(**kwargs)
    restore_stochastic_depth(model)
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    p=args.source_root/'training/zoo/classifiers.py'
    expected=next(r['sha256'] for r in pins['files'] if r['path']=='training/zoo/classifiers.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    tree=ast.parse(p.read_text())
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepFakeClassifier')
    current=next(n for n in ast.parse((ROOT/'sciona/dfdc_classifier.py').read_text()).body if isinstance(n,ast.ClassDef))
    get_forward=lambda cls:next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert ast.dump(get_forward(original))==ast.dump(get_forward(current))
    parameters=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='encoder_params' for t in n.targets))
    selected=next(v for k,v in zip(parameters.keys,parameters.values) if k.value=='tf_efficientnet_b7_ns')
    feature=next(v.value for k,v in zip(selected.keys,selected.values) if k.value=='features')
    operation=next(v for k,v in zip(selected.keys,selected.values) if k.value=='init_op')
    assert feature==2560 and {k.arg:ast.literal_eval(k.value) for k in operation.keywords}=={'pretrained':True,'drop_path_rate':.2}
    namespace={'nn':nn,'AdaptiveAvgPool2d':nn.AdaptiveAvgPool2d,'Dropout':nn.Dropout,'Linear':nn.Linear,
               'encoder_params':{'tf_efficientnet_b7_ns':{'features':2560,'init_op':partial(historical_encoder,pretrained=False,drop_path_rate=.2)}}}
    exec(compile(ast.Module(body=[original],type_ignores=[]),'<pinned-dfdc-classifier>', 'exec'),namespace)
    torch.set_num_threads(2)
    with warnings.catch_warnings(), patch('torch.load',side_effect=AssertionError('implicit weight load')):
        warnings.simplefilter('ignore',UserWarning)
        before=torch.random.get_rng_state().clone()
        model=build_classifier(initialization='random',seed=940)
        assert torch.equal(before,torch.random.get_rng_state())
        source=namespace['DeepFakeClassifier']('tf_efficientnet_b7_ns')
        source.load_state_dict(model.state_dict(),strict=True)
        model.train();source.train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(708)
            x=torch.rand(1,3,380,380,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            torch.manual_seed(801)
            expected=source(x)
            expected_loss=nn.functional.binary_cross_entropy_with_logits(expected,torch.ones_like(expected)*.99)
            expected_loss.backward()
            torch.manual_seed(801)
            actual=model(y)
            actual_loss=nn.functional.binary_cross_entropy_with_logits(actual,torch.ones_like(actual)*.99)
            actual_loss.backward()
        assert actual.shape==(1,1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        gradients=0;inactive=[]
        for (name,p),(other,q) in zip(model.named_parameters(),source.named_parameters()):
            assert name==other
            if p.grad is None:
                assert q.grad is None;inactive.append(name)
            else:
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                assert torch.isfinite(p.grad).all();gradients+=1
        assert inactive==['encoder.classifier.weight','encoder.classifier.bias']
        for name,value in model.state_dict().items():torch.testing.assert_close(value,source.state_dict()[name],rtol=0,atol=0)
        del source,x,y,expected,actual,expected_loss,actual_loss
        gc.collect()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001,nesterov=True)
        previous=model.fc.bias.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(previous,model.fc.bias)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        del optimizer
        model.eval()
        with torch.no_grad():
            prediction=model(torch.zeros(1,3,380,380))
        assert prediction.shape==(1,1) and torch.isfinite(prediction).all()
        restored=build_classifier(initialization='state',seed=97,state=model.state_dict())
        for key,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        del restored;gc.collect()
        restored=build_classifier(initialization='encoder',seed=98,state=model.encoder.state_dict())
        for key,value in model.encoder.state_dict().items():torch.testing.assert_close(value,restored.encoder.state_dict()[key],rtol=0,atol=0)
        assert not torch.equal(model.fc.weight,restored.fc.weight)
        del restored;gc.collect()
        invalid=dict(model.state_dict());key='fc.bias';invalid[key]=torch.full_like(invalid[key],float('nan'))
        try:build_classifier(initialization='state',seed=98,state=invalid)
        except ValueError:pass
        else:raise AssertionError('nonfinite state accepted')
        count=sum(p.numel() for p in model.parameters())
    files=['sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py','docs/reviews/competition_dfdc_stochastic_depth.json','scripts/validate_dfdc_classifier.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-classifier-validation.v1','result':'passed','source_commit':pins['commit'],
       'dependency':{'timm':timm.__version__,'torch':torch.__version__},
       'checks':{'source_forward_ast':True,'source_encoder_selection':True,'parameters':count,
                 'full_resolution_forward_backward':True,'input_gradient_exact':True,
                 'parameter_gradients_exact':gradients,'inactive_source_encoder_head_tensors':len(inactive),
                 'batchnorm_state_exact':True,'clipped_sgd_updates':1,'eval_forward':True,
                 'full_state_restoration':True,'encoder_state_restoration':True,'nonfinite_state_rejected':True},
       'limits':'Synthetic 380x380 batch1 and explicit random weights. Both source-head oracle and adaptation use installed timm0.9.2 with separately source-validated historical stochastic depth. Full historical encoder CPU parity is in competition_dfdc_stochastic_depth.json; historical binary runtime equivalence is not claimed. Optimizer step is isolated, not full source class-balanced loss/scheduler/training lifecycle evidence. No pretrained accuracy claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_classifier.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
