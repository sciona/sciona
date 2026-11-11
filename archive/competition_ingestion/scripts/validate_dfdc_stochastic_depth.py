"""Historical/current full B7 parity measurement with identical synthetic state."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--historical-root',type=Path,required=True)
    args=parser.parse_args()
    pins_path=ROOT/'docs/reviews/competition_dfdc_timm_dependency_pin.json'
    pins=json.loads(pins_path.read_text())
    for path,digest in pins['files'].items():
        assert hashlib.sha256((args.historical_root/path).read_bytes()).hexdigest()==digest
    torch.set_num_threads(2)
    with patch('torch.load',side_effect=AssertionError('implicit checkpoint load')), warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        sys.path.insert(0,str(args.historical_root))
        import timm as historical
        assert historical.__version__=='0.4.12'
        torch.manual_seed(717)
        old=historical.create_model('tf_efficientnet_b7_ns',pretrained=False,drop_path_rate=.2)
        sys.path.pop(0)
        for key in tuple(sys.modules):
            if key=='timm' or key.startswith('timm.'):del sys.modules[key]
        # Remove old timm-only TorchScript overload registrations before loading
        # another version under the same module name. B7 uses eager execution.
        import torch._jit_internal as jit_internal
        for key in tuple(jit_internal._overloaded_methods):
            if 'timm.' in key: del jit_internal._overloaded_methods[key]
        for key in tuple(jit_internal._overloaded_method_class_fileno):
            if 'timm.' in key[0]: del jit_internal._overloaded_method_class_fileno[key]
        import timm as current
        assert current.__version__=='0.9.2'
        new=current.create_model('tf_efficientnet_b7_ns',pretrained=False,drop_path_rate=.2)
        from sciona.dfdc_stochastic_depth import restore_stochastic_depth
        replacements=restore_stochastic_depth(new)
        assert replacements > 0
        def function_body(path):
            fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='drop_path')
            return ast.dump(fn,include_attributes=False)
        assert function_body(args.historical_root/'timm/models/layers/drop.py') == function_body(ROOT/'sciona/dfdc_stochastic_depth.py')
        a,b=old.state_dict(),new.state_dict()
        assert a.keys()==b.keys()
        assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
        new.load_state_dict(a,strict=True)
        del a,b
        torch.manual_seed(719)
        x=torch.rand(1,3,380,380)
        old.eval();new.eval()
        with torch.no_grad():
            left=old.forward_features(x);right=new.forward_features(x)
        eval_exact=torch.equal(left,right)
        eval_error=float((left-right).abs().max())
        del left,right
        old.train();new.train()
        torch.manual_seed(720)
        left=old.forward_features(x)
        torch.manual_seed(720)
        right=new.forward_features(x)
        train_exact=torch.equal(left,right)
        train_error=float((left.detach()-right.detach()).abs().max())
        del left,right
        # Compare full stochastic forward/backward with identical draws and state.
        old.load_state_dict(new.state_dict(),strict=True)
        lx=x.clone().requires_grad_();rx=x.clone().requires_grad_()
        torch.manual_seed(721)
        left=old.forward_features(lx);left.square().mean().backward()
        old_rng=torch.get_rng_state()
        torch.manual_seed(721)
        right=new.forward_features(rx);right.square().mean().backward()
        assert torch.equal(old_rng,torch.get_rng_state())
        deterministic_exact=torch.equal(left,right)
        input_grad_exact=torch.equal(lx.grad,rx.grad)
        oldparams=dict(old.named_parameters());newparams=dict(new.named_parameters())
        assert oldparams.keys()==newparams.keys()
        grads=0;grad_mismatches=0
        for key,value in oldparams.items():
            other=newparams[key]
            if value.grad is None:
                assert other.grad is None
            else:
                grads+=1
                if not torch.equal(value.grad,other.grad):grad_mismatches+=1
        files={}
        for module in tuple(sys.modules.values()):
            name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
            if name.startswith('timm') and path and path.endswith('.py'):
                files[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert eval_exact and train_exact and deterministic_exact and input_grad_exact and grad_mismatches==0
    report={'format':'dfdc-stochastic-depth-validation.v1','result':'passed',
            'harness_adaptation':'Removed old timm-only TorchScript overload registrations to import both versions; all encoder execution eager.',
            'historical_timm':'0.4.12','installed_timm':current.__version__,
            'checks':{'source_function_ast_exact':True,'replaced_modules':replacements,'rng_continuation_exact':True,'state_keys_shapes_dtypes_exact':True,'evaluation_features_exact':eval_exact,
                      'evaluation_max_abs_error':eval_error,'training_same_seed_features_exact':train_exact,
                      'training_max_abs_error':train_error,'stochastic_backward_features_exact':deterministic_exact,
                      'stochastic_input_gradient_exact':input_grad_exact,
                      'parameter_gradients_compared':grads,'parameter_gradient_mismatches':grad_mismatches},
            'finding':'Restoring historical stochastic depth yields exact synthetic full B7 evaluation, training features, input/parameter gradients and RNG continuation under current Torch CPU.',
            'limits':'Actual full B7 feature extractor at380, synthetic random shared state, current Torch CPU float32. Historical Torch/CUDA/AMP runtime and pretrained quality are not verified. The adapter is tested here; pipeline default integration is separate.',
            'installed_imported_python_sha256':files,
            'sha256':{'scripts/validate_dfdc_stochastic_depth.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'sciona/dfdc_stochastic_depth.py':hashlib.sha256((ROOT/'sciona/dfdc_stochastic_depth.py').read_bytes()).hexdigest(),
                      'docs/licenses/Timm-Apache-2.0.txt':hashlib.sha256((ROOT/'docs/licenses/Timm-Apache-2.0.txt').read_bytes()).hexdigest(),
                      'docs/reviews/competition_dfdc_timm_dependency_pin.json':hashlib.sha256(pins_path.read_bytes()).hexdigest()}}
    (ROOT/'docs/reviews/competition_dfdc_stochastic_depth.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
