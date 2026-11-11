"""Actual synthetic MTCNN network/cascade parity against pinned source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_detector import build_detector
from sciona.dfdc_faces import extract_faces
from sciona import dfdc_mtcnn_cascade as cascade
from sciona import dfdc_mtcnn_networks as networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / 'docs/reviews/competition_dfdc_facenet_source_pins.json').read_text())
    for item in pins['files']:
        assert hashlib.sha256((args.source_root / item['path']).read_bytes()).hexdigest() == item['sha256']
    torch.set_num_threads(2)
    source = ast.parse((args.source_root / 'models/mtcnn.py').read_text())
    classes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in ('PNet','RNet','ONet','MTCNN')]
    namespace = {'torch': torch, 'nn': nn, 'np': np, 'detect_face': cascade.detect_face}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<source-mtcnn-networks>', 'exec'), namespace)
    current = ast.parse(Path(networks.__file__).read_text())
    current_classes = {n.name: n for n in current.body if isinstance(n, ast.ClassDef)}
    for original in classes:
        adapted = current_classes[original.name]
        if original.name == 'MTCNN':
            original.body = [m for m in original.body if isinstance(m, ast.FunctionDef) and m.name in ('__init__','detect')]
        else:
            ctor = next(m for m in original.body if isinstance(m, ast.FunctionDef) and m.name == '__init__')
            ctor.args.args = ctor.args.args[:1]
            ctor.args.defaults = []
            ctor.body = [n for n in ctor.body if not (isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'pretrained')]
        assert ast.dump(original) == ast.dump(adapted)
    original_cascade = ast.parse((args.source_root / 'models/utils/detect_face.py').read_text())
    current_cascade = ast.parse(Path(cascade.__file__).read_text())
    functions = {n.name:n for n in current_cascade.body if isinstance(n, ast.FunctionDef)}
    originals = [n for n in original_cascade.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert len(originals) == 9
    for n in originals:
        assert ast.dump(n) == ast.dump(functions[n.name])
    source_helpers = dict(vars(cascade))
    exec(compile(ast.Module(body=originals, type_ignores=[]), '<source-cascade>', 'exec'), source_helpers)
    namespace['detect_face'] = source_helpers['detect_face']
    network_cases = 0
    with patch('torch.load', side_effect=AssertionError('implicit state load')):
        before = torch.random.get_rng_state().clone()
        detector = build_detector(initialization='random', seed=607)
        assert torch.equal(before, torch.random.get_rng_state())
        for name, size in [('PNet', 37), ('RNet',24), ('ONet',48)]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(705)
                original = namespace[name](pretrained=False).eval()
                adapted = getattr(networks, name)().eval()
                adapted.load_state_dict(original.state_dict())
                x = torch.rand(2,3,size,size,requires_grad=True)
                y = x.detach().clone().requires_grad_()
                a, b = original(x), adapted(y)
                for u,v in zip(a,b): torch.testing.assert_close(u,v,rtol=0,atol=0)
                sum(t.square().sum() for t in a).backward()
                sum(t.square().sum() for t in b).backward()
                torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
                for u,v in zip(original.parameters(), adapted.parameters()):
                    torch.testing.assert_close(u.grad,v.grad,rtol=0,atol=0)
                network_cases += 1
        # Original constructor builds the same actual networks with file loads disabled.
        for name in ('PNet','RNet','ONet'):
            cls = namespace[name]
            namespace[name] = lambda cls=cls: cls(pretrained=False)
        original = namespace['MTCNN'](margin=0,thresholds=[.7,.8,.8],device=torch.device('cpu')).eval()
        image = Image.fromarray(np.random.default_rng(403).integers(0,256,(64,64,3),dtype=np.uint8))
        cascade_cases = 0
        for positive in (False, True):
            with torch.no_grad():
                for parameter in detector.parameters(): parameter.zero_()
                for model,head in [(detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')]:
                    getattr(model,head).bias.copy_(torch.tensor([-2.,2.] if positive else [2.,-2.]))
                detector.onet.dense6_3.bias.fill_(.5)
            original.load_state_dict(detector.state_dict())
            calls = {'pnet':0,'rnet':0,'onet':0}
            hooks=[]
            for name in calls:
                def track(module, inputs, output, name=name):calls[name]+=1
                hooks.append(getattr(detector,name).register_forward_hook(track))
            try:actual = detector.detect(image,landmarks=True)
            finally:
                for hook in hooks:hook.remove()
            expected = original.detect(image,landmarks=True)
            for a,b in zip(actual,expected):
                if a is None: assert b is None
                else:np.testing.assert_array_equal(a,b)
            assert calls['pnet'] > 0
            if positive:
                assert calls['rnet'] > 0 and calls['onet'] > 0 and len(actual[0]) > 0
                prepared = extract_faces(np.asarray(image)[None].copy(),detector)
                assert len(prepared)==1 and len(prepared[0]['faces'])>0
            else: assert actual[0] is None and calls['rnet']==calls['onet']==0
            cascade_cases += 1
        state={k:v.clone() for k,v in detector.state_dict().items()}
        restored=build_detector(initialization='state',seed=803,state=state)
        for k,v in restored.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        rejections=0
        key = next(iter(state))
        invalid_states = [
            {}, dict(state, **{key:torch.tensor(0.)}),
            dict(state, **{key:state[key].double()}),
            dict(state, **{key:torch.full_like(state[key], float('nan'))}),
        ]
        for invalid in invalid_states:
            try:build_detector(initialization='state',seed=1,state=invalid)
            except ValueError:rejections+=1
            else:raise AssertionError('invalid state accepted')
    files=['sciona/dfdc_detector.py','sciona/dfdc_mtcnn_networks.py','sciona/dfdc_mtcnn_cascade.py',
           'sciona/dfdc_faces.py','scripts/validate_dfdc_detector.py','docs/reviews/competition_dfdc_facenet_source_pins.json']
    report={'format':'dfdc-detector-validation.v1','result':'passed','source_commit':pins['commit'],
        'checks':{'normalized_class_ast_parity':4,'unchanged_cascade_functions':9,
                  'actual_network_forward_backward_cases':network_cases,'actual_cascade_cases':cascade_cases,
                  'explicit_state_restoration':True,'invalid_state_rejections':rejections,'implicit_loads_forbidden':True},
        'limits':'Synthetic random subnetworks and deliberately constructed constant-bias cascade states. Verifies execution and parity, not pretrained face-detection quality or historical dependency equivalence. Full competition lifecycle still unvalidated.',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_detector.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
