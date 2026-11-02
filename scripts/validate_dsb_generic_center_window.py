"""Pinned-source center-feature value/gradient qualification on synthetic tensors."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from sciona.atoms.dl.spatial import centered_window as candidate
from scripts.audit_competition_dsb_semantics import checked_source
ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError('Explicit single-thread environment required')
    torch.set_num_threads(1)
    pins = json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree = ast.parse(checked_source(source_root,pins,'net_classifier.py'))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CaseNet')
    forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    statements = [n for n in forward.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('featshape','centerFeat')]
    if [n.targets[0].id for n in statements] != ['featshape','centerFeat','centerFeat']: raise ValueError('Source expression inventory changed')
    divisions = [n for statement in statements for n in ast.walk(statement) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div)]
    if len(divisions)!=6: raise ValueError('Source integer division inventory changed')
    for node in divisions: node.op=ast.FloorDiv()
    code=compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),'<pinned-center-feature>','exec')
    cases=0
    for dtype in (torch.float32,torch.float64):
        for shape in ((2,3,4,6,8),(1,2,5,7,3),(1,1,2,2,2)):
            for tied in (False,True):
                values=(torch.ones(shape,dtype=dtype) if tied else torch.arange(torch.tensor(shape).prod().item(),dtype=dtype).reshape(shape)).requires_grad_()
                reference=values.detach().clone().requires_grad_()
                pool=torch.nn.MaxPool3d(2)
                scope=dict(noduleFeat=reference,self=SimpleNamespace(pool=pool))
                exec(code,scope)
                actual=pool(candidate.centered_window(values,(2,3,4),(2,2,2)))[:,:,0,0,0]
                expected=scope['centerFeat']
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                torch.testing.assert_close(torch.autograd.grad(actual.sum(),values)[0],torch.autograd.grad(expected.sum(),reference)[0],rtol=0,atol=0)
                cases+=1
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,cases=cases,
        exact_features=True,exact_gradients=True,native_threads=1,source_commit=pins['commit'],
        source_hashes={'net_classifier.py':pins['files']['net_classifier.py']},
        adaptations=['Six source Python 2 integer slice divisions translated to floor division.'],
        provider_sha256=sha(Path(inspect.getfile(candidate))),validator_sha256=sha(Path(__file__)),
        test_sha256=sha(ROOT/'tests/test_centered_window.py'),
        limitations=['CPU float32/float64 qualified; source max pooling remains recipe policy.',
                    'No architecture transfer or predictive-quality claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    report=validate(parser.parse_args().source_root)
    (ROOT/'docs/reviews/dsb_generic_center_window_source_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
