"""Compare generic differentiable pooling directly with pinned source expressions."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from sciona.atoms.dl import probability_aggregation as candidate
from scripts.audit_competition_dsb_semantics import checked_source

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    torch.set_num_threads(1)
    pins=json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree=ast.parse(checked_source(source_root,pins,'net_classifier.py'))
    statements=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in {'base_prob','casePred'}]
    if [n.targets[0].id for n in statements]!=['base_prob','casePred']:
        raise ValueError('Source aggregation expressions changed')
    code=compile(ast.Module(body=statements,type_ignores=[]),'<pinned-public-noisy-or>','exec')
    cases=0
    for dtype in [torch.float32,torch.float64]:
        for baseline in [-30.,0.,2.]:
            for shape in [(1,1),(2,3),(3,5)]:
                p=torch.linspace(.05,.85,shape[0]*shape[1],dtype=dtype).reshape(shape).requires_grad_()
                b=torch.tensor([baseline],dtype=dtype,requires_grad=True)
                rp=p.detach().clone().requires_grad_();rb=b.detach().clone().requires_grad_()
                scope=dict(torch=torch,out=rp,self=SimpleNamespace(baseline=rb))
                exec(code,scope)
                expected=scope['casePred'];actual=candidate.noisy_or_with_baseline(p,b)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                for a,e in zip(torch.autograd.grad(actual.sum(),(p,b)),torch.autograd.grad(expected.sum(),(rp,rb))):
                    torch.testing.assert_close(a,e,rtol=0,atol=0)
                cases+=1
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        source_commit=pins['commit'],source_hashes={'net_classifier.py':pins['files']['net_classifier.py']},
        cases=cases,exact_probabilities=True,exact_input_gradients=True,exact_baseline_gradients=True,native_threads=1,
        provider_sha256=sha(Path(inspect.getfile(candidate))),validator_sha256=sha(Path(__file__)),
        source_extractor_sha256=sha(ROOT/'scripts/audit_competition_dsb_semantics.py'),
        test_sha256=sha(ROOT/'tests/test_probability_aggregation.py'),
        limitations=['CPU float32 and float64 qualified; no accelerator or low-precision qualification.',
            'No event independence, calibration or predictive-quality claim.',
            'Model integration and catalog approval remain pending.'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=validate(parser.parse_args().source_root)
    (ROOT/'docs/reviews/dsb_generic_noisy_or_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
