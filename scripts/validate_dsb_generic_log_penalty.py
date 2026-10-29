"""Direct pinned-source value/gradient qualification for thresholded log penalty."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from sciona.atoms.dl.loss import thresholded_log as candidate
from scripts.audit_competition_dsb_semantics import checked_source

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    torch.set_num_threads(1)
    pins=json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    name='training/classifier/trainval_classifier.py'
    tree=ast.parse(checked_source(source_root,pins,name));cases=0
    for function in ['train_casenet','val_casenet']:
        body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        statements=[n for n in ast.walk(body) if isinstance(n,ast.Assign) and len(n.targets)==1
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'missMask','missLoss'}]
        if [n.targets[0].id for n in statements]!=['missMask','missLoss']:raise ValueError('Source loss inventory changed')
        code=compile(ast.Module(body=statements,type_ignores=[]),'<pinned-public-log-penalty>','exec')
        for dtype in [torch.float32,torch.float64]:
            for threshold in [.03,.4,1.]:
                p=torch.tensor([[0.,.02,.03],[.2,.5,1.]],dtype=dtype,requires_grad=True)
                w=torch.tensor([[1.,.5,1.],[0.,1.,1.]],dtype=dtype,requires_grad=True)
                rp=p.detach().clone().requires_grad_();rw=w.detach().clone().requires_grad_()
                scope=dict(torch=torch,casePred_each=rp,isnod=rw,xsize=rp.shape,args=SimpleNamespace(miss_thresh=threshold))
                exec(code,scope)
                expected=scope['missLoss'];actual=candidate.thresholded_log_penalty(p,w,threshold,.001)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                for a,e in zip(torch.autograd.grad(actual,(p,w)),torch.autograd.grad(expected,(rp,rw))):
                    torch.testing.assert_close(a,e,rtol=0,atol=0)
                cases+=1
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,
        source_commit=pins['commit'],source_hashes={name:pins['files'][name]},cases=cases,
        exact_values=True,exact_probability_gradients=True,exact_weight_gradients=True,native_threads=1,
        provider_sha256=sha(Path(inspect.getfile(candidate))),validator_sha256=sha(Path(__file__)),
        source_extractor_sha256=sha(ROOT/'scripts/audit_competition_dsb_semantics.py'),
        test_sha256=sha(ROOT/'tests/test_thresholded_log_penalty.py'),
        limitations=['CPU float32/float64 source expressions with offset .001 qualified.',
            'Other positive offsets have separate synthetic contract evidence; gradient is undefined at the strict threshold.',
            'Training integration and catalog approval remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=validate(parser.parse_args().source_root)
    (ROOT/'docs/reviews/dsb_generic_log_penalty_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
