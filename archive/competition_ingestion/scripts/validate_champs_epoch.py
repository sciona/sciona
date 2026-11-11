"""Probe the pinned CHAMPS epoch controller with a synthetic analytic model.

This validates controller semantics, not graph-transformer topology (separate
full-model evidence). Original source bodies execute without epoch edits.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_epoch_corrections import correct_chunked_evaluation


class AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, atoms, positions, bonds, *rest):
        return self.weight.expand(atoms.shape[0], 68, bonds.shape[1]), None


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.trace = []

    def step(self, closure=None):
        self.trace.append({'lr': self.param_groups[0]['lr'],
                           'gradient': float(self.param_groups[0]['params'][0].grad)})
        return super().step(closure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo/'docs/reviews/competition_champs_source_pins.json').read_text())
    hashes = {p['software_path']:p['sha256'] for p in pins['pins']}
    ns = {'torch':torch,'np':np,'NUM_BOND_ORIG_TYPES':8,'MAX_BOND_COUNT':406,
          'APEX_AVAILABLE':False,'logging':lambda *_:None}
    for relative, names in [('src/graph_transformer.py',{'sqdist'}),
                            ('src/utils/filters.py',{'subgraph_filter'}),
                            ('src/train.py',{'loss','epoch'})]:
        raw = (args.source/relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashes[relative]
        nodes = [n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(nodes)==len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,'exec'),ns)
    # Every molecule has all eight supervised types plus an auxiliary pair.
    atoms = torch.ones(4, 3, 3, dtype=torch.long)
    positions = torch.zeros(4,3,5)
    bonds = torch.zeros(4,9,5,dtype=torch.long)
    bonds[:,:,0] = torch.arange(1,10)
    bonds[:,:,1] = torch.arange(1,10)
    targets = torch.zeros(4,9,4)
    targets[:,:8,0] = 10.
    targets[:,:8,2] = 2.
    targets[:,:8,3] = 1.
    batch = (torch.arange(4), atoms, positions, bonds, torch.zeros(4,9),
             torch.zeros(4,1,7,dtype=torch.long),torch.zeros(4,1),
             torch.zeros(4,1,10,dtype=torch.long),torch.zeros(4,1),targets)
    results=[]
    for chunks, log_loss in [(1,False),(2,False),(1,True)]:
        model = AnalyticModel()
        opt = RecordingSGD(model.parameters(),lr=.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,4,eta_min=.01)
        ns.update(args=SimpleNamespace(log_interval=99,batch_chunk=chunks,batch_size=4,
            champs_loss=log_loss,warmup_step=2,lr=.1,scheduler='cosine',clip=.1,
            cutout=0.,use_quad=False),train_step=0,para_model=model,scheduler=scheduler)
        output = ns['epoch']([batch]*4,model,opt)
        rates=[.05,.1,.01+.09*(1+math.cos(3*math.pi/4))/2,.01]
        np.testing.assert_allclose([r['lr'] for r in opt.trace],rates,rtol=1e-7)
        # With all predictions below target, global derivative is -2 and
        # log derivative is -2/error. Both exceed the clipping threshold.
        expected=.25
        for rate,item in zip(rates,opt.trace):
            grad = -2/(10-2*expected) if log_loss else -2.
            clipped=grad*min(1.,.1/(abs(grad)+1e-6))
            np.testing.assert_allclose(item['gradient'],clipped,rtol=2e-6)
            expected-=rate*clipped
        np.testing.assert_allclose(float(model.weight.detach()),expected,rtol=1e-6)
        assert ns['train_step']==4 and all(torch.isfinite(x).all() for x in output)
        results.append({'chunks':chunks,'objective':'log_type_mae' if log_loss else 'global_mae',
                        'warmup_cosine_matches':True,'clipped_updates_match':True})
    # Source chunked validation performs backward inside no_grad: reproduce
    # explicitly so the corrected runtime must not inherit this defect.
    ns['args'].batch_chunk=2
    ns['args'].champs_loss=False
    try:
        ns['epoch']([batch],model,None)
    except RuntimeError as exc:
        assert 'does not require grad' in str(exc)
    else:
        raise AssertionError('Expected original chunked validation defect')
    raw = (args.source/'src/train.py').read_bytes()
    corrected = correct_chunked_evaluation(raw,hashes['src/train.py'])
    nodes = [n for n in ast.parse(corrected).body if isinstance(n,ast.FunctionDef) and n.name=='epoch']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<corrected-epoch>','exec'),ns)
    before = model.weight.detach().clone()
    ns['args'].batch_chunk=1
    reference = ns['epoch']([batch],model,None)
    ns['args'].batch_chunk=2
    for log_loss in (False,True):
        ns['args'].champs_loss=log_loss
        actual = ns['epoch']([batch],model,None)
        for a,b in zip(actual,reference):
            torch.testing.assert_close(a,b)
    assert torch.equal(before,model.weight)
    args.output.write_text(json.dumps({'source_commit':pins['commit'],'results':results,
        'source_defects':['Chunked evaluation calls backward inside no_grad and fails.'],
        'corrected_chunked_evaluation_matches_full_batch':True,
        'correction_sha256':hashlib.sha256((repo/'sciona/champs_epoch_corrections.py').read_bytes()).hexdigest(),
        'scope':'Original epoch with analytic synthetic model and cutout disabled; optimizer, accumulation, clipping and warmup/cosine checks. No full-network training/checkpoint claim.',
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Three original epoch training paths match analytic updates; chunked validation defect reproduced')


if __name__=='__main__':
    main()
