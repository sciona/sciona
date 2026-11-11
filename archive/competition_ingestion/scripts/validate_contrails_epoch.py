"""Compare extracted epochs with executable, hash-pinned original loop AST."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import time
from pathlib import Path

import torch

from sciona.contrails_epoch import train_epoch
from sciona.contrails_losses import BCELoss
from sciona.contrails_scheduler import Scheduler


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 2, 1)

    def forward(self, x):
        y = self.layer(x)
        return y[:, :1], y[:, 1:]


def validate(root, source_root):
    torch.set_num_threads(2)
    pins = json.loads((root / 'docs/reviews/competition_contrails_source_pins.json').read_text())
    source = 'src/vit4/evaluate.py'
    raw = (source_root / source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(f['sha256'] for f in pins['files'] if f['path'] == source)
    tree = ast.parse(raw)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple)
                and isinstance(n.target.elts[0], ast.Name) and n.target.elts[0].id == 'ibatch')
    oracle = compile(ast.Module(body=[loop], type_ignores=[]), '<pinned-training-loop>', 'exec')
    cases = []
    for nbatch, accumulation, validations in [(8, 2, 1), (9, 2, 2), (3, 2, 2), (1, 2, 1), (4, 1, 1)]:
        torch.manual_seed(192)
        actual = SyntheticModel()
        expected = copy.deepcopy(actual)
        opts = [torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.01) for m in (actual, expected)]
        schedule = [{'linear': {'epoch_end': .5, 'lr_start': 1e-8, 'lr_end': 8e-4}},
                    {'cosine': {'epoch_end': 3, 'lr_end': 1e-6}}]
        schedulers = [Scheduler(o, schedule) for o in opts]
        batches = [dict(x=torch.randn(2, 1, 3, 3), y=torch.rand(2, 1, 3, 3),
                        y_sym=torch.rand(2, 1, 3, 3), w=torch.tensor([0., 1.])) for _ in range(nbatch)]
        for epoch in range(2):
            result = train_epoch(actual, batches, BCELoss(), opts[0], schedulers[0],
                                 epoch=epoch, accumulate=accumulation, validations_per_epoch=validations)
            validation_calls = []
            def evaluate(model, loader, **kwargs):
                validation_calls.append(True)
                return dict(loss=0., dice=0., score=0., dt=0.)
            opts[1].zero_grad()
            ns = dict(loader_train=batches, model=expected, criterion=BCELoss(),
                      optimizer=opts[1], scheduler=schedulers[1], accumulate=accumulation,
                      device='cpu', nbatch=nbatch, iepoch=epoch, istep=0,
                      icheck=[(nbatch // accumulation) * (i + 1) // validations - 1 for i in range(validations)],
                      n_sum=0, loss_sum=0., lrs=[], epochs_log=[], losses_train=[], losses_val=[],
                      evaluate=evaluate, loader_val=None, loader_test=None, th_val=.4, th_test=.4,
                      time=time, tb=time.time(), dt_val=0., nn=torch.nn)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(oracle, ns)
            assert result['optimizer_steps'] == ns['istep']
            assert result['batches_processed'] == ns['ibatch'] + 1
            assert result['learning_rates'] == ns['lrs']
            assert result['validation_epochs'] == ns['epochs_log']
            assert len(validation_calls) == 2 * len(result['validation_epochs'])
            assert opts[0].param_groups[0]['lr'] == opts[1].param_groups[0]['lr']
            for a, b in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                if a.grad is None or b.grad is None:
                    assert a.grad is None and b.grad is None
                else:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for key in opts[0].state[a]:
                    torch.testing.assert_close(opts[0].state[a][key], opts[1].state[b][key], rtol=0, atol=0)
            cases.append(dict(batches=nbatch, accumulation=accumulation, validations=validations,
                              epoch=epoch, processed=result['batches_processed'], steps=result['optimizer_steps']))
    return {'approved': False, 'cases': cases, 'exact_epoch_comparisons': len(cases),
            'hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                       ['sciona/contrails_epoch.py', 'sciona/contrails_scheduler.py', 'scripts/validate_contrails_epoch.py']},
            'scope': 'Original loop AST versus extracted epoch: parameters, gradients, AdamW moments, validation timing and LR. Synthetic small model; full model training and checkpoint lifecycle pending.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root / 'docs/reviews/competition_contrails_epoch.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'exact_epoch_comparisons': result['exact_epoch_comparisons'], 'cases': result['cases']}))
