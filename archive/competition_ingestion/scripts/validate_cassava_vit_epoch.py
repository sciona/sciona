"""Compare CPU epoch reconstruction to isolated pinned winning epoch code."""
import contextlib
import copy
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sciona.cassava_vit_epoch import train_epoch
from scripts.validate_cassava_source_components import source_helpers


class Progress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def close(self):
        pass


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/vit.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['vit']['code_sha256']
    namespace = source_helpers(source, {
        'log_t', 'exp_t', 'compute_normalization_fixed_point',
        'compute_normalization_binary_search', 'ComputeNormalization',
        'compute_normalization', 'tempered_softmax', 'bi_tempered_logistic_loss',
        'train_one_epoch',
    }, {'torch': torch, 'time': time, 'gc': gc, 'tqdm': Progress,
        'autocast': contextlib.nullcontext,
        'xm': SimpleNamespace(optimizer_step=lambda opt: opt.step()),
        'CFG': {'t1': .8, 't2': 1.4, 'smoothing': .06, 'accum_iter': 2}})
    comparisons = 0
    for batch_count in (1, 2, 3, 5):
        torch.manual_seed(82 + batch_count)
        model = torch.nn.Linear(4, 5)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 / 7)
        ref_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4 / 7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-4)
        ref_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ref_optimizer, T_0=10, eta_min=1e-4)
        batches = [(torch.randn(3, 4), torch.arange(3, dtype=torch.int64)) for _ in range(batch_count)]
        for epoch in range(2):
            result = train_epoch(model, batches, optimizer, scheduler)
            namespace['train_one_epoch'](epoch, reference, ref_optimizer, batches, 'cpu', scheduler=ref_scheduler)
            assert result['optimizer_steps'] == (batch_count + 1) // 2
            assert scheduler.get_last_lr() == ref_scheduler.get_last_lr()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(optimizer.state[actual][key], ref_optimizer.state[expected][key], atol=2e-7, rtol=2e-5)
            comparisons += 1
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'epoch_comparisons': comparisons, 'adam_state_compared': True,
              'scheduler_compared': True,
              'limitations': ['CPU optimizer substitutes for TPU replica reduction.',
                              'No mixed precision, backbone, transforms, folds or checkpoints qualified.']}
    paths = ['sciona/cassava_vit_epoch.py', 'sciona/cassava_loss.py',
             'scripts/validate_cassava_vit_epoch.py', 'scripts/validate_cassava_source_components.py',
             'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_vit_epoch_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
