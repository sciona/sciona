"""Replay all source base-training rates using historical scheduler classes."""
import argparse
import ast
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import types
import warnings
import weakref

import torch

from sciona.wheat_base_schedule import base_learning_rate


def main(scheduler_source, warmup_source, output):
    source_hashes = {}
    trees = []
    for path, digest in [(scheduler_source, '1efe08d7f86f12bbe2c1ce722646ae4945c6fe19611db5c9058ffbf396bc94c3'),
                         (warmup_source, '631edf613cfc0b098bbe4e8f9a249c30488b9a12c1fe7e5e55f08c9367f9d8db')]:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('historical scheduler source drift')
        source_hashes[path.name] = digest
        trees.append(ast.parse(raw))
    namespace = dict(Optimizer=torch.optim.Optimizer, wraps=wraps, weakref=weakref,
        warnings=warnings, types=types, math=math, EPOCH_DEPRECATION_WARNING='source explicit epoch',
        ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau)
    nodes = [n for n in trees[0].body if isinstance(n, ast.ClassDef) and n.name in ('_LRScheduler', 'CosineAnnealingLR')]
    nodes += [n for n in trees[1].body if isinstance(n, ast.ClassDef) and n.name == 'GradualWarmupScheduler']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-base-schedulers>', 'exec'), namespace)
    rows = []
    for detector, optimizer_class, requested in [('effdet', torch.optim.Adam, .0005), ('fasterrcnn', torch.optim.SGD, .005)]:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = optimizer_class([parameter], lr=requested/10)
        cosine = namespace['CosineAnnealingLR'](optimizer, 99)
        scheduler = namespace['GradualWarmupScheduler'](optimizer, multiplier=10, total_epoch=1, after_scheduler=cosine)
        rates = []
        for epoch in range(100):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                scheduler.step(epoch)
            rate = optimizer.param_groups[0]['lr']
            if rate != base_learning_rate(epoch, detector):
                raise ValueError(f'source learning rate differs at {detector} epoch {epoch}: {rate}')
            rates.append(rate)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        rows.append(dict(detector=detector, exact_training_rates=rates))
    files = ['sciona/wheat_base_schedule.py', 'scripts/validate_wheat_base_schedule.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=source_hashes, cases=rows, exact_rate_count=200,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Source scheduler composition only; historical optimizer numerical updates and complete training remain separate.',
                'Epoch 2 retains the cosine scheduler initial learning rate before its first explicit step; this source behavior is preserved.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_rate_count=200, first_four_rates=[row['exact_training_rates'][:4] for row in rows])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scheduler-source',type=Path,required=True)
    parser.add_argument('--warmup-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.scheduler_source,args.warmup_source,args.output)
