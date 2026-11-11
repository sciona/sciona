"""Compare checkpoint events with pinned source outer loop and evaluate_val."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_checkpoints import finish_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / 'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path = args.source_root / 'training/pipelines/train_classifier.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == next(
        entry['sha256'] for entry in pins['files'] if entry['path'] == 'training/pipelines/train_classifier.py')
    tree = ast.parse(path.read_text())
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_val')
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main_fn.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    rank_block = next(n for n in loop.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'args.local_rank == 0')
    # Execute the original save/validation block, including modulo cadence.
    executable = ast.Module(body=[evaluate, *copy.deepcopy(rank_block.body)], type_ignores=[])
    executable = compile(ast.fix_missing_locations(executable), '<source-checkpoint-order>', 'exec')
    cases = 0
    for epoch in (0, 1, 39):
        for every in (1, 2):
            for score in (.2, .5, .8):
                model = torch.nn.Linear(2, 1).eval()
                events = []
                def save(payload, name):
                    events.append(dict(copy.deepcopy(payload), kind=name.removeprefix('snapshot_')))
                namespace = {
                    'torch': SimpleNamespace(save=save), 'model': model,
                    'args': SimpleNamespace(local_rank=0, output_dir='', fold=0, test_every=every),
                    'epoch': epoch, 'current_epoch': epoch, 'bce_best': .5,
                    'snapshot_name': 'snapshot', 'val_data_loader': (),
                    'summary_writer': SimpleNamespace(add_scalar=lambda *a, **k: None),
                    'validate': lambda *a, **k: (score, {}, {}),
                    'open': lambda *a, **k: io.StringIO(), 'json': json,
                }
                with redirect_stdout(io.StringIO()):
                    exec(executable, namespace)
                # The source's first last-save path inserts an extra slash.
                for event in events:
                    if event['kind'] == '/snapshot_last': event['kind'] = 'last'
                validating = (epoch + 1) % every == 0
                result = finish_epoch(model, epoch=epoch, best_loss=.5,
                                      validation_loss=score if validating else None)
                assert result['best_loss'] == namespace['bce_best']
                assert result['improved'] == (validating and score < .5)
                assert len(result['events']) == len(events)
                for actual, expected in zip(result['events'], events):
                    assert {k:v for k,v in actual.items() if k != 'state_dict'} == {
                        k:v for k,v in expected.items() if k != 'state_dict'}
                    assert actual['state_dict'].keys() == expected['state_dict'].keys()
                    for key in actual['state_dict']:
                        assert torch.equal(actual['state_dict'][key], expected['state_dict'][key])
                cases += 1
    model = torch.nn.Linear(2, 1).eval()
    result = finish_epoch(model, epoch=0, best_loss=100, validation_loss=.3)
    assert [e['kind'] for e in result['events']] == ['last', '0', 'best_dice', 'last']
    assert [e['bce_best'] for e in result['events']] == [100, 100, .3, .3]
    before = result['events'][0]['state_dict']['weight'].clone()
    with torch.no_grad(): model.weight.add_(1)
    assert torch.equal(result['events'][0]['state_dict']['weight'], before)
    result['events'][1]['state_dict']['weight'].zero_()
    assert torch.equal(result['events'][0]['state_dict']['weight'], before)
    rejected = 0
    for kwargs in ({'epoch': -1}, {'epoch': True}, {'best_loss': None},
                   {'validation_loss': float('nan')}, {'validation_loss': -1}):
        inputs = dict(epoch=0, best_loss=100, validation_loss=.3); inputs.update(kwargs)
        try: finish_epoch(model, **inputs)
        except ValueError: rejected += 1
        else: raise AssertionError('invalid checkpoint metadata accepted')
    files = ['sciona/dfdc_checkpoints.py', 'scripts/validate_dfdc_checkpoints.py',
             'docs/reviews/competition_dfdc_source_pins.json']
    report = {'format': 'dfdc-checkpoint-validation.v1', 'result': 'passed',
              'source_commit': pins['commit'],
              'checks': {'source_save_order_cases': cases, 'snapshot_isolation': True,
                         'independent_metadata_order': True, 'invalid_metadata_rejections': rejected},
              'semantics': ['Numbered and initial last snapshots contain the previous best loss.',
                            'Strict validation improvement writes best_dice; every validation overwrites last.',
                            'Snapshot epoch metadata is zero-based loop epoch plus one.'],
              'adaptations': ['CPU unwrapped state dictionaries; in-memory snapshot events; no pickle or raw prediction output.'],
              'limits': 'Synthetic linear model. No historical weights, optimizer resume, full training lifecycle or suffix-40 provenance claim.',
              'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_checkpoints.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__': main()
