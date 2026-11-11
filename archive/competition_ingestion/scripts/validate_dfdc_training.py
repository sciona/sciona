"""Pinned outer-loop parity over synthetic records and actual preparation."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_training import train
from sciona.dfdc_dataset import PreparedDataset, make_loader
from sciona.dfdc_evaluation import ValidationDataset, evaluate
from sciona.dfdc_epoch import train_epoch
from sciona.dfdc_scheduler import PolyLR


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.fc = torch.nn.Linear(3, 1)

    @property
    def module(self):
        return self

    def forward(self, images):
        return self.fc(self.encoder(images))


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path = args.source_root/'training/pipelines/train_classifier.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == next(
        f['sha256'] for f in pins['files'] if f['path'] == 'training/pipelines/train_classifier.py')
    tree = ast.parse(path.read_text())
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main_fn.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    reset = next(n for n in main_fn.body if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'data_val.reset')
    assert ast.unparse(reset) == 'data_val.reset(1, args.seed)'
    eval_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_val')
    executable = compile(ast.fix_missing_locations(ast.Module(
        body=[eval_fn, reset, loop], type_ignores=[])), '<source-outer-training>', 'exec')
    # All records are invented here. Validation has both classes; train balances 4+4.
    rows = []
    for fold in (0, 1):
        for label in (0, 1):
            for i in range(4):
                pixels = np.arange(21*25*3, dtype=np.uint16).reshape(21,25,3)
                rows.append({'image': ((pixels+label*39+i*7)%256).astype(np.uint8),
                             'label': label, 'fold': fold, 'frame': i*20,
                             'clip_position': fold*2+label, 'mask': None, 'landmarks': None})
    torch.set_num_threads(2)
    cases = updates = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for seed in (111, 555):
            for every in (1, 2):
                torch.manual_seed(601)
                source = Model(); adapted = copy.deepcopy(source)
                def optimizer(m):
                    return torch.optim.SGD(m.parameters(), lr=.01, momentum=.9,
                                           weight_decay=.0001, nesterov=True)
                aopt, bopt = optimizer(source), optimizer(adapted)
                asched, bsched = PolyLR(aopt, max_iter=100500), PolyLR(bopt, max_iter=100500)
                expected_events = []; expected_metrics = []; expected_val = []
                def save(payload, name):
                    kind = name.lstrip('/').removeprefix('snapshot_')
                    expected_events.append(dict(copy.deepcopy(payload), kind=kind))
                def source_train(epoch, losses, model, opt, sched, loader, writer, conf, rank, changed):
                    expected_metrics.append(train_epoch(model,opt,sched,loader,epoch=epoch,batches_per_epoch=2))
                def source_validate(model, data_loader):
                    value = evaluate(model, data_loader); expected_val.append(value)
                    return value['loss'], {}, {}
                def loader(dataset, **kwargs):
                    assert kwargs['num_workers'] == 0 and not kwargs['pin_memory']
                    assert kwargs['shuffle'] and kwargs['drop_last']
                    assert kwargs['sampler'] is None
                    return make_loader(dataset, batch_size=kwargs['batch_size'])
                random.seed(seed); torch.manual_seed(seed)
                data_train = PreparedDataset(rows, mode='train', detector=lambda *a: [], predictor=lambda *a: None)
                data_val = ValidationDataset(rows)
                # Source constructs val loader before reset; actual iteration occurs after reset.
                val_loader = torch.utils.data.DataLoader(data_val, batch_size=4, num_workers=0,
                                                         shuffle=False, pin_memory=False)
                namespace = {'torch': SimpleNamespace(save=save), 'model':source,
                             'args':SimpleNamespace(seed=seed,distributed=False,freeze_epochs=0,
                                workers=0,local_rank=0,only_changed_frames=False,output_dir='',test_every=every,fold=0),
                             'data_train':data_train, 'data_val':data_val,'val_data_loader':val_loader,
                             'start_epoch':0,'max_epochs':2,'current_epoch':0,'batch_size':2,
                             'loss_functions':{},'optimizer':aopt,'scheduler':asched,'conf':{},
                             'summary_writer':SimpleNamespace(add_scalar=lambda *a,**k:None),
                             'train_epoch':source_train,'validate':source_validate,'DataLoader':loader,
                             'bce_best':100.,'snapshot_name':'snapshot',
                             'open':lambda *a,**k:io.StringIO(),'json':json}
                with redirect_stdout(io.StringIO()): exec(executable, namespace)
                surrounding = (random.getstate(),np.random.get_state(),torch.get_rng_state())
                observed_events = []
                result = train(adapted,bopt,bsched,rows,detector=lambda *a:[],predictor=lambda *a: None,
                               seed=seed,checkpoint_sink=observed_events.append,
                               epochs=2,batch_size=2,batches_per_epoch=2,test_every=every)
                assert_equal(surrounding,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
                assert_equal(expected_events,observed_events)
                assert_equal(source.state_dict(),adapted.state_dict())
                assert_equal(aopt.state_dict(),bopt.state_dict())
                assert_equal(asched.state_dict(),bsched.state_dict())
                assert_equal(expected_metrics,[e['training'] for e in result['epochs']])
                assert_equal(expected_val,[e['validation'] for e in result['epochs'] if e['validation'] is not None])
                assert result['best_loss'] == namespace['bce_best']
                updates += sum(e['training']['optimizer_updates'] for e in result['epochs'])
                cases += 1
        # Failure during persistence must also restore the surrounding streams.
        surrounding = (random.getstate(),np.random.get_state(),torch.get_rng_state())
        model=Model();opt=optimizer(model);sched=PolyLR(opt,max_iter=100500)
        surrounding = (random.getstate(),np.random.get_state(),torch.get_rng_state())
        def failing_sink(event): raise RuntimeError('synthetic persistence failure')
        try:
            train(model,opt,sched,rows,detector=lambda *a:[],predictor=lambda *a: None,seed=111,
                  checkpoint_sink=failing_sink,epochs=1,batch_size=2,batches_per_epoch=1)
        except RuntimeError as error:
            assert str(error) == 'synthetic persistence failure'
        else: raise AssertionError('sink failure was swallowed')
        assert_equal(surrounding,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
    files=['sciona/dfdc_training.py','sciona/dfdc_dataset.py','sciona/dfdc_evaluation.py',
           'sciona/dfdc_epoch.py','sciona/dfdc_checkpoints.py','sciona/dfdc_scheduler.py',
           'scripts/validate_dfdc_training.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-training-lifecycle-validation.v1','result':'passed','source_commit':pins['commit'],
            'checks':{'source_outer_loop_cases':cases,'connected_optimizer_updates':updates,
                      'model_optimizer_scheduler_snapshots_metrics_exact':True,
                      'caller_rng_restored_success_and_sink_failure':True},
            'adaptations':['CPU float32, workers zero, no distributed/freeze/resume.',
                           'Explicit Python/Torch seeding and caller RNG isolation.',
                           'In-memory unwrapped tensor snapshots; no raw prediction output.'],
            'limits':'Source outer loop is executed unchanged with separately validated inner epoch/evaluation adapters. Actual sample preparation and augmentation on synthetic records, small classifier. Full B7 connected lifecycle, historical encoder and checkpoint suffix-40 provenance remain unverified.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__': main()
