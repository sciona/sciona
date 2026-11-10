"""Trace pinned training control flow without importing legacy frameworks."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inventory_mercari_models import inventory

ROOT = Path(__file__).resolve().parents[1]


def qualify(directory):
    models = inventory(directory)  # verifies all model and implementation hashes
    triage = json.loads((ROOT/'docs/reviews/competition_mercari_source_triage.json').read_text())
    config_sha = next(r['sha256'] for r in triage['source_files'] if r['path'] == 'mercari/config.py')
    if hashlib.sha256((directory/'mercari/config.py').read_bytes()).hexdigest() != config_sha:
        raise ValueError('Pinned configuration source changed')
    models['source_sha256']['mercari/config.py'] = config_sha
    traces = {}
    for backend, filename, classname in [('tensorflow', 'tf_sparse.py', 'Regression'),
                                         ('mxnet', 'mx_sparse.py', 'MXRegression')]:
        tree = ast.parse((directory / 'mercari' / filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
        fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        loops = [n for n in ast.walk(fit) if isinstance(n, ast.For)
                 and ast.unparse(n.iter) == 'range(self.n_epoch)']
        if len(loops) != 1:
            raise ValueError('Expected one source epoch loop')
        loop = loops[0]
        decay = [n for n in loop.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'n_epoch < self.decay_epochs']
        if len(decay) != 1 or [ast.unparse(n) for n in decay[0].body] != ['bs *= self.bs_decay', 'lr *= self.lr_decay']:
            raise ValueError('Unqualified source schedule')
        # Execute the actual source loop control and decay statements, replacing
        # model computation with a recorder. No framework operation is executed.
        trace_loop = copy.deepcopy(loop)
        trace_loop.body = ast.parse('observed.append((n_epoch, bs, lr))').body + [copy.deepcopy(decay[0])]
        code = compile(ast.fix_missing_locations(ast.Module(body=[trace_loop], type_ignores=[])), '<source-schedule>', 'exec')
        traces[backend] = []
        for model in models['variants'][backend]:
            controls = model['resolved_constructor_controls']
            for epochs in (1, 3):
                state = dict(self=SimpleNamespace(**{**controls, 'n_epoch': epochs}),
                             bs=controls['batch_size'], lr=controls['learning_rate'], observed=[])
                exec(code, state)
                expected = [(i, controls['batch_size'] * controls['bs_decay'] ** min(i, controls['decay_epochs']),
                             controls['learning_rate'] * controls['lr_decay'] ** min(i, controls['decay_epochs']))
                            for i in range(epochs)]
                np.testing.assert_allclose(state['observed'], expected, rtol=1e-15, atol=0)
                traces[backend].append(dict(seed=model['seed'], epochs=epochs,
                    schedule=[dict(epoch=i, batch_size=bs, learning_rate=lr) for i, bs, lr in state['observed']]))
        if backend == 'tensorflow':
            batches = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'index_batches' for t in n.targets)]
            if len(batches) != 1:
                raise ValueError('Expected source TensorFlow batch construction')
            tf_code = compile(ast.Expression(batches[0].value), '<source-batches>', 'eval')
        else:
            iterator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CSRIter')
            reset = copy.deepcopy(next(n for n in iterator.body if isinstance(n, ast.FunctionDef) and n.name == 'reset'))
            namespace = {'np': np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[reset], type_ignores=[])), '<source-iterator>', 'exec'), namespace)
            mx_reset = namespace['reset']
    batch_cases = []
    for bs in (2048, 4096, 8192, 16384):
        for n in (17, bs-1, bs, bs+1, 2*bs+1):
            indices = np.arange(n)
            tf_batches = eval(tf_code, dict(indices=indices, n=n, bs=bs))
            state = SimpleNamespace(X=SimpleNamespace(shape=(n, 1)), batch_size=bs, shuffle=True)
            np.random.seed(827)
            mx_reset(state)
            mx_batches = state._index_batches
            assert len(tf_batches) == (n+bs-1)//bs
            assert sum(map(len, tf_batches)) == n
            assert len(mx_batches) == n//bs
            assert sum(map(len, mx_batches)) == n - n % bs
            batch_cases.append(dict(synthetic_population=n, batch_size=bs,
                tensorflow_updates=len(tf_batches), mxnet_updates=len(mx_batches), mxnet_discarded=n % bs))
    minimum_population = max(epoch['batch_size'] for case in traces['mxnet'] for epoch in case['schedule'])
    assert minimum_population == 8192
    return dict(format='mercari-training-schedule.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, source_commit=models['source_commit'], source_sha256=models['source_sha256'],
        schedule_traces=traces, batch_cases=batch_cases,
        mxnet_minimum_training_population_for_one_update_each_epoch=minimum_population,
        tensorflow_partial_batches_retained=True, mxnet_partial_batches_discarded=True,
        limits=['Control-flow qualification only; no optimizer updates or legacy-framework runtime parity established.',
                'MXNet needs at least 8192 training examples for a nonzero update in every epoch of each declared model; validation examples do not count.',
                'Do not shrink source batch sizes or drop final epochs to make a small synthetic test pass.',
                'TEST_SIZE controls simulated test volume; source chooses three epochs when it equals one and one epoch otherwise.',
                'The complete winning ensemble, feature transforms, optimizer semantics and source-runtime reproduction remain unqualified.'],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    result = qualify(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_mercari_training_schedule.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, schedule_cases=sum(map(len, result['schedule_traces'].values())),
                          batch_cases=len(result['batch_cases']), approved=False)))
