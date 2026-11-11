"""Synthetic decision comparison with externally supplied, pinned Keras source."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import argparse
import numpy as np
from sciona.tgs_callbacks import ValidationControl


def main(path, expected_hash):
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected_hash
    tree = ast.parse(content)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ('EarlyStopping', 'ReduceLROnPlateau')]
    assert len(selected) == 2
    class Callback:
        def __init__(self):
            pass
    class NumpyCompatibility:
        Inf = np.inf
        def __getattr__(self, name):
            return getattr(np, name)
    backend = SimpleNamespace(get_value=lambda value: value.item(),
                              set_value=lambda value, new: value.__setitem__(Ellipsis, new))
    namespace = dict(np=NumpyCompatibility(), K=backend, Callback=Callback, warnings=warnings)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-reference-callbacks>', 'exec'), namespace)
    cases = []
    sequences = [[.5,.50004,.50008,.49,.48,.47,.46], [.5]*7,
                 [.5,.6,.7,.70001,.70002,.8,.79,.78,.77,.76]]
    for scores in sequences:
        model = SimpleNamespace(stop_training=False, optimizer=SimpleNamespace(lr=np.array(.01)))
        early = namespace['EarlyStopping'](monitor='metric', patience=4, mode='max')
        plateau = namespace['ReduceLROnPlateau'](monitor='metric', patience=2, factor=.5, min_lr=.001, mode='max')
        for callback in [early, plateau]:
            callback.model = model
            callback.on_train_begin()
        actual = ValidationControl(learning_rate=.01, stop_patience=4, reduce_patience=2, factor=.5, minimum=.001)
        decisions = []
        best = -np.inf
        for epoch, score in enumerate(scores):
            logs = {'metric':score}
            early.on_epoch_end(epoch, logs)
            plateau.on_epoch_end(epoch, logs)
            decision = actual.observe(score)
            assert decision['stop'] == model.stop_training
            assert decision['save_best'] == (score > best)
            assert decision['learning_rate'] == float(model.optimizer.lr)
            best = max(best, score)
            decisions.append(decision)
            if model.stop_training:
                break
        cases.append(decisions)
    root = Path(__file__).resolve().parents[1]
    report = dict(passed=True, synthetic_only=True, catalog_mutations=0,
                  reference='https://raw.githubusercontent.com/keras-team/keras/2.2.0/keras/callbacks.py',
                  reference_sha256=expected_hash, cases=cases,
                  scope='Reference callback decision code executes with scalar backend storage; no full historical Keras training runtime claim.',
                  sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                          ['sciona/tgs_callbacks.py','scripts/validate_tgs_callback_reference.py']})
    (root/'docs/reviews/competition_tgs_callback_reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Reference decision comparison passed:', len(cases), 'sequences')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('sha256')
    args=parser.parse_args()
    main(args.source, args.sha256)
