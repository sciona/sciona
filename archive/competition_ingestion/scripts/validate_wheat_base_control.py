"""Replay source checkpoint and stopping decisions over synthetic histories."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.wheat_base_control import BaseTrainingControl


def main(root, output):
    cases = []
    for detector, filename, digest, current, best in [
        ('effdet', 'effdet_train.py', 'd469f63af50abdb8a9e704bb756a569fd5f06383d3a5d09bd36b951220dbec4c', 'val_loss', 'val_loss_min'),
        ('fasterrcnn', 'faster_rcnn_fpn_train.py', '46086931f010b57fdb31e0f613dfa4fdd4a76461d39c742edd288144dbc5a588', 'ap', 'ap_max')]:
        raw = (root/filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('base training source drift')
        conditions = [n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.If)]
        select = next(n for n in conditions if ast.unparse(n.test) == f'{current} '+('<' if detector=='effdet' else '>')+f' {best}')
        stop = next(n for n in conditions if ast.unparse(n.test) == 'pat == args.patience or epoch == args.epochs - 1')
        select_code = compile(ast.Module(body=[select],type_ignores=[]), '<source-checkpoint>', 'exec')
        stop_code = compile(ast.Expression(body=stop.test), '<source-stop>', 'eval')
        for seed in range(64):
            metrics = np.random.default_rng(seed).integers(0, 11, size=100).astype(float)/10
            if seed == 0:
                metrics[:] = 0
            if seed == 1:
                metrics = np.arange(100)/100 if detector=='fasterrcnn' else 1-np.arange(100)/100
            saved = []
            model = SimpleNamespace(state_dict=lambda:{})
            model.model = model
            namespace = dict(print=lambda *args:None, CHECKPOINT='synthetic', model=model,
                args=SimpleNamespace(patience=40,epochs=100), pat=0, **{best:float('inf') if detector=='effdet' else 0.})
            namespace['torch'] = SimpleNamespace(save=lambda *args: saved.append(namespace['epoch']))
            control = BaseTrainingControl(detector)
            for epoch, metric in enumerate(metrics):
                namespace.update(epoch=epoch, **{current:float(metric)})
                old_count = len(saved)
                exec(select_code, namespace)
                expected_stop = eval(stop_code, namespace)
                decision = control.observe(epoch, float(metric))
                assert decision == dict(save_checkpoint=len(saved)>old_count, stop=expected_stop,
                    best_epoch=saved[-1] if saved else None, best_metric=namespace[best], patience=namespace['pat'])
                if expected_stop:
                    break
            if saved:
                assert control.selected_epoch() == saved[-1]
            else:
                try:
                    control.selected_epoch()
                except ValueError:
                    pass
                else:
                    raise AssertionError('missing source checkpoint was accepted')
            cases.append(dict(detector=detector,history=seed,completed_epochs=epoch+1,selected_epoch=saved[-1] if saved else None))
    files=['sciona/wheat_base_control.py','scripts/validate_wheat_base_control.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,exact_history_cases=len(cases),cases=cases,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Synthetic validation histories only; actual source metrics and training populations require separate qualification.',
                'Nonfinite metrics are rejected rather than allowing a bad execution to qualify.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_history_cases=len(cases))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source_root,args.output)
