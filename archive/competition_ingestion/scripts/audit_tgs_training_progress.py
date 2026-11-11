"""Validate recorded source-budget fit histories; partial progress is not approval."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from sciona.tgs_callbacks import ValidationControl
from sciona.tgs_checkpoint_store import CheckpointStore
from sciona.tgs_workflow_state import WorkflowStateStore


def audit_fit(fit, result):
    digest = hashlib.sha256(json.dumps(fit, sort_keys=True).encode()).hexdigest()
    if result.get('fit') != fit['key'] or result.get('fit_contract_sha256') != digest:
        raise ValueError('fit contract mismatch')
    history, epochs = result['history'], result['actual_epochs']
    if type(epochs) is not int or not 0 < epochs <= fit['epoch_ceiling'] or len(history) != epochs:
        raise ValueError('invalid recorded epoch count')
    for index, entry in enumerate(history):
        if entry['epoch'] != index or any(not math.isfinite(float(v)) for v in entry.values()):
            raise ValueError('invalid epoch history')
    counts = result['population_counts']
    if any(type(counts[k]) is not int or counts[k] < 0 for k in ('training_labeled','training_query','validation_labeled')) or not counts['validation_labeled']:
        raise ValueError('invalid population counts')
    training = counts['training_labeled'] + counts['training_query']
    controls, batch = fit['controls'], int(fit['controls']['batch_size'])
    if not training:
        raise ValueError('empty recorded training population')
    if fit['branch'] == 'keras':
        steps = 2 * (training // batch)
        if not steps or result['steps_per_epoch'] != steps or result['optimizer_updates'] != steps * epochs:
            raise ValueError('source Keras update count mismatch')
        receipts = [result['best_checkpoint']]
        if controls['callback'] == 'reduce_lr':
            monitor = ValidationControl(learning_rate=float(controls['learning_rate']),
                stop_patience=int(controls['early_stop_patience']), reduce_patience=int(controls['reduce_lr_patience']),
                factor=float(controls['reduce_lr_factor']), minimum=float(controls['reduce_lr_min']))
            for entry in history:
                if not math.isclose(entry['learning_rate'], monitor.rate, rel_tol=1e-12, abs_tol=0):
                    raise ValueError('recorded plateau learning rate mismatch')
                monitor.observe(entry['source_metric'])
            if epochs < fit['epoch_ceiling'] and not monitor.stopped:
                raise ValueError('shortened fit without source early stopping')
            if result['periodic_checkpoints']:
                raise ValueError('unexpected periodic checkpoints')
        else:
            if epochs != fit['epoch_ceiling']:
                raise ValueError('incomplete source snapshot schedule')
            period = epochs // int(controls['n_snapshots'])
            expected = {n for n in range(period, epochs+1, period) if n != 1}
            if {int(n) for n in result['periodic_checkpoints']} != expected:
                raise ValueError('periodic checkpoint boundaries differ')
            for entry in history:
                rate = float(controls['learning_rate']) * (1 + math.cos(math.pi*(entry['epoch']%period)/period))/2
                if not math.isclose(entry['learning_rate'], rate, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError('recorded cosine learning rate mismatch')
            receipts.extend(result['periodic_checkpoints'].values())
    else:
        if epochs != fit['epoch_ceiling'] or result['optimizer_updates'] != epochs * math.ceil(training / batch):
            raise ValueError('incomplete source PyTorch updates')
        period = epochs // int(controls['snapshot'])
        if {int(n) for n in result['cycle_checkpoints']} != set(range(period, epochs+1, period)):
            raise ValueError('cycle checkpoint boundaries differ')
        receipts = list(result['cycle_checkpoints'].values())
    return receipts


def main(runtime, output):
    # Context validation against current code/input identity is owned by the
    # launcher; this audit verifies the already context-bound recorded state.
    manifest = json.loads((runtime/'state/state.json').read_text())
    state = WorkflowStateStore(runtime/'state', context_sha256=manifest['context_sha256']).load()
    plan = json.loads(Path('docs/reviews/competition_tgs_training_plan.json').read_text())
    inventory = {fit['key']:fit for fit in plan['fits']}
    checkpoints = CheckpointStore(runtime/'checkpoints')
    verified = 0
    for key, result in state['fits'].items():
        for receipt in audit_fit(inventory[key], result):
            checkpoints.get(receipt)
            verified += 1
    report = dict(approved=False, catalog_mutations=0, recorded_fit_checks_passed=True,
                  completed_fits=len(state['fits']), completed_rounds=len(state['rounds']),
                  verified_checkpoint_files=verified,
                  optimizer_updates=sum(r['optimizer_updates'] for r in state['fits'].values()),
                  full_workflow_complete=len(state['fits'])==63 and set(state['rounds'])=={1,2,3},
                  limits=['History/control and checkpoint-integrity audit only; no independent retraining or prediction-quality claim.',
                          'Final publication requires complete workflow, provenance and served-execution gates.'],
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.runtime_directory,args.output)
