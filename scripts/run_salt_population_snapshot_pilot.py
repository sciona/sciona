"""Continue the synthetic population pilot through all three source snapshot fits.

This completes one fold's first-round Keras chain, not the 63-fit workflow.
Each phase uses its predecessor's validation-best weights and unmodified controls.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from sciona.tgs_checkpoint_store import CheckpointStore
from sciona.tgs_fit_execution import run_fit

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['keras.r1.p2.f0', 'keras.r1.p3.f0', 'keras.r1.p4.f0']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(previous, runtime, output):
    previous, runtime = previous.resolve(), runtime.resolve()
    if runtime == ROOT or ROOT in runtime.parents or runtime == previous:
        raise ValueError('Independent private runtime required')
    context = json.loads((previous / 'context.json').read_text())
    pins = dict(context['source_sha256'])
    pins[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    predecessor_path = previous / 'result.json'
    predecessor_digest = digest(predecessor_path)

    def verify():
        for name, expected in pins.items():
            if digest(ROOT / name) != expected:
                raise ValueError('Bound source changed: ' + name)
        if digest(predecessor_path) != predecessor_digest:
            raise ValueError('Predecessor result changed')

    verify()
    predecessor = json.loads(predecessor_path.read_text())
    plan = json.loads((ROOT / 'docs/reviews/competition_tgs_training_plan.json').read_text())
    previous_fit = next(f for f in plan['fits'] if f['key'] == 'keras.r1.p1.f0')
    expected = hashlib.sha256(json.dumps(previous_fit, sort_keys=True).encode()).hexdigest()
    if predecessor['fit'] != previous_fit['key'] or predecessor['fit_contract_sha256'] != expected:
        raise ValueError('Completed Lovasz predecessor contract required')
    receipt = predecessor['best_checkpoint']
    CheckpointStore(previous / 'checkpoints').get(receipt)

    # Recreate precisely the original synthetic generator; no new data recipe.
    tree = ast.parse((ROOT / 'scripts/run_salt_population_pilot.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    selected, active = [], False
    for node in function.body:
        names = [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else []
        if 'seed' in names:
            active = True
        if active:
            selected.append(node)
        if 'population' in names:
            break
    namespace = dict(np=np)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original-pilot-population>', 'exec'), namespace)
    population = namespace['population']
    if (namespace['seed'] != context['seed']
            or len(population['labeled_images']) != context['labeled_count']
            or len(population['query_images']) != context['query_count']):
        raise ValueError('Synthetic population identity differs')

    runtime.mkdir(parents=True, exist_ok=False)
    store = CheckpointStore(runtime / 'checkpoints')
    filename = receipt['sha256'] + '.pt'
    shutil.copyfile(previous / 'checkpoints' / filename, runtime / 'checkpoints' / filename)
    store.get(receipt)
    write_json(runtime / 'context.json', dict(seed=context['seed'], source_sha256=pins,
        predecessor_result_sha256=predecessor_digest, predecessor_checkpoint=receipt,
        fits=KEYS, labeled_count=context['labeled_count'], query_count=context['query_count']))
    completed = {predecessor['fit']: predecessor}
    phases = []
    torch.set_num_threads(2)
    for key in KEYS:
        verify()
        fit = next(f for f in plan['fits'] if f['key'] == key)
        if fit['controls']['callback'] != 'snapshot' or fit['epoch_ceiling'] != 40:
            raise ValueError('Expected full source snapshot contract')
        seed = int.from_bytes(hashlib.sha256(f"{context['seed']}:{key}".encode()).digest()[:4], 'little')
        torch.manual_seed(seed)
        print(json.dumps(dict(started=True, fit=key, epoch_ceiling=40,
            predecessor=fit['weight_predecessor'], full_source_controls=True)), flush=True)
        result = run_fit(key, plan, population, completed, {}, {}, store, rng=np.random.default_rng(seed))
        verify()
        if result['actual_epochs'] != 40 or result['optimizer_updates'] != 400:
            raise ValueError('Source snapshot schedule did not complete')
        if {str(epoch) for epoch in result['periodic_checkpoints']} != {'40'}:
            raise ValueError('Source cycle-end checkpoint required')
        if result['initialization']['checkpoint'] != completed[fit['weight_predecessor']]['best_checkpoint']:
            raise ValueError('Validation-best handoff differs')
        store.get(result['best_checkpoint'])
        for snapshot in result['periodic_checkpoints'].values():
            store.get(snapshot)
        write_json(runtime / (key + '.json'), result)
        completed[key] = result
        phases.append(dict(fit=key, actual_epochs=result['actual_epochs'],
            optimizer_updates=result['optimizer_updates'], best_source_metric=max(h['source_metric'] for h in result['history']),
            best_checkpoint_sha256=result['best_checkpoint']['sha256'],
            periodic_checkpoint_count=len(result['periodic_checkpoints']), validation_best_handoff_verified=True))
        report = dict(passed=len(phases) == len(KEYS), approved=False, synthetic_only=True,
            phases=phases, completed_fits=len(phases), required_fits=len(KEYS), source_sha256=pins,
            predecessor_result_sha256=predecessor_digest, catalog_mutations=0,
            limitations=['One fold of the first-round Keras branch; other folds, PyTorch branch and pseudo rounds remain required.',
                         'Synthetic execution does not establish empirical performance or authorize a full-workflow promotion.'])
        write_json(output, report)
        print(json.dumps(phases[-1]), flush=True)
    write_json(runtime / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-runtime', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.previous_runtime, args.runtime_directory, args.output)
