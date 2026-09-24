"""Continue the completed synthetic population pilot through source Lovasz phase."""
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


def main(previous, runtime, output):
    previous, runtime = previous.resolve(), runtime.resolve()
    if runtime == ROOT or ROOT in runtime.parents or runtime == previous:
        raise ValueError('Independent private runtime required')
    context = json.loads((previous / 'context.json').read_text())
    digests = dict(context['source_sha256'])
    digests[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def verify():
        for name, digest in digests.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Bound source changed: ' + name)
    verify()
    result_path = previous / 'result.json'
    result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    predecessor = json.loads(result_path.read_text())
    plan = json.loads((ROOT / 'docs/reviews/competition_tgs_training_plan.json').read_text())
    key = 'keras.r1.p1.f0'
    fit = next(f for f in plan['fits'] if f['key'] == key)
    prior_fit = next(f for f in plan['fits'] if f['key'] == fit['weight_predecessor'])
    expected = hashlib.sha256(json.dumps(prior_fit, sort_keys=True).encode()).hexdigest()
    if predecessor['fit'] != prior_fit['key'] or predecessor['fit_contract_sha256'] != expected:
        raise ValueError('Predecessor contract differs')
    receipt = predecessor['best_checkpoint']
    CheckpointStore(previous / 'checkpoints').get(receipt)
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
    if namespace['seed'] != context['seed'] or len(population['labeled_images']) != context['labeled_count']:
        raise ValueError('Population identity differs')
    runtime.mkdir(parents=True, exist_ok=False)
    store = CheckpointStore(runtime / 'checkpoints')
    filename = receipt['sha256'] + '.pt'
    shutil.copyfile(previous / 'checkpoints' / filename, runtime / 'checkpoints' / filename)
    store.get(receipt)
    bound = dict(seed=context['seed'], source_sha256=digests, predecessor_result_sha256=result_digest,
                 predecessor_checkpoint=receipt, fit=key, labeled_count=160, query_count=64)
    (runtime / 'context.json').write_text(json.dumps(bound, indent=2) + '\n')
    seed = int.from_bytes(hashlib.sha256(f"{context['seed']}:{key}".encode()).digest()[:4], 'little')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    print(json.dumps(dict(started=True, fit=key, loss=fit['controls']['loss_function'],
        epoch_ceiling=fit['epoch_ceiling'], early_stop_patience=fit['controls']['early_stop_patience'],
        predecessor_verified=True, full_source_controls=True)), flush=True)
    result = run_fit(key, plan, population, {predecessor['fit']: predecessor}, {}, {}, store, rng=np.random.default_rng(seed))
    verify()
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != result_digest:
        raise ValueError('Predecessor result changed')
    (runtime / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    report = dict(passed=True, approved=False, synthetic_only=True, fit=key,
        actual_epochs=result['actual_epochs'], optimizer_updates=result['optimizer_updates'],
        best_source_metric=max(h['source_metric'] for h in result['history']),
        source_sha256=digests, predecessor_result_sha256=result_digest,
        catalog_mutations=0, limitations=['Single-fold full source Lovasz phase; complete ensemble and 63-fit workflow remain pending.',
            'Exact synthetic population and validation-best predecessor; optimizer reset follows source phase handoff.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-runtime', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.previous_runtime, args.runtime_directory, args.output)
