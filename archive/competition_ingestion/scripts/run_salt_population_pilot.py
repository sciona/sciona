"""Independent full-budget first-phase pilot for synthetic population adequacy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sciona.tgs_fit_execution import run_fit
from sciona.tgs_checkpoint_store import CheckpointStore


def main(runtime, reference_root, output):
    root = Path(__file__).resolve().parents[1]
    runtime = runtime.resolve()
    if runtime == root or root in runtime.parents:
        raise ValueError('private runtime outside repository required')
    runtime.mkdir(parents=True, exist_ok=False)
    seed = 7913
    rng = np.random.default_rng(seed)
    masks = np.zeros((224, 101, 101), dtype=bool)
    for index in range(len(masks)):
        if index % 2:
            start = int(rng.integers(10, 50))
            masks[index, :, start:start+30] = True
    images = (32 + masks.astype(np.uint8)*180 + rng.integers(0, 16, masks.shape, dtype=np.uint8)).astype(np.uint8)
    population = dict(labeled_images=np.repeat(images[:160, ..., None], 3, axis=-1),
        query_images=np.repeat(images[160:, ..., None], 3, axis=-1), labeled_masks=masks[:160],
        folds=np.tile(np.arange(5), 32), labeled_nonconstant=np.ones(160, dtype=bool),
        query_nonconstant=np.ones(64, dtype=bool), pseudo_validation=np.arange(5))
    plan_path = root / 'docs/reviews/competition_tgs_training_plan.json'
    plan = json.loads(plan_path.read_text())
    sources = list((root/'sciona').glob('tgs_*.py')) + [root/'sciona/hubmap_losses.py', Path(__file__), plan_path]
    digests = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    references = {'resnext50': dict(path=str(reference_root/'resnext50'),
        sha256='3bcb9dedb226c5e5cdd3510d25cdc33297f59016a6d7069758024caa13e3172d')}
    context = dict(seed=seed, labeled_count=160, query_count=64, source_sha256=digests,
                   fit='keras.r1.p0.f0', independent_population=True)
    (runtime/'context.json').write_text(json.dumps(context, indent=2)+'\n')
    fit_seed = int.from_bytes(hashlib.sha256(f"{seed}:keras.r1.p0.f0".encode()).digest()[:4], 'little')
    torch.set_num_threads(2)
    torch.manual_seed(fit_seed)
    print(json.dumps(dict(started=True, fit='keras.r1.p0.f0', labeled_count=160,
                          training_count=128, validation_count=32, steps_per_epoch=10,
                          source_epoch_ceiling=125, source_early_stopping=True)), flush=True)
    result = run_fit('keras.r1.p0.f0', plan, population, {}, {}, references,
                     CheckpointStore(runtime/'checkpoints'), rng=np.random.default_rng(fit_seed))
    for name, digest in digests.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('pilot source changed during execution')
    (runtime/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    history = result['history']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        fit='keras.r1.p0.f0', independent_population=True, labeled_count=160,
        actual_epochs=result['actual_epochs'], optimizer_updates=result['optimizer_updates'],
        steps_per_epoch=result['steps_per_epoch'],
        best_source_metric=max(row['source_metric'] for row in history),
        best_uncapped_metric=max(row['uncapped_metric'] for row in history),
        source_sha256=digests,
        limits=['Population-adequacy pilot for one full source phase; not a replacement for the 63-fit workflow.',
                'Existing completed fits and pseudo-round results were not reused or changed.',
                'Pseudo-label confidence and remaining phases require separate execution.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime_directory, args.reference_directory, args.output)
