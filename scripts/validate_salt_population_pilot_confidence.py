"""Inspect the completed population pilot; this is not a full ensemble round."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sciona.tgs_checkpoint_store import CheckpointStore
from sciona.tgs_keras_inference import predict_checkpoints
from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_pseudo_selection import select_pseudo_labels

ROOT = Path(__file__).resolve().parents[1]


def validate(runtime):
    context = json.loads((runtime / 'context.json').read_text())
    for name, digest in context['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pilot-bound source changed: ' + name)
    result = json.loads((runtime / 'result.json').read_text())
    if result['fit'] != 'keras.r1.p0.f0' or result['optimizer_updates'] != 500:
        raise ValueError('Unexpected completed pilot')
    # Execute the original synthetic generator statements, avoiding a second data recipe.
    tree = ast.parse((ROOT / 'scripts/run_salt_population_pilot.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    selected, active = [], False
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'seed' for t in node.targets):
            active = True
        if active:
            selected.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'population' for t in node.targets):
            break
    namespace = dict(np=np)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original-synthetic-generator>', 'exec'), namespace)
    population = namespace['population']
    if len(population['query_images']) != context['query_count']:
        raise ValueError('Population differs')
    torch.set_num_threads(2)
    store = CheckpointStore(runtime / 'checkpoints')
    receipt = result['best_checkpoint']
    state = store.get(receipt)
    predictions = predict_checkpoints(TGSResNeXt50(), [state], population['query_images'], batch_size=8)[0]
    confidence = ((predictions < .2) | (predictions > .8)).mean(axis=(1, 2))
    area = (predictions > .5).sum(axis=(1, 2))
    selected = select_pseudo_labels(confidence, area, population['query_nonconstant'])
    return dict(format='salt-population-pilot-confidence.v1', passed=True, approved=False, synthetic_only=True,
        checkpoint_sha256=receipt['sha256'], checkpoint_integrity_verified=True, query_count=len(predictions),
        prediction_min=float(predictions.min()), prediction_max=float(predictions.max()),
        confidence_min=float(confidence.min()), confidence_max=float(confidence.max()),
        confidence_mean=float(confidence.mean()), keras_candidate_count=len(selected['keras_indices']),
        torch_candidate_counts=[len(fold) for fold in selected['torch_folds']],
        source_sha256=context['source_sha256'],
        fit_result_sha256=hashlib.sha256((runtime / 'result.json').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), catalog_mutations=0,
        limits=['Single validation-best first-phase checkpoint with source TTA and quantization, not the complete source ensemble.',
                'Candidate counts apply source thresholds to this diagnostic checkpoint only; they do not authorize a pseudo-label round.',
                'Full 63-fit workflow, branch blending and complete pseudo-round selection remain required.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.runtime_directory)
    (ROOT / 'docs/reviews/competition_tgs_population160_confidence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}))
