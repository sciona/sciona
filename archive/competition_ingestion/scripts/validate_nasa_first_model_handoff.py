"""Verify existing synthetic native checkpoints through explicit slot bindings."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from sciona.model_slot_bank import assemble_model_slot_bank, select_model_slots
from sciona.nasa_first_prediction import nasa_first_prediction
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_native_wrapper import source_function, TRAIN_SOURCE, TRAIN_PIN, ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source, checkpoint, *, graph_selector=None):
    report = json.loads((ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json').read_text())
    if json.loads((checkpoint/'qualification.json').read_text()) != report or not report['passed']:
        raise ValueError('Qualified synthetic checkpoint evidence differs')
    for name, digest in report['local_source_closure_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Qualified native source drift')
    fits = json.loads((checkpoint/'fit_progress.json').read_text())
    if fits != report['fits'] or len(fits) != 21:
        raise ValueError('Twenty-one qualified fits required')
    models = {}
    for index, fit in enumerate(fits):
        path = checkpoint/f'model_{index}.cbm'
        if sha(path) != fit['model_sha256']:
            raise ValueError('Native checkpoint integrity differs')
        model = CatBoostRegressor(thread_count=1)
        model.load_model(str(path))
        models[str(index)] = model
    solution = source/'1st Place/Phase 1/submission/solution.py'
    if sha(solution) != '854eb1f0874fa77748b26673cc848c8fe4dde38c76e7b474694069c05741e917':
        raise ValueError('Pinned source dispatcher differs')
    loader = next(node for node in ast.parse(solution.read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name == 'load_model')
    labels = ast.literal_eval(loader.body[0].value)
    residual = source_function(source, TRAIN_SOURCE, TRAIN_PIN, 'train_catboost_diff')
    cutoff = next(node for node in ast.walk(residual) if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.Gt))
    start = pd.Timestamp(cutoff.comparators[0].value) + pd.Timedelta(days=1)
    bindings = {label.upper(): {0: str(2*i), 1: str(2*i+1), 2: str(2*i)} for i, label in enumerate(labels)}
    bank = assemble_model_slot_bank(models, bindings, {'global_model': '20'})
    if len({id(model) for slots in bank.values() for model in slots.values()}) != 21:
        raise ValueError('Independent model state count differs')
    queries_checked = 0
    for index, label in enumerate(labels):
        _, _, queries, raw, policy = population_inputs(index, start)
        queries = queries.iloc[128:].copy()
        selected = (select_model_slots(bank, label.upper()) if graph_selector is None else
                    graph_selector(models, bindings, {'global_model': '20'}, label.upper()))
        if selected[0] is not selected[2] or selected['global_model'] is not models['20']:
            raise ValueError('Model alias handoff differs')
        expected, route = nasa_first_prediction(queries, {0: models[str(2*index)], 1: models[str(2*index+1)],
            2: models[str(2*index)], 'global_model': models['20']}, label, raw_options=raw, **policy)
        actual, actual_route = nasa_first_prediction(queries, selected, label, raw_options=raw, **policy)
        if route != actual_route or route != 'primary':
            raise ValueError('Native handoff fell back')
        pd.testing.assert_frame_equal(actual, expected)
        queries_checked += len(queries)
        # A sparse bank can serve the same source single-population invocation.
        sparse_models = {name: models[name] for name in [str(2*index), str(2*index+1), '20']}
        sparse = assemble_model_slot_bank(sparse_models, {label.upper(): bindings[label.upper()]}, {'global_model': '20'})
        sparse_selected = (select_model_slots(sparse, label.upper()) if graph_selector is None else
                           graph_selector(sparse_models, {label.upper(): bindings[label.upper()]}, {'global_model': '20'}, label.upper()))
        for slot, model in selected.items():
            if sparse_selected[slot] is not model:
                raise ValueError('Sparse population state differs')
    try:
        select_model_slots(bank, 'synthetic-missing-population')
    except KeyError:
        missing_rejected = True
    else:
        raise ValueError('Unknown population silently routed')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True, native_threads=1,
        native_models=21, local_slots=30, global_model_shared=True, alias_pairs=10,
        selection_mode='direct' if graph_selector is None else 'injected_graph_selector',
        distinct_query_comparisons=queries_checked, sparse_population_banks_checked=10,
        missing_population_rejected=missing_rejected,
        qualification_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json'),
        implementation_sha256={name: sha(ROOT/name) for name in ['sciona/model_slot_bank.py',
            'tests/test_model_slot_bank.py', 'scripts/validate_nasa_first_model_handoff.py']},
        limitations=['Existing qualified synthetic checkpoints reused without retraining.',
            'This establishes in-process model identity and slot handoff, not a serialized complete CDG.',
            'Source prediction remains one population per call; no added multi-population prediction interface.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_directory, args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_model_handoff.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, native_models=21, distinct_query_comparisons=result['distinct_query_comparisons'])))
