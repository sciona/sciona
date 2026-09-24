"""Execute the complete primary graph using qualified synthetic native state."""
import argparse
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from catboost import CatBoostRegressor
import pandas as pd

from sciona.nasa_first_primary_graph import build_nasa_first_primary_graph
from sciona.nasa_first_prediction import nasa_first_prediction
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_model_handoff import ROOT, sha
from scripts.validate_nasa_first_native_wrapper import source_function, TRAIN_SOURCE, TRAIN_PIN


def validate(source, checkpoint):
    native = json.loads((ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json').read_text())
    if json.loads((checkpoint/'qualification.json').read_text()) != native or not native['passed']:
        raise ValueError('Qualified native evidence differs')
    for name, digest in native['local_source_closure_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Qualified native source drift')
    fits = json.loads((checkpoint/'fit_progress.json').read_text())
    if fits != native['fits'] or len(fits) != 21:
        raise ValueError('Qualified fit topology differs')
    models = []
    for index, fit in enumerate(fits):
        path = checkpoint/f'model_{index}.cbm'
        if sha(path) != fit['model_sha256']:
            raise ValueError('Checkpoint integrity differs')
        model = CatBoostRegressor(thread_count=1)
        model.load_model(str(path))
        models.append(model)
    solution = source/'1st Place/Phase 1/submission/solution.py'
    if sha(solution) != '854eb1f0874fa77748b26673cc848c8fe4dde38c76e7b474694069c05741e917':
        raise ValueError('Pinned dispatcher differs')
    loader = next(n for n in ast.parse(solution.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    labels = ast.literal_eval(loader.body[0].value)
    residual = source_function(source, TRAIN_SOURCE, TRAIN_PIN, 'train_catboost_diff')
    cutoff = next(n for n in ast.walk(residual) if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Gt))
    start = pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    original = build_nasa_first_primary_graph()
    digest, nodes, edges = encode_execution_graph(original)
    graph = decode_execution_graph(nodes, edges, digest)
    if graph != original:
        raise ValueError('Primary graph codec differs')
    compared = 0
    with tempfile.TemporaryDirectory(prefix='sciona-first-primary-') as directory:
        for index, label in enumerate(labels):
            _, _, queries, raw, policy = population_inputs(index, start)
            queries = queries.iloc[128:].copy()
            slots = {0: models[2*index], 1: models[2*index+1], 2: models[2*index], 'global_model': models[20]}
            for selected in (queries, queries.iloc[[63, 0, 63]]):
                expected, route = nasa_first_prediction(selected, slots, label, raw_options=raw, **policy)
                if route != 'primary':
                    raise ValueError('Reference primary fell back')
                captured = {}
                def capture(path, node, port, value):
                    if node == 'format' and port == 'out_predictions':
                        captured['predictions'] = value
                with patch.object(runner, 'RUNS_DIR', Path(directory)), patch.object(runner, 'save_intermediate_value', side_effect=capture), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-first-primary', str(compared)).execute(
                        dict(queries=selected, raw_options=raw, models=slots, population=label, integer_dtype='int16', **policy), cdg=graph))
                if result['status'] != 'completed' or 'predictions' not in captured:
                    raise ValueError('Primary graph failed')
                pd.testing.assert_frame_equal(captured['predictions'], expected)
                compared += len(selected)
    provider_paths = ['model_selection/materialized_prediction.py', 'domain_adapters/first_place_prediction.py',
                      'domain_adapters/first_place_features.py']
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True, native_threads=1,
        nodes=len(graph.nodes), edges=len(graph.edges), graph_sha256=digest, native_models=21,
        graph_cases=20, distinct_query_comparisons=640, duplicate_query_comparisons=30, total_compared=compared,
        exact_frame_equality=True, codec_roundtrip=True, production_executor=True,
        native_qualification_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json'),
        implementation_sha256={name: sha(ROOT/name) for name in ['sciona/nasa_first_primary_graph.py',
            'sciona/nasa_first_feature_graph.py', 'sciona/nasa_first_graph_adapters.py', 'sciona/materialized_prediction.py',
            'scripts/validate_nasa_first_primary_graph.py', 'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py']},
        provider_sha256={name: sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in provider_paths},
        limitations=['Primary prediction only; fallback routing, empty-query short circuit and training graph remain pending.',
            'Previously qualified synthetic models reused without retraining; no empirical performance claim.',
            'Intermediate persistence replaced by capture; all calculations and graph dispatch use the production executor.',
            'New provider catalog binding and publication remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_directory, args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_primary_graph.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, nodes=result['nodes'], edges=result['edges'], total_compared=result['total_compared'])))
