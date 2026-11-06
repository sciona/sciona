"""Record complete synthetic graph execution without persisting runtime data."""
import asyncio
import contextlib
import copy
import importlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_info
from sciona.architect.models import EdgeKind
from sciona.decomposed_plasticc_graph import build_decomposed_plasticc_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.atoms.ml import decomposed_plasticc_numeric as numeric
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from sciona import decomposed_plasticc_observations as observations
from sciona import decomposed_plasticc_training as training
from sciona import decomposed_plasticc_prediction as prediction
from sciona import plasticc_training as reference_training
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from sciona.visualizer import runner
from scripts.validate_plasticc_runtime import synthetic_payload
from scripts.audit_decomposed_plasticc_readiness import ROOT, sha


def validate():
    paths = [str(p.relative_to(ROOT)) for pattern in ('plasticc_*.py', 'decomposed_plasticc_*.py')
             for p in sorted((ROOT/'sciona').glob(pattern))]
    paths += ['scripts/validate_decomposed_plasticc_graph_v2.py', 'scripts/validate_plasticc_runtime.py',
              'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py']
    before = {p: sha(ROOT/p) for p in paths}
    graph = build_decomposed_plasticc_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest or len(nodes) != 26:
        raise ValueError('Complete graph roundtrip required')
    functions = {}; hashes = {}
    for node in graph.nodes:
        module, symbol = node.matched_primitive.rsplit('.', 1)
        fn = getattr(importlib.import_module(module), symbol)
        if list(inspect.signature(fn).parameters) != [p.name for p in node.inputs]:
            raise ValueError('Graph/provider input signature differs')
        functions[node.matched_primitive] = fn
        hashes[node.matched_primitive] = sha(inspect.getfile(inspect.unwrap(fn)))
    if len(functions) != 25:
        raise ValueError('Exactly24 new providers plus the existing kernel required')
    callable_nodes = {e.source_id for e in graph.edges if e.edge_kind == EdgeKind.CALLABLE_INJECTION}
    if len(callable_nodes) != 12:
        raise ValueError('Twelve explicit callable nodes required')
    expected_operations = dict(amplitude=numeric.initial_amplitude, fit=numeric.fit_gp,
        kernel=matern_kernel_matrix, scale=numeric.scale_covariance, noise=numeric.observation_variance,
        condition=numeric.condition, predict=numeric.predict_gp)
    original_check = observations.checked_operations
    original_folds = training.assign_folds
    original_weights = training.training_weights
    original_average = prediction.average_folds
    original_fit = reference_training.fit_lightgbm_classifier
    runner._ensure_atoms_imported()
    cases = []
    for weighting, class_weighting in [('flat', 'flat'), ('redshift', 'kaggle')]:
        payload = synthetic_payload()
        payload['config'].update(weighting=weighting, class_weighting=class_weighting)
        saved = copy.deepcopy(payload); captured = {}; counts = dict(gp_bindings=0, folds=0, weights=0, average=0, models=0)
        def checked(operations):
            if operations != expected_operations: raise ValueError('Exact registered GP bindings required')
            counts['gp_bindings'] += 1
            return original_check(operations)
        def folds(population, prepared, operation):
            if operation is not numeric.reference_stratified_folds: raise ValueError('Exact fold binding required')
            counts['folds'] += 1
            return original_folds(population, prepared, operation)
        def weights(population, prepared, operation):
            if operation is not numeric.balanced_weights: raise ValueError('Exact balance binding required')
            counts['weights'] += 1
            return original_weights(population, prepared, operation)
        def average(models, features, classes, score, normalize):
            if score is not numeric.predict_scores or normalize is not numeric.normalize_scores:
                raise ValueError('Exact prediction bindings required')
            if len(models) != payload['config']['num_folds']: raise ValueError('All fitted models required')
            counts['average'] += 1
            return original_average(models, features, classes, score, normalize)
        def fit(*args, **kwargs):
            if kwargs != payload['config']['training_parameters']: raise ValueError('Configured training controls changed')
            if kwargs['n_jobs'] != 1: raise ValueError('One-worker fitting required')
            counts['models'] += 1
            return original_fit(*args, **kwargs)
        def capture(directory, node, name, value):
            if node == 'encode' and name == 'out_result': captured['result'] = value
        with tempfile.TemporaryDirectory(prefix='synthetic-plasticc-graph-') as directory:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(runner, 'RUNS_DIR', Path(directory)), patch.object(runner, 'save_intermediate_value', side_effect=capture), \
                     patch.object(observations, 'checked_operations', side_effect=checked), \
                     patch.object(training, 'assign_folds', side_effect=folds), patch.object(training, 'training_weights', side_effect=weights), \
                     patch.object(prediction, 'average_folds', side_effect=average), patch.object(reference_training, 'fit_lightgbm_classifier', side_effect=fit):
                    status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-plasticc', 'decomposed').execute({'payload': payload}, cdg=graph))
                expected = predict_runtime(train_runtime(prepare_runtime(payload)))
        if status['status'] != 'completed': raise ValueError('Complete registered graph failed')
        traces = {t['node_id']: t for t in status['trace']}
        if set(traces) != {n.node_id for n in graph.nodes} or any(t['cached'] for t in traces.values()):
            raise ValueError('Every graph node must execute uncached')
        if not all(traces[n].get('callable_reference') is True for n in callable_nodes):
            raise ValueError('Callable references must be explicit')
        if counts['gp_bindings'] <= 0 or any(counts[k] != 1 for k in ('folds', 'weights', 'average')) or counts['models'] != payload['config']['num_folds']:
            raise ValueError('Complete numeric binding and model execution required')
        actual = captured['result']
        if actual['classes'] != expected['classes'] or actual['object_ids'] != expected['object_ids'] or payload != saved:
            raise ValueError('Output alignment or root preservation differs')
        np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-8, rtol=1e-8)
        cases.append(dict(weighting=weighting, class_weighting=class_weighting, models=counts['models'],
            configured_tree_ceiling=payload['config']['training_parameters']['n_estimators'],
            all_nodes_uncached=True, exact_numeric_bindings=True, root_input_unchanged=True,
            output_alignment_equal=True, probabilities_match=True, atol=1e-8, rtol=1e-8))
    if any(p['num_threads'] != 1 for p in threadpool_info()): raise ValueError('Native thread budget exceeded')
    if before != {p: sha(ROOT/p) for p in paths}: raise ValueError('Implementation changed during execution')
    if any(sha(inspect.getfile(inspect.unwrap(functions[n]))) != h for n, h in hashes.items()):
        raise ValueError('Provider changed during execution')
    return dict(format='decomposed-plasticc-graph.v2', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, graph_sha256=digest, nodes=len(nodes), edges=len(edges), distinct_providers=len(hashes),
        new_providers=24, callable_nodes=sorted(callable_nodes), cases=cases, provider_sha256=hashes,
        implementation_sha256=before, resource_policy=dict(serial_cases=True, serial_models=True, native_threads=1),
        limits=['Complete configured synthetic executions, not historical population size or competition accuracy.',
                'Synthetic configuration uses three folds, one augmentation attempt per original and a60-tree ceiling.',
                'Exact callable bindings and numeric output parity do not replace semantic interface qualification.',
                'No catalog mutation or approval; original CDG and new providers remain pending qualification.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/decomposed_plasticc_graph_execution_v2.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, nodes=result['nodes'], edges=result['edges'], cases=len(result['cases']), approved=False)))
