"""Execute all native meta bags through the draft 12-node graph, synthetic only."""
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

import sciona.atoms.ml.stacking_meta as provider
from sciona import stacking_meta_operations as operations
from sciona import otto_meta_tuning as tuning
from sciona.stacking_meta_graph import build_meta_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.review_otto_candidate_runtime import review as runtime_review

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture():
    rng = np.random.default_rng(12)
    y = np.tile(np.arange(9), 4)
    x = np.eye(9)[y]*10 + rng.uniform(0, .2, (len(y), 9))
    q = np.eye(9)*10 + .1
    layout = dict(tree_training=x, tree_query=q,
                  neural_training=np.column_stack((x, np.ones((len(y), 2)))),
                  neural_query=np.column_stack((q, np.ones((9, 2)))))
    candidates = {
        'xgboost': [dict(rounds=n, max_depth=3, eta=.4, subsample=.8, colsample_bytree=.8) for n in (4, 8)],
        'lasagne_neural': [dict(hidden=[16], epochs=n, batch_size=64, learning_rate=.1, momentum=.9, ddof=1) for n in (1, 20)],
        'adaboost_extratrees': [dict(algorithm='SAMME', boost_rounds=4, learning_rate=.7, trees=3,
                                   max_depth=n, min_samples_leaf=2, max_features=9) for n in (1, 3)]}
    return dict(layout=layout, labels=y, folds=np.arange(len(y)) % 4,
                identities=[f'synthetic-meta-{i}' for i in range(len(y))], seed=12, candidates=candidates)


def main():
    if not __debug__:
        raise RuntimeError('Validation assertions required')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError('Explicit single-threaded native pools required')
    runtime = runtime_review()
    source_paths = list((ROOT/'sciona').glob('otto_*.py')) + [
        ROOT/'sciona/stacking_meta_operations.py', ROOT/'sciona/stacking_meta_graph.py',
        ROOT/'sciona/probability_ensemble.py', Path(__file__).resolve()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    provider_path = Path(inspect.getfile(provider))
    provider_hash = sha(provider_path)
    pools = threadpool_info()
    if any(p['num_threads'] != 1 for p in pools):
        raise ValueError('Loaded numerical pool exceeds the one-core allocation')
    inputs = fixture()
    digest, nodes, edges = encode_execution_graph(build_meta_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    outputs, calls = {}, []
    original = tuning._learner

    def instrumented(family):
        learner, runs = original(family)
        def fit(x, y, q, *, seed, controls):
            bag = learner(x, y, q, seed=seed, controls=controls)
            assert bag.shape == (runs, len(q), 9)
            calls.append(dict(family=family, fitted_models=runs,
                              fitting_rows=len(x), prediction_rows=len(q)))
            print(json.dumps(dict(completed_native_bag=len(calls), **calls[-1])), flush=True)
            return bag
        return fit, runs

    def capture(directory, node, port, value):
        if port.startswith('out_'):
            outputs[(node, port)] = value
            print('Completed meta node: '+node, flush=True)

    with tempfile.TemporaryDirectory(prefix='synthetic-stacking-meta-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture), \
             patch.object(tuning, '_learner', side_effect=instrumented), \
             patch.object(operations, '_learner', side_effect=instrumented):
            result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-meta', 'native').execute(inputs, cdg=graph))
    assert result['status'] == 'completed'
    assert {node for node, port in outputs} == {node.node_id for node in graph.nodes}
    final = outputs[('result', 'out_result')]
    probabilities = np.asarray(final['blend']['probabilities'])
    assert probabilities.shape == (9, 9) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0)
    for family in operations.FAMILIES:
        family_calls = [c for c in calls if c['family'] == family]
        assert len(family_calls) == 9
        assert [(c['fitting_rows'], c['prediction_rows']) for c in family_calls] == [(27, 9)]*8+[(36, 9)]
        assert sum(c['fitted_models'] for c in family_calls) == 9*operations.RUNS[family]
    assert sum(s['tuning_model_fits'] for s in final['selection'].values()) == 8800
    assert sum(s['refit_models'] for s in final['selection'].values()) == 1100
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert sha(provider_path) == provider_hash
    assert runtime_review() == runtime
    report = dict(status='passed', approved=False, catalog_mutations=0,
        serialized_graph_sha256=digest, actual_runner_nodes=len(graph.nodes),
        native_tuning_fits=8800, native_refits=1100, completed_native_bags=calls,
        provider_sha256=provider_hash, implementation_sha256=before,
        candidate_runtime_evidence_sha256=runtime['evidence_sha256'],
        selection=final['selection'], strict_json_result=True,
        limits=['Synthetic meta-stage execution only; upstream first-level and supplemental producers are not executed.',
                'No original-intake completion, new provider approval or empirical effectiveness claim.',
                'Prepared objects are qualified only within one in-process uncached graph execution.'])
    json.dumps(final, allow_nan=False)
    (ROOT/'docs/reviews/competition_otto_decomposed_meta_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='passed', actual_runner_nodes=len(graph.nodes), native_model_fits=9900)), flush=True)


if __name__ == '__main__':
    main()
