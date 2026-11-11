#!/usr/bin/env python3
"""Execute every ensemble node on shared-identity synthetic raw clips."""
import argparse
import ast
import asyncio
import hashlib
import importlib
import inspect
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


def synthetic_inputs():
    result = {name: [] for name in [
        'alex_training_populations', 'alex_prediction_populations', 'alex_label_populations', 'alex_identity_populations',
        'feng_training_populations', 'feng_prediction_populations', 'feng_label_populations', 'feng_identity_populations',
        'csp_candidates', 'candidate_sequences', 'primary_positive', 'auxiliary_positive', 'negative',
        'prediction', 'sampling_frequencies', 'andriy_identity_populations']}
    fs = 400
    for population in range(3):
        rng = np.random.default_rng(9041 + population)
        def clip():
            value = rng.standard_normal((16, 600*fs), dtype=np.float32)
            value *= np.repeat(np.exp(rng.normal(0, .2, (16, 20))), 30*fs, axis=1).astype(np.float32)
            return value
        candidates = [clip() for _ in range(4)]
        positive = [clip() for _ in range(14)]
        negative = [clip() for _ in range(14)]
        prediction = [clip() for _ in range(2)]
        training = [positive[0], negative[0], positive[1], negative[1]]
        labels = np.array([1, 0, 1, 0])
        ids = np.array([101+2*population, 102+2*population])
        values = [training, prediction, labels, ids,
            [x.T for x in training], [x.T for x in prediction[::-1]], labels, ids[::-1],
            candidates, np.array([6, 1, 1, 6]), positive[:-1], positive[-1:], negative,
            prediction, fs, ids]
        for key, value in zip(result, values):
            result[key].append(value)
    result['output_ids'] = np.array([103, 101, 102, 104, 106, 105])
    result['baseline'] = np.zeros(6)
    return result


def source_blend(reference_dir, scores, identities, output_ids, baseline):
    source_bytes = (reference_dir/'make_blend.py').read_bytes()
    sha = '6a64446123cfa6d3a867837057cda256cf13442c674317bc239117d00520374b'
    assert hashlib.sha256(source_bytes).hexdigest() == sha
    source = source_bytes.decode()
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'weights' for t in n.targets)]
    weights = ast.literal_eval(assignments[0].value)
    assert len(weights) == 11 and set(weights.values()) == {1.}
    for before, after in {
        "default = pd.read_csv('./csv_files/sample_submission.csv', index_col=0)": 'default = fixture_default.copy(deep=True)',
        "res = pd.read_csv(base + fn, index_col='File')": 'res = fixture_predictions[fn].copy(deep=True)',
        'np.sum(weights.values())': 'np.sum(list(weights.values()))',
        "default.to_csv('./submissions/Winning_submission.csv')": '',
    }.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    namespace = dict(fixture_default=pd.DataFrame({'Class': baseline}, index=output_ids),
        fixture_predictions={name: pd.DataFrame({'Class': values}, index=pd.Index(ids, name='File'))
                             for name, values, ids in zip(weights, scores, identities)})
    exec(compile(source, '<synthetic-source-final-blend>', 'exec'), namespace)
    return namespace['default']['Class'].to_numpy()


async def validate(args):
    root = Path(__file__).resolve().parents[1]
    graph = build_ensemble_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({
        'cdg_nodes': [{**n, 'version_id': 'validation'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'validation'} for e in edges]},
        version_id='validation', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    provider_paths = {}
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        function = getattr(importlib.import_module(module), name)
        provider_paths[node.matched_primitive] = Path(inspect.getsourcefile(function))
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in provider_paths.items()}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    inputs = synthetic_inputs()
    old_dir, old_save, old_run = runner.RUNS_DIR, runner.save_intermediate_value, subprocess.run
    last_outputs = {n.node_id: 'out_'+n.outputs[-1].name for n in graph.nodes}
    completed = []
    def save(directory, node_id, name, value):
        if node_id == 'andriy_populations' and name.startswith('out_'):
            np.savez(args.run_dir/(name+'.npz'), **{f'population_{i}': x for i, x in enumerate(value)})
        result = old_save(directory, node_id, name, value)
        if name == last_outputs[node_id]:
            completed.append(node_id)
            print(f'Completed {len(completed)}/{len(nodes)}: {node_id}', flush=True)
        return result
    def run(*a, **kw):
        result = old_run(*a, **kw)
        if result.returncode:
            message = result.stderr or b''
            (args.run_dir/'failed_subprocess_stderr.txt').write_bytes(message.encode() if isinstance(message, str) else message)
        return result
    runner.RUNS_DIR, runner.save_intermediate_value, subprocess.run = args.run_dir, save, run
    started = time.monotonic()
    try:
        result = await runner.CDGExecutionSession(None, 'synthetic-eleven-model-ensemble', 'check').execute(inputs, cdg=restored)
    finally:
        runner.RUNS_DIR, runner.save_intermediate_value, subprocess.run = old_dir, old_save, old_run
    assert result['status'] == 'completed' and len(result['trace']) == len(nodes)
    assert all(not item['cached'] for item in result['trace'])
    pack = next(n for n in graph.nodes if n.node_id == 'pack')
    scores = [np.load(args.run_dir/'check/pack'/('in_'+port.name+'.npy')) for port in pack.inputs[:11]]
    ids = [np.load(args.run_dir/'check/pack'/('in_'+name+'_ids.npy')) for name in ['alex', 'feng', 'andriy']]
    identities = [ids[0]]*4 + [ids[1]]*4 + [ids[2]]*3
    assert all(value.shape == (6,) and np.isfinite(value).all() for value in scores)
    expected = source_blend(args.reference_dir, scores, identities, inputs['output_ids'], inputs['baseline'])
    actual = np.load(args.run_dir/'check/blend/out_ensemble_scores.npy')
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (6,) and np.isfinite(actual).all() and np.ptp(actual) > 0
    for name, path in provider_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
    report = dict(graph_digest=digest, nodes=len(nodes), edges=len(edges), executed_nodes=len(result['trace']),
        full_runner_cases=1, synthetic_only=True, populations=3, prediction_segments=6,
        exact_final_source_blend=True, final_score_spread=float(np.ptp(actual)),
        model_score_spreads=[float(np.ptp(value)) for value in scores],
        duration_seconds=time.monotonic()-started, provider_sha256=hashes,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        graph_source_sha256=hashlib.sha256((root/'sciona/ensemble_execution.py').read_bytes()).hexdigest(),
        limitations=['Full raw graph execution with shared physical synthetic prediction clips, permuted Feng order and explicit output order.',
            'Four training clips per Alex/Feng population and 28 per Andriy population; explicit synthetic training groups, no real source membership inference.',
            'Exact final source blending of actual branch outputs; independent upstream numerical reference reports remain separate evidence.',
            'Current runtime adaptations apply; no historical-engine equivalence or predictive-quality claim.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(full_ensemble_completed=True, executed_nodes=len(nodes), final_score_spread=report['final_score_spread'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    asyncio.run(validate(parser.parse_args()))
