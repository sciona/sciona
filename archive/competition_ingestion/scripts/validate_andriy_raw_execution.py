#!/usr/bin/env python3
"""Execute the complete Andriy graph on synthetic raw population clips."""
import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import numpy as np
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


@contextmanager
def synthetic_diagnostics(directory):
    """Retain synthetic intermediate arrays and failed subprocess diagnostics."""
    if directory is None:
        yield
        return
    directory.mkdir(parents=True, exist_ok=False)
    original_save = runner.save_intermediate_value
    original_run = subprocess.run
    def save(run_dir, node_id, name, value):
        if node_id == 'populations' and name.startswith('out_'):
            np.savez(directory / (name + '.npz'), **{f'population_{i}': x for i, x in enumerate(value)})
        return original_save(run_dir, node_id, name, value)
    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if result.returncode:
            message = result.stderr or b''
            if isinstance(message, str):
                message = message.encode()
            (directory / 'failed_subprocess_stderr.txt').write_bytes(message)
        return result
    runner.save_intermediate_value = save
    subprocess.run = run
    try:
        yield
    finally:
        runner.save_intermediate_value = original_save
        subprocess.run = original_run


def synthetic_inputs(training_clips):
    if training_clips < 4 or training_clips % 2:
        raise ValueError('even training clip count of at least four required')
    result = {name: [] for name in ['csp_candidates', 'candidate_sequences',
        'primary_positive', 'auxiliary_positive', 'negative', 'prediction', 'sampling_frequencies']}
    fs = 256
    for population in range(3):
        rng = np.random.default_rng(9041 + population)
        def clip():
            # Independent channel/window amplitudes add feature diversity without
            # planting labels in the features or forcing perfect separation.
            value = rng.standard_normal((16, 600 * fs), dtype=np.float32)
            scales = np.exp(rng.normal(0, .2, (16, 20)))
            value *= np.repeat(scales, 30 * fs, axis=1).astype(np.float32)
            return value
        candidates = [clip() for _ in range(4)]
        positive = [clip() for _ in range(training_clips // 2)]
        negative = [clip() for _ in range(training_clips // 2)]
        predictions = [clip() for _ in range(2)]
        values = [candidates, np.array([6, 1, 1, 6]), positive[:-1],
                  positive[-1:], negative, predictions, fs]
        for group, value in zip(result.values(), values):
            group.append(value)
    return result


async def validate(args):
    root = Path(__file__).resolve().parents[1]
    graph = build_andriy_execution_graph()
    provider_hashes = {}
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        impl = getattr(importlib.import_module(module), name)
        provider_hashes[node.matched_primitive] = hashlib.sha256(Path(inspect.getsourcefile(impl)).read_bytes()).hexdigest()
    reference_hashes = {}
    for name in ['andriy_population_parity.json', 'andriy_execution_parity.json']:
        path = root/'docs/reviews'/name
        reference_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({
        'cdg_nodes': [{**n, 'version_id': 'validation'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'validation'} for e in edges]},
        version_id='validation', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    inputs = synthetic_inputs(args.training_clips)
    print(f'Executing raw graph: three populations, {args.training_clips} training and two prediction clips each', flush=True)
    started = time.monotonic()
    with TemporaryDirectory(prefix='sciona-andriy-raw-graph-') as temporary:
        old = runner.RUNS_DIR
        runner.RUNS_DIR = Path(temporary)
        try:
            with synthetic_diagnostics(args.diagnostics_dir):
                result = await runner.CDGExecutionSession(None, 'synthetic-andriy-raw', 'check').execute(inputs, cdg=restored)
            if result['status'] != 'completed':
                raise RuntimeError('Full raw graph did not complete: '+str(result))
            shapes, spreads = {}, {}
            for node in graph.nodes[1:]:
                for index, port in enumerate(node.outputs):
                    value = np.load(Path(temporary)/'check'/node.node_id/f'out_{port.name}.npy')
                    assert np.all(np.isfinite(value))
                    shapes[port.name] = list(value.shape)
                    if index == 0:
                        assert value.shape == (6,) and np.all((value >= 0) & (value <= 1))
                        spreads[node.node_id] = float(np.ptp(value))
            assert all(spread > 0 for spread in spreads.values())
        finally:
            runner.RUNS_DIR = old
    report = dict(graph_digest=digest, nodes=len(nodes), edges=len(edges),
        full_runner_cases=1, synthetic_only=True, training_clips_per_population=args.training_clips,
        prediction_clips_per_population=2, populations=3, output_shapes=shapes, score_spreads=spreads,
        duration_seconds=time.monotonic()-started, provider_sha256=provider_hashes,
        reference_report_sha256=reference_hashes,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Full graph execution smoke check; independent numerical reference evidence is separate.',
            'Synthetic random class assignment; nonconstant predictions are not predictive quality evidence.',
            'Documented current-runtime adaptations remain; no historical MATLAB or R equivalence claim.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(full_raw_graph_completed=True, score_spreads=spreads)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-clips', type=int, default=6)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--diagnostics-dir', type=Path, help='New private directory for synthetic-only debugging artifacts')
    asyncio.run(validate(parser.parse_args()))
