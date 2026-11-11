#!/usr/bin/env python3
"""Check actual R model execution through the serialized Andriy model subgraph."""
import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.architect.handoff import CDGExport
from sciona.ghost.registry import REGISTRY
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


async def validate(output):
    root = Path(__file__).resolve().parents[1]
    full = build_andriy_execution_graph()
    graph = CDGExport(nodes=full.nodes[1:], edges=[], metadata={
        'scope': 'Actual three-branch model subgraph using synthetic model-ready features; raw preprocessing excluded.'})
    reference_hashes = {}
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        impl = getattr(importlib.import_module(module), name)
        report_path = root / f'docs/reviews/andriy_r_{node.node_id}_parity.json'
        evidence = json.loads(report_path.read_text())
        assert hashlib.sha256(Path(inspect.getsourcefile(impl)).read_bytes()).hexdigest() == evidence['provider_sha256']
        for relative, expected in evidence['compiled_library_sha256'].items():
            assert hashlib.sha256((Path(os.environ['R_LIBS_USER']) / relative).read_bytes()).hexdigest() == expected
        reference_hashes[report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({
        'cdg_nodes': [{**n, 'version_id': 'validation'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'validation'} for e in edges]},
        version_id='validation', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    groups = [[], [], [], []]
    for population, segments in enumerate([6, 5, 4]):
        rng = np.random.default_rng(671 + population)
        train = rng.normal(size=(512, 1965))
        labels = np.r_[np.ones(256), np.zeros(256)]
        train[:, :1000] += np.where(labels, .15, -.15)[:, None]
        prediction = rng.normal(size=(19 * segments, 1965))
        prediction[:, :1000] += np.repeat(np.linspace(-.8, .8, segments), 19)[:, None]
        valid = np.ones(len(prediction), dtype=bool)
        valid[0] = False
        valid[-19:] = False
        prediction[-19:] = np.nan
        for group, value in zip(groups, [train, labels, prediction, valid]):
            group.append(value)
    expected = {}
    for node in graph.nodes:
        expected[node.node_id] = REGISTRY[node.matched_primitive]['impl'](*groups)
        print(f'Direct {node.node_id} complete', flush=True)
    with TemporaryDirectory(prefix='sciona-andriy-model-graph-') as temporary:
        old = runner.RUNS_DIR
        runner.RUNS_DIR = Path(temporary)
        try:
            inputs = dict(zip([p.name for p in graph.nodes[0].inputs], groups))
            result = await runner.CDGExecutionSession(None, 'synthetic-andriy-models', 'check').execute(inputs, cdg=restored)
            assert result['status'] == 'completed', result['status']
            for node in graph.nodes:
                for port, wanted in zip(node.outputs, expected[node.node_id]):
                    actual = np.load(Path(temporary) / 'check' / node.node_id / f'out_{port.name}.npy')
                    np.testing.assert_array_equal(actual, wanted)
        finally:
            runner.RUNS_DIR = old
    spreads = {name: float(np.ptp(values[0])) for name, values in expected.items()}
    assert all(value > .2 for value in spreads.values())
    full_digest, full_nodes, full_edges = encode_execution_graph(full)
    files = [Path(__file__), root/'sciona/andriy_execution.py', root/'tests/test_andriy_execution.py']
    report = dict(full_graph_hash=full_digest, full_nodes=len(full_nodes), full_edges=len(full_edges),
        executed_model_graph_hash=digest, executed_nodes=3, exact_outputs=6,
        population_segment_counts=[6, 5, 4], score_spreads=spreads,
        reference_report_sha256=reference_hashes,
        validation_files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limitations=['Model subgraph only: synthetic normalized features, actual pinned R training and prediction.',
            'Direct-provider versus serialized-runner parity is routing evidence, not independent algorithm validation; hash-bound source reports supply separate algorithm evidence.',
            'Full raw three-population graph and full eleven-model ensemble have not been executed by this validator.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(exact_outputs=6, score_spreads=spreads)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(validate(args.output))
