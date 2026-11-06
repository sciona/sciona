"""Execute every decomposed Santander node uncached on generated replay inputs."""
import asyncio
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from scripts.santander_synthetic import payload
from sciona.santander_execution import prepare, execute
from sciona.decomposed_santander_graph import build_decomposed_santander_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    graph = build_decomposed_santander_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    paths = {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('santander_*.py')}
    paths |= {'sciona/decomposed_santander_graph.py', 'sciona/decomposed_santander_runtime.py',
              'scripts/santander_synthetic.py', 'scripts/validate_decomposed_santander_graph_execution.py',
              'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py'}
    before = {name: sha(ROOT/name) for name in sorted(paths)}
    provider_paths = {n.matched_primitive: Path(inspect.getfile(importlib.import_module(
        n.matched_primitive.rsplit('.',1)[0]))) for n in graph.nodes}
    provider_before = {name: sha(path) for name,path in provider_paths.items()}
    packages = [line.split('==')[0] for line in (ROOT/'requirements/santander-execution.txt').read_text().splitlines() if line]
    versions = {name: metadata.version(name) for name in packages}
    for node in graph.nodes:
        importlib.import_module(node.matched_primitive.rsplit('.',1)[0])
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'): captured[node,name] = value
    inputs = payload()
    print('Starting full uncached thirty-model Santander graph', flush=True)
    with tempfile.TemporaryDirectory(prefix='santander-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-santander', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed' or len(status['trace']) != 7 or any(t['cached'] for t in status['trace']):
        raise ValueError('Every graph node must execute uncached')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    print('Graph complete; comparing full independent reconstruction', flush=True)
    if result != execute(prepare(inputs)):
        raise ValueError('Complete reconstruction result differs')
    if result['model_counts'] != dict(initial_neural=10, neural=10, tree=10):
        raise ValueError('Incomplete thirty-model lifecycle')
    if result['selection_counts'] != {
        'neural': dict(positive=5000,negative=3000,labeled_rows=8060,query_reference_rows=8500),
        'tree': dict(positive=2700,negative=2000,labeled_rows=4760,query_reference_rows=8500)}:
        raise ValueError('Full pseudo-label populations required')
    if len(result['scores']) != 8500 or result['score_kind'] != 'rank_blend':
        raise ValueError('Complete query result required')
    if before != {name:sha(ROOT/name) for name in paths} or provider_before != {name:sha(path) for name,path in provider_paths.items()}:
        raise ValueError('Implementation changed during execution')
    if versions != {name:metadata.version(name) for name in versions}:
        raise ValueError('Runtime changed during execution')
    return dict(format='decomposed-santander-execution.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, all_nodes_uncached=True,
        serialized_graph_sha256=digest, code_sha256=before, bound_provider_sha256=provider_before,
        runtime=versions, resource_policy=dict(native_threads=1,lightgbm_threads=1,torch_threads=1,torch_interop_threads=1,serial_models=True),
        checks=dict(actual_runner_nodes=7,actual_runner_edges=13,models_fitted=30,
            synthetic_query_rows=8500,full_default_pseudo_counts=True,
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        limits=['Synthetic full reconstruction equivalence does not establish historical winner strength or training-population equivalence.',
                'Current source, package closure, provider eligibility and original-intake publication remain to be qualified.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_santander_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
