"""Execute every decomposed Santa node uncached on generated replay inputs."""
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

from scripts.santa_synthetic import payload
from sciona.santa_execution import prepare, execute
from sciona.decomposed_santa_graph import build_decomposed_santa_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    graph = build_decomposed_santa_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    paths = {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('santa_*.py')}
    paths |= {'sciona/decomposed_santa_graph.py', 'sciona/decomposed_santa_runtime.py',
              'scripts/santa_synthetic.py', 'scripts/validate_decomposed_santa_graph_execution.py',
              'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py'}
    before = {name: sha(ROOT/name) for name in sorted(paths)}
    provider_paths = {n.matched_primitive: Path(inspect.getfile(importlib.import_module(
        n.matched_primitive.rsplit('.',1)[0]))) for n in graph.nodes}
    provider_before = {name: sha(path) for name,path in provider_paths.items()}
    packages = [line.split('==')[0] for line in (ROOT/'requirements/santa-execution.txt').read_text().splitlines() if line]
    versions = {name: metadata.version(name) for name in packages}
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'): captured[node,name] = value
    inputs = payload()
    with tempfile.TemporaryDirectory(prefix='santa-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-santa', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed' or len(status['trace']) != 6 or any(t['cached'] for t in status['trace']):
        raise ValueError('Every graph node must execute uncached')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    if result != execute(prepare(inputs)):
        raise ValueError('Complete reconstruction result differs')
    models = captured['fit','out_models']
    if result['models'] != 2 or result['training_rows'] != 3142 or result['validation_rows'] != 3186:
        raise ValueError('Incomplete synthetic model lifecycle')
    if set(models.evaluation) != {'normal','transformed'} or any(
        not 1 <= e['best_iteration'] <= len(e['iteration_rmse']) <= 24 for e in models.evaluation.values()):
        raise ValueError('Held-out model selection missing')
    if before != {name:sha(ROOT/name) for name in paths} or provider_before != {name:sha(path) for name,path in provider_paths.items()}:
        raise ValueError('Implementation changed during execution')
    if versions != {name:metadata.version(name) for name in versions}:
        raise ValueError('Runtime changed during execution')
    return dict(format='decomposed-santa-execution.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, all_nodes_uncached=True,
        serialized_graph_sha256=digest, code_sha256=before, bound_provider_sha256=provider_before,
        runtime=versions, resource_policy=dict(native_threads=1,lightgbm_threads=1,serial_models=True),
        checks=dict(actual_runner_nodes=6,actual_runner_edges=10,models_fitted=2,
            training_rows=result['training_rows'],validation_rows=result['validation_rows'],
            heldout_selected_iterations=True,action_count=len(result['scores']),
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        limits=['Synthetic full reconstruction equivalence does not establish historical winner strength or training-population equivalence.',
                'Current source, package closure, provider eligibility and original-intake publication remain to be qualified.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_santa_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
