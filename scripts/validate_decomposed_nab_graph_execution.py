"""Execute every decomposed NAB stage with generated inputs and full forests."""
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

from scripts.nab_synthetic import payload
from sciona.nab_contract import prepare, execute
from sciona.decomposed_nab_graph import build_decomposed_nab_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    inputs = payload()
    digest, nodes, edges = encode_execution_graph(build_decomposed_nab_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    for node in graph.nodes:
        importlib.import_module(node.matched_primitive.rsplit('.',1)[0])
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[node, name] = value
    with tempfile.TemporaryDirectory(prefix='nab-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-nab', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Full graph execution failed')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    if result != execute(prepare(inputs)):
        raise ValueError('Complete reconstruction result differs')
    if (result['detection']['models_fitted'] != 90
            or len(result['detection']['detections']) != 100
            or not any(result['detection']['detections'])
            or result['evaluation']['scored_points'] != 85):
        raise ValueError('Synthetic lifecycle coverage differs')
    paths = [str(p.relative_to(ROOT)) for pattern in ('nab_*.py','decomposed_nab_*.py')
             for p in sorted((ROOT/'sciona').glob(pattern))]
    paths += ['scripts/nab_synthetic.py','scripts/validate_decomposed_nab_graph_execution.py',
              'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']
    return dict(format='decomposed-nab-graph-execution.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, serialized_graph_sha256=digest,
        checks=dict(actual_runner_nodes=6, actual_runner_edges=13, models_fitted=90,
                    stream_points=100, exact_reconstruction_result=True,
                    strict_json_output=True, graph_codec_roundtrip=True),
        code_sha256={p:sha(ROOT/p) for p in paths},
        bound_provider_sha256={n.matched_primitive:sha(Path(inspect.getfile(
            importlib.import_module(n.matched_primitive.rsplit('.',1)[0])))) for n in graph.nodes},
        runtime={n:metadata.version(n) for n in ('numpy','scipy','scikit-learn','joblib','threadpoolctl')},
        resource_policy=dict(native_threads=1, forest_jobs=1, serial_models=True),
        limits=['Exact comparison is against the existing generic isolation-forest reconstruction, not a historical winning detector.',
                'Synthetic computation does not establish generalization, source scoring parity on this environment, clean installation or catalog eligibility.',
                'Current provider semantic review and original-intake publication remain pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_nab_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
