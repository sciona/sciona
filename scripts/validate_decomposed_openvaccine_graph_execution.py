"""Execute the complete feature-level graph and retrain the unchanged reference."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('KERAS_HOME', '/private/tmp/sciona_openvaccine_keras')

import asyncio
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
from scripts.openvaccine_synthetic import payload
from scripts.validate_openvaccine_environment import dependency_closure
from scripts.provision_openvaccine_reviewed_software import provision
from sciona.openvaccine_contract import prepare, execute
from sciona.decomposed_openvaccine_graph import build_decomposed_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.keras.layers.Layer.__module__.split('.')[0] != 'tf_keras':
        raise ValueError('Reviewed legacy Keras required')
    versions = dependency_closure()
    if (np.__version__, pd.__version__, tf.__version__, tf_keras.__version__) != tuple(
            versions[n] for n in ('numpy', 'pandas', 'tensorflow', 'tf-keras')):
        raise ValueError('Actual imports differ from installed metadata')
    software = provision()
    baseline = json.loads((ROOT/'docs/reviews/decomposed_openvaccine_baseline_execution.json').read_text())
    if any(sha(ROOT/p) != h for p, h in baseline['code_sha256'].items()):
        raise ValueError('Unchanged baseline reference required')
    graph = build_decomposed_openvaccine_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    paths = set(baseline['code_sha256']) | {
        'sciona/decomposed_openvaccine_runtime.py', 'sciona/decomposed_openvaccine_ensemble.py',
        'sciona/decomposed_openvaccine_graph.py', 'scripts/validate_decomposed_openvaccine_graph_execution.py',
        'tests/test_decomposed_openvaccine_runtime.py', 'tests/test_decomposed_openvaccine_ensemble.py'}
    before = {p: sha(ROOT/p) for p in sorted(paths)}
    providers = {n.matched_primitive: Path(inspect.getfile(importlib.import_module(
        n.matched_primitive.rsplit('.', 1)[0]))) for n in graph.nodes}
    provider_before = {n: sha(p) for n, p in providers.items()}
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[node, name] = value
    environment = dict(SCIONA_OPENVACCINE_SOURCE_DIR='/private/tmp/sciona_openvaccine_source',
        SCIONA_OPENVACCINE_FOLD_BINARY='/private/tmp/sciona_openvaccine_contrafold/src/contrafold',
        SCIONA_OPENVACCINE_FOLD_PARAMETERS='/private/tmp/sciona_openvaccine_eternafold_parameters/parameters/EternaFoldParams.v1')
    inputs = payload()
    print('Executing six-node graph: folding, pretraining, all 20 models, refinement and external sum', flush=True)
    with tempfile.TemporaryDirectory(prefix='openvaccine-synthetic-decomposed-') as temporary:
        with patch.dict(os.environ, environment), patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-openvaccine', 'decomposed').execute(
                {'payload': inputs}, cdg=graph))
        if status['status'] != 'completed' or len(status['trace']) != 6 or any(t['cached'] for t in status['trace']):
            raise ValueError('All six nodes must execute uncached')
        if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
            raise ValueError('Incomplete actual graph outputs')
        result = json.loads(json.dumps(captured['result', 'out_result'], allow_nan=False))
        if (result['members'] != 20 or result['pretraining_steps'] != 1
                or result['initial_supervised_steps'] != 20 or len(result['rounds']) != 1):
            raise ValueError('Complete configured lifecycle required')
        progress = result['rounds'][0]
        if (progress['members'] != 20 or progress['supervised_steps'] != 20
                or progress['pseudo_label_steps'] != 20 or not 0 <= progress['rollbacks'] <= 20):
            raise ValueError('Complete member refinement required')
        if any(np.asarray(result[k]).shape != (2, 14, 5) or not np.isfinite(result[k]).all()
               for k in ('predictions', 'teacher_mean', 'teacher_std')):
            raise ValueError('Complete finite predictions and teacher moments required')
        print('Graph passed; retraining complete unchanged reference for exact comparison', flush=True)
        with patch.dict(os.environ, environment):
            reference = json.loads(json.dumps(execute(prepare(inputs)), allow_nan=False))
        if result != reference:
            raise ValueError('Complete reconstruction result differs from unchanged reference')
        for key in ('predictions', 'teacher_mean', 'teacher_std'):
            if np.asarray(result[key], np.float32).tobytes() != np.asarray(reference[key], np.float32).tobytes():
                raise ValueError('Prediction or teacher float32 bytes differ')
    if (before != {p: sha(ROOT/p) for p in paths}
            or provider_before != {n: sha(p) for n, p in providers.items()}
            or versions != dependency_closure() or software != provision()):
        raise ValueError('Code, providers, runtime or software changed during execution')
    return dict(format='decomposed-openvaccine-execution.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, all_nodes_uncached=True, serialized_graph_sha256=digest,
        code_sha256=before, bound_provider_sha256=provider_before, runtime=versions,
        provisioned_software=software, actual_imports_verified=True, active_dependency_constraints_satisfied=True,
        resource_policy=dict(native_threads=1, tensorflow_intra_threads=1, tensorflow_inter_threads=1, serial_models=True),
        checks=dict(actual_runner_nodes=6, actual_runner_edges=7, models=20, refinement_rounds=1,
            pretraining_steps=1, initial_supervised_steps=20, refinement_supervised_steps=20,
            pseudo_label_steps=20, full_reference_retrained=True, exact_reconstruction_result=True,
            prediction_and_teacher_bytes_equal=True, strict_json_output=True, graph_codec_roundtrip=True),
        limits=['Complete synthetic reconstruction parity does not establish historical accuracy, schedule or winning clustered blend equivalence.',
                'Synthetic execution uses one explicit refinement round; multi-round control flow is separately tested.',
                'Installed metadata closure and actual root imports are checked; installed RECORD hashes and clean deployment are separate.',
                'Provider semantics and original-intake publication remain to be qualified.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_openvaccine_graph_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']), flush=True)
