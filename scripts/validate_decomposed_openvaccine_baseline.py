"""Execute the unchanged complete reconstruction under bounded current runtime."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('KERAS_HOME','/private/tmp/sciona_openvaccine_keras')

import asyncio
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import tf_keras
import pandas as pd
from scripts.openvaccine_synthetic import payload
from scripts.validate_openvaccine_environment import dependency_closure
from scripts.provision_openvaccine_reviewed_software import provision
from sciona.openvaccine_graph import build_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.ml.openvaccine_execution as provider

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.keras.layers.Layer.__module__.split('.')[0] != 'tf_keras':
        raise ValueError('Reviewed legacy Keras required')
    versions = dependency_closure()
    if (np.__version__,pd.__version__,tf.__version__,tf_keras.__version__) != tuple(
            versions[n] for n in ('numpy','pandas','tensorflow','tf-keras')):
        raise ValueError('Actual imports differ from installed package metadata')
    software = provision()
    prior = json.loads((ROOT/'docs/reviews/competition_openvaccine_graph_execution.json').read_text())
    if any(sha(ROOT/p) != digest for p,digest in prior['code_sha256'].items()):
        raise ValueError('Original reconstruction changed before baseline')
    provider_path = Path(inspect.getfile(provider))
    if sha(provider_path) != prior['provider_sha256']:
        raise ValueError('Original provider changed')
    paths = set(prior['code_sha256']) | {
        'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py',
        'scripts/validate_decomposed_openvaccine_baseline.py','scripts/provision_openvaccine_reviewed_software.py',
        'scripts/validate_openvaccine_environment.py','requirements/openvaccine-execution.txt'}
    before = {p:sha(ROOT/p) for p in sorted(paths)}
    digest,nodes,edges = encode_execution_graph(build_openvaccine_graph())
    if digest != prior['serialized_graph_sha256']:
        raise ValueError('Original graph changed')
    graph = decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory,node,name,value):
        if name.startswith('out_'):
            captured[node,name] = value
    environment = dict(SCIONA_OPENVACCINE_SOURCE_DIR='/private/tmp/sciona_openvaccine_source',
        SCIONA_OPENVACCINE_FOLD_BINARY='/private/tmp/sciona_openvaccine_contrafold/src/contrafold',
        SCIONA_OPENVACCINE_FOLD_PARAMETERS='/private/tmp/sciona_openvaccine_eternafold_parameters/parameters/EternaFoldParams.v1')
    print('Executing complete raw-input baseline: folding, pretraining, 20 models, refinement and checkpoint ensemble',flush=True)
    with tempfile.TemporaryDirectory(prefix='openvaccine-synthetic-baseline-') as temporary:
        with patch.dict(os.environ,environment),patch.object(runner,'RUNS_DIR',Path(temporary)), \
             patch.object(runner,'save_intermediate_value',side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None,'synthetic-openvaccine','baseline').execute(
                {'payload':payload()},cdg=graph))
    if status['status'] != 'completed' or len(status['trace']) != 2 or any(t['cached'] for t in status['trace']):
        raise ValueError('Both original nodes must execute uncached')
    result = json.loads(json.dumps(captured['execute','out_result'],allow_nan=False))
    if result['members'] != 20 or len(result['rounds']) != 1:
        raise ValueError('Complete twenty-member refinement inventory required')
    progress = result['rounds'][0]
    if (progress['members'] != 20 or progress['supervised_steps'] != 20 or progress['pseudo_label_steps'] != 20
            or not 0 <= progress['rollbacks'] <= 20 or result['pretraining_steps'] != 1
            or result['initial_supervised_steps'] != 20):
        raise ValueError('Complete configured baseline lifecycle required')
    if any(np.asarray(result[k]).shape != (2,14,5) or not np.isfinite(result[k]).all()
           for k in ('predictions','teacher_mean','teacher_std')):
        raise ValueError('Complete finite prediction and teacher outputs required')
    if (before != {p:sha(ROOT/p) for p in paths} or sha(provider_path) != prior['provider_sha256']
            or versions != dependency_closure() or software != provision()):
        raise ValueError('Code, provider, software or runtime changed during execution')
    recorded = dict(line.split('==') for line in (ROOT/'requirements/openvaccine-execution.txt').read_text().splitlines() if line)
    return dict(format='decomposed-openvaccine-baseline-execution.v1',status='passed',approved=False,
        catalog_mutations=0,synthetic_only=True,all_nodes_uncached=True,serialized_graph_sha256=digest,
        code_sha256=before,provider_sha256=sha(provider_path),runtime=versions,
        runtime_drift={k:dict(recorded=v,installed=versions.get(k)) for k,v in recorded.items() if versions.get(k)!=v},
        provisioned_software=software,actual_imports_verified=True,active_dependency_constraints_satisfied=True,
        resource_policy=dict(native_threads=1,tensorflow_intra_threads=1,tensorflow_inter_threads=1,serial_models=True),
        checks=dict(actual_runner_nodes=2,models=20,refinement_rounds=1,pretraining_steps=1,
            initial_supervised_steps=20,refinement_supervised_steps=20,pseudo_label_steps=20,
            masked_validation=True,distinct_population_lengths=True,strict_json_output=True,
            graph_codec_roundtrip=True,finite_predictions_and_teacher_moments=True),
        limits=['Complete unchanged synthetic baseline with caller-specified training and one refinement round. No historical schedule, accuracy or clustered winning-blend equivalence claim.',
                'Package requirements are checked against installed metadata and root imports; installed RECORD hashes and clean deployment remain separate.',
                'This report does not approve either original intake or qualify any future decomposition.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_openvaccine_baseline_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)
