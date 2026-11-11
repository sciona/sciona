"""Execute unchanged complete March reference on the current bounded runtime."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import asyncio
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from packaging.requirements import Requirement
import networkx as nx
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info
from sciona.march_contract import prepare,execute
from sciona.march_graph import build_march_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.ml.march_execution as provider
from scripts.march_synthetic import payload
from scripts.import_flavours_provider_drafts import ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    names=['numpy','scipy','scikit-learn','joblib','threadpoolctl','networkx']
    versions={name:metadata.version(name) for name in names}
    for name in names:
        for text in metadata.requires(name) or []:
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            if not requirement.specifier.contains(metadata.version(requirement.name),prereleases=True):
                raise ValueError('Active dependency constraint violated')
    if (np.__version__,scipy.__version__,sklearn.__version__,nx.__version__)!=tuple(versions[n] for n in ('numpy','scipy','scikit-learn','networkx')):
        raise ValueError('Actual imports differ from installed metadata')
    return versions


def validate():
    old=json.loads((ROOT/'docs/reviews/competition_march_graph_execution.json').read_text())
    if any(sha(ROOT/p)!=digest for p,digest in old['code_sha256'].items()):
        raise ValueError('Historical reference changed')
    provider_path=Path(inspect.getfile(provider))
    if sha(provider_path)!=old['provider_sha256']:raise ValueError('Reference provider changed')
    versions=runtime()
    graph=build_march_graph();digest,nodes,edges=encode_execution_graph(graph)
    if digest!=old['serialized_graph_sha256']:raise ValueError('Historical graph changed')
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest:raise ValueError('Graph codec mismatch')
    paths=set(old['code_sha256'])|{'scripts/validate_decomposed_march_baseline.py',
        'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py',
        'tests/test_march_features.py','tests/test_march_training.py','tests/test_march_contract.py'}
    before={p:sha(ROOT/p) for p in sorted(paths)}
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'):captured[node,name]=value
    inputs=payload()
    with tempfile.TemporaryDirectory(prefix='march-synthetic-baseline-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-march','baseline').execute({'payload':inputs},cdg=graph))
    if status['status']!='completed' or len(status['trace'])!=2 or any(t['cached'] for t in status['trace']):
        raise ValueError('Both complete baseline nodes must execute uncached')
    if set(captured)!={(n.node_id,'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Missing actual graph output')
    actual=json.loads(json.dumps(captured['execute','out_result'],allow_nan=False))
    reference=json.loads(json.dumps(execute(prepare(inputs)),allow_nan=False))
    if actual!=reference or np.asarray(actual['probabilities'],np.float64).tobytes()!=np.asarray(reference['probabilities'],np.float64).tobytes():
        raise ValueError('Complete fresh reference result differs')
    if (actual['training_rows'],actual['calibration_rows'],actual['predicted_matchups'])!=(4,4,2):
        raise ValueError('Full configured lifecycle required')
    probabilities=np.asarray(actual['probabilities'])
    if probabilities.shape!=(2,) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)) or abs(probabilities.sum()-1)>1e-14:
        raise ValueError('Finite orientation-symmetric probability output required')
    pools=threadpool_info()
    if not pools or any(p['num_threads']!=1 for p in pools):raise ValueError('One-thread native pools required')
    if before!={p:sha(ROOT/p) for p in paths} or sha(provider_path)!=old['provider_sha256'] or versions!=runtime():
        raise ValueError('Implementation or environment changed during execution')
    return dict(format='decomposed-march-baseline.v1',status='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,reference_source_unchanged=True,all_nodes_uncached=True,serialized_graph_sha256=digest,
        code_sha256=before,provider_sha256=sha(provider_path),runtime=versions,
        checks=dict(actual_runner_nodes=2,actual_runner_edges=1,training_rows=4,calibration_rows=4,predicted_matchups=2,
            strict_json_output=True,graph_codec_roundtrip=True,full_reference_retrained=True,
            exact_reconstruction_result=True,probability_bytes_equal=True,orientation_symmetry=True),
        actual_import_versions_verified=True,active_dependency_constraints_satisfied=True,
        resource_policy=dict(native_threads=1,calibration_workers=1,serial_execution=True),
        native_pools=[dict(internal_api=p['internal_api'],num_threads=p['num_threads']) for p in pools],
        limits=['Complete current synthetic execution of unchanged generic benchmark; no historical winning-source or real-world accuracy claim.',
                'The current NumPy version differs from the historical environment pin. Active dependency constraints and actual imports pass; historical environment and output equivalence are not claimed.',
                'Installed package file authentication, extracted operation interfaces and original-CDG publication remain separate requirements.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/decomposed_march_baseline_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
