"""Record reproducible synthetic graph evidence without catalog mutations."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from sciona.decomposed_video_multilabel_graph import build_decomposed_video_multilabel_graph
from sciona.services.execution_graph_codec import encode_execution_graph

ROOT=Path(__file__).resolve().parents[1]
TESTS=[
    'tests/test_sparse_binary_logistic.py',
    'tests/test_sparse_numeric_vocabulary.py',
    'tests/test_decomposed_video_multilabel_graph.py',
    'tests/test_masked_attention_provider.py',
    'tests/test_empirical_f1_provider.py',
    'tests/test_video_multilabel_shards.py',
    'tests/test_video_multilabel_sharded_training.py',
]


def validate():
    paths=TESTS+[
        'scripts/validate_decomposed_video_multilabel_graph.py',
        'scripts/video_multilabel_synthetic.py',
        'sciona/decomposed_video_multilabel_graph.py',
        'sciona/video_multilabel_shards.py',
        'sciona/video_multilabel_sharded_training.py',
        'sciona/video_multilabel_reusable_network.py',
        'sciona/video_multilabel_contract.py','sciona/video_multilabel_training.py',
        'sciona/video_multilabel_network.py','sciona/video_multilabel_thresholds.py',
        'sciona/ghost/registry.py','sciona/visualizer/runner.py',
        'sciona/services/execution_graph_codec.py',
        'sciona/materialized_prediction.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/model_selection/materialized_prediction.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/sklearn/feature_extraction/sparse_numeric.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/sklearn/linear_model/sparse_binary.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/masked_attention.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/calibration/empirical_f1.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/video_multilabel_sharded_execution.py',
    ]
    def fingerprints():
        return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    def versions():
        return {name:metadata.version(name) for name in ('numpy','torch','scipy','scikit-learn','pytest')}
    before=fingerprints();installed=versions()
    graph_digest=encode_execution_graph(build_decomposed_video_multilabel_graph())[0]
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-sharded-graph-') as temporary:
        xml=Path(temporary)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*TESTS,
            '--junitxml='+str(xml)],cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:
            raise ValueError('Synthetic graph suite failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=100,failures=0,errors=0,skipped=0):
        raise ValueError('Complete 100-test suite required')
    if before!=fingerprints() or installed!=versions():
        raise ValueError('Software changed during validation')
    return dict(format='decomposed-video-multilabel-graph.v4',passed=True,approved=False,
        catalog_mutations=0,synthetic_only=True,tests=counts,software_sha256=before,
        observed_package_versions=installed,graph_sha256=graph_digest,
        native_graph_executions=3,graph_nodes=10,explicit_callable_edges=9,
        exact_registered_bindings=True,all_nodes_uncached=True,
        full_configured_lifecycle_parity=True,resource_policy=dict(serial_tests=True,native_threads=1),
        limits=['No catalog staging or approval; original CDG remains separately qualified.',
                'Nine numeric operations are explicit; provider source and interface qualification remain pending.',
                'Package versions are observations; dependency closure and installed RECORD qualification remain pending.',
                'Synthetic full-population parity does not establish competition accuracy or production-scale memory bounds.',
                'Shard stores are caller-owned private runtime objects; no paths or data are embedded in the graph.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/decomposed_video_multilabel_graph_execution_v4.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],tests=result['tests'],approved=False)))
