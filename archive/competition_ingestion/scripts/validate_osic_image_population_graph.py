"""Qualify sequential image population graph against direct source routing."""
import argparse
import asyncio
import importlib
import inspect
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from scripts import validate_osic_image_population as overlay
from sciona.osic_image_population_graph import build_image_population_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph


def worker(state_directory,dependencies):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import tf_keras as keras
    import numpy as np
    import pandas as pd
    import cv2
    import pydicom
    from pydicom.dataset import Dataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,SecondaryCaptureImageStorage,generate_uid
    from sciona.atoms.ml.domain_adapters import osic_image,osic_image_lifecycle
    from sciona.atoms.ml import keras_bounded_prediction
    from sciona.atoms.ml.domain_adapters import osic_image_routing
    from sciona.osic_image_population import predict_from_encoded
    from sciona.atoms.ml.pipeline import ordered_fields
    from sciona.ghost.registry import REGISTRY
    from sciona.visualizer import runner
    from scripts.validate_osic_image_state import COORDINATES
    cv2.setNumThreads(0)
    assert cv2.getNumThreads()==1
    for module in (cv2,cv2._native,pydicom):
        assert Path(module.__file__).resolve().is_relative_to(Path(dependencies).resolve())
    def forbidden(*a,**kw):raise AssertionError('No training or executable model loading during graph inference')
    for name in ('fit','train_step','train_on_batch'):setattr(keras.Model,name,forbidden)
    keras.models.load_model=forbidden
    keras.optimizers.legacy.Adam.apply_gradients=forbidden
    def encode(offset):
        ds=Dataset();ds.file_meta=FileMetaDataset()
        ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds.SOPClassUID=SecondaryCaptureImageStorage;ds.SOPInstanceUID=generate_uid()
        ds.Rows=13;ds.Columns=19;ds.SamplesPerPixel=1
        ds.PhotometricInterpretation='MONOCHROME2'
        ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
        ds.RescaleIntercept='-17.25';ds.RescaleSlope='1.5'
        ds.PixelData=(np.arange(247).reshape(13,19)+offset-123).astype('<i2').tobytes()
        stream=BytesIO();pydicom.dcmwrite(stream,ds,enforce_file_format=True)
        return stream.getvalue()
    records=[('-1.dcm',b'unselected-invalid'),('0.dcm',encode(0)),('1.dcm',encode(11)),
             ('2.dcm',encode(22)),('4.dcm',encode(33)),('9.dcm',b'unselected-invalid')]
    files=[state_directory/'state/manifest.json',state_directory/'state/weights.npz']
    before=[overlay.sha(path) for path in files]
    baseline=pd.DataFrame(dict(entity=['b','a'],time=[3.,-1.],target=[8.,2.],numeric=[47.,53.],
        category_a=['Male','female'],category_b=['other','other'],uncertainty=[1.,2.]))
    query=pd.DataFrame(dict(key=['q0','q1','q2','q3'],entity=['a','b','a','b'],time=[8,-3,0,3]))
    planes={'a':records,'b':records}
    reference_model=osic_image_lifecycle.restore_image_prediction_state(state_directory/'state',**COORDINATES)
    expected=predict_from_encoded(baseline,query,planes,
        lambda inputs:keras_bounded_prediction.predict_bounded(reference_model,inputs,32))
    digest,nodes,edges=encode_execution_graph(build_image_population_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    def explicit_discovery():
        for node in graph.nodes:
            module,symbol=node.matched_primitive.rsplit('.',1)
            fn=getattr(importlib.import_module(module),symbol)
            if REGISTRY[node.matched_primitive]['impl'] is not fn:
                raise ValueError('Graph registry binding differs')
            if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:
                raise ValueError('Graph callable signature differs')
    captured={}
    def capture(path,node,name,value):
        if name.startswith('out_'):captured[node,name[4:]]=value
    payload=dict(directory=state_directory/'state',baselines=baseline,queries=query,planes=planes,
                 batch_size=32,**COORDINATES)
    with tempfile.TemporaryDirectory(prefix='sciona-slice-graph-runs-',dir='/private/tmp') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)), \
             patch.object(runner,'save_intermediate_value',side_effect=capture), \
             patch.object(runner,'_ensure_atoms_imported',explicit_discovery):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-slice-graph','valid').execute(payload,cdg=graph))
        if result['status']!='completed' or len(captured)!=10:
            raise ValueError('All population outputs, restoration and reduction must execute')
        trace=json.loads((Path(temporary)/'valid/execution_trace.json').read_text())
    np.testing.assert_array_equal(captured['reduce','result'],expected['values'])
    assert captured['population','query_keys']==expected['keys']
    for actual,wanted in zip(captured['population','selected_positions'],expected['selected_positions']):
        np.testing.assert_array_equal(actual,wanted)
    np.testing.assert_array_equal(captured['population','query_subject'],[1,0,1,0])
    assert not any(row['cached'] for row in trace)
    assert {row['node_id'] for row in trace if row.get('callable_reference')}=={'prepare','select_inputs','predict'}
    assert getattr(captured['restore','result'],'optimizer',None) is None
    if before!=[overlay.sha(path) for path in files]:raise ValueError('Private state changed')
    return dict(passed=True,graph_sha256=digest,nodes=6,edges=10,entities=2,queries=4,selected_slices_per_entity=4,
        unselected_invalid_payloads_ignored=4,exact_direct_population_predictions=True,
        query_order_preserved=True,explicit_callable_operations=True,
        node_order=[row['node_id'] for row in trace],fitting_disabled=True,hdf5_loading_disabled=True,
        private_state_unchanged=True,provider_sha256={
            'preparation':overlay.sha(inspect.getfile(osic_image)),
            'lifecycle':overlay.sha(inspect.getfile(osic_image_lifecycle)),
            'prediction':overlay.sha(inspect.getfile(keras_bounded_prediction)),
            'ordered_fields':overlay.sha(inspect.getfile(ordered_fields)),
            'routing':overlay.sha(inspect.getfile(osic_image_routing))},
        resource_policy=dict(native_threads=1,dataset_workers=1,reference_and_graph_sequential=True))


def validate(dependencies,source_directory,state_directory):
    files=['sciona/osic_image_population_graph.py','sciona/osic_image_routing.py','sciona/osic_image_slice_graph.py','sciona/osic_image_lifecycle_graphs.py',
           'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py',
           'scripts/validate_osic_image_population_graph.py']
    before={name:overlay.sha(overlay.ROOT/name) for name in files}
    original_run=subprocess.run
    redirects=0
    def redirect(command,*args,**kwargs):
        nonlocal redirects
        command=list(command)
        target=str(Path(overlay.__file__).resolve())
        if target in command and '--worker' in command:
            command[command.index(target)]=str(Path(__file__).resolve());redirects+=1
        return original_run(command,*args,**kwargs)
    with patch.object(overlay.subprocess,'run',redirect):
        result=overlay.validate(dependencies,source_directory,state_directory)
    if redirects!=1 or before!={name:overlay.sha(overlay.ROOT/name) for name in files}:
        raise ValueError('Graph inputs or isolated worker dispatch changed')
    if result['execution']['graph_sha256']!=encode_execution_graph(build_image_population_graph())[0]:
        raise ValueError('Decoded worker graph differs')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        execution=result['execution'],software_sha256=before,
        dependency_archive_sha256=result['dependency_archive_sha256'],shared_environment_mutations=0,
        limitations=['Two-entity population routing and trajectory reduction over intermediate numeric state only; quantile branch, final blending and complete training remain pending.',
            'Direct qualified provider composition is the graph-wiring oracle; this run does not independently replay the source notebook.',
            'The population dispatcher is locally qualified but not catalog-approved; the graph and original CDG remain unapproved.',
            'Explicit provider discovery and intercepted intermediate persistence only; no HTTP transport, default discovery or caching qualification.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dependencies',type=Path,required=True)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--state-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.worker:print(json.dumps(worker(args.state_directory,args.dependencies)))
    else:
        result=validate(args.dependencies,args.source_directory,args.state_directory)
        (overlay.ROOT/'docs/reviews/osic_image_population_graph.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=6,edges=10,exact_predictions=True,approved=False)))
