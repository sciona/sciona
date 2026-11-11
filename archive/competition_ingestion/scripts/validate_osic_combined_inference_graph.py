"""Qualify combined inference graph against direct branch execution and keyed blending."""
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
from sciona.osic_combined_inference_graph import build_combined_inference_graph
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
    query=pd.DataFrame(dict(key=['z','a','m','b'],entity=['a','b','a','b'],time=[8,-3,0,3]))
    planes={'a':records,'b':records}
    from sciona import osic_feature_state,osic_quantile_features,osic_quantile_training
    feature_coordinate='synthetic-joint-features.v1'
    response_coordinate='synthetic-response.v1'
    training=pd.DataFrame(dict(entity=['b','a','b','a','b','a'],time=[3.,0.,-1.,4.,1.,8.],
        target=[8.,-1.,0.,11.,4.,7.],numeric=[47.,53.,99.,18.,100.,28.],
        category_a=['Male','female','unknown','Male','female','female'],
        category_b=['other','Ex-smoker','Currently smokes','other','other','other']))
    raw_queries=query[['entity','time']]
    base=baseline[osic_quantile_features.FIELDS]
    preparation=osic_feature_state.export_preparation_state(training,base,raw_queries,
        feature_coordinate=feature_coordinate,response_coordinate=response_coordinate,allow_transductive=True)
    quantile=osic_quantile_training.train_quantile_branch(training,base,raw_queries,
        feature_coordinate=feature_coordinate,response_coordinate=response_coordinate,
        seeds=(910,911),allow_transductive=True)
    bundle=json.loads(json.dumps(dict(preparation=preparation,models=quantile['fold_states']),allow_nan=False))
    bundle_before=json.dumps(bundle,sort_keys=True)
    input_copies=[frame.copy(deep=True) for frame in (baseline,query)]
    expected_features=osic_feature_state.transform_queries(bundle['preparation'],raw_queries,
        feature_coordinate=feature_coordinate,response_coordinate=response_coordinate)['features']
    def forbidden(*a,**kw):raise AssertionError('No training or executable model loading during graph inference')
    for name in ('fit','train_step','train_on_batch'):setattr(keras.Model,name,forbidden)
    keras.models.load_model=forbidden
    keras.optimizers.legacy.Adam.apply_gradients=forbidden
    for module,names in ((osic_feature_state,('export_preparation_state','assemble_populations','joint_population_features')),
                         (osic_quantile_features,('prepare_features','assemble_populations','joint_population_features')),
                         (osic_quantile_training,('train_quantile_branch','train_prepared_quantile_folds','prepare_features'))):
        for name in names:setattr(module,name,forbidden)
    reference_model=osic_image_lifecycle.restore_image_prediction_state(state_directory/'state',**COORDINATES)
    expected=predict_from_encoded(baseline,query,planes,
        lambda inputs:keras_bounded_prediction.predict_bounded(reference_model,inputs,32))
    expected_keys,expected_values=osic_image.blend_keyed_predictions(expected['keys'],expected['values'],
        query['key'].tolist()[::-1],quantile['predictions'][::-1])
    order=np.argsort(query['key'].to_numpy())
    np.testing.assert_array_equal(expected_values,.45*expected['values'][order]+.55*quantile['predictions'][order])
    digest,nodes,edges=encode_execution_graph(build_combined_inference_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    provider_hashes={}
    def explicit_discovery():
        for node in graph.nodes:
            module,symbol=node.matched_primitive.rsplit('.',1)
            imported=importlib.import_module(module)
            fn=getattr(imported,symbol)
            provider_hashes[module]=overlay.sha(inspect.getfile(imported))
            if REGISTRY[node.matched_primitive]['impl'] is not fn:
                raise ValueError('Graph registry binding differs')
            if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:
                raise ValueError('Graph callable signature differs')
    captured={}
    def capture(path,node,name,value):
        if name.startswith('out_'):captured[node,name[4:]]=value
    payload=dict(directory=state_directory/'state',baselines=baseline,queries=query,planes=planes,
                 batch_size=32,dtype='float32',state=bundle['preparation'],states=bundle['models'],
                 feature_coordinate=feature_coordinate,image_coordinate=COORDINATES['image_coordinate'],
                 metadata_coordinate=COORDINATES['metadata_coordinate'],
                 image_response_coordinate=COORDINATES['response_coordinate'],quantile_response_coordinate=response_coordinate)
    with tempfile.TemporaryDirectory(prefix='sciona-slice-graph-runs-',dir='/private/tmp') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)), \
             patch.object(runner,'save_intermediate_value',side_effect=capture), \
             patch.object(runner,'_ensure_atoms_imported',explicit_discovery):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-slice-graph','valid').execute(payload,cdg=graph))
        if result['status']!='completed' or len(captured)!=25:
            raise ValueError('All population outputs, restoration and reduction must execute')
        trace=json.loads((Path(temporary)/'valid/execution_trace.json').read_text())
    np.testing.assert_array_equal(captured['image_reduce','result'],expected['values'])
    assert captured['image_population','query_keys']==expected['keys']
    for actual,wanted in zip(captured['image_population','selected_positions'],expected['selected_positions']):
        np.testing.assert_array_equal(actual,wanted)
    np.testing.assert_array_equal(captured['image_population','query_subject'],[1,0,1,0])
    assert not any(row['cached'] for row in trace)
    assert {row['node_id'] for row in trace if row.get('callable_reference')}=={'image_prepare','image_select_inputs','image_predict'}
    assert getattr(captured['image_restore','result'],'optimizer',None) is None
    if before!=[overlay.sha(path) for path in files]:raise ValueError('Private state changed')
    np.testing.assert_array_equal(captured['quantile_project','features'],expected_features)
    np.testing.assert_array_equal(captured['quantile_reduce','result'],quantile['predictions'])
    assert captured['quantile_query_route','keys']==query['key'].tolist()
    assert captured['blend','keys']==expected_keys
    np.testing.assert_array_equal(captured['blend','values'],expected_values)
    assert captured['coordinates','image']==COORDINATES['response_coordinate']
    assert captured['coordinates','quantile']==response_coordinate
    assert captured['coordinates','image']!=captured['coordinates','quantile']
    assert json.dumps(bundle,sort_keys=True)==bundle_before
    for actual,copy in zip((baseline,query),input_copies):pd.testing.assert_frame_equal(actual,copy)
    assert len(trace)==15
    return dict(passed=True,graph_sha256=digest,nodes=15,edges=26,entities=2,queries=4,
        exact_image_predictions=True,exact_quantile_features=True,exact_quantile_predictions=True,
        exact_keyed_blend=True,independent_weighted_blend_exact=True,
        query_order_preserved=True,distinct_response_coordinates_preserved=True,
        node_order=[row['node_id'] for row in trace],fitting_disabled_during_replay=True,
        hdf5_loading_disabled=True,private_state_unchanged=True,numeric_bundle_unchanged=True,
        input_tables_unchanged=True,quantile_epochs=[row['epochs'] for row in quantile['folds']],
        image_training_complete=False,provider_sha256=provider_hashes,
        resource_policy=dict(native_threads=1,dataset_workers=1,reference_and_graph_sequential=True))



def validate(dependencies,source_directory,state_directory):
    from scripts.validate_osic_combined_graph_structure import validate as structural_check
    structure=structural_check()
    files=list(structure['software_sha256'])+['scripts/validate_osic_combined_inference_graph.py',
        'sciona/visualizer/runner.py','sciona/osic_image_routing.py','sciona/osic_image_state.py',
        'sciona/osic_feature_state.py','sciona/osic_quantile_features.py','sciona/osic_quantile_training.py',
        'sciona/osic_quantile_state.py','sciona/osic_quantile_model.py','sciona/osic_quantile_objective.py',
        'sciona/tensorflow_distribution_losses.py','sciona/osic_predictions.py']
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
    if result['execution']['graph_sha256']!=encode_execution_graph(build_combined_inference_graph())[0]:
        raise ValueError('Decoded worker graph differs')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        execution=result['execution'],software_sha256=before,structural_qualification=structure,
        overlay_software_sha256=result['software_sha256'],numeric_provider_sha256=result['numeric_provider_sha256'],
        dependency_archive_sha256=result['dependency_archive_sha256'],shared_environment_mutations=0,
        limitations=['Synthetic combined inference over an intermediate image checkpoint and freshly trained quantile states; complete image training remains pending.',
            'Direct qualified branch execution and an independent sorted weighted sum are the graph-wiring oracles; this run does not independently replay the source notebook.',
            'Coordinate routing is locally qualified, not catalog-approved; graph and original CDG remain unapproved.',
            'Explicit discovery and intercepted intermediate persistence; HTTP transport, default discovery and caching remain unqualified.',
            'Caller owns compatibility of image slope, time, response units and uncertainty semantics; no empirical effectiveness claim.'])



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
        (overlay.ROOT/'docs/reviews/osic_combined_inference_graph.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=15,edges=26,exact_predictions=True,approved=False)))
