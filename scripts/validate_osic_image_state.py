"""Numeric state qualification using a stable intermediate synthetic checkpoint."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COORDINATES=dict(image_coordinate='synthetic-inference-intensity.v1',
    metadata_coordinate='synthetic-inference-vector.v1',response_coordinate='synthetic-slope.v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def runtime():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import efficientnet.tfkeras
    import tf_keras as keras
    import numpy as np
    return keras,np


def replay(directory):
    keras,np=runtime()
    from sciona.osic_image_state import restore_prediction_state
    from sciona.osic_image_predictor import make_slice_predictor
    def forbidden(*args,**kwargs):
        raise AssertionError('Fitting or HDF5 loading forbidden in numeric replay')
    keras.Model.fit=forbidden
    keras.Model.train_step=forbidden
    keras.Model.train_on_batch=forbidden
    keras.optimizers.legacy.Adam.apply_gradients=forbidden
    keras.models.load_model=forbidden
    model=restore_prediction_state(directory/'state',**COORDINATES)
    assert getattr(model,'optimizer',None) is None
    with np.load(directory/'queries.npz',allow_pickle=False) as values:
        actual=make_slice_predictor(model)([values['image'],values['metadata']])
        np.testing.assert_array_equal(actual,values['expected'])
    return dict(passed=True,queries=3,exact_predictions=True,fitting_disabled=True,
                hdf5_loading_disabled=True,optimizer_state_absent=True)


def worker(directory):
    keras,np=runtime()
    from sciona.osic_image_state import export_prediction_state,restore_prediction_state
    from sciona.osic_image_predictor import make_slice_predictor
    frozen=json.loads((directory/'snapshot.json').read_text())
    if frozen.get('stable_copy') is not True or sha(directory/'source.h5')!=frozen['source_sha256']:
        raise ValueError('Stable private synthetic checkpoint copy required')
    model=keras.models.load_model(directory/'source.h5')
    updates=int(model.optimizer.iterations)
    if not 0<updates<960:
        raise ValueError('This diagnostic expects an intermediate, not complete, training checkpoint')
    manifest=export_prediction_state(model,directory/'state',**COORDINATES)
    rng=np.random.default_rng(915)
    inputs=[rng.normal(size=(3,512,512,1)),rng.normal(size=(3,4))]
    expected=make_slice_predictor(model)(inputs)
    np.savez(directory/'queries.npz',image=inputs[0],metadata=inputs[1],expected=expected)
    private_paths=[directory/'state/manifest.json',directory/'state/weights.npz',directory/'queries.npz',directory/'source.h5']
    before={p.name:sha(p) for p in private_paths}
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay',
        '--runtime-directory',str(directory)],env=os.environ.copy(),capture_output=True,text=True)
    if child.returncode:
        raise RuntimeError('Numeric replay failed:\n'+child.stderr)
    restored=json.loads(child.stdout)
    assert before=={p.name:sha(p) for p in private_paths}
    rejected=[]
    def reject(name,change):
        altered=copy.deepcopy(manifest)
        change(altered)
        path=directory/'state/manifest.json'
        path.write_text(json.dumps(altered)+'\n')
        try:
            restore_prediction_state(directory/'state',**COORDINATES)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Corrupt state accepted: '+name)
        finally:
            path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    reject('unknown_format',lambda s:s.update(format='unknown'))
    reject('executable_field',lambda s:s.update(model_config='forbidden'))
    reject('architecture',lambda s:s.update(architecture='other'))
    reject('image_coordinate',lambda s:s.update(image_coordinate='wrong'))
    reject('metadata_coordinate',lambda s:s.update(metadata_coordinate='wrong'))
    reject('response_coordinate',lambda s:s.update(response_coordinate='wrong'))
    reject('code_identity',lambda s:s.update(model_code_sha256='0'*64))
    reject('archive_identity',lambda s:s.update(weights_sha256='0'*64))
    reject('graph_identity',lambda s:s.update(graph_sha256='0'*64))
    reject('weight_inventory',lambda s:s['weight_shapes'].pop())
    assert before=={p.name:sha(p) for p in private_paths}
    return dict(intermediate_optimizer_updates=updates,weight_arrays=len(manifest['weight_shapes']),
        parameters=int(model.count_params()),fresh_process_replay=restored,
        corruption_cases_rejected=rejected,private_artifacts_unchanged=True,
        runtime=dict(legacy_keras=keras.__version__,numpy=np.__version__),
        resource_policy=dict(native_threads=1,dataset_workers=1,export_and_replay_sequential=True))


def validate(dependencies,directory):
    from scripts.validate_osic_image_model import PINS
    dependencies=Path(dependencies)
    for name,digest in PINS.items():
        if sha(dependencies/name)!=digest:
            raise ValueError('Dependency archive changed')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-state-overlay-') as temporary:
        overlay=Path(temporary)
        for name in PINS:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as wheel:
                    wheel.extractall(overlay)
        env=dict(os.environ,PYTHONPATH=str(overlay)+os.pathsep+str(ROOT),TF_USE_LEGACY_KERAS='1',
            TF_CPP_MIN_LOG_LEVEL='3',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key]='1'
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',
            '--runtime-directory',str(directory)],env=env,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError('Intermediate checkpoint qualification failed:\n'+process.stderr)
        execution=json.loads(process.stdout)
    files=['sciona/osic_image_state.py','sciona/osic_image_model.py','sciona/osic_image_predictor.py',
           'sciona/keras_bounded_prediction.py','scripts/validate_osic_image_state.py']
    return dict(format='osic-image-state.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,full_training_schedule_complete=False,execution=execution,
        dependency_archive_sha256=PINS,software_sha256={name:sha(ROOT/name) for name in files},
        limitations=['Intermediate synthetic training checkpoint only; complete 30-epoch training is still pending.',
            'Numeric inference state excludes optimizer/callback/random state and cannot resume training.',
            'Coordinate labels are caller declarations; authenticity requires external artifact provenance.',
            'Restore invokes only the fixed local model constructor and numeric weight loading; no serialized model graph is executed.',
            'This does not complete raw-input trained inference, the final two-branch graph or publication gates.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',required=True,type=Path)
    parser.add_argument('--dependencies')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--worker',action='store_true')
    modes.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.runtime_directory)))
    elif args.worker:
        print(json.dumps(worker(args.runtime_directory)))
    else:
        if not args.dependencies:
            parser.error('--dependencies required')
        result=validate(args.dependencies,args.runtime_directory)
        (ROOT/'docs/reviews/osic_image_state.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['execution']))
