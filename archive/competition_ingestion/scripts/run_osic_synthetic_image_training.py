"""Run the full image schedule in an isolated, single-core synthetic worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=['sciona/group_member_sampling.py','sciona/keras_serial_steps.py',
       'sciona/osic_image_batches.py','sciona/osic_image_model.py',
       'sciona/osic_image_training_controls.py','sciona/osic_image_training.py',
       'scripts/run_osic_synthetic_image_training.py']


def identities():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def runtime_imports():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import efficientnet.tfkeras  # register the source custom objects
    import tf_keras
    import numpy as np
    return tf_keras,np


def replay(directory):
    keras,np=runtime_imports()
    def forbidden(*args,**kwargs):
        raise RuntimeError('Fitting is forbidden during checkpoint replay')
    keras.Model.fit=forbidden
    keras.Model.train_on_batch=forbidden
    keras.Model.train_step=forbidden
    model=keras.models.load_model(directory/'checkpoints/best.h5',compile=False)
    with np.load(directory/'replay.npz',allow_pickle=False) as data:
        result=model([data['image'],data['metadata']],training=False).numpy()
        np.testing.assert_array_equal(result,data['expected'])
        assert np.isfinite(result).all()
    (directory/'replay.json').write_text(json.dumps(dict(passed=True,queries=2,
        maximum_absolute_error=0.,fresh_process=True,fit_disabled=True))+'\n')


def worker(directory):
    keras,np=runtime_imports()
    from sciona.osic_image_training import train
    pinned=identities()
    rng=np.random.default_rng(805)
    images=[rng.normal(size=(2,512,512,1)).astype('float32') for _ in range(3)]
    metadata=np.array([[.2,0,0,0],[.8,1,1,1],[.4,0,0,1]],dtype='float32')
    slopes=np.array([-.2,.4,.8],dtype='float32')
    (directory/'launch.json').write_text(json.dumps(dict(format='osic-synthetic-image-run.v1',
        synthetic_only=True,code_sha256=pinned,epochs=30,batch_size=8,training_steps=32,
        validation_steps=16,training_seed=517,fixture_seed=805,native_threads=1),indent=2)+'\n')
    def progress(entry):
        with (directory/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps(entry,allow_nan=False),flush=True)
    model,history,checkpoint=train(images,metadata,slopes,seed=517,
        checkpoint_directory=directory/'checkpoints',progress=progress)
    assert len(history)==30 and history[-1]['optimizer_iterations']==960
    # Full-model private HDF5 is used only for this runtime qualification.
    selected=keras.models.load_model(checkpoint,compile=False)
    query_image=images[0]
    query_metadata=np.stack([metadata[0],metadata[0]])
    expected=selected([query_image,query_metadata],training=False).numpy()
    assert np.isfinite(expected).all()
    np.savez(directory/'replay.npz',image=query_image,metadata=query_metadata,expected=expected)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay',
        '--runtime-directory',str(directory)],check=True)
    if identities()!=pinned:
        raise ValueError('Code changed during training; result is not qualified')
    result=dict(format='osic-synthetic-image-run.v1',completed=True,approved=False,
        catalog_mutations=0,synthetic_only=True,code_sha256=pinned,
        epochs=30,optimizer_updates=960,validation_batches=480,batch_size=8,
        best_validation_loss=min(e['val_loss'] for e in history),
        replay=json.loads((directory/'replay.json').read_text()),
        limitations=['Prepared synthetic images, metadata and targets; raw source preparation is not covered.',
            'Same-population validation is not generalization evidence.',
            'Sequential sampling and B5-only construction do not reproduce historical random streams.',
            'Full schedule execution is not source-versus-adapter stochastic training parity.',
            'Private HDF5 replay is not a portable numeric-state codec or original CDG approval.'])
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def launch(dependencies,directory):
    from scripts.validate_osic_image_model import PINS
    directory.mkdir(parents=True,exist_ok=False)
    dependencies=Path(dependencies)
    for name,digest in PINS.items():
        if hashlib.sha256((dependencies/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Dependency archive changed')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-training-') as overlay:
        for name in PINS:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as wheel:
                    wheel.extractall(overlay)
        env=dict(os.environ,TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',
            MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=overlay+os.pathsep+str(ROOT))
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key]='1'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',
            '--runtime-directory',str(directory)],env=env,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',required=True)
    parser.add_argument('--dependencies')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--worker',action='store_true')
    mode.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    directory=Path(args.runtime_directory).resolve()
    if args.replay:
        replay(directory)
    elif args.worker:
        worker(directory)
    else:
        if not args.dependencies:
            parser.error('--dependencies required for launch')
        launch(args.dependencies,directory)
