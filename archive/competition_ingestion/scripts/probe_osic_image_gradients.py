"""Bounded full-batch numerical diagnostics; not a training qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(case):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np
    from sciona.osic_image_model import build_model
    from sciona.osic_image_training_controls import compile_model
    tf.keras.utils.set_random_seed(517)
    model = build_model()
    compile_model(model)
    assert not model.losses

    @tf.function
    def gradients(image, metadata, target):
        with tf.GradientTape() as tape:
            prediction = model([image, metadata], training=True)
            loss = tf.reduce_mean(tf.abs(target-prediction))
        values = tape.gradient(loss, model.trainable_variables)
        if any(value is None for value in values):
            raise ValueError('Disconnected trainable parameter')
        finite = tf.stack([tf.reduce_all(tf.math.is_finite(value)) for value in values])
        return loss, values, finite

    rng = np.random.default_rng(17)
    if case == 'zero':
        image = np.zeros((8,512,512,1), np.float32)
        metadata = np.zeros((8,4), np.float32)
        target = np.ones((8,1), np.float32)
    elif case == 'varied':
        image = rng.normal(size=(8,512,512,1)).astype('float32')
        metadata = rng.normal(size=(8,4)).astype('float32')
        target = np.linspace(.1,.8,8,dtype='float32').reshape(-1,1)
    else:
        raise ValueError('Unknown synthetic case')
    loss, values, finite = gradients(image, metadata, target)
    flags = finite.numpy()
    accepted = bool(flags.all() and np.isfinite(float(loss)))
    if accepted:
        model.optimizer.apply_gradients(zip(values, model.trainable_variables))
    return dict(case=case, batch_size=8, image_shape=[512,512,1],
        loss=float(loss) if np.isfinite(float(loss)) else None,
        gradient_tensors=len(flags), finite_gradient_tensors=int(flags.sum()),
        optimizer_update_applied=accepted, optimizer_iterations=int(model.optimizer.iterations),
        all_model_weights_finite=all(np.isfinite(value).all() for value in model.get_weights()),
        tensorflow=tf.__version__, native_threads=1)


def probe(dependencies):
    from scripts.validate_osic_image_model import PINS
    dependencies = Path(dependencies)
    for name, digest in PINS.items():
        if sha(dependencies/name) != digest:
            raise ValueError('Dependency archive changed')
    results = []
    with tempfile.TemporaryDirectory(prefix='sciona-osic-gradients-') as tmp:
        for name in PINS:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as wheel:
                    wheel.extractall(tmp)
        env = dict(os.environ, TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='3',
            MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=tmp+os.pathsep+str(ROOT))
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key] = '1'
        for case in ('zero', 'varied'):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--worker', case], env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Synthetic gradient probe failed:\n'+result.stderr)
            results.append(json.loads(result.stdout))
            print(json.dumps(dict(completed_case=case)), flush=True)
    return dict(format='osic-image-gradient-probe.v1', probe_completed=True,
        training_qualified=False, approved=False, catalog_mutations=0, synthetic_only=True,
        model_sha256=sha(ROOT/'sciona/osic_image_model.py'),
        controls_sha256=sha(ROOT/'sciona/osic_image_training_controls.py'),
        probe_sha256=sha(__file__), dependency_archive_sha256=PINS, cases=results,
        limitations=[
            'One full-size batch per isolated process, not the complete 30-epoch schedule.',
            'The probe withholds optimizer updates on nonfinite gradients; source training has no such explicit guard.',
            'Training-mode BatchNorm state updates occur even when the optimizer update is withheld; each process is discarded.',
            'No gradient clipping, batch-size reduction, model substitution or pretrained weights.',
            'No source-versus-adapter training parity or empirical effectiveness claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies')
    parser.add_argument('--worker', choices=['zero','varied'])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
    else:
        if not args.dependencies:
            parser.error('--dependencies is required outside worker mode')
        report = probe(args.dependencies)
        (ROOT/'docs/reviews/osic_image_gradient_probe.json').write_text(
            json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report['cases'], allow_nan=False))
