"""Synchronous single-core bridge to the known sparse neural worker module."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.sparse import save_npz


def run_sparse_neural(request, features, targets=None):
    with tempfile.TemporaryDirectory(prefix='sciona-neural-job-') as temporary:
        directory = Path(temporary)
        (directory/'request.json').write_text(json.dumps(request, allow_nan=False))
        save_npz(directory/'features.npz', features, compressed=False)
        if targets is not None:
            np.save(directory/'targets.npy', targets, allow_pickle=False)
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2')
        process = subprocess.Popen([sys.executable, '-m', 'sciona.sparse_neural_worker', str(directory)],
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        if process.returncode or not (directory/'result.json').is_file():
            raise RuntimeError('Sparse neural worker failed before completing the numeric protocol')
        result = json.loads((directory/'result.json').read_text())
        if result.get('passed') is not True:
            raise ValueError('Sparse neural operation rejected: '+result.get('error_type', 'UnknownError'))
        if request['operation'] == 'fit':
            controls = request['controls']
            bs, lr = controls['batch_size'], controls['learning_rate']
            expected = []
            for epoch in range(controls['n_epoch']):
                expected.append(dict(epoch=epoch, batch_size=bs, learning_rate=lr,
                                     updates=(features.shape[0]+bs-1)//bs))
                if epoch < controls['decay_epochs']:
                    bs *= controls['bs_decay']
                    lr *= controls['lr_decay']
            if (result.get('epoch_schedule') != expected
                    or result.get('optimizer_updates') != sum(e['updates'] for e in expected)):
                raise ValueError('Worker did not complete the exact requested training schedule')
            return json.loads((directory/'state.json').read_text())
        return np.load(directory/'prediction.npy', allow_pickle=False)
