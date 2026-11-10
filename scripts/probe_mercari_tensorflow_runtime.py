"""Execute the 12 declared source TensorFlow models on synthetic sparse inputs."""
import argparse
import ast
import faulthandler
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import time

# Initialize TensorFlow before sklearn/pandas/Arrow. The inverse order stalled
# this machine's graph executor; the minimal graph succeeds with this order.
import tensorflow.compat.v1 as tf
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

from inventory_mercari_models import inventory

ROOT = Path(__file__).resolve().parents[1]


class InstrumentTraining(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'ConfigProto':
            for keyword in node.keywords:
                if keyword.arg == 'device_count':
                    keyword.value = ast.parse("{'CPU': 1, 'GPU': 0}", mode='eval').body
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == 'fit':
            node.body.insert(0, ast.parse('self.update_count_ = 0').body[0])
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, ast.Tuple) and ast.unparse(t) == '(_, loss)' for t in node.targets):
            return [node, ast.parse('self.update_count_ += 1').body[0]]
        return node


def probe(directory):
    models = inventory(directory)
    triage = json.loads((ROOT/'docs/reviews/competition_mercari_source_triage.json').read_text())
    quantile_sha = next(r['sha256'] for r in triage['source_files'] if r['path'] == 'mercari/regression_clf.py')
    quantile_source = directory/'mercari/regression_clf.py'
    if hashlib.sha256(quantile_source.read_bytes()).hexdigest() != quantile_sha:
        raise ValueError('Pinned quantile source changed')
    tf.disable_eager_execution()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    namespace = dict(tf=tf, np=np, time=time, logging=logging, StandardScaler=StandardScaler,
                     BaseEstimator=BaseEstimator, RegressorMixin=RegressorMixin,
                     mean_squared_log_error=mean_squared_log_error, memory_info=lambda: 'synthetic probe')
    exec(compile(quantile_source.read_text(), '<source-quantile>', 'exec'), namespace)
    tree = ast.parse((directory/'mercari/tf_sparse.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    instrumented = ast.fix_missing_locations(InstrumentTraining().visit(ast.Module(body=definitions, type_ignores=[])))
    exec(compile(instrumented, '<source-tensorflow>', 'exec'), namespace)
    rng = np.random.default_rng(853)
    dense = rng.uniform(0, 1, (8192, 23)).astype(np.float32)
    dense[rng.random(dense.shape) < .7] = 0
    X = sparse.csr_matrix(dense)
    y = np.log1p(1 + dense[:, 0] + 2 * dense[:, 1] + rng.uniform(0, .2, len(dense)))
    query = X[:13].copy()
    results = []
    for declared in models['variants']['tensorflow']:
        controls = dict(declared['resolved_constructor_controls'])
        controls['actfunc'] = namespace['prelu']
        instance = namespace[declared['model_class']](**controls)
        case = dict(group=declared['group'], seed=declared['seed'], passed=False)
        try:
            faulthandler.dump_traceback_later(120, exit=True)
            instance.fit(X, y, use_gpu=False, verbose=False)
            predicted = np.asarray(instance.predict(query))
            replay = np.asarray(instance.predict(query))
            expected_updates = sum((len(y) + controls['batch_size'] * controls['bs_decay'] ** min(i, controls['decay_epochs']) - 1)
                                   // (controls['batch_size'] * controls['bs_decay'] ** min(i, controls['decay_epochs']))
                                   for i in range(controls['n_epoch']))
            assert instance.update_count_ == expected_updates
            assert predicted.shape == (13,) and np.isfinite(predicted).all()
            np.testing.assert_array_equal(predicted, replay)
            case.update(passed=True, epochs=controls['n_epoch'], optimizer_updates=instance.update_count_, finite_query_predictions=True)
        except Exception as error:
            case.update(error_type=type(error).__name__, error=str(error))
        finally:
            faulthandler.cancel_dump_traceback_later()
            if hasattr(instance, '_session'):
                instance._session.close()
        results.append(case)
        print(json.dumps(case), flush=True)
    return dict(format='mercari-tensorflow-runtime-probe.v1', approved=False, catalog_mutations=0,
        synthetic_only=True, synthetic_training_population=8192, cases=results,
        source_commit=models['source_commit'], source_sha256={**models['source_sha256'], 'mercari/regression_clf.py': quantile_sha},
        runtime_versions={name: version(name) for name in ('tensorflow', 'numpy', 'scipy', 'scikit-learn')},
        resource_policy=dict(concurrent_models=1, intra_op_threads=1, inter_op_threads=1, cpu_devices=1, gpu_devices=0),
        limits=['Synthetic sparse features bypass the six application feature graphs; full graph execution and ensemble calibration remain outstanding.',
                'Uses TensorFlow compatibility graph API in the current runtime; historical TensorFlow 1.4 numerical equivalence is not established.',
                'Source Huber-named class retains its inherited MSE builder; no silent loss correction.',
                'Source model sizes, epoch counts, batch schedules and seeds retained; device count restricted and update counter instrumented.',
                'TensorFlow imported before sklearn/Arrow; the opposite order stalled in the local graph executor. Each fit has a 120-second diagnostic watchdog.',
                'No MXNet runtime, additional four winning models, application accuracy or original CDG approval is claimed.'],
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    result = probe(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_mercari_tensorflow_runtime_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['cases']))
