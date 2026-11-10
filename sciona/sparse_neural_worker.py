"""Private numeric job protocol; initialize TensorFlow before other libraries."""
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu, tf
from sciona.sparse_neural_state import export_prediction_state, restore_prediction_state

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz


def execute(directory):
    request = json.loads((directory/'request.json').read_text())
    X = load_npz(directory/'features.npz')
    identities = [request[k] for k in ('feature_columns', 'feature_coordinate', 'response_coordinate')]
    model = None
    result = dict(passed=True)
    try:
        if request['operation'] == 'fit':
            controls = dict(request['controls'])
            activation = controls.pop('activation')
            if activation not in ('relu', 'prelu') or request['mode'] not in ('mse', 'quantile'):
                raise ValueError('Unknown model configuration')
            cls = SparseRegressor if request['mode'] == 'mse' else QuantileSparseRegressor
            model = cls(**controls, actfunc=prelu if activation == 'prelu' else tf.nn.relu)
            model.fit(X, np.load(directory/'targets.npy', allow_pickle=False))
            state = export_prediction_state(model, *identities)
            (directory/'state.json').write_text(json.dumps(state, allow_nan=False))
            result.update(optimizer_updates=model.update_count_, epoch_schedule=model.epoch_schedule_)
        elif request['operation'] == 'predict':
            model = restore_prediction_state(request['state'], *identities)
            prediction = model.predict(X)
            dtype = np.float64 if request['state']['mode'] == 'quantile' else np.float32
            np.save(directory/'prediction.npy', prediction.astype(dtype), allow_pickle=False)
        else:
            raise ValueError('Unknown numerical operation')
        return result
    finally:
        if model is not None:
            model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    try:
        result = execute(directory)
    except Exception as error:
        # Do not expose observation values through library exception messages.
        result = dict(passed=False, error_type=type(error).__name__)
    (directory/'result.json').write_text(json.dumps(result))
