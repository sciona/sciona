"""Save and replay trusted, locally generated source-model checkpoints.

Private diagnostic artifacts only. Never load arbitrary external pickle or
TensorFlow graph files through this helper.
"""
import json
import pickle

import numpy as np


def save_and_replay(model, model_class, controls, tf, directory, matrices):
    directory.mkdir(parents=True, exist_ok=False)
    prefix = str(directory/'model')
    metadata = dict(output=model._output.name, indices=model._xs.indices.name,
                    values=model._xs.values.name, shape=model._xs.shape.name,
                    binary_X=model.binary_X)
    state = dict(target_scaler=model.target_scaler)
    if hasattr(model, 'percentiles'):
        state['percentiles'] = model.percentiles
    (directory/'metadata.json').write_text(json.dumps(metadata))
    with (directory/'inverse-state.pkl').open('wb') as stream:
        pickle.dump(state, stream)
    with model._graph.as_default():
        saver = tf.train.Saver()
        saver.save(model._session, prefix)
    model._session.close()

    # Reconstruct target inversion separately from the trained instance.
    # Only files just produced above are read, never caller-provided artifacts.
    restored = model_class(**controls)
    with (directory/'inverse-state.pkl').open('rb') as stream:
        state = pickle.load(stream)
    for key, value in state.items():
        setattr(restored, key, value)
    metadata = json.loads((directory/'metadata.json').read_text())
    graph = tf.Graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                           device_count={'CPU': 1, 'GPU': 0})
    predictions = []
    with graph.as_default(), tf.Session(graph=graph, config=config) as session:
        saver = tf.train.import_meta_graph(prefix+'.meta')
        saver.restore(session, prefix)
        output = graph.get_tensor_by_name(metadata['output'])
        for matrix in matrices:
            coo = matrix.tocoo()
            feed = {
                graph.get_tensor_by_name(metadata['indices']): np.stack([coo.row, coo.col]).T,
                graph.get_tensor_by_name(metadata['values']): np.ones_like(coo.data) if metadata['binary_X'] else coo.data,
                graph.get_tensor_by_name(metadata['shape']): np.array(coo.shape),
            }
            raw = session.run(output, feed_dict=feed)
            predictions.append(np.expm1(restored._invert_target(raw)))
    return predictions
