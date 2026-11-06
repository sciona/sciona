"""Complete Porto fitting with an independently bound numeric reduction."""
import hashlib
import json
import numpy as np
from sciona.porto_execution import Prepared, prepare
from sciona.porto_preparation import prepare_populations
from sciona.porto_neural_branch import fit_branch
from sciona.porto_tree_blend import fit_tree


def checked_payload(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest() != prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    p = json.loads(prepared.configuration)
    if prepare(p) != prepared:
        raise ValueError('Canonical prepared input required')
    return p


def features(prepared):
    p = checked_payload(prepared)
    return prepare_populations(p['training']['values'], p['query']['values'], **p['controls']['preparation'])


def fit(prepared, transformed):
    p = checked_payload(prepared)
    if type(transformed) is not dict or set(transformed) != {
            'tree_training','tree_query','dae_training','dae_query','prepared_width','binary_output_columns'}:
        raise ValueError('Complete same-invocation feature preparation required')
    predictions, records = [], []
    for controls in p['controls']['neural_models']:
        fitted = fit_branch(transformed['dae_training'], p['training']['labels'],
                            transformed['dae_query'], **controls)
        predictions.append(fitted['probabilities'])
        records.append({k:v for k,v in fitted.items() if k != 'probabilities'})
    tree = fit_tree(transformed['tree_training'], p['training']['labels'],
                    transformed['tree_query'], **p['controls']['tree'])
    return dict(neural_probabilities=np.stack(predictions), tree_probabilities=tree['probabilities'],
        neural_records=records, tree_iterations=tree['boosting_iterations'],
        models=6, dae_models=5, prepared_width=transformed['prepared_width'])


def blend_inputs(fitted):
    if type(fitted) is not dict:
        raise ValueError('Complete fitted ensemble required')
    neural, tree = np.asarray(fitted['neural_probabilities']), np.asarray(fitted['tree_probabilities'])
    if neural.ndim != 2 or neural.shape[0] != 5 or not neural.shape[1] or tree.shape != (neural.shape[1],):
        raise ValueError('Five aligned neural vectors and one tree vector required')
    for values in (neural, tree):
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('Finite probabilities in [0,1] required')
    # The original reduces five neural rows in float64 before adding the tree.
    return np.array(neural, dtype=np.float64, order='C', copy=True), np.ones(5)


def result(prepared, fitted, neural_sum):
    p = checked_payload(prepared)
    blend_inputs(fitted)
    values = np.asarray(neural_sum)
    if values.shape != (len(p['query']['values']),) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Aligned finite neural sum required')
    tree = np.asarray(fitted['tree_probabilities'])
    if tree.shape != values.shape:
        raise ValueError('Tree scores differ from query population')
    probabilities = (values + tree)/6
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Invalid ensemble probabilities')
    output = dict(probabilities=probabilities.tolist(), models=fitted['models'], dae_models=fitted['dae_models'],
        prepared_width=fitted['prepared_width'], learned_widths=[r['learned_width'] for r in fitted['neural_records']],
        score_kind='equal_probability_mean',
        scope='Independent explicit six-model configuration; transductive feature preparation and DAE training, historical ensemble settings unqualified.')
    json.dumps(output, allow_nan=False)
    return output
