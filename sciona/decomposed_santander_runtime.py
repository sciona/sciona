"""Complete Santander lifecycle with explicit stages and reusable numeric sum."""
import hashlib
import json
import numpy as np
from scipy.stats import rankdata
from sciona.santander_execution import Prepared, prepare
from sciona.santander_retraining import expanded_population
from sciona.santander_training import train_neural_cv
from sciona.santander_tree import train_tree_cv


def checked_payload(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest() != prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    payload = json.loads(prepared.configuration)
    if prepare(payload) != prepared:
        raise ValueError('Canonical prepared input required')
    return payload


def populations(payload):
    return (np.asarray(payload['training']['values'], dtype=float),
            payload['training']['labels'], np.asarray(payload['query']['values'], dtype=float))


def initial(prepared):
    p = checked_payload(prepared)
    return train_neural_cv(*populations(p), **p['controls']['initial_neural'])


def selections(prepared, initial_models):
    p = checked_payload(prepared)
    if type(initial_models) is not dict or 'mean_ranks' not in initial_models:
        raise ValueError('Complete initial neural result required')
    x, y, q = populations(p)
    # Validate both full tails before starting either branch refit.
    return {name: expanded_population(x, y, q, initial_models['mean_ranks'],
        positives=positive, negatives=negative, reference_policy=p['controls']['reference_policy'])
        for name, positive, negative in [('neural', 5000, 3000), ('tree', 2700, 2000)]}


def branches(prepared, selected):
    p = checked_payload(prepared)
    if type(selected) is not dict or set(selected) != {'neural', 'tree'}:
        raise ValueError('Both same-invocation selected populations required')
    result = {}
    for name, learner in [('neural', train_neural_cv), ('tree', train_tree_cv)]:
        population = selected[name]
        result[name] = learner(population['training'], population['labels'], population['query'],
                               **p['controls'][name])
    return result


def blend_inputs(fitted):
    if type(fitted) is not dict or set(fitted) != {'neural', 'tree'}:
        raise ValueError('Both fitted branches required')
    arrays = [np.asarray(fitted[name]['mean_ranks']) for name in ('neural', 'tree')]
    if any(a.ndim != 1 or not len(a) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonempty finite real branch ranks required')
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('Branch query populations differ')
    return np.stack([rankdata(a, method='average') for a in arrays]), np.array([2.1, 1.])


def result(prepared, initial_models, selected, fitted, weighted):
    p = checked_payload(prepared)
    if type(selected) is not dict or type(fitted) is not dict or set(selected) != {'neural', 'tree'} or set(fitted) != {'neural', 'tree'}:
        raise ValueError('Both same-invocation branches required')
    values = np.asarray(weighted)
    if values.shape != (len(p['query']['values']),) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Aligned finite weighted scores required')
    # Divide after adding, preserving the complete original floating operation order.
    output = dict(scores=(values / 3.1).tolist(),
        selection_counts={name: dict(positive=int(np.sum(v['selected_labels'] == 1)),
            negative=int(np.sum(v['selected_labels'] == 0)), labeled_rows=len(v['labels']),
            query_reference_rows=len(v['query'])) for name, v in selected.items()},
        model_counts={'initial_neural': len(initial_models['models']),
                      **{name: len(v['models']) for name, v in fitted.items()}},
        score_kind='rank_blend', reference_policy=p['controls']['reference_policy'],
        validation_scope='Pseudo-labeled query rows remain in the count reference; global encoding precedes CV and scores are not independent validation.')
    json.dumps(output, allow_nan=False)
    return output
