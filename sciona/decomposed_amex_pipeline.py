"""Explicit source-recipe stages; numerical combination remains a graph dependency.

The qualified original implementation stays unchanged. These stages retain its
complete learner controls and population ordering. They are not yet approved.
"""
import hashlib
import json
import numpy as np
from sciona.amex_execution import Prepared, prepare
from sciona.amex_raw import preprocess
from sciona.amex_manual import build
from sciona.amex_sequence import encode_series, prediction_slots
from sciona.amex_assembly import assemble
from sciona.amex_row_model import fit as row_fit
from sciona.amex_tree import fit as tree_fit
from sciona.amex_neural_training import fit as neural_fit


def populations(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    p=json.loads(prepared.configuration)
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    controls=p['controls']
    training=preprocess(p['training']['sequences'],controls['category_rules'])
    query=preprocess(p['query']['sequences'],controls['category_rules'])
    return dict(joint={k:training[k]+query[k] for k in training},
        training_count=len(training['numerics']),labels=p['training']['labels'],
        zero_fill_columns=controls['zero_fill_columns'],seed=controls['seed'])


def features(population):
    joint=population['joint']
    blocks=build(**joint,zero_fill_columns=population['zero_fill_columns'])
    encoded=encode_series(joint['numerics'],joint['categories'])['sequences']
    rows=[np.concatenate((x,c),axis=1) for x,c in zip(joint['numerics'],joint['categories'])]
    return dict(blocks=blocks,encoded=encoded,rows=rows)


def row_models(population,feature_values):
    n=population['training_count'];rows=feature_values['rows']
    return row_fit(rows[:n],population['labels'],rows[n:],seed=population['seed'])


def assembly(feature_values,row_result):
    predictions=row_result['training_predictions']+row_result['query_predictions']
    assembled=assemble(feature_values['blocks'],predictions)
    return dict(manual=assembled['tree_features'][:,:-13],slots=prediction_slots(predictions),
        neural=assembled['neural_features'])


def tree_models(population,assembled):
    n=population['training_count'];manual=assembled['manual'];slots=assembled['slots']
    return tree_fit(manual[:n],slots[:n],population['labels'],manual[n:],slots[n:],seed=population['seed'])


def neural_models(population,feature_values,assembled):
    n=population['training_count'];encoded=feature_values['encoded'];values=assembled['neural']
    return neural_fit(encoded[:n],values[:n],population['labels'],encoded[n:],values[n:],seed=population['seed'])


def combination_inputs(tree_result,neural_result):
    branches=[tree_result['variants']['manual']['query_predictions'],
        tree_result['variants']['manual_with_slots']['query_predictions'],
        neural_result['variants']['series']['query_predictions'],
        neural_result['variants']['combined']['query_predictions']]
    values=[np.asarray(p) for p in branches]
    if any(p.ndim!=1 or not len(p) or p.shape!=values[0].shape or p.dtype.kind not in 'iuf'
           or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in values):
        raise ValueError('Four aligned probability vectors required')
    return np.stack([p.astype(np.float64) for p in values]),np.array([.3,.35,.15,.1])


def result(scores):
    if (not isinstance(scores,np.ndarray) or scores.ndim!=1 or not len(scores)
            or scores.dtype!=np.float64 or not np.isfinite(scores).all()
            or np.any((scores<0)|(scores>1))):
        raise ValueError('Finite combined score vector required')
    out=dict(scores=scores.tolist(),models=25,model_counts=dict(row=5,downstream_tree=10,neural=10),
        score_kind='literal_weighted_sum_0.9',
        scope='Independent full CPU learner pipeline; joint population features and source literal nonnormalized blend. Historical accuracy and native parity unqualified.')
    json.dumps(out,allow_nan=False)
    return out
