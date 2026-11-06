"""Draft separate-field recipe stages; no catalog approval is implied.

These are recipe adapters, not domain-independent numerical primitives. Text
policy and model choices stay explicit in the validated request. Fitted state
is execution-local and must not be shared between concurrent callers.
"""
from dataclasses import dataclass

from threadpoolctl import threadpool_limits

from sciona.text_pair_complete_contract import prepare, PreparedCompletePairs
from sciona.text_pair_contract import _json
from sciona.text_pair_fields import FieldPairFeatures, normalize_populations
from sciona.text_pair_complete_recipe import fit_feature_models, calibrate_and_predict
from sciona.text_pair_normalization_policy import NormalizationPolicy
from sciona.text_pair_training import _labels


@dataclass(frozen=True)
class Normalized:
    payload: dict
    training: list
    calibration: list
    query: list


@dataclass(frozen=True)
class Features:
    populations: Normalized
    fitted: object


@dataclass(frozen=True)
class Models:
    populations: Normalized
    fitted: object


@dataclass(frozen=True)
class Predictions:
    populations: Normalized
    values: dict


def prepare_fields(payload):
    prepared = prepare(payload)
    if prepared.payload['field_role'] != 'title_description':
        raise ValueError('Original topology requires separate title and description fields')
    return prepared


def normalize_fields(prepared):
    if not isinstance(prepared, PreparedCompletePairs):
        raise ValueError('Prepared complete pairs required')
    p = prepare_fields(prepared.payload).payload
    training, calibration, query = normalize_populations(
        p['training_pairs'], p['calibration_pairs'], p['query_pairs'],
        NormalizationPolicy(**p['normalization']))
    return Normalized(p, training, calibration, query)


def fit_features(normalized):
    if not isinstance(normalized, Normalized):
        raise ValueError('Normalized record populations required')
    controls = normalized.payload['controls']
    with threadpool_limits(limits=1):
        fitted = FieldPairFeatures().fit(normalized.training, **{
            k: controls[k] for k in ('components', 'seed', 'iterations', 'oversamples', 'k1', 'b')})
    return Features(normalized, fitted)


def fit_models(features):
    if not isinstance(features, Features):
        raise ValueError('Fitted record features required')
    p = features.populations
    fitted = fit_feature_models(features.fitted, _labels(p.payload['training_labels'], len(p.training)),
        **{k: p.payload['controls'][k] for k in ('seed', 'trees', 'max_depth', 'min_leaf')})
    return Models(p, fitted)


def predict(models):
    if not isinstance(models, Models):
        raise ValueError('Fitted record models required')
    p = models.populations
    with threadpool_limits(limits=1):
        values = calibrate_and_predict(models.fitted, p.calibration,
            _labels(p.payload['calibration_labels'], len(p.calibration)), p.query)
    return Predictions(p, values)


def format_rows(predictions):
    if not isinstance(predictions, Predictions):
        raise ValueError('Record predictions required')
    p, result = predictions.populations, predictions.values
    rows = [dict(row_index=i, raw_score=raw, scaled_score=scaled, relevant=decision)
        for i, (raw, scaled, decision) in enumerate(zip(
            result['scores'], result['scaled_scores'], result['matches'], strict=True))]
    if len(rows) != len(p.query) or any(
        not 0 <= r['raw_score'] <= 1 or not 0 <= r['scaled_score'] <= 1
        or type(r['relevant']) is not int or r['relevant'] not in (0, 1)
        or r['relevant'] != int(r['scaled_score'] >= result['threshold']) for r in rows):
        raise ValueError('Invalid ordered prediction rows')
    output = dict(version=2, task='binary_relevance', field_role='title_description', rows=rows,
        threshold=result['threshold'], calibration_selection_f1=result['calibration_f1'],
        metric=result['metric'], score_scaling=result['score_scaling'], feature_count=result['feature_count'],
        training_rows=len(p.training), calibration_rows=len(p.calibration), query_rows=len(rows))
    _json(output)
    return output
