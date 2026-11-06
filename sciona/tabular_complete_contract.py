"""Versioned finite-JSON boundary for the pending expanded tabular recipe."""
from dataclasses import dataclass
import copy
import json

from sciona.tabular_ensemble_contract import _json, prepare as prepare_populations
from sciona.tabular_category_features import _interactions
from sciona.tabular_complete_training import fit_complete_stacker
from sciona.tabular_ensemble_calibration import calibrate_stack, predict_calibrated


@dataclass(frozen=True)
class PreparedComplete:
    configuration: str


def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Exact policy fields required')


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError('Positive integer policy required')


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError('Valid random seed required')


def _columns(value, width, *, nonempty=False):
    if (type(value) is not list or (nonempty and not value)
            or any(type(i) is not int or not 0 <= i < width for i in value)
            or len(set(value)) != len(value)):
        raise ValueError('Distinct valid original column indices required')


def _number(value, *, positive=False):
    if type(value) not in (int, float) or (value <= 0 if positive else value < 0):
        raise ValueError('Valid numeric policy required')


def validate_feature_policy(policy, numeric_width, categorical_width):
    _fields(policy, ('clip_low', 'clip_high', 'max_mode_fraction', 'log_columns', 'interactions',
        'smoothing', 'group_aggregates', 'text_topics', 'entity_embeddings', 'selection'))
    if (any(type(policy[k]) not in (int, float) for k in ('clip_low', 'clip_high', 'max_mode_fraction'))
            or not 0 <= policy['clip_low'] < policy['clip_high'] <= 1
            or not 0 < policy['max_mode_fraction'] <= 1):
        raise ValueError('Valid clipping and modal-share policies required')
    _columns(policy['log_columns'], numeric_width)
    _interactions(policy['interactions'], categorical_width)
    _number(policy['smoothing'])
    for name in ('group_aggregates', 'text_topics', 'entity_embeddings'):
        if type(policy[name]) is not list:
            raise ValueError('Feature specifications must be explicit lists')
    for spec in policy['group_aggregates']:
        _fields(spec, ('key_column', 'value_columns', 'statistics', 'unknown_value'))
        _columns([spec['key_column']], categorical_width)
        _columns(spec['value_columns'], numeric_width, nonempty=True)
        stats = spec['statistics']
        if (type(stats) is not list or not stats or any(type(v) is not str or v not in
                {'count', 'sum', 'mean', 'var', 'std', 'min', 'max'} for v in stats)
                or len(set(stats)) != len(stats) or type(spec['unknown_value']) not in (int, float)):
            raise ValueError('Valid aggregate statistics and fill required')
    for spec in policy['text_topics']:
        _fields(spec, ('column', 'topics', 'max_features', 'iterations', 'lowercase', 'seed'))
        _columns([spec['column']], categorical_width)
        for key in ('topics', 'max_features', 'iterations'): _positive(spec[key])
        _seed(spec['seed'])
        if type(spec['lowercase']) is not bool: raise ValueError('Explicit case policy required')
    for spec in policy['entity_embeddings']:
        _fields(spec, ('columns', 'dimension', 'iterations', 'regularization', 'tolerance', 'seed'))
        _columns(spec['columns'], categorical_width, nonempty=True)
        _positive(spec['dimension']); _positive(spec['iterations']); _seed(spec['seed'])
        _number(spec['regularization']); _number(spec['tolerance'], positive=True)
    selection = policy['selection']
    if selection is not None:
        _fields(selection, ('method', 'keep', 'step'))
        if selection['method'] != 'rfe': raise ValueError('RFE selection required')
        _positive(selection['keep']); _positive(selection['step'])
    # Fold-specific vocabulary, surviving width and optimization convergence
    # are runtime conditions, not facts inferred from the input schema.


def prepare(payload):
    _json(payload)
    _fields(payload, ('version', 'training', 'calibration', 'query', 'controls'))
    if type(payload['version']) is not int or payload['version'] != 2:
        raise ValueError('Expanded tabular version 2 required')
    controls = payload['controls']
    _fields(controls, ('seed', 'trees', 'max_depth', 'min_leaf', 'features'))
    training = payload['training']
    # Reuse the retained population/split contract without invoking its trainer.
    legacy = copy.deepcopy(payload)
    legacy['version'] = 1
    legacy['controls'] = {key: controls[key] for key in ('seed', 'trees', 'max_depth', 'min_leaf')}
    legacy['controls'].update(clip_low=0., clip_high=1.)
    prepare_populations(legacy)
    validate_feature_policy(controls['features'], len(training['numeric'][0]), len(training['categorical'][0]))
    return PreparedComplete(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(',', ':')))


def execute(prepared):
    if not isinstance(prepared, PreparedComplete): raise ValueError('Prepared complete recipe required')
    p = json.loads(prepare(json.loads(prepared.configuration)).configuration)
    controls = dict(p['controls']); controls['feature_controls'] = controls.pop('features')
    fitted = fit_complete_stacker(**p['training'], **controls)
    calibrated = calibrate_stack(fitted, **p['calibration'])
    result = predict_calibrated(calibrated, **p['query'])
    result.update(training_rows=len(p['training']['numeric']), calibration_rows=len(p['calibration']['numeric']),
        query_rows=len(p['query']['numeric']), folds=len(fitted.fold_models), models=2,
        forest_trees=len(fitted.base.trees.estimators_), oof_rows=len(fitted.oof_predictions))
    _json(result)
    return result
