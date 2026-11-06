"""Strict finite JSON boundary for the pending complete binary-relevance recipe."""
import copy
from dataclasses import dataclass
import json

from sciona.bm25_scoring import bm25_score
from sciona.text_pair_contract import _json
from sciona.text_pair_complete_recipe import execute as run_recipe
from sciona.text_pair_normalization_policy import NormalizationPolicy
from sciona.text_pair_training import _labels


@dataclass(frozen=True)
class PreparedCompletePairs:
    payload: dict


def prepare(payload):
    """Validate a copied request before any model fitting.

    shared_text declares that both positions are comparable text representations,
    used consistently for every lexical feature; separate title/description
    records are not silently concatenated. Each output ordinal corresponds to
    the same input query ordinal. External identifiers remain caller-owned.
    """
    _json(payload)
    fields = {'version', 'task', 'field_role', 'normalization', 'controls',
              'training_pairs', 'training_labels', 'calibration_pairs', 'calibration_labels', 'query_pairs'}
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError('Exact complete text-pair fields required')
    if type(payload['version']) is not int or payload['version'] != 2:
        raise ValueError('Complete text-pair version 2 required')
    if payload['task'] != 'binary_relevance' or payload['field_role'] != 'shared_text':
        raise ValueError('Explicit binary relevance and shared text roles required')
    normalization = payload['normalization']
    if type(normalization) is not dict or set(normalization) != {'vocabulary', 'domain_tokens', 'synonyms'}:
        raise ValueError('Exact normalization policy fields required')
    policy = NormalizationPolicy(**normalization)
    controls = payload['controls']
    integer_fields = {'components', 'seed', 'iterations', 'oversamples', 'trees', 'max_depth', 'min_leaf'}
    if type(controls) is not dict or set(controls) != integer_fields | {'k1', 'b'}:
        raise ValueError('Exact numerical controls required')
    for name in integer_fields:
        value = controls[name]
        if type(value) is not int or value < (0 if name == 'seed' else 1):
            raise ValueError('Valid integer numerical control required')
    if controls['seed'] >= 2**32:
        raise ValueError('Seed out of range')
    bm25_score([], [], 0, 1, 1., controls['k1'], controls['b'])
    populations = [payload[name + '_pairs'] for name in ('training', 'calibration', 'query')]
    policy.normalize_populations(*populations)
    for name in ('training', 'calibration'):
        labels = payload[name + '_labels']
        if type(labels) is not list or any(type(v) is not int for v in labels):
            raise ValueError('Integer binary label lists required')
        _labels(labels, len(payload[name + '_pairs']))
    return PreparedCompletePairs(copy.deepcopy(payload))


def execute(prepared):
    """Revalidate intermediates and emit finite, ordered generic relevance rows.

    The rows form a generic submission table, not any named competition's file
    format. Raw ensemble and isotonic-scaled scores are distinct. The reported
    selection F1 reuses mapping-fit rows and is not independent performance.
    """
    if not isinstance(prepared, PreparedCompletePairs):
        raise ValueError('Prepared complete pairs required')
    p = prepare(prepared.payload).payload
    result = run_recipe(p['training_pairs'], p['training_labels'], p['calibration_pairs'],
        p['calibration_labels'], p['query_pairs'], policy=NormalizationPolicy(**p['normalization']), **p['controls'])
    rows = [dict(row_index=i, raw_score=raw, scaled_score=scaled, relevant=decision)
            for i, (raw, scaled, decision) in enumerate(zip(
                result['scores'], result['scaled_scores'], result['matches'], strict=True))]
    if len(rows) != len(p['query_pairs']) or any(
            not 0 <= row['raw_score'] <= 1 or not 0 <= row['scaled_score'] <= 1
            or type(row['relevant']) is not int or row['relevant'] not in (0, 1)
            or row['relevant'] != int(row['scaled_score'] >= result['threshold']) for row in rows):
        raise ValueError('Invalid ordered prediction rows')
    output = dict(version=2, task=p['task'], field_role=p['field_role'], rows=rows,
        threshold=result['threshold'], calibration_selection_f1=result['calibration_f1'],
        metric=result['metric'], score_scaling=result['score_scaling'], feature_count=result['feature_count'],
        training_rows=len(p['training_pairs']), calibration_rows=len(p['calibration_pairs']), query_rows=len(rows))
    _json(output)
    return output


def execute_json(request):
    """Decode strict JSON (including duplicate-key rejection) and encode output."""
    if type(request) is not str:
        raise ValueError('JSON request string required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON constant')
    payload = json.loads(request, object_pairs_hook=pairs, parse_constant=constant)
    return json.dumps(execute(prepare(payload)), allow_nan=False, sort_keys=True, separators=(',', ':'))
