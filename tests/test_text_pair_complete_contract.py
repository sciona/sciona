"""Synthetic serialization, row alignment and pre-fit rejection checks."""
import json
import pytest

from sciona.text_pair_complete_contract import prepare, execute, execute_json
from sciona.text_pair_complete_recipe import execute as run_recipe
from sciona.text_pair_normalization_policy import NormalizationPolicy
from tests.test_text_pair_complete_recipe import CONTROLS, policy
from tests.test_text_pair_training import data


def payload():
    train, y, cal, cy, query = data()
    p = policy()
    return dict(version=2, task='binary_relevance', field_role='shared_text',
        training_pairs=train, training_labels=y, calibration_pairs=cal, calibration_labels=cy,
        query_pairs=query, controls=dict(CONTROLS), normalization=dict(vocabulary=dict(p.vocabulary),
            domain_tokens={k: list(v) for k, v in p.domain_tokens.items()},
            synonyms={k: list(v) for k, v in p.synonyms.items()}))


def test_complete_serialized_round_trip_matches_direct_and_preserves_repeated_rows():
    p = payload()
    p['query_pairs'] += [p['query_pairs'][0]]
    result = json.loads(execute_json(json.dumps(p)))
    reference = run_recipe(p['training_pairs'], p['training_labels'], p['calibration_pairs'],
        p['calibration_labels'], p['query_pairs'], policy=NormalizationPolicy(**p['normalization']), **p['controls'])
    assert [r['row_index'] for r in result['rows']] == [0, 1, 2]
    assert [r['raw_score'] for r in result['rows']] == reference['scores']
    assert [r['scaled_score'] for r in result['rows']] == reference['scaled_scores']
    assert [r['relevant'] for r in result['rows']] == reference['matches']
    assert result['threshold'] == reference['threshold']
    assert result['calibration_selection_f1'] == reference['calibration_f1']
    assert [result[k] for k in ('training_rows', 'calibration_rows', 'query_rows', 'feature_count')] == [8, 4, 3, 13]


def test_copy_isolates_request_and_mutated_prepared_data_is_revalidated(monkeypatch):
    p = payload()
    prepared = prepare(p)
    p['normalization']['vocabulary'].clear()
    assert prepared.payload['normalization']['vocabulary']
    prepared.payload['query_pairs'] = prepared.payload['training_pairs']
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid request reached execution')
    monkeypatch.setattr('sciona.text_pair_complete_contract.run_recipe', forbidden)
    with pytest.raises(ValueError, match='Overlapping'):
        execute(prepared)


@pytest.mark.parametrize('case', ['extra', 'version', 'task', 'role', 'tuple', 'nan',
    'seed', 'boolean_control', 'missing_control', 'extra_policy', 'labels', 'overlap', 'b', 'k1'])
def test_invalid_requests_rejected_before_training(case, monkeypatch):
    p = payload()
    if case == 'extra': p['extra'] = 1
    if case == 'version': p['version'] = True
    if case == 'task': p['task'] = 'ranking'
    if case == 'role': p['field_role'] = 'unknown_role'
    if case == 'tuple': p['query_pairs'][0] = tuple(p['query_pairs'][0])
    if case == 'nan': p['controls']['b'] = float('nan')
    if case == 'seed': p['controls']['seed'] = 2**32
    if case == 'boolean_control': p['controls']['trees'] = True
    if case == 'missing_control': del p['controls']['iterations']
    if case == 'extra_policy': p['normalization']['extra'] = {}
    if case == 'labels': p['calibration_labels'][0] = True
    if case == 'overlap': p['query_pairs'] = [['bird colour', 'red bird']]
    if case == 'b': p['controls']['b'] = 2
    if case == 'k1': p['controls']['k1'] = 0
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid request reached execution')
    monkeypatch.setattr('sciona.text_pair_complete_contract.run_recipe', forbidden)
    with pytest.raises(ValueError):
        execute(prepare(p))


@pytest.mark.parametrize('text', ['{"version":2,"version":2}', '{"a":{"x":1,"x":2}}',
                                '{"value":NaN}', '{"value":Infinity}', '{"value":1e400}'])
def test_strict_json_decoder(text):
    with pytest.raises(ValueError):
        execute_json(text)


def test_separate_fields_serialized_lifecycle():
    from tests.test_text_pair_fields import records
    from sciona.text_pair_fields import execute as run_fields
    p = payload()
    p['field_role'] = 'title_description'
    for name in ('training_pairs', 'calibration_pairs', 'query_pairs'):
        p[name] = records(p[name])
    p['query_pairs'][0][0]['title'] = 'purple cloud'
    result = json.loads(execute_json(json.dumps(p)))
    reference = run_fields(p['training_pairs'], p['training_labels'], p['calibration_pairs'],
        p['calibration_labels'], p['query_pairs'], policy=NormalizationPolicy(**p['normalization']), **p['controls'])
    assert result['field_role'] == 'title_description'
    assert [r['raw_score'] for r in result['rows']] == reference['scores']
    assert [r['scaled_score'] for r in result['rows']] == reference['scaled_scores']
    assert [r['relevant'] for r in result['rows']] == reference['matches']
