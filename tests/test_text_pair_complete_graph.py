"""Synthetic native graph parity, stage routing and training isolation."""
import copy

import pytest

from sciona.text_pair_complete_contract import execute, prepare
from sciona.text_pair_complete_graph import build_complete_text_pair_graph, STAGES
from sciona.text_pair_complete_stages import prepare_fields
from tests.test_text_pair_complete_contract import payload
from tests.test_text_pair_fields import records
from tests.test_text_pair_provider_graphs import run


def field_payload():
    p = payload()
    p['field_role'] = 'title_description'
    for name in ('training_pairs', 'calibration_pairs', 'query_pairs'):
        p[name] = records(p[name])
    return p


@pytest.mark.parametrize('change', ['baseline', 'distinct_titles', 'repeated_query'])
def test_native_complete_graph_parity_and_stage_identity(change, tmp_path, monkeypatch):
    p = field_payload()
    if change == 'distinct_titles':
        p['query_pairs'][0][0]['title'] = 'purple cloud'
    if change == 'repeated_query':
        p['query_pairs'].append(copy.deepcopy(p['query_pairs'][0]))
    before = copy.deepcopy(p)
    expected = execute(prepare(p))
    captured = run(build_complete_text_pair_graph(), {'payload': p}, tmp_path, monkeypatch)
    assert captured['format_rows', 'out_result'] == expected
    assert p == before
    for source, target, input_name in zip(STAGES, STAGES[1:],
            ('prepared', 'normalized', 'features', 'models', 'predictions')):
        assert captured[source, 'out_result'] is captured[target, 'in_' + input_name]
    normalized = captured['normalize_fields', 'out_result']
    features = captured['fit_features', 'out_result'].fitted
    models = captured['fit_models', 'out_result'].fitted
    assert features.rows == normalized.training
    assert features.descriptions.bm25 is models.features.descriptions.bm25
    assert models.features is features
    assert features.rows != normalized.calibration
    assert features.rows != normalized.query
    assert models.forest.n_jobs == 1


def test_original_graph_rejects_shared_field_scope_before_fitting():
    with pytest.raises(ValueError, match='separate title and description'):
        prepare_fields(payload())


def test_mutated_prepared_population_rejected_before_feature_fitting():
    from sciona.text_pair_complete_stages import normalize_fields
    prepared = prepare_fields(field_payload())
    prepared.payload['query_pairs'] = prepared.payload['training_pairs']
    with pytest.raises(ValueError, match='Overlapping'):
        normalize_fields(prepared)
