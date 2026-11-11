"""Execute the original candidate retrieved from the catalog document RPC."""
import copy

import pytest

from scripts import stage_text_pair_corrected_intake as intake
from sciona.text_pair_complete_contract import execute, prepare
from tests.test_text_pair_complete_graph import field_payload
from tests.test_text_pair_provider_graphs import run


@pytest.fixture(scope='module')
def stored_graph():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db, plan, providers)


@pytest.mark.parametrize('change', ['baseline', 'distinct_titles', 'repeated_query'])
def test_stored_original_matches_full_recipe(stored_graph, change, tmp_path, monkeypatch):
    p = field_payload()
    if change == 'distinct_titles':
        p['query_pairs'][0][0]['title'] = 'purple cloud'
    if change == 'repeated_query':
        p['query_pairs'].append(copy.deepcopy(p['query_pairs'][0]))
    before = copy.deepcopy(p)
    captured = run(stored_graph, {'payload': p}, tmp_path, monkeypatch)
    assert captured['format_rows', 'out_result'] == execute(prepare(p))
    assert p == before
    assert len(stored_graph.nodes) == 6
    for edge in stored_graph.edges:
        assert captured[edge.source_id, 'out_'+edge.output_name] is captured[edge.target_id, 'in_'+edge.input_name]
