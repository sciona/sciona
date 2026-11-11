"""Run all configured lifecycle cases from the stored original CDG candidate."""
import pytest
from tests import test_decomposed_video_multilabel_graph as baseline
from scripts import stage_video_corrected_intake as intake


@pytest.fixture(scope='module')
def stored_graph():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db, plan, providers)


@pytest.mark.parametrize('seed,epochs,batch_size', [(12,80,4),(3,5,3),(29,3,16)])
def test_stored_original_full_lifecycle(tmp_path, monkeypatch, stored_graph, seed, epochs, batch_size):
    monkeypatch.setattr(baseline, 'build_decomposed_video_multilabel_graph', lambda: stored_graph.model_copy(deep=True))
    baseline.test_serialized_native_graph_full_lifecycle(tmp_path, monkeypatch, seed, epochs, batch_size)
