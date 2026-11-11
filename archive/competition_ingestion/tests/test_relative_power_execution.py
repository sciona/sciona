import numpy as np
import pytest
from sciona.relative_power_execution import build_relative_power_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer.runner import CDGExecutionSession
from sciona.atoms.riemannian_bci.signal_processing.source_inputs import prepare_segment_windows
from sciona.atoms.riemannian_bci.covariance_features.relative_power import relative_log_band_power
from sciona.atoms.riemannian_bci.signal_processing.feature_vectorization import vectorize_feature_partitions
from sciona.atoms.ml.xgboost.competition_bagging import bagged_window_probabilities
from sciona.atoms.riemannian_bci.signal_processing.segment_scores import segment_probability_max


@pytest.mark.asyncio
async def test_serialized_relative_power_graph_preserves_partition_and_segment_wiring(tmp_path, monkeypatch):
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR', tmp_path)
    rng = np.random.default_rng(5921)
    training = [rng.normal(size=(3, 16000)) for _ in range(12)]
    prediction = [rng.normal(size=(3, length)) for length in [8000, 16000, 24000]]
    labels = np.tile([0, 1], 6)
    train, ids, pred, pred_ids, count = prepare_segment_windows(training, prediction, 8000, 8000)
    bands = [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]]
    features, pred_features = vectorize_feature_partitions(relative_log_band_power(train, bands), relative_log_band_power(pred, bands))
    expected = segment_probability_max(bagged_window_probabilities(features, ids, labels, pred_features, 10), pred_ids, count)
    graph = build_relative_power_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': 'test'} for n in nodes],
        'cdg_edges': [{**e, 'version_id': 'test'} for e in edges]}, version_id='test', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    assert len({(e.target_id, e.input_name) for e in graph.edges}) == len(graph.edges)
    result = await CDGExecutionSession(None, 'synthetic-relative-power', 'check').execute(
        {'training_segments': training, 'prediction_segments': prediction, 'segment_labels': labels}, cdg=restored)
    assert result['status'] == 'completed'
    actual = np.load(tmp_path / 'check' / 'segments' / 'out_relative_power_segment_probabilities.npy')
    np.testing.assert_array_equal(actual, expected)
