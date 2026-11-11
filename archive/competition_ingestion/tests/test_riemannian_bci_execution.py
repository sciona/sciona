from collections import Counter
import numpy as np
import pytest
from sciona.riemannian_bci_execution import build_riemannian_branch_graph
from sciona.atoms.riemannian_bci.signal_processing.source_inputs import prepare_segment_windows


def test_graph_has_no_ambiguous_ports_and_only_intended_runtime_inputs():
    graph = build_riemannian_branch_graph()
    destinations = Counter((e.target_id, e.input_name) for e in graph.edges)
    assert all(count == 1 for count in destinations.values())
    roots = {i.name for n in graph.nodes for i in n.inputs if (n.node_id, i.name) not in destinations}
    assert roots == {'training_segments', 'prediction_segments', 'segment_labels'}
    edges = {(e.target_id, e.input_name): (e.source_id, e.output_name) for e in graph.edges}
    for branch in ['autocorrelation', 'coherence']:
        assert edges[(branch + '_tangent', 'metric')] == ('configuration', branch + '_metric')
        assert edges[(branch + '_classifier', 'n_bags')] == ('configuration', branch + '_bags')
        assert edges[(branch + '_segments', 'segment_indices')] == ('windows', 'prediction_indices')


def test_window_boundary_adapter_preserves_separate_segment_identities():
    result = prepare_segment_windows([np.ones((2, 9)), np.ones((2, 5)) * 2],
                                     [np.ones((2, 13)) * 3], 4, 4)
    train, train_ids, prediction, pred_ids, count = result
    np.testing.assert_array_equal(train_ids, [0, 0, 1])
    np.testing.assert_array_equal(pred_ids, [0, 0, 0])
    assert count == 1 and train.shape == prediction.shape == (3, 2, 4)
    assert np.all(prediction == 3) and np.all(train < 3)
    with pytest.raises(ValueError, match='channel counts'):
        prepare_segment_windows([np.ones((2, 8))], [np.ones((3, 8))], 4, 4)
