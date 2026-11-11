import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feng_expanded_partitions import feng_expanded_partitions
from sciona.feng_expanded_execution import build_feng_expanded_execution_graph


@pytest.mark.parametrize('parts', [([], [np.zeros((10, 16))]), ([np.zeros((10, 16))], []),
                                 ([np.zeros((20000, 16))], [np.zeros((240000, 16))]),
                                 ([np.zeros((240000, 15))], [np.zeros((240000, 16))])])
def test_invalid_partition_boundaries(parts):
    with pytest.raises(ValueError):
        feng_expanded_partitions(*parts)


def test_shared_graph_has_distinct_terminal_outputs_and_unique_input_producers():
    graph = build_feng_expanded_execution_graph()
    targets = [(e.target_id, e.input_name) for e in graph.edges]
    assert len(targets) == len(set(targets)) == 6
    consumers = {e.source_id for e in graph.edges}
    terminals = {p.name for n in graph.nodes if n.node_id not in consumers for p in n.outputs}
    assert terminals == {'feng_expanded_knn_segment_probabilities', 'feng_glm_segment_probabilities'}
    roots = {p.name for n in graph.nodes for p in n.inputs if (n.node_id, p.name) not in targets}
    assert roots == {'training_segments', 'prediction_segments', 'segment_labels'}
