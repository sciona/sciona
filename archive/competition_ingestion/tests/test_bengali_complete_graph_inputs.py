import numpy as np

from scripts.grapheme_synthetic_domains import complete_populations, UNSEEN
from scripts.validate_bengali_complete_graph import inputs_for
from sciona.grapheme_execution import build_grapheme_graph


def test_root_inputs_keep_zero_shot_training_and_query_boundaries(tmp_path):
    data = complete_populations()
    inputs = inputs_for(data, tmp_path/'runtime', tmp_path/'initialization.pth')
    graph = build_grapheme_graph()
    bound = {(edge.target_id, edge.input_name) for edge in graph.edges}
    roots = {port.name for node in graph.nodes for port in node.inputs if (node.node_id, port.name) not in bound}
    assert set(inputs) == roots
    assert inputs['validation_images'] is data['calibration'].images
    assert inputs['query_images'] is data['query'].images
    for branch in ('ood', 'seen'):
        spec = inputs[branch+'_spec']
        assert spec['population'] is data['hand']
        assert UNSEEN not in spec['population'].labels
        assert UNSEEN not in spec['validation'].labels
        if branch == 'ood':
            known = np.flatnonzero(data['known_class_mask'])
            np.testing.assert_array_equal(known[spec['targets']], spec['population'].labels)
            np.testing.assert_array_equal(known[spec['validation_targets']], spec['validation'].labels)
    for suffix in ('a', 'b'):
        assert UNSEEN in inputs['font_'+suffix+'_spec']['population'].labels
        assert inputs['gan_'+suffix+'_spec']['hand'] is data['hand']
    assert len(inputs['truth_components']) == len(data['calibration'].images)
    assert 'query_targets' not in inputs
