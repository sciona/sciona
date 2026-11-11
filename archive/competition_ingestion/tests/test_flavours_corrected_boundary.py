"""The corrected catalog boundary must retain diagnostic sinks and shared roots."""
import pytest
from scripts.stage_flavours_corrected_intake import boundaries,build_graph


def test_catalog_boundary_retains_every_terminal_loss_and_one_shared_label_input():
    graph=build_graph();inputs,outputs=boundaries(graph)
    assert len(inputs)==6 and sum(p['name']=='labels' for p in inputs)==1
    assert len(outputs)==51
    sources=graph.metadata['catalog_output_sources']
    assert sources['result']==dict(node_id='report',output_name='result')
    assert {source['node_id'] for alias,source in sources.items() if alias!='result'}=={
        f'neural_f{fold}_s{seed}_validation' for fold in range(5) for seed in range(10)}
    assert all(p['type_desc']=='float' for p in outputs if p['name']!='result')


def test_shared_input_with_conflicting_type_is_rejected():
    graph=build_graph()
    node=next(n for n in graph.nodes if n.node_id=='evaluation_inputs')
    next(p for p in node.inputs if p.name=='labels').type_desc='dict'
    with pytest.raises(ValueError,match='Repeated root input'):boundaries(graph)


def test_dropping_terminal_validation_loss_is_rejected():
    graph=build_graph()
    del graph.metadata['catalog_output_sources']['neural_f0_s0_validation__loss']
    with pytest.raises(ValueError,match='Complete root boundary'):boundaries(graph)
