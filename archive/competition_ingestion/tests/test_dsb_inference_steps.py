import numpy as np
import pytest
from sciona.atoms.dl.detection.dsb_inference_steps import (
    prepare_inference_geometry,normalize_and_align_tiles,filter_classifier_proposals,
    predict_detector_tiles,predict_classifier_crops,
)
from sciona.atoms.numerical.spatial.volume_tiles import split_volume_tiles
from sciona.dsb_components import prepare_detector_tiles
from sciona.dsb_decomposed_inference import build_dsb_inference_steps_graph


@pytest.mark.parametrize('shape,side,margin',[((1,1,1,1),16,0),((1,19,17,21),32,16),((2,33,29,17),32,0)])
def test_explicit_geometry_matches_source_recipe(shape,side,margin):
    volume=np.arange(np.prod(shape),dtype=np.uint32).reshape(shape).astype(np.uint8)
    padded,coordinates,s,m,b,cs,cm,cb,stride=prepare_inference_geometry(volume,side,margin)
    tiles,grid=split_volume_tiles(padded,s,m,b)
    ct,cg=split_volume_tiles(coordinates,cs,cm,cb)
    actual=normalize_and_align_tiles(tiles,grid,ct,cg)
    expected=prepare_detector_tiles(volume,side_len=side,margin=margin)
    assert stride==4 and actual[2]==expected[2]
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])


def test_strict_filter_and_empty_shapes():
    rows=np.array([[-1.,0,0,0,2],[-.999,0,0,0,2],[-2.,0,0,0,2]])
    np.testing.assert_array_equal(filter_classifier_proposals(rows),rows[[1]])
    assert filter_classifier_proposals(np.empty((0,5))).shape==(0,5)
    with pytest.raises(ValueError):filter_classifier_proposals(np.ones((2,4)))
    with pytest.raises(ValueError):normalize_and_align_tiles(np.zeros(1),(1,1,1),np.zeros(1),(2,1,1))


def test_training_mode_models_rejected_before_inference():
    class TrainingModel:
        training=True
    with pytest.raises(ValueError,match='Eval detector'):
        predict_detector_tiles(np.ones((1,1)),np.ones((1,1)),TrainingModel())
    with pytest.raises(ValueError,match='Eval classifier'):
        predict_classifier_crops(np.ones(1),np.ones(1),TrainingModel())


def test_explicit_graph_roundtrip_and_required_stage_inventory():
    from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
    graph=build_dsb_inference_steps_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph
    assert len(graph.nodes)==18
    by_id={n.node_id:n for n in graph.nodes}
    for name in ['image_tiles','coordinate_tiles']:
        assert by_id[name].matched_primitive.endswith('volume_tiles.split_volume_tiles')
    assert by_id['reconstruct'].matched_primitive.endswith('volume_tiles.combine_volume_tiles')
    assert {p.name for p in by_id['prediction'].outputs}=={'case_probability','proposal_probabilities'}
    assert all(n.matched_primitive!='sciona.atoms.dl.detection.dsb_execution.dsb_predict' for n in graph.nodes)

    for edge in graph.edges:
        source=next(p for p in by_id[edge.source_id].outputs if p.name==edge.output_name)
        target=next(p for p in by_id[edge.target_id].inputs if p.name==edge.input_name)
        assert edge.source_type==source.type_desc and edge.target_type==target.type_desc
