"""Reuse full lifecycle assertions with the exact catalog graph."""
import pytest
from tests import test_decomposed_video_multilabel_graph as baseline
from sciona.catalog_video_multilabel_graph import build_catalog_video_multilabel_graph


@pytest.mark.parametrize('seed,epochs,batch_size', [(12,80,4),(3,5,3),(29,3,16)])
def test_catalog_graph_full_lifecycle(tmp_path,monkeypatch,seed,epochs,batch_size):
    graph=build_catalog_video_multilabel_graph()
    mean=next(n for n in graph.nodes if n.node_id=='mean')
    assert [(p.name,p.type_desc) for p in mean.outputs]==[('values','NDArray[np.float64]')]
    assert all(p.constraints for node in graph.nodes for p in node.inputs+node.outputs)
    monkeypatch.setattr(baseline,'build_decomposed_video_multilabel_graph',lambda:graph)
    baseline.test_serialized_native_graph_full_lifecycle(tmp_path,monkeypatch,seed,epochs,batch_size)
