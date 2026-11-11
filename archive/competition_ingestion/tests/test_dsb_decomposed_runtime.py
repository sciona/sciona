from types import SimpleNamespace
import pytest
from sciona.atoms.dl.detection.dsb_training_runtime import train_shared_runtime
from sciona.dsb_decomposed_runtime import build_dsb_runtime_graph


def fixture():
    shared,head=object(),object()
    detector=SimpleNamespace(parameters=lambda:iter([shared]))
    classifier=SimpleNamespace(NoduleNet=detector,parameters=lambda:iter([shared,head]))
    return dict(detector=detector,classifier=classifier,
        detector_optimizer=SimpleNamespace(param_groups=[{'params':[shared]}],state={'momentum':'retained'}),
        classifier_optimizer=SimpleNamespace(param_groups=[{'params':[shared,head]}],state={'momentum':'retained'}),start_epoch=49)


def test_loop_receives_original_runtime_factories_and_epoch_callback(monkeypatch):
    import sciona.dsb_training_range as loop
    runtime=fixture();training=SimpleNamespace(topk=2);validation=object();callback=object();metrics=[{'epoch':51}]
    def run(actual,t,v,**kwargs):
        assert actual is runtime and t is training and v is validation
        assert kwargs==dict(end_epoch=51,classifier_variant=3,freeze_batchnorm=False,prepare_epoch=callback)
        assert actual['detector_optimizer'].state=={'momentum':'retained'}
        assert actual['classifier_optimizer'].state=={'momentum':'retained'}
        return dict(checkpoint=b'synthetic-test-token',epochs=metrics)
    monkeypatch.setattr(loop,'run_training_range',run)
    checkpoint,returned=train_shared_runtime(runtime,training,validation,51,topk=2,prepare_epoch=callback)
    assert checkpoint==b'synthetic-test-token' and returned is metrics


@pytest.mark.parametrize('fault',['model_alias','detector_optimizer','classifier_optimizer','topk'])
def test_invalid_runtime_rejected_before_training(monkeypatch,fault):
    import sciona.dsb_training_range as loop
    def unexpected(*args,**kwargs):raise AssertionError('Training ran with invalid runtime')
    monkeypatch.setattr(loop,'run_training_range',unexpected)
    runtime=fixture();training=SimpleNamespace(topk=2)
    if fault=='model_alias':runtime['classifier'].NoduleNet=object()
    elif fault=='topk':training.topk=3
    else:runtime[fault].param_groups=[{'params':[object()]}]
    with pytest.raises(ValueError):train_shared_runtime(runtime,training,object(),51,topk=2)


def test_runtime_edge_preserves_single_restoration_boundary():
    from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
    graph=build_dsb_runtime_graph();digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph and len(nodes)==19
    by_id={n.node_id:n for n in graph.nodes}
    assert by_id['training_runtime'].matched_primitive.endswith('restore_shared_training_runtime')
    assert by_id['training'].matched_primitive.endswith('train_shared_runtime')
    assert 'start_epoch' not in {p.name for p in by_id['training'].inputs}
    inbound=[e for e in graph.edges if e.target_id=='training_runtime']
    assert [(e.source_id,e.input_name) for e in inbound]==[('initialize_classifier','initial_classifier_checkpoint')]
    assert any(e.source_id=='training_runtime' and e.target_id=='training' and e.input_name=='runtime' for e in graph.edges)
