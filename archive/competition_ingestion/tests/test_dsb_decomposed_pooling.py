import torch
from sciona.dsb_network import CaseNet as OriginalCaseNet
from sciona.dsb_decomposed_network import CaseNet,shared_training_models
from sciona.dsb_checkpoint import checkpoint_bytes,restore_checkpoint
from sciona.dsb_decomposed_pooling_graph import build_dsb_pooling_graph


def test_initialization_checkpoint_keys_and_rng_are_unchanged():
    torch.set_num_threads(1)
    torch.manual_seed(71);expected=OriginalCaseNet(2);expected_rng=torch.get_rng_state()
    torch.manual_seed(71);actual=CaseNet(2);actual_rng=torch.get_rng_state()
    assert actual.state_dict().keys()==expected.state_dict().keys()
    for name,value in actual.state_dict().items():
        torch.testing.assert_close(value,expected.state_dict()[name],rtol=0,atol=0)
    torch.testing.assert_close(actual_rng,expected_rng,rtol=0,atol=0)
    payload=checkpoint_bytes(expected,4)
    with torch.no_grad():actual.baseline.fill_(3.)
    assert restore_checkpoint(actual,payload)==5
    for name,value in actual.state_dict().items():
        torch.testing.assert_close(value,expected.state_dict()[name],rtol=0,atol=0)


def test_shared_model_feature_parameters_retain_identity():
    detector,classifier=shared_training_models(2)
    assert classifier.NoduleNet is detector
    shared={id(p) for p in detector.parameters()}
    assert shared <= {id(p) for p in classifier.parameters()}
    assert id(classifier.baseline) not in shared


def test_graph_restores_both_classifier_paths_with_generic_pooling():
    from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
    graph=build_dsb_pooling_graph();digest,nodes,edges=encode_execution_graph(graph)
    assert len(nodes)==19 and decode_execution_graph(nodes,edges,digest)==graph
    by_id={n.node_id:n for n in graph.nodes}
    for name in ['training_runtime','models']:
        assert '.dsb_pooling_runtime.' in by_id[name].matched_primitive
    assert graph.metadata['publication_status']=='draft'
