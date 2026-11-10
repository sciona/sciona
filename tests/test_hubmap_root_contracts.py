"""Guard catalog boundaries using synthetic contracts on the full lifecycle graph."""
import pytest
from sciona.catalog_hubmap_graph import validate_root_contracts
from sciona.hubmap_execution import build_hubmap_lifecycle_graph


@pytest.fixture
def boundary():
    graph=build_hubmap_lifecycle_graph()
    for node in graph.nodes:
        for port in node.inputs+node.outputs:
            port.constraints='Synthetic contract for '+port.name
    graph.metadata['catalog_output_sources']={
        'initial_metrics':dict(node_id='initial_training',output_name='initial_metrics'),
        'retraining_metrics':dict(node_id='retraining',output_name='retraining_metrics'),
        'predictions':dict(node_id='final_prediction',output_name='predictions')}
    names=['slides','groups','initial_stages','initial_selectors','tile_size','input_resolution',
           'pseudo_sources','pseudo_rng','inference_geometry','retraining_stages','final_selectors',
           'final_images','final_rng']
    inputs={p.name:p for n in graph.nodes for p in n.inputs}
    outputs={p.name:p for n in graph.nodes for p in n.outputs}
    ports=[]
    for direction,ordered,available in [('input',names,inputs),
            ('output',['initial_metrics','retraining_metrics','predictions'],outputs)]:
        for ordinal,name in enumerate(ordered):
            port=available[name]
            ports.append(dict(direction=direction,ordinal=ordinal,
                **{k:getattr(port,k) for k in ('name','type_desc','constraints','required','default_value_repr')}))
    return graph,ports


def test_complete_shared_inputs_and_three_outputs(boundary):
    validate_root_contracts(*boundary)


@pytest.mark.parametrize('field,value', [('constraints','incorrect'),('required',True),
    ('default_value_repr','512'),('type_desc','str'),('ordinal',99)])
def test_reject_changed_root_contract(boundary,field,value):
    graph,ports=boundary
    next(p for p in ports if p['name']=='tile_size')[field]=value
    with pytest.raises(ValueError,match='Root contracts differ'):
        validate_root_contracts(graph,ports)


def test_shared_input_cannot_disagree_between_training_stages(boundary):
    graph,ports=boundary
    next(p for p in graph.nodes[3].inputs if p.name=='slides').constraints='Different meaning'
    with pytest.raises(ValueError,match='Shared root input contracts disagree'):
        validate_root_contracts(graph,ports)


def test_metrics_output_cannot_be_omitted(boundary):
    graph,ports=boundary
    del graph.metadata['catalog_output_sources']['initial_metrics']
    with pytest.raises(ValueError,match='Every leaf output'):
        validate_root_contracts(graph,ports)


def test_internal_checkpoint_cannot_replace_root_output(boundary):
    graph,ports=boundary
    graph.metadata['catalog_output_sources']['initial_metrics']['output_name']='initial_checkpoints'
    with pytest.raises(ValueError,match='Root outputs must select each leaf'):
        validate_root_contracts(graph,ports)


def test_duplicate_upstream_binding_rejected(boundary):
    graph,ports=boundary
    graph.edges.append(graph.edges[0].model_copy())
    with pytest.raises(ValueError,match='One upstream binding'):
        validate_root_contracts(graph,ports)


def test_dangling_edge_rejected(boundary):
    graph,ports=boundary
    graph.edges[0].output_name='missing'
    with pytest.raises(ValueError,match='Valid edge endpoints'):
        validate_root_contracts(graph,ports)
