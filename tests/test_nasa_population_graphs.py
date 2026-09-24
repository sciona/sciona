import pytest

from sciona.nasa_population_graphs import build_population_graph,select_population_graph
from sciona.nasa_lifecycle_graphs import build_nasa_training_graph,build_nasa_inference_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer.runner import get_topo_sorted_leaves
from sciona.atoms.ml.pipeline.keyed_routing import pop_first,singleton,insert_unique,require_exhausted
from sciona.atoms.ml.domain_adapters.airport_population import training_jobs
from tests.test_nasa_population_dispatch import records


@pytest.mark.parametrize('kind,builder',[('training',build_nasa_training_graph),('inference',build_nasa_inference_graph)])
@pytest.mark.parametrize('count',[1,2,10])
def test_each_compiled_branch_preserves_lifecycle_nodes_and_internal_edges(kind,builder,count):
    base=builder();graph=build_population_graph(kind,count)
    for index in range(count):
        prefix=f'p{index:02d}_'
        for original in base.nodes:
            actual=next(node for node in graph.nodes if node.node_id==prefix+original.node_id).model_copy(deep=True)
            actual.node_id=original.node_id
            assert actual==original
        for original in base.edges:
            assert any(edge.source_id==prefix+original.source_id and edge.target_id==prefix+original.target_id
                and edge.output_name==original.output_name and edge.input_name==original.input_name for edge in graph.edges)
    digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph
    assert len(get_topo_sorted_leaves(graph.nodes,graph.edges))==len(graph.nodes)
    connected={(edge.target_id,edge.input_name) for edge in graph.edges}
    roots={port.name for node in graph.nodes for port in node.inputs if (node.node_id,port.name) not in connected}
    assert roots==({'training_records','airports','maximum_errors','train_fraction','seed','threshold','prediction_feature_name'}
                   if kind=='training' else {'prediction_records','model_states'})
    if kind=='inference':assert not any(node.matched_primitive.endswith(('.fit_regressor','.fit_classifier')) for node in graph.nodes)


@pytest.mark.parametrize('source',[{'line-b':{'duration':3.0},'line-a':{'duration':7.0}},
                                  {'solar':{'energy':12.5},'wind':{'energy':2.1}}])
def test_keyed_routing_preserves_values_order_and_input_ownership_across_domains(source):
    original=dict(source)
    key,value,remaining=pop_first(source)
    result=singleton(key,value)
    key,value,empty=pop_first(remaining)
    completed=insert_unique(result,key,value)
    assert require_exhausted(empty,completed)==source
    assert list(completed)==list(source) and source==original
    assert len(result)==1 and len(remaining)==1
    with pytest.raises(ValueError):pop_first(empty)
    with pytest.raises(ValueError):require_exhausted(remaining,completed)
    with pytest.raises(ValueError):insert_unique(completed,key,value)


@pytest.mark.parametrize('cutoffs',[{}, {'KAAA':2}, {'KAAA':2,'KBBB':float('nan')},
                                  {'KAAA':2,'KBBB':-1},{'KAAA':2,'KBBB':True}])
def test_training_requires_complete_finite_explicit_policy(cutoffs):
    with pytest.raises(ValueError):training_jobs(records(),('KAAA','KBBB'),cutoffs)


def test_training_cannot_silently_skip_a_declared_population():
    with pytest.raises(ValueError):
        training_jobs(records(),('KAAA','KBBB','KZZZ'),{'KAAA':2,'KBBB':3,'KZZZ':4})


def test_training_selection_uses_declared_populations_without_query_or_model_inputs():
    graph=select_population_graph('training',dict(training_records=records(),
        airports=('KBBB','KAAA'),maximum_errors={'KAAA':20,'KBBB':30}))
    assert graph.metadata['active_population_count']==2
    assert graph==build_population_graph('training',2)


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_compiler_rejects_invalid_counts(count):
    with pytest.raises(ValueError):build_population_graph('inference',count)
