"""Graph boundaries must retain every learner branch and explicit combination."""
import importlib
import inspect
import numpy as np
import pytest
from sciona.decomposed_amex_graph import build_decomposed_amex_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.decomposed_amex_pipeline import combination_inputs,result


def test_serialized_graph_preserves_registered_interfaces_and_root_boundary():
    graph=build_decomposed_amex_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph
    assert len(nodes)==10 and len(edges)==16
    by_id={n.node_id:n for n in graph.nodes}
    consumed={(e.target_id,e.input_name) for e in graph.edges}
    assert [(n.node_id,p.name) for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in consumed]==[('prepare','payload')]
    for node in graph.nodes:
        module,name=node.matched_primitive.rsplit('.',1)
        fn=getattr(importlib.import_module(module),name)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
    for edge in graph.edges:
        source=next(p for p in by_id[edge.source_id].outputs if p.name==edge.output_name)
        target=next(p for p in by_id[edge.target_id].inputs if p.name==edge.input_name)
        assert source.type_desc==target.type_desc==edge.source_type==edge.target_type
    assert {(e.source_id,e.target_id) for e in graph.edges if e.target_id=='combination_inputs'}=={
        ('tree_models','combination_inputs'),('neural_models','combination_inputs')}
    assert graph.metadata['catalog_output_sources']=={'result':{'node_id':'result','output_name':'result'}}


def branches(values):
    return (dict(variants={name:dict(query_predictions=p) for name,p in zip(('manual','manual_with_slots'),values[:2])}),
        dict(variants={name:dict(query_predictions=p) for name,p in zip(('series','combined'),values[2:])}))


def test_recipe_supplies_exact_order_and_weights_without_aliasing():
    original=np.arange(12,dtype=np.float64).reshape(4,3)/12
    values,weights=combination_inputs(*branches(original))
    np.testing.assert_array_equal(values,original)
    np.testing.assert_array_equal(weights,[.3,.35,.15,.1])
    assert not np.shares_memory(values,original)


@pytest.mark.parametrize('bad',[np.array([]),np.ones((1,2)),np.array([np.nan,0.]),
    np.array([np.inf,0.]),np.array([-1.,0.]),np.array([1.1,0.]),np.ones(3),np.array(['a','b'])])
def test_recipe_probability_boundary_rejects_bad_branch(bad):
    inputs=[np.array([.2,.8]) for _ in range(4)];inputs[2]=bad
    with pytest.raises(ValueError):combination_inputs(*branches(inputs))


@pytest.mark.parametrize('bad',[[.2],np.array([]),np.ones((1,1)),np.array([1],dtype=np.int32),
    np.array([np.nan]),np.array([np.inf]),np.array([-.1]),np.array([1.1])])
def test_encoded_boundary_rejects_bad_scores(bad):
    with pytest.raises(ValueError):result(bad)
