"""Preserve callable ordering, tuple outputs and requiredness in catalog intake."""
import importlib
import pytest
from sciona.decomposed_aptos_graph import build_decomposed_aptos_graph
from scripts.stage_aptos_decomposed_providers import interfaces


@pytest.mark.parametrize('node_id',[n.node_id for n in build_decomposed_aptos_graph().nodes])
def test_every_registered_interface_roundtrips(node_id):
    node=next(n for n in build_decomposed_aptos_graph().nodes if n.node_id==node_id)
    module,name=node.matched_primitive.rsplit('.',1)
    ports=interfaces(getattr(importlib.import_module(module),name),[node])
    assert [(p['name'],p['type_desc']) for p in ports if p['direction']=='input']==[(p.name,p.type_desc) for p in node.inputs]
    assert [(p['name'],p['type_desc']) for p in ports if p['direction']=='output']==[(p.name,p.type_desc) for p in node.outputs]
    assert [p['constraints'] for p in ports]==[p.constraints for p in node.inputs+node.outputs]
    assert all(p['required'] and p['default_value_repr']=='' for p in ports)


@pytest.mark.parametrize('fault',['input_order','input_type','optional','output_missing','output_type'])
def test_incompatible_tuple_provider_declaration_rejected(fault):
    node=next(n for n in build_decomposed_aptos_graph().nodes if n.node_id=='cut_inputs')
    from sciona.atoms.ml.decomposed_aptos_execution import cut_inputs
    if fault=='input_order':node.inputs[0].name='wrong_input'
    elif fault=='input_type':node.inputs[0].type_desc='float'
    elif fault=='optional':node.inputs[0].required=False
    elif fault=='output_missing':node.outputs.pop()
    else:node.outputs[1].type_desc='dict'
    with pytest.raises(ValueError):interfaces(cut_inputs,[node])
