"""Synthetic guards against dropping branches or conflating source identities."""
import json
import pytest
from scripts.audit_physics_projection_lineage import grouped,signature


def node(name,inputs,outputs,step='step',rule='combine',feeds=None):
    return dict(node_id=name,type_signature=json.dumps(dict(inference_rule_id=rule,inputs=inputs,output=outputs[0],outputs=outputs,
        variable_bindings=dict(derivation_id='synthetic',step_id=step,feeds=feeds or []))))


def test_complete_split_scope_matches_one_source_step():
    legacy=[node('a',['left'],['result']),node('b',['right'],['result'])]
    projected=[node('whole',['left','right'],['result'])]
    assert signature(grouped(legacy))==signature(grouped(projected))


def test_missing_input_or_alternative_root_does_not_match():
    complete=signature(grouped([node('whole',['a','b'],['lower','upper'])]))
    assert complete!=signature(grouped([node('missing-input',['a'],['lower','upper'])]))
    assert complete!=signature(grouped([node('missing-root',['a','b'],['lower'])]))


def test_changed_step_identity_does_not_match():
    assert signature(grouped([node('a',['x'],['y'])]))!=signature(grouped([node('b',['x'],['y'],step='different')]))


@pytest.mark.parametrize('change',[dict(rule='different'),dict(feeds=[{'value':'different'}])])
def test_inconsistent_split_rule_or_feed_is_rejected(change):
    with pytest.raises(ValueError,match='Inconsistent split-step'):
        grouped([node('a',['x'],['y']),node('b',['x'],['y'],**change)])
