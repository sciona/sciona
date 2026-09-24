"""Synthetic edge guards for legacy/source projection comparison."""
import json
import pytest
from scripts.audit_legacy_physics_execution_scope import scope


def node(name,step,inputs,output):
    return dict(node_id=name,type_signature=json.dumps(dict(inputs=inputs,output=output,inference_rule_id='synthetic-rule',
        variable_bindings=dict(derivation_id='synthetic-derivation',step_id=step,feeds=[]))))


def fixture():
    nodes=[node('a','first',['premise'],'shared'),node('b','second',['shared'],'terminal')]
    return nodes,[dict(source_id='a',target_id='b')]


def test_complete_dependency_relation():
    nodes,edges=fixture()
    groups,pairs,relations=scope(nodes,edges)
    assert len(groups)==2 and len(relations)==1
    assert pairs[('synthetic-derivation','first')]==[('premise','shared')]


@pytest.mark.parametrize('edges',[[],[dict(source_id='b',target_id='a')],
    [dict(source_id='a',target_id='b'),dict(source_id='b',target_id='a')]])
def test_missing_reversed_or_extra_dependency_rejected(edges):
    with pytest.raises(ValueError,match='dependency edges differ'):
        scope(fixture()[0],edges)


def test_unknown_endpoint_rejected():
    with pytest.raises(ValueError,match='Unknown edge endpoint'):
        scope(fixture()[0],[dict(source_id='absent',target_id='b')])
