from types import SimpleNamespace
import numpy as np
import pytest
from scripts.review_causal_provider_interfaces import interfaces


def function(values:np.ndarray,weights:np.ndarray|None=None,enabled:bool=True)->np.ndarray:
    return values


def node(inputs=None,output_type='numpy.ndarray'):
    return SimpleNamespace(inputs=inputs if inputs is not None else [SimpleNamespace(name='values',type_desc='numpy.ndarray',required=True)],
        outputs=[SimpleNamespace(name='result',type_desc=output_type)])


def test_optional_omission_keeps_native_defaults_and_nullability():
    ports=interfaces(function,[node()])
    weight=next(p for p in ports if p['name']=='weights')
    assert weight['nullable_annotation'] and not weight['required'] and weight['default_value_repr']=='None'
    enabled=next(p for p in ports if p['name']=='enabled')
    assert enabled['default_value_repr']=='True'


def test_required_omission_is_rejected():
    with pytest.raises(ValueError,match='input mismatch'):interfaces(function,[node(inputs=[])])


def test_optional_native_parameter_can_be_explicit_required_graph_input():
    n=node()
    n.inputs.append(SimpleNamespace(name='weights',type_desc='numpy.ndarray',required=True))
    assert interfaces(function,[n])


def test_output_type_mismatch_is_rejected():
    with pytest.raises(ValueError,match='Output annotation'):interfaces(function,[node(output_type='list')])
