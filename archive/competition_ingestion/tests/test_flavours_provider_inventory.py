"""Dependency closure must see wrapped calls and avoid attribute-name aliases."""
from functools import wraps
from scripts.inventory_flavours_decomposed_providers import registered_call_dependencies


def _callee():return 1
_callee.__module__='sciona.atoms.synthetic'


def _attribute_only(state):return state._callee


def _calls_helper():return _callee()
_calls_helper.__module__='sciona.atoms.synthetic'


def _root():return _calls_helper()


def test_attribute_name_is_not_a_global_function_dependency():
    assert registered_call_dependencies(_attribute_only,{_callee:'synthetic.callee'})==set()


def test_wrapped_and_private_helper_calls_are_followed():
    @wraps(_root)
    def wrapper():return _root()
    assert registered_call_dependencies(wrapper,{_callee:'synthetic.callee'})=={'synthetic.callee'}
