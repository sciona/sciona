from scripts.inventory_andriy_provider_closure import registered_dependencies
from scripts.inventory_andriy_provider_closure import provider_interfaces
import numpy as np
import pytest
from sciona.architect.models import IOSpec


def test_dependency_inventory_follows_comprehension_and_helper_cycle():
    namespace = {'__name__': 'sciona.atoms.synthetic_closure'}
    exec('''
def atom(value):
    return value
def helper(value):
    if value:
        return atom(value)
    return helper(value)
def root(values):
    return [helper(value) for value in values]
''', namespace)
    assert registered_dependencies(namespace['root'], {namespace['atom']: 'synthetic.atom'}) == {'synthetic.atom'}


def test_dependency_inventory_stops_at_registered_boundary():
    namespace = {'__name__': 'sciona.atoms.synthetic_closure'}
    exec('''
def leaf(value):
    return value
def intermediate(value):
    return leaf(value)
def root(value):
    return intermediate(value)
''', namespace)
    registered = {namespace['leaf']: 'synthetic.leaf', namespace['intermediate']: 'synthetic.intermediate'}
    assert registered_dependencies(namespace['root'], registered) == {'synthetic.intermediate'}
    assert registered_dependencies(namespace['intermediate'], registered) == {'synthetic.leaf'}


def test_interface_inventory_preserves_tuple_order_and_defaults():
    def provider(samples: np.ndarray, limit: int = 7) -> tuple[np.ndarray, list]:
        raise AssertionError('Inventory must not execute the provider')
    outputs = [IOSpec(name='scores', type_desc='numpy.ndarray'), IOSpec(name='groups', type_desc='list')]
    ports = provider_interfaces(provider, outputs)
    assert [(p['name'], p['ordinal']) for p in ports if p['direction'] == 'output'] == [('scores', 0), ('groups', 1)]
    assert ports[1]['required'] is False and ports[1]['default_value_repr'] == '7'
    with pytest.raises(ValueError, match='Explicit tuple output names'):
        provider_interfaces(provider)
    with pytest.raises(ValueError, match='Graph output type differs'):
        provider_interfaces(provider, list(reversed(outputs)))


def test_interface_inventory_rejects_untyped_input():
    def provider(samples) -> np.ndarray:
        raise AssertionError('Inventory must not execute the provider')
    with pytest.raises(ValueError, match='Unsupported provider port annotation'):
        provider_interfaces(provider)
