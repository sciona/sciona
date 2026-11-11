import pytest
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from scripts.inventory_ensemble_adapters import interfaces


def node():
    return AlgorithmicNode(node_id='synthetic', name='synthetic', description='Synthetic contract',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive='synthetic.adapter',
        inputs=[IOSpec(name=name, type_desc='list') for name in ['a', 'b']],
        outputs=[IOSpec(name=name, type_desc='list') for name in ['first', 'second']])


def test_adapter_inventory_rejects_swapped_unannotated_family_return():
    def swapped(a: list, b: list) -> tuple:
        return b, a
    with pytest.raises(ValueError, match='unchanged input references in order'):
        interfaces(swapped, node())


def test_adapter_inventory_rejects_output_type_mismatch():
    def wrong_type(a: list, b: list) -> tuple[list, int]:
        return a, len(b)
    with pytest.raises(ValueError, match='output arity/types'):
        interfaces(wrong_type, node())
