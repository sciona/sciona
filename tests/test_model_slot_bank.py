import pytest

from sciona.model_slot_bank import assemble_model_slot_bank, select_model_slots


def test_local_aliases_shared_state_and_owned_containers():
    residual, direct, shared = object(), object(), object()
    models = dict(residual=residual, direct=direct, shared=shared)
    bindings = {'plant-A': {'base': 'residual', 'correction': 'direct', 'alias': 'residual'},
                'plant-B': {'base': 'direct'}}
    bank = assemble_model_slot_bank(models, bindings, {'global': 'shared'})
    assert bank['plant-A']['base'] is bank['plant-A']['alias'] is residual
    assert bank['plant-A']['global'] is bank['plant-B']['global'] is shared
    selected = select_model_slots(bank, 'plant-A')
    selected.clear()
    bindings['plant-A'].clear()
    models.clear()
    assert len(bank['plant-A']) == 4
    assert len(bank['plant-B']) == 2


@pytest.mark.parametrize('models,local,shared', [
    ({'m': object()}, {'p': {'slot': 'missing'}}, {}),
    ({'m': object()}, {'p': {'slot': 'm'}}, {'slot': 'm'}),
    ({'m': object(), 'unused': object()}, {'p': {'slot': 'm'}}, {}),
    ({'m': None}, {'p': {'slot': 'm'}}, {}),
    ({'m': object()}, {'p': {}}, {}),
    ({'m': object()}, {'': {'slot': 'm'}}, {}),
])
def test_incomplete_or_conflicting_bindings_rejected(models, local, shared):
    with pytest.raises(ValueError):
        assemble_model_slot_bank(models, local, shared)


def test_empty_and_unknown_population_are_explicit():
    assert assemble_model_slot_bank({}, {}, {}) == {}
    with pytest.raises(ValueError):
        assemble_model_slot_bank({}, {}, {'unused': 'missing'})
    with pytest.raises(KeyError):
        select_model_slots({}, 'missing')
    with pytest.raises(ValueError):
        select_model_slots({'p': None}, 'p')
