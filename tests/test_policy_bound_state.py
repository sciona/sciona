import copy
import pytest
from sciona.policy_bound_state import bind_policy_state,unbind_policy_state


def test_owned_state_and_policy():
    model={'opaque':'synthetic'};policy={'columns':['load'],'fills':{'load':0}}
    state=bind_policy_state(model,policy)
    policy['columns'].append('mutated')
    restored,settings=unbind_policy_state(state)
    assert restored==model and settings['columns']==['load']
    settings['columns'].clear()
    assert unbind_policy_state(state)[1]['columns']==['load']


@pytest.mark.parametrize('field',['model_state','policy'])
def test_either_side_tampering_is_rejected(field):
    state=bind_policy_state({'x':1},{'x':2});bad=copy.deepcopy(state)
    bad['payload'][field]['x']=3
    with pytest.raises(ValueError,match='integrity'):unbind_policy_state(bad)


def test_nonfinite_policy_rejected():
    with pytest.raises(ValueError):bind_policy_state({}, {'fill':float('nan')})
