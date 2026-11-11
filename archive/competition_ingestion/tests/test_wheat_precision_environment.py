import pytest

from scripts.check_wheat_precision_environment import assess_environment


def environment():
    return dict(cuda_available=True, cuda_build='10.1', amp_initialize=True,
                amp_scale_loss=True, python='3.7.6', torch='1.4.0', cudnn=7501)


@pytest.mark.parametrize('key', ['cuda_available', 'cuda_build', 'amp_initialize', 'amp_scale_loss'])
def test_missing_capability_prevents_ready_state(key):
    observed = environment()
    observed.pop(key)
    result = assess_environment(observed)
    assert not result['runtime_ready']
    assert result['missing_requirements']


def test_capability_does_not_imply_historical_equivalence_or_approval():
    observed = environment()
    observed.update(torch='2.11.0', cuda_build='12.8')
    result = assess_environment(observed)
    assert result['runtime_ready']
    assert not result['historical_core_versions_match']
    assert not result['historical_apex_revision_qualified']
    assert not result['numerical_precision_qualified']
    assert not result['approved']


def test_historical_core_versions_still_require_precision_qualification():
    result = assess_environment(environment())
    assert result['historical_core_versions_match']
    assert not result['numerical_precision_qualified']
