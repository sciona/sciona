"""Run the numerical and lifecycle suite through exact registered providers."""
import pytest
from sciona.atoms.ml.sklearn.decomposition import sparse_projection as provider
from sciona.ghost.registry import REGISTRY, get_witness
from tests import test_sparse_randomized_projection as baseline
from tests.test_sparse_randomized_projection import (
    test_factors_and_projection,
    test_invalid_controls,
    test_invalid_values_and_transform,
    test_retained_lifecycle_with_new_projection,
    test_unsorted_duplicate_storage_preserved,
)


@pytest.fixture(autouse=True)
def registered_calls(monkeypatch):
    monkeypatch.setattr(baseline,'fit_projection',provider.fit_sparse_projection)
    monkeypatch.setattr(baseline,'transform_projection',provider.transform_sparse_projection)


@pytest.mark.parametrize('function',[provider.fit_sparse_projection,provider.transform_sparse_projection])
def test_exact_registration_and_abstract_rejection(function):
    name=function.__module__+'.'+function.__name__
    assert REGISTRY[name]['impl'] is function
    with pytest.raises(ValueError,match='Concrete sparse matrix'):
        get_witness(name)()
