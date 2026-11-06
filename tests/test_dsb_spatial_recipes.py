"""Recipe adapters require concrete arrays and caller-owned random state."""
import inspect
import pytest
from sciona.atoms.dl.detection import dsb_spatial_recipes as recipes
from sciona.ghost.registry import REGISTRY


@pytest.mark.parametrize('name,required', [
    ('crop_detection', ('volume','target','boxes')),
    ('augment_detection', ('sample','target','boxes','coord')),
    ('augment_classifier', ('sample','coord')),
    ('crop_classifier_proposal', ('volume','target')),
    ('assign_anchor_labels', ('input_size','target','boxes')),
])
@pytest.mark.parametrize('rng_arguments', [{}, {'rng': None}])
def test_registered_recipe_rejects_missing_rng_before_processing(name, required, rng_arguments):
    function=getattr(recipes,name)
    assert 'sciona.atoms.dl.detection.dsb_spatial_recipes.'+name in REGISTRY
    assert 'rng' in inspect.signature(function).parameters
    with pytest.raises(ValueError,match='Explicit caller-owned RNG required'):
        function(**{key:None for key in required}, **rng_arguments)


def test_recipe_witness_rejects_unsupported_abstract_execution():
    with pytest.raises(ValueError,match='Concrete arrays'):
        recipes.concrete_only()
