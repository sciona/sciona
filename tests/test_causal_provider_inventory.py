import inspect
from sciona.ghost.registry import REGISTRY
from scripts.inventory_causal_decomposed_providers import provider_closure,registered_dependencies


def test_dynamic_statistic_table_and_module_attributes_are_included():
    from sciona.atoms.causal_inference.feature_primitives.batched_statistics import PAIR_FUNCTIONS,typed_pair_statistic
    _,_,closure=provider_closure()
    registered={inspect.unwrap(e['impl']):name for name,e in REGISTRY.items()}
    deps=registered_dependencies(typed_pair_statistic,registered)
    assert {registered[inspect.unwrap(f)] for f in PAIR_FUNCTIONS.values()}<=deps
    for name in ['discretize_and_bin','knn_entropy_estimator','uniform_divergence','hsic_independence_test','empirical_entropy','mixed_power_moment']:
        assert any(d.endswith('.'+name) for d in deps)
    assert all(d in closure for d in deps)
    assert not any('selected_features' in d for d in deps)


def test_recipe_assembly_tracks_generic_stack_and_unregistered_partition_helper():
    _,roots,closure=provider_closure()
    assemble=next(r for r in roots if r.endswith('.assemble_selected_columns'))
    assert any(d.endswith('.stack_feature_columns') for d in closure[assemble])
    partitions=next(r for r in roots if r.endswith('.prepare_causal_training_partitions'))
    # The source helper partitions arrays but must not pull in the five-fit wrapper.
    assert not any(d.endswith('.train_causal_classifiers') for d in closure[partitions])
    assert all(r in closure for r in roots)
