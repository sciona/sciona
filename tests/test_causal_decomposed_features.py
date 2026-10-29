import numpy as np
import pytest
from sciona.atoms.causal_inference.feature_primitives.selected_features import FEATURE_ORDER,extract_causal_selected_features
from sciona.atoms.causal_inference.feature_primitives.pair_orientation import orient_typed_pairs
from sciona.atoms.causal_inference.feature_primitives.batched_statistics import typed_pair_statistic
from sciona.atoms.causal_inference.feature_primitives.feature_recipe_ports import feature_statistic_configuration,assemble_selected_columns


def test_all_43_columns_match_existing_extractor_across_all_type_pairs():
    from scripts.validate_causal_execution import synthetic_pairs
    for seed in [731,732,733]:
        pairs,types=synthetic_pairs(np.random.default_rng(seed),9)
        xs,xt,ys,yt=orient_typed_pairs(pairs,types)
        columns={}
        for name,statistic in zip(FEATURE_ORDER,feature_statistic_configuration()):
            args=(ys,yt,xs,xt) if name.endswith(('_ba','_b')) else (xs,xt,ys,yt)
            columns[name]=typed_pair_statistic(*args,statistic)
        actual=assemble_selected_columns(**columns)
        np.testing.assert_array_equal(actual,extract_causal_selected_features(pairs,types))


@pytest.mark.parametrize('change',[
    {'statistic':'unknown'},{'x_types':['Unknown']},{'ys':[np.arange(4.)]},
    {'xs':[np.ones(3)]},{'xs':[np.array([1.,np.nan,2.])]},{'x_types':['Binary']},
    {'ys':[]},
])
def test_statistic_rejects_invalid_inputs(change):
    kwargs=dict(xs=[np.arange(3.)],x_types=['Numerical'],ys=[np.array([2.,4.,3.])],y_types=['Numerical'],statistic='moment21')
    kwargs.update(change)
    with pytest.raises(ValueError):typed_pair_statistic(**kwargs)


@pytest.mark.asyncio
async def test_full_feature_graph_executes_all_nodes_and_preserves_scores(tmp_path,monkeypatch):
    import runpy
    from pathlib import Path
    import sciona.causal_decomposed_training as training
    from sciona.causal_decomposed_features import build_explicit_causal_feature_graph
    graph=build_explicit_causal_feature_graph()
    assert len(graph.nodes)==114
    assert not any(n.node_id=='features' for n in graph.nodes)
    assert len([n for n in graph.nodes if n.matched_primitive.endswith('.typed_pair_statistic')])==86
    monkeypatch.setattr(training,'build_explicit_causal_training_graph',lambda:graph)
    oracle=runpy.run_path(str(Path(__file__).with_name('test_causal_decomposed_training.py')))
    await oracle['check_full_default_five_fit_graph'](tmp_path,monkeypatch,expected_nodes=114)
