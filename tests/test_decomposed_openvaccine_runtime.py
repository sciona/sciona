"""Synthetic lifecycle preservation and dense-feature preflight tests."""
import ast
import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts.openvaccine_synthetic import payload
from sciona import decomposed_openvaccine_runtime as runtime
from sciona import openvaccine_lifecycle as original
from sciona.openvaccine_contract import _arguments, prepare
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.decomposed_openvaccine_ensemble import blend_inputs


def dense_fixture():
    raw = payload()
    args = _arguments(raw)
    rng = np.random.default_rng(981)
    for key, prefix in [('sequences', ''), ('pseudo_sequences', 'pseudo_'),
                        ('prediction_sequences', 'prediction_')]:
        sequences = args.pop(key)
        batch, length = len(sequences), len(sequences[0])+2
        args[prefix+'nodes'] = rng.normal(size=(batch, length, 55)).astype(np.float32)
        args[prefix+'adjacency'] = rng.normal(size=(batch, length, length, 8)).astype(np.float32)
    return dict(args, seed=raw['seed'])


def test_complete_training_body_preserved():
    reference = ast.parse(inspect.getsource(original.execute_population)).body[0]
    candidate = ast.parse(inspect.getsource(runtime.fit_forecast)).body[0]
    def training_body(function):
        start = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'batches')
        return ast.Module(body=function.body[start:-1], type_ignores=[])
    expected = training_body(reference)
    class RenameFinalCollector(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'predict_teacher':
                node.id = 'collect_checkpoints'
            return node
    expected = RenameFinalCollector().visit(expected)
    assert ast.dump(training_body(candidate)) == ast.dump(expected)


def test_dense_features_require_no_raw_sequence_fields():
    featured = dense_fixture()
    checked, arrays, targets, weights, membership = runtime.checked_features(featured)
    assert not any('sequence' in name for name in checked)
    assert set(membership) == EXPECTED_MEMBERS
    assert len(arrays) == 6 and targets.shape[-1] == 5 and weights.shape == (4,)
    checked['nodes'][:] = 0
    assert np.any(featured['nodes'])


@pytest.mark.parametrize('failure', ['extra', 'seed', 'steps', 'empty_rounds', 'later_round',
    'target_length', 'feature_dtype', 'adjacency', 'nan', 'ddof', 'rollback', 'weights'])
def test_all_preflight_before_tensorflow_or_model_work(failure):
    featured = dense_fixture()
    if failure == 'extra':
        featured['sequences'] = []
    elif failure == 'seed':
        featured['seed'] = True
    elif failure == 'steps':
        featured['initial_supervised_steps'] = 0
    elif failure == 'empty_rounds':
        featured['pseudo_rounds'] = []
    elif failure == 'later_round':
        featured['pseudo_rounds'].append({})
    elif failure == 'target_length':
        featured['targets'] = featured['targets'][:, :-1]
    elif failure == 'feature_dtype':
        featured['nodes'] = featured['nodes'].astype(np.float64)
    elif failure == 'adjacency':
        featured['adjacency'] = featured['adjacency'][:, :-1]
    elif failure == 'nan':
        featured['pseudo_nodes'][0, 0, 0] = np.nan
    elif failure == 'ddof':
        featured['std_ddof'] = True
    elif failure == 'rollback':
        featured['rollback_policy'] = 'ignore'
    elif failure == 'weights':
        featured['proximity_factors'][:] = 0
    with patch.dict(sys.modules, {'tensorflow': None}), patch.object(runtime, 'create_model') as create:
        with pytest.raises(ValueError):
            runtime.fit_forecast(featured)
        create.assert_not_called()


def test_configured_multistep_multiround_lifecycle_and_cleanup(monkeypatch):
    featured = dense_fixture()
    featured['pretraining_steps'] = 3
    featured['initial_supervised_steps'] = 2
    featured['pseudo_rounds'] = [copy.deepcopy(featured['pseudo_rounds'][0]) for _ in range(3)]
    seed = Mock()
    fake_tf = SimpleNamespace(keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=seed)))
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)
    monkeypatch.setenv('SCIONA_OPENVACCINE_SOURCE_DIR', 'synthetic-source')
    created, saved, refined, roots = [], [], [], []
    def create(source, family, **kwargs):
        model = SimpleNamespace(family=family, base=SimpleNamespace(get_weights=lambda: ['base']), train_step=Mock())
        assert source == 'synthetic-source'
        if family != 'autoencoder':
            assert kwargs == dict(base_weights=['base'])
        created.append(model)
        return model
    def save(model, directory):
        directory.mkdir(parents=True)
        saved.append((model.family, directory))
        return 'a'*64
    def refine(source, records, pn, pa, **kwargs):
        assert len(records) == 20
        assert set(kwargs['recipes']) == EXPECTED_MEMBERS
        assert kwargs['rollback_policy'] == featured['rollback_policy']
        assert kwargs['absolute_tolerance'] == featured['absolute_tolerance']
        assert kwargs['std_ddof'] == featured['std_ddof']
        np.testing.assert_array_equal(pn, featured['pseudo_nodes'])
        np.testing.assert_array_equal(pa, featured['pseudo_adjacency'])
        assert all(len(v['supervised_batches']) == 1 for v in kwargs['recipes'].values())
        refined.append(kwargs['output_directory'].name)
        return records, [dict(supervised_steps=1, pseudo_label_section=dict(steps=1, rolled_back=(i % 2 == 0)))
                         for i in range(20)]
    def collect(source, records, qn, qa, **kwargs):
        assert all(record.directory.is_dir() for record in records)
        roots.append(records[0].directory.parent.parent)
        np.testing.assert_array_equal(qn, featured['prediction_nodes'])
        np.testing.assert_array_equal(qa, featured['prediction_adjacency'])
        shape = (len(qn), qn.shape[1]-2, 5)
        return dict(clipped_members=np.zeros((20,)+shape, np.float64),
                    teacher_mean=np.zeros(shape, np.float32), teacher_std=np.ones(shape, np.float32))
    with patch.object(runtime, 'create_model', side_effect=create), \
         patch.object(runtime, 'save_checkpoint', side_effect=save), \
         patch.object(runtime, 'refine_round', side_effect=refine), \
         patch.object(runtime, 'collect_checkpoints', side_effect=collect):
        forecast = runtime.fit_forecast(featured)
    seed.assert_called_once_with(featured['seed'])
    assert len(created) == 21 and created[0].train_step.call_count == 3
    assert all(model.train_step.call_count == 2 for model in created[1:])
    assert len(saved) == 20 and refined == ['round0', 'round1', 'round2']
    assert all(not path.exists() for path in roots)
    assert forecast['pretraining_steps'] == 3 and forecast['initial_supervised_steps'] == 40
    assert forecast['rounds'] == [dict(members=20, supervised_steps=20, pseudo_label_steps=20, rollbacks=10)]*3
    rows, coefficients = blend_inputs(forecast)
    output = runtime.result(forecast, ordered_weighted_sum(rows, coefficients))
    assert output['rounds'] == forecast['rounds']
    assert np.asarray(output['predictions']).shape == (2, 14, 5)
    output['rounds'][0]['members'] = 0
    assert forecast['rounds'][0]['members'] == 20


def test_raw_folding_three_populations_in_original_order(monkeypatch):
    raw = payload()
    featured = dense_fixture()
    monkeypatch.setenv('SCIONA_OPENVACCINE_SOURCE_DIR', 'synthetic-source')
    monkeypatch.setenv('SCIONA_OPENVACCINE_FOLD_BINARY', 'synthetic-binary')
    monkeypatch.setenv('SCIONA_OPENVACCINE_FOLD_PARAMETERS', 'synthetic-parameters')
    arrays = [(featured[p+'nodes'], featured[p+'adjacency']) for p in ('', 'pseudo_', 'prediction_')]
    prepared = prepare(raw)
    with patch.object(runtime, 'fold_features', side_effect=arrays) as fold:
        actual = runtime.features(prepared)
    assert prepared.payload == raw
    assert [call.args[1] for call in fold.call_args_list] == [raw[k] for k in
        ('sequences', 'pseudo_sequences', 'prediction_sequences')]
    assert all(call.kwargs == dict(binary='synthetic-binary', parameters='synthetic-parameters')
               for call in fold.call_args_list)
    assert set(actual) == set(featured)
