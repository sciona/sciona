"""Synthetic parity and boundary failures for the explicit ensemble reduction."""
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona import decomposed_openvaccine_ensemble as extracted
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS, predict_twenty
from sciona.openvaccine_teacher import MemberCheckpoint


class Runtime:
    def __init__(self, family, forward, reverse):
        self.family = family
        self.outputs = (forward.copy(), reverse.copy())
        self.calls = []

    def predict(self, nodes, adjacency):
        self.calls.append((nodes.copy(), adjacency.copy()))
        return self.outputs[len(self.calls)-1].copy()


def fixture(batch=2, length=7, order='sorted'):
    rng = np.random.default_rng(7823)
    nodes = rng.normal(size=(batch, length+2, 55)).astype(np.float32)
    adjacency = rng.normal(size=(batch, length+2, length+2, 8)).astype(np.float32)
    keys = sorted(EXPECTED_MEMBERS)
    if order == 'reverse':
        keys.reverse()
    if order == 'permuted':
        keys = [keys[i] for i in rng.permutation(20)]
    members = [(family, slot, Runtime(family,
        rng.uniform(-20, 20, (batch, length, 5)).astype(np.float32),
        rng.uniform(-20, 20, (batch, length, 5)).astype(np.float32))) for family, slot in keys]
    return members, nodes, adjacency


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('length', [2, 7, 14])
@pytest.mark.parametrize('order', ['sorted', 'reverse', 'permuted'])
@pytest.mark.parametrize('ddof', [0, 1])
def test_exact_prediction_and_teacher_parity(batch, length, order, ddof):
    members, nodes, adjacency = fixture(batch, length, order)
    original_members = copy.deepcopy(members)
    before_nodes, before_adj = nodes.copy(), adjacency.copy()
    expected = predict_twenty(original_members, nodes, adjacency, std_ddof=ddof)
    collected = extracted.collect_members(members, nodes, adjacency, std_ddof=ddof)
    before = copy.deepcopy(collected)
    rows, coefficients = extracted.blend_inputs(collected)
    actual = extracted.finish_prediction(collected, ordered_weighted_sum(rows, coefficients))
    np.testing.assert_array_equal(actual, expected['prediction'], strict=True)
    assert actual.tobytes() == expected['prediction'].tobytes()
    for name in ('teacher_mean', 'teacher_std'):
        np.testing.assert_array_equal(collected[name], expected[name], strict=True)
        assert collected[name].tobytes() == expected[name].tobytes()
    for key in before:
        np.testing.assert_array_equal(collected[key], before[key], strict=True)
    np.testing.assert_array_equal(nodes, before_nodes, strict=True)
    np.testing.assert_array_equal(adjacency, before_adj, strict=True)
    for (_, _, runtime), (_, _, original) in zip(members, original_members):
        assert len(runtime.calls) == 2
        for pair, original_pair in zip(runtime.calls, original.calls):
            for value, reference in zip(pair, original_pair):
                np.testing.assert_array_equal(value, reference, strict=True)
    rows[:] = 0
    np.testing.assert_array_equal(collected['clipped_members'], before['clipped_members'])


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'family', 'bool_slot', 'nan', 'shape'])
def test_reject_invalid_members(failure):
    members, nodes, adjacency = fixture()
    if failure == 'missing':
        members.pop()
    elif failure == 'duplicate':
        members[-1] = copy.deepcopy(members[0])
    elif failure == 'family':
        members[0][2].family = 'unknown'
    elif failure == 'bool_slot':
        family, _, runtime = members[0]
        members[0] = family, False, runtime
    elif failure == 'nan':
        members[0][2].outputs[0][0, 0, 0] = np.nan
    elif failure == 'shape':
        members[0][2].outputs = tuple(v[..., :4] for v in members[0][2].outputs)
    with pytest.raises(ValueError):
        extracted.collect_members(members, nodes, adjacency, std_ddof=0)


@pytest.mark.parametrize('ddof', [True, -1, 2, 0., '0'])
def test_ddof_rejected_before_prediction(ddof):
    members, nodes, adjacency = fixture()
    with pytest.raises(ValueError):
        extracted.collect_members(members, nodes, adjacency, std_ddof=ddof)
    assert all(not runtime.calls for _, _, runtime in members)


@pytest.mark.parametrize('failure', ['dtype', 'count', 'clip', 'nan', 'mean_shape', 'std_negative'])
def test_collected_boundary_failures(failure):
    members, nodes, adjacency = fixture()
    collected = extracted.collect_members(members, nodes, adjacency, std_ddof=0)
    if failure == 'dtype':
        collected['clipped_members'] = collected['clipped_members'].astype(np.float32)
    elif failure == 'count':
        collected['clipped_members'] = collected['clipped_members'][:-1]
    elif failure == 'clip':
        collected['clipped_members'][0, 0, 0, 0] = 6.1
    elif failure == 'nan':
        collected['clipped_members'][0, 0, 0, 0] = np.nan
    elif failure == 'mean_shape':
        collected['teacher_mean'] = collected['teacher_mean'][:, :-1]
    else:
        collected['teacher_std'][0, 0, 0] = -1
    with pytest.raises(ValueError):
        extracted.blend_inputs(collected)


@pytest.mark.parametrize('failure', ['dtype', 'shape', 'nan', 'low', 'high'])
def test_external_total_boundary(failure):
    members, nodes, adjacency = fixture()
    collected = extracted.collect_members(members, nodes, adjacency, std_ddof=0)
    rows, coefficients = extracted.blend_inputs(collected)
    total = ordered_weighted_sum(rows, coefficients)
    if failure == 'dtype':
        total = total.astype(np.float32)
    elif failure == 'shape':
        total = total[:-1]
    else:
        total[0] = {'nan': np.nan, 'low': -10.1, 'high': 120.1}[failure]
    with pytest.raises(ValueError):
        extracted.finish_prediction(collected, total)


def test_checkpoint_identity_and_load_order():
    members, nodes, adjacency = fixture(order='reverse')
    records = [MemberCheckpoint(f, s, Path('synthetic')/str(i), format(i, '064x'))
               for i, (f, s, _) in enumerate(members)]
    with patch.object(extracted, 'load_checkpoint', side_effect=[m[2] for m in members]) as load:
        extracted.collect_checkpoints('synthetic-source', records, nodes, adjacency, std_ddof=1)
    assert len(load.call_args_list) == 20
    for call, record in zip(load.call_args_list, records):
        assert call.args == ('synthetic-source', record.directory)
        assert call.kwargs == dict(expected_manifest_sha256=record.manifest_sha256,
                                   expected_family=record.family)
    with patch.object(extracted, 'load_checkpoint') as load:
        with pytest.raises(ValueError):
            extracted.collect_checkpoints('synthetic-source', records[:-1]+records[:1],
                                          nodes, adjacency, std_ddof=1)
        load.assert_not_called()


def test_negative_zero_preserves_reference_accumulator():
    members, nodes, adjacency = fixture()
    for _, _, runtime in members:
        for output in runtime.outputs:
            output[:] = -0.
    expected = predict_twenty(copy.deepcopy(members), nodes, adjacency, std_ddof=0)
    collected = extracted.collect_members(members, nodes, adjacency, std_ddof=0)
    rows, coefficients = extracted.blend_inputs(collected)
    actual = extracted.finish_prediction(collected, ordered_weighted_sum(rows, coefficients))
    assert actual.tobytes() == expected['prediction'].tobytes()
