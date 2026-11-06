"""Synthetic pinned-source parity and independent cylinder geometry contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection as intersect
from scripts.review_trackml_intersection_reuse import source_classes


def inputs(n=32, seed=0, scale=1., direction=1.):
    rng = np.random.RandomState(seed)
    phase = rng.uniform(-np.pi, np.pi, n)
    xm = np.full(n, scale)
    ym = np.full(n, .5*scale)
    radius = np.full(n, 2.*scale)
    return [xm+radius*np.cos(phase), ym+radius*np.sin(phase), np.zeros(n),
            xm, ym, radius, np.full(n, 3.*scale), np.full(n, direction),
            np.full(n, 4.*scale**2), np.full(n, .1)]


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('scale', [.01, 1., 100.])
@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('pitch_sign', [-1., 1.])
@pytest.mark.parametrize('minimum_phase', [-.1, .1])
def test_pinned_source_and_geometric_invariants(seed, scale, direction, pitch_sign, minimum_phase):
    values = inputs(seed=seed, scale=scale, direction=direction)
    values[6] *= pitch_sign
    values[9][:] = minimum_phase
    saved = [v.copy() for v in values]
    x, y, z, xm, ym, radius, pitch, uz, target, tolerance = values
    classes, _ = source_classes()
    reference = classes['CylinderIntersector'](SimpleNamespace(cyl_rsqr=np.array([target[0]])))
    expected = reference.intersectHelices(x, y, z, uz, xm, ym, radius, pitch,
                                         pre_move=tolerance*radius, missable=False)
    actual = intersect(*values)
    for a, b in zip(actual[:4], (expected[0], expected[1], expected[2], expected[4]), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[4], expected[3] >= 0)
    hit = actual[4]
    np.testing.assert_allclose((actual[0]**2+actual[1]**2)[hit], target[hit], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[2][hit], (z+actual[3]*(pitch/(2*np.pi)))[hit])
    assert np.all((actual[3]*np.sign(uz*pitch))[hit] >= tolerance[hit])
    for a, b in zip(values, saved, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('mode', ['unreachable', 'exhausted'])
def test_explicit_miss(mode):
    values = inputs()
    values[8 if mode == 'unreachable' else 9][:] = 100.
    result = intersect(*values)
    assert not result[4].any()
    assert all(np.isnan(v).all() for v in result[:4])


def test_empty_batch():
    result = intersect(*inputs(n=0))
    assert all(v.shape == (0,) for v in result)
    assert result[4].dtype == np.bool_


@pytest.mark.parametrize('index,value', [(5, 0.), (6, 0.), (7, 0.), (8, -1.), (0, np.inf), (9, np.nan)])
def test_invalid_contract(index, value):
    args = inputs()
    args[index][:] = value
    with pytest.raises(ValueError):
        intersect(*args)


def test_concentric_geometry_rejects():
    args = inputs()
    args[3][:] = args[4][:] = 0.
    with pytest.raises(ValueError):
        intersect(*args)


@pytest.mark.parametrize('value', [np.ones(32, dtype=np.float32), np.ones((32, 1)), np.ones(31)])
def test_invalid_array(value):
    args = inputs()
    args[0] = value
    with pytest.raises(ValueError):
        intersect(*args)
