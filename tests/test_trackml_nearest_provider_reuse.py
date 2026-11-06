"""Independent synthetic geometric invariants, without tracking-domain policy."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_nearest_point_distance as project


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('scale', [.01, 1., 100.])
def test_projection_geometry_and_unit_scaling(seed, scale):
    rng = np.random.RandomState(seed)
    n = 24
    center = rng.normal(size=(3, n))
    radius = rng.uniform(.5, 2, n)
    pitch = rng.uniform(30, 60, n) * rng.choice([-1., 1.], n)
    phase = rng.uniform(-np.pi, np.pi, n)
    params = [center[0] + radius * np.cos(phase), center[1] + radius * np.sin(phase),
              center[2], center[0], center[1], radius, pitch]
    target = list(center + rng.normal(size=(3, n)))
    inputs = params + target
    saved = [a.copy() for a in inputs]
    base = project(*inputs, iterations=6)
    actual = project(*[a * scale for a in inputs], iterations=6)
    for a, b in zip(actual, base, strict=True):
        np.testing.assert_allclose(a, b * scale, rtol=1e-11, atol=1e-11 * scale)
    x, y, z, distance = base
    np.testing.assert_allclose((x-center[0])**2 + (y-center[1])**2, radius**2, atol=1e-12)
    phase_at_projection = phase + (z-center[2]) * (2*np.pi) / pitch
    np.testing.assert_allclose(x, center[0]+radius*np.cos(phase_at_projection), atol=1e-12)
    np.testing.assert_allclose(y, center[1]+radius*np.sin(phase_at_projection), atol=1e-12)
    residual = np.stack(target) - np.stack((x, y, z))
    np.testing.assert_allclose(distance, np.linalg.norm(residual, axis=0), atol=1e-12)
    # This well-conditioned subset converges to a stationary distance; this is
    # not asserted for arbitrary eccentricity or the default iteration count.
    tangent = np.stack((-(y-center[1]), x-center[0], pitch/(2*np.pi)))
    np.testing.assert_allclose(np.sum(residual*tangent, axis=0), 0, atol=1e-11)
    for a, b in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(a, b)


def test_zero_radius_reduces_to_axial_line_projection():
    targets = [np.array([3., -2.]), np.array([4., 5.]), np.array([-7., 9.])]
    x, y, z, distance = project(0., 0., 1., 0., 0., 0., 12., *targets)
    np.testing.assert_allclose(x, 0, atol=1e-12)
    np.testing.assert_allclose(y, 0, atol=1e-12)
    np.testing.assert_allclose(z, targets[2], atol=1e-12)
    np.testing.assert_allclose(distance, np.hypot(targets[0], targets[1]), atol=1e-12)
