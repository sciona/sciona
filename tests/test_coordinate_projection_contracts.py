"""Additional public-domain boundary cases for reusable coordinate projections."""
import numpy as np
import pytest
from sciona.atoms.physics.coordinate_projection import cylindrical_projection, plane_projection


@pytest.mark.parametrize('operation', [cylindrical_projection, plane_projection])
@pytest.mark.parametrize('points', [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.int64),
                                   np.ones(3), np.ones((2, 4)), np.full((2, 3), np.nan)])
def test_invalid_coordinate_contract(operation, points):
    with pytest.raises(ValueError):
        operation(points, 2.)


@pytest.mark.parametrize('radius', [0., -1.])
def test_reference_radius_must_be_positive(radius):
    with pytest.raises(ValueError):
        cylindrical_projection(np.ones((1, 3)), radius)


@pytest.mark.parametrize('weight', [-2., 0., 1.])
def test_transverse_weight_is_explicit_policy(weight):
    points = np.array([[3., 4., 10.], [-3., 4., -10.]])
    actual = cylindrical_projection(points, 5., weight)
    expected = points.copy()
    expected[:, :2] *= weight
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('height', [-2., 0., 2.])
def test_plane_projection_supports_both_line_directions_and_zero_height(height):
    points = np.array([[3., 4., 2.], [3., 4., -2.]])
    expected = np.array([[1.5*height, 2.*height], [-1.5*height, -2.*height]])
    np.testing.assert_array_equal(plane_projection(points, height), expected)
