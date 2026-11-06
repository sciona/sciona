"""Independent synthetic covariance oracle and registered boundary checks."""
import math
import numpy as np
import pytest
from icontract import ViolationError
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('cross', [False, True])
@pytest.mark.parametrize('anisotropic', [False, True])
def test_independent_pairwise_covariance(width, cross, anisotropic):
    rng = np.random.default_rng(28)
    x = rng.normal(size=(7, width)); x[1] = x[0]
    y = rng.normal(size=(3, width)) if cross else None
    scales = np.arange(1, width+1, dtype=float) if anisotropic else 2.5
    saved = x.copy(), None if y is None else y.copy()
    actual = matern_kernel_matrix(x, y, length_scale=scales, nu=1.5)
    target = x if y is None else y
    expanded = np.broadcast_to(scales, (width,))
    expected = np.empty((len(x), len(target)))
    for i in range(len(x)):
        for j in range(len(target)):
            radius = math.sqrt(3*math.fsum(((float(x[i,k])-float(target[j,k]))/float(expanded[k]))**2 for k in range(width)))
            expected[i,j] = (1+radius)*math.exp(-radius)
    np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=3e-15)
    np.testing.assert_array_equal(x, saved[0])
    if y is not None:
        np.testing.assert_array_equal(y, saved[1])
    else:
        np.testing.assert_array_equal(np.diag(actual), np.ones(len(x)))
        np.testing.assert_array_equal(actual, actual.T)
        assert np.linalg.eigvalsh(actual).min() >= -1e-12


def test_singleton_and_duplicate_coordinates():
    x = np.array([[3., -7.]])
    np.testing.assert_array_equal(matern_kernel_matrix(x, length_scale=np.array([2.])), [[1.]])
    np.testing.assert_array_equal(matern_kernel_matrix(x, np.repeat(x, 3, axis=0)), [[1., 1., 1.]])


@pytest.mark.parametrize('fault', ['rank', 'nonfinite_x', 'nonfinite_y', 'width', 'zero_scale', 'scale_width', 'smoothness'])
def test_invalid_registered_boundary(fault):
    x = np.ones((3, 2)); y = np.zeros((4, 2)); scale = 1.; nu = 1.5
    if fault == 'rank': x = x.ravel()
    elif fault == 'nonfinite_x': x[0,0] = np.nan
    elif fault == 'nonfinite_y': y[0,0] = np.inf
    elif fault == 'width': y = y[:,:1]
    elif fault == 'zero_scale': scale = 0.
    elif fault == 'scale_width': scale = np.array([1., 2., 3.])
    else: nu = 0.
    with pytest.raises(ViolationError):
        matern_kernel_matrix(x, y, length_scale=scale, nu=nu)
