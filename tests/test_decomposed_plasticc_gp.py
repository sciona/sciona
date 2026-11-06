"""Synthetic GP prediction equivalence with explicit registered covariance."""
import numpy as np
import pytest
import george
from george import kernels
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from sciona.decomposed_plasticc_gp import scale_covariance, observation_variance, condition, predict


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('seed', [8, 39])
def test_generic_numeric_conditioning_matches_reference(width, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(17, width)); q = rng.normal(size=(11, width))
    y = rng.normal(size=len(x)); error = rng.uniform(.1, .4, len(x))
    lengths = np.arange(1, width+1, dtype=float); amplitude = 2.3
    original = [v.copy() for v in (x, q, y, error)]
    kernel = amplitude*kernels.Matern32Kernel(lengths**2, ndim=width)
    reference = george.GP(kernel); reference.compute(x, error)
    expected_mean, expected_var = reference.predict(y, q, return_var=True)
    covariance = scale_covariance(matern_kernel_matrix(x, length_scale=lengths), amplitude)
    cross = scale_covariance(matern_kernel_matrix(q, x, length_scale=lengths), amplitude)
    noise = observation_variance(error, np.exp(reference.white_noise.get_value(x)))
    factor, alpha, logdet = condition(covariance, y, noise)
    mean, variance = predict(factor, alpha, cross, np.full(len(q), amplitude))
    np.testing.assert_allclose(mean, expected_mean, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(variance, expected_var, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(logdet, reference.solver.log_determinant, atol=1e-11, rtol=1e-11)
    for value, saved in zip((x, q, y, error), original):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize('fix_scale', [False, True])
def test_retained_source_fit_with_registered_prediction_blocks(fix_scale):
    from sciona.plasticc_runtime import decode_objects
    from scripts.validate_plasticc_runtime import synthetic_payload
    from sciona.plasticc_observations import get_band_central_wavelength
    obj = decode_objects(synthetic_payload()['training'][:1], True)[0]
    fitted, obs, reported_parameters = obj.fit_gaussian_process(fix_scale=fix_scale)
    gp = fitted.func.__self__
    parameters = gp.kernel.get_parameter_dict(include_frozen=True)
    lengths = np.sqrt(np.exp([parameters['k2:metric:log_M_0_0'], parameters['k2:metric:log_M_1_1']]))
    amplitude = float(gp.kernel.get_value(np.zeros((1,2)), diag=True)[0])
    x = np.column_stack([obs.time, obs.band.map(get_band_central_wavelength)])
    q = x[::5].copy(); q[:,0] += 1.5
    covariance = scale_covariance(matern_kernel_matrix(x, length_scale=lengths), amplitude)
    cross = scale_covariance(matern_kernel_matrix(q, x, length_scale=lengths), amplitude)
    noise = observation_variance(obs.flux_error.to_numpy(), np.exp(gp.white_noise.get_value(x)))
    factor, alpha, _ = condition(covariance, obs.flux.to_numpy(), noise)
    mean, variance = predict(factor, alpha, cross, np.full(len(q), amplitude))
    expected_mean, expected_var = fitted(q, return_var=True)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-7, rtol=1e-8)
    np.testing.assert_allclose(variance, expected_var, atol=1e-7, rtol=1e-8)
    assert np.isfinite(reported_parameters).all()


@pytest.mark.parametrize('fault', ['asymmetry', 'targets', 'noise', 'singular', 'factor', 'cross', 'prior'])
def test_rejects_invalid_numeric_boundaries(fault):
    covariance = np.eye(3); targets = np.arange(3.); noise = np.zeros(3)
    if fault == 'asymmetry': covariance[0,1] = .1
    elif fault == 'targets': targets = targets[:2]
    elif fault == 'noise': noise[0] = -1
    elif fault == 'singular': covariance[:] = 0
    if fault in ('asymmetry', 'targets', 'noise', 'singular'):
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            condition(covariance, targets, noise)
        return
    factor, alpha, _ = condition(covariance, targets, noise)
    cross = np.ones((2,3)); prior = np.ones(2)
    if fault == 'factor': factor[2,0] = .1
    elif fault == 'cross': cross = cross[:,:2]
    else: prior[0] = -1
    with pytest.raises(ValueError): predict(factor, alpha, cross, prior)


def test_does_not_repair_inconsistent_posterior_variance():
    mean, variance = predict(np.eye(1), np.ones(1), np.array([[2.]]), np.ones(1))
    np.testing.assert_array_equal(mean, [2.])
    np.testing.assert_array_equal(variance, [-3.])


def test_explicit_noise_policy_no_hidden_floor():
    error = np.array([0., .1, 2.]); white = np.array([0., 1.25e-12, .4])
    np.testing.assert_array_equal(observation_variance(error, white), np.sqrt(error**2+white)**2)
    assert observation_variance(np.zeros(1), np.zeros(1))[0] == 0
    with pytest.raises(ValueError): observation_variance(error, -np.ones(3))
