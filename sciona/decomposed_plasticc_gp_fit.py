"""Numeric extraction of Avocado's bounded Matérn fitting and fallback policy.

Derived from kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Coordinates, scales, noise and optimization bounds are explicit; no astronomical
band lookup, object identifiers or table columns enter this numeric component.
"""
from dataclasses import dataclass
import george
from george import kernels
import numpy as np
from scipy.optimize import minimize
from .decomposed_plasticc_gp import _real


@dataclass(repr=False)
class Fitted:
    """Invocation-local mutable model; reported optimizer values may be inactive."""
    model: object
    coordinates: np.ndarray
    targets: np.ndarray
    errors: np.ndarray
    reported_parameters: np.ndarray
    effective_parameters: np.ndarray
    success: bool


def initial_amplitude(targets, errors, relative_floor, amplitude_fraction):
    """Estimate variance from the largest noise-adjusted absolute response.

    Preserve first-maximum selection and the source's maximum-response noise
    floor. Both dimensionless constants are explicit, independent of domain.
    """
    y = _real(targets); error = _real(errors)
    if y.ndim != 1 or not len(y) or error.shape != y.shape or np.any(error <= 0):
        raise ValueError('Nonempty aligned targets and positive errors required')
    controls = _real([relative_floor, amplitude_fraction])
    if controls.shape != (2,) or controls[0] < 0 or controls[1] <= 0:
        raise ValueError('Nonnegative relative floor and positive amplitude fraction required')
    scores = np.abs(y)/np.sqrt(error**2+(controls[0]*np.max(y))**2)
    amplitude = float((controls[1]*np.abs(y[np.argmax(scores)]))**2)
    if not np.isfinite(scores).all() or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Finite nonzero amplitude estimate required')
    return amplitude


def fit(coordinates, targets, errors, amplitude, lengths, optimized_axis,
        metric_log_bounds, amplitude_log_radius, fix_amplitude, white_noise_variance):
    """Fit one coordinate length and optionally amplitude with analytic gradients.

    Metric bounds apply to log(length**2). Amplitude bounds apply around george's
    initial log-constant parameter. Freeze every other coordinate metric. On
    unsuccessful optimization restore the full initial active parameter vector,
    but retain optimizer.x separately, including any nonfinite diagnostic value.
    No optimizer limit, restart, jitter repair or alternate algorithm is added.
    """
    x = _real(coordinates); y = _real(targets); error = _real(errors); scales = _real(lengths)
    bounds = _real(metric_log_bounds)
    if x.ndim != 2 or not all(x.shape) or y.shape != (len(x),) or error.shape != y.shape or np.any(error <= 0):
        raise ValueError('Finite aligned numeric coordinates, targets and positive errors required')
    if scales.shape != (x.shape[1],) or np.any(scales <= 0):
        raise ValueError('Positive coordinate-matched length scales required')
    if type(optimized_axis) is not int or not 0 <= optimized_axis < x.shape[1] or type(fix_amplitude) is not bool:
        raise ValueError('Valid optimized axis and boolean amplitude policy required')
    if bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError('Increasing finite log-metric bounds required')
    for value in (amplitude, amplitude_log_radius, white_noise_variance):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or _real(value) <= 0:
            raise ValueError('Positive finite amplitude, radius and white-noise variance required')
    metric = scales**2
    if not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError('Finite positive squared length scales required')
    kernel = float(amplitude)*kernels.Matern32Kernel(metric, ndim=x.shape[1])
    if fix_amplitude:
        kernel.freeze_parameter('k1:log_constant')
    for axis in range(x.shape[1]):
        if axis != optimized_axis:
            kernel.freeze_parameter(f'k2:metric:log_M_{axis}_{axis}')
    model = george.GP(kernel, white_noise=np.log(float(white_noise_variance)), fit_white_noise=False)
    guess = model.get_parameter_vector().copy()
    model.compute(x, error)

    def negative_likelihood(parameters):
        model.set_parameter_vector(parameters)
        return -model.log_likelihood(y)

    def negative_gradient(parameters):
        model.set_parameter_vector(parameters)
        return -model.grad_log_likelihood(y)

    optimizer_bounds = [tuple(bounds)]
    if not fix_amplitude:
        optimizer_bounds = [(guess[0]-amplitude_log_radius, guess[0]+amplitude_log_radius)] + optimizer_bounds
    outcome = minimize(negative_likelihood, model.get_parameter_vector(), jac=negative_gradient, bounds=optimizer_bounds)
    model.set_parameter_vector(outcome.x if outcome.success else guess)
    return Fitted(model, x, y, error, np.asarray(outcome.x).copy(),
                  model.get_parameter_vector().copy(), bool(outcome.success))
