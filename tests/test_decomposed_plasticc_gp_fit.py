"""Retained numerical fitting parity, fallback distinction and generic axes."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from sciona import decomposed_plasticc_gp_fit as numeric
from sciona import plasticc_observations as retained


def source_case():
    from sciona.plasticc_runtime import decode_objects
    from scripts.validate_plasticc_runtime import synthetic_payload
    obj = decode_objects(synthetic_payload()['training'][:1], True)[0]
    # Nonconsecutive labels ensure numeric first-maximum selection preserves
    # source idxmax semantics without treating a table label as an array index.
    obj.observations.index = np.arange(len(obj.observations))*3+5
    obs = obj.preprocess_observations()
    x = np.column_stack([obs.time, obs.band.map(retained.get_band_central_wavelength)])
    y = obs.flux.to_numpy(); error = obs.flux_error.to_numpy()
    return obj, x, y, error


def invoke(x, y, error, fixed):
    amplitude = numeric.initial_amplitude(y, error, .01, .5)
    return numeric.fit(x, y, error, amplitude, np.array([20., 6000.]), 0,
                       np.array([0., np.log(1000**2)]), 10., fixed, 1.25e-12)


@pytest.mark.parametrize('fixed', [False, True])
def test_retained_optimizer_and_predictions(fixed):
    obj, x, y, error = source_case()
    reference, _, reported = obj.fit_gaussian_process(fix_scale=fixed)
    actual = invoke(x, y, error, fixed)
    np.testing.assert_allclose(actual.reported_parameters, reported, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual.effective_parameters, reference.func.__self__.get_parameter_vector(), atol=1e-10, rtol=1e-10)
    q = x[::4].copy(); q[:,0] += 3
    expected = reference(q, return_var=True)
    result = actual.model.predict(actual.targets, q, return_var=True)
    for a, b in zip(result, expected): np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('diagnostic_nan', [False, True])
def test_failed_optimizer_preserves_reported_and_effective_distinction(fixed, diagnostic_nan):
    obj, x, y, error = source_case(); guesses = []
    def failure(fun, start, jac, bounds):
        guesses.append(start.copy())
        # Exercise both callbacks so fallback must undo optimizer mutations.
        fun(start+.1); jac(start+.1)
        diagnostic = np.full_like(start, np.nan) if diagnostic_nan else start+.25
        return SimpleNamespace(success=False, x=diagnostic)
    with patch.object(retained, 'minimize', side_effect=failure):
        reference, _, reported = obj.fit_gaussian_process(fix_scale=fixed)
    with patch.object(numeric, 'minimize', side_effect=failure):
        actual = invoke(x, y, error, fixed)
    assert actual.success is False
    np.testing.assert_array_equal(actual.reported_parameters, reported)
    np.testing.assert_array_equal(actual.effective_parameters, guesses[1])
    np.testing.assert_array_equal(guesses[0], guesses[1])
    expected = reference(x[::6], return_var=True)
    result = actual.model.predict(actual.targets, x[::6], return_var=True)
    for a, b in zip(result, expected): np.testing.assert_array_equal(a, b)


def test_generic_three_coordinate_fit_and_input_preservation():
    rng = np.random.default_rng(5); x = rng.normal(size=(18,3)); y = np.sin(x[:,2]); errors = np.full(18,.2)
    original = [a.copy() for a in (x,y,errors)]
    fitted = numeric.fit(x,y,errors,2.,np.array([1.,2.,3.]),2,np.array([-2.,4.]),5.,False,1e-10)
    assert fitted.effective_parameters.shape == (2,)
    parameters = fitted.model.kernel.get_parameter_dict(include_frozen=True)
    assert parameters['k2:metric:log_M_0_0'] == 0.
    assert parameters['k2:metric:log_M_1_1'] == np.log(4.)
    for a,b in zip((x,y,errors),original): np.testing.assert_array_equal(a,b)
    fitted.coordinates[0,0] = 999
    assert x[0,0] != 999


def test_invalid_controls_reject_before_optimizer():
    with patch.object(numeric,'minimize',side_effect=AssertionError('Must not optimize')):
        with pytest.raises(ValueError):
            numeric.fit(np.ones((3,2)),np.ones(3),np.ones(3),1.,np.ones(2),2,np.array([0.,1.]),10.,False,1e-12)
    with pytest.raises(ValueError): numeric.initial_amplitude(np.zeros(3),np.ones(3),.01,.5)
