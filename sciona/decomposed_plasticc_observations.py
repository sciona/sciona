"""Explicit astronomy policy over supplied reusable numeric GP operations.

Retains Avocado's wavelength mapping, default controls and per-object fit cache.
Derived from kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
This adapter is not the complete decomposed CDG or a publication approval.
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import cho_solve
from .plasticc_observations import AstronomicalObject, get_band_central_wavelength
from .plasticc_population import Dataset, PopulationAugmentor
from .plasticc_runtime import prepare_runtime
from .plasticc_defaults import DEFAULTS

OPERATIONS = {'amplitude', 'fit', 'kernel', 'scale', 'noise', 'condition', 'predict'}


def checked_operations(operations):
    if type(operations) is not dict or set(operations) != OPERATIONS or not all(callable(f) for f in operations.values()):
        raise ValueError('Complete explicit numeric GP callables required')
    return dict(operations)


class Predictor:
    """Invocation-local numeric prediction state, conditioned once per fit.

    Qualified bindings must authenticate the supplied operations. State and
    callable semantics must remain unchanged while cached by an observation.
    """
    def __init__(self, fitted, operations):
        self.operations = checked_operations(operations)
        self.coordinates = fitted.coordinates.copy()
        parameters = fitted.model.kernel.get_parameter_dict(include_frozen=True)
        width = self.coordinates.shape[1]
        self.lengths = np.sqrt(np.exp([parameters[f'k2:metric:log_M_{i}_{i}'] for i in range(width)]))
        self.amplitude = float(fitted.model.kernel.get_value(np.zeros((1,width)), diag=True)[0])
        self.reported_parameters = fitted.reported_parameters.copy()
        covariance = self.covariance(self.coordinates)
        white = np.exp(fitted.model.white_noise.get_value(self.coordinates))
        noise = self.operations['noise'](fitted.errors, white)
        self.factor, self.alpha, self.logdet = self.operations['condition'](covariance, fitted.targets, noise)

    def covariance(self, coordinates, other=None):
        normalized = self.operations['kernel'](coordinates, other, length_scale=self.lengths, nu=1.5)
        return self.operations['scale'](normalized, self.amplitude)

    def __call__(self, coordinates, return_cov=True, return_var=False):
        cross = self.covariance(coordinates, self.coordinates)
        mean, variance = self.operations['predict'](self.factor, self.alpha, cross, np.full(len(cross), self.amplitude))
        if return_var:
            return mean, variance
        if return_cov:
            prior = self.covariance(coordinates)
            return mean, prior-np.dot(cross, cho_solve((self.factor, False), cross.T))
        return mean


class NumericObject(AstronomicalObject):
    def __init__(self, metadata, observations, operations):
        super().__init__(metadata, observations.copy(deep=True))
        self.operations = checked_operations(operations)

    def fit_gaussian_process(self, fix_scale=False, verbose=False, guess_length_scale=20.0, **preprocessing_kwargs):
        observations = self.preprocess_observations(**preprocessing_kwargs)
        targets = observations.flux.to_numpy(); errors = observations.flux_error.to_numpy()
        coordinates = np.column_stack([observations.time, observations.band.map(get_band_central_wavelength)])
        amplitude = self.operations['amplitude'](targets, errors, .01, .5)
        fitted = self.operations['fit'](coordinates, targets, errors, amplitude,
            np.array([guess_length_scale,6000.]), 0, np.array([0.,np.log(1000**2)]), 10., fix_scale, 1.25e-12)
        # The inherited default accessor caches this predictor by object. An
        # explicit fit call still fits afresh, as in the retained implementation.
        return Predictor(fitted, self.operations), observations, fitted.reported_parameters.copy()


class NumericAugmentor(PopulationAugmentor):
    def __init__(self, reference, operations, *, augment_retries, seed):
        super().__init__(reference, augment_retries=augment_retries, seed=seed)
        self.operations = checked_operations(operations)

    def augment_object(self, reference_object, force_success=True):
        # Preserve every source RNG draw and rejection/retry. Conversion after
        # augmentation makes feature fitting use the same explicit operations.
        augmented = super().augment_object(reference_object, force_success=force_success)
        if augmented is None:
            return None
        return NumericObject(augmented.metadata, augmented.observations, self.operations)


def prepare(payload, operations):
    """Retain root validation, then bind numeric operations for both populations."""
    operations = checked_operations(operations)
    prepared = prepare_runtime(payload)
    def population(source):
        objects = [NumericObject(o.metadata, o.observations, operations) for o in source.objects]
        return Dataset.from_objects(source.name, objects)
    augmentor = NumericAugmentor(payload['photoz_reference'], operations,
        augment_retries=DEFAULTS['augment_retries'], seed=payload['config'].get('augmentation_seed',0))
    return replace(prepared, training=population(prepared.training), prediction=population(prepared.prediction), augmentor=augmentor)
