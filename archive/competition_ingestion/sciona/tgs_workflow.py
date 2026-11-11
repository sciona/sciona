"""Full63-fit TGS orchestration; publication remains a separate verified gate."""
import hashlib
import json

import numpy as np
import torch

from sciona.tgs_fit_execution import run_fit
from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_torch_models import TGSResNet34
from sciona.tgs_round_inference import predict_round
from sciona.tgs_round_outputs import complete_round
from sciona.tgs_schedule import next_action


def run_workflow(plan, populations, references, store, *, seed, record, state=None):
    """Execute unchanged fit controls and record each completed action.

    `record(state)` must persist into private runtime storage and raise on
    failure. State includes synthetic or private query arrays and must never
    be embedded in source-controlled reports. A returned state proves runtime
    completion only; provenance, qualification and publication gates remain.

    Callers provide aligned Keras BGR and PyTorch grayscale populations, their
    explicit fold assignments and the pseudo-only labeled validation split.
    """
    if type(seed) is not int or not 0 <= seed < 2**32 or not callable(record):
        raise ValueError('explicit uint32 seed and persistence callback required')
    if state is None:
        state = dict(fits={}, rounds={}, seed=seed)
    if state['seed'] != seed:
        raise ValueError('workflow seed differs from resumed state')
    fits_by_key = {fit['key']: fit for fit in plan['fits']}
    for key, result in state['fits'].items():
        if key not in fits_by_key:
            raise ValueError('unknown resumed fit')
        digest = hashlib.sha256(json.dumps(fits_by_key[key], sort_keys=True).encode()).hexdigest()
        if result.get('fit') != key or result.get('fit_contract_sha256') != digest:
            raise ValueError('resumed fit contract differs from reviewed plan')
    while True:
        action = next_action(plan, state['fits'], state['rounds'])
        if action['kind'] == 'complete':
            return state
        if action['kind'] == 'fit':
            key = action['key']
            fit = fits_by_key[key]
            fit_seed = int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:4], 'little')
            torch.manual_seed(fit_seed)
            result = run_fit(key, plan, populations[fit['branch']], state['fits'], state['rounds'],
                             references, store, rng=np.random.default_rng(fit_seed))
            state['fits'][key] = result
        else:
            stage = action['stage']
            keras_population, torch_population = populations['keras'], populations['pytorch']
            keras_model = TGSResNeXt50()
            torch_model = TGSResNet34({1: 4, 2: 3, 3: 5}[stage])
            predictions = predict_round(stage, state['fits'], store, keras_model, torch_model,
                keras_population['query_images'], torch_population['query_images'],
                ~np.asarray(torch_population['query_nonconstant']),
                keras_batch_size=24 if stage == 1 else 28,
                torch_batch_size=36 if stage == 3 else 18)
            del keras_model, torch_model
            state['rounds'][stage] = complete_round(stage, predictions['scores'], torch_population['query_images'],
                training_images=torch_population['labeled_images'], training_masks=torch_population['labeled_masks'])
        # Stop on failed persistence rather than allowing downstream actions to
        # depend on results that the caller could not durably record.
        record(state)
