"""Single reviewed TGS fit: dependencies, populations, training and handoff.

The full workflow owns the reviewed plan and ordered branch representations.
This runner executes the plan's controls unchanged; it does not approve a CDG.
"""
import hashlib
import json
import numpy as np

from sciona.tgs_checkpoint_store import persist_phase_result
from sciona.tgs_initialization import load_resnet34_reference, load_resnext50_reference
from sciona.tgs_keras_phase import train_phase as train_keras
from sciona.tgs_torch_phase import train_phase as train_torch
from sciona.tgs_population import select_fit_population
from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_torch_models import TGSResNet34


def predecessor_receipt(fit, completed):
    predecessor = fit['weight_predecessor']
    if predecessor is None:
        return None
    if predecessor not in completed:
        raise ValueError('missing predecessor fit: ' + predecessor)
    previous = completed[predecessor]
    if fit['branch'] == 'keras':
        if 'best_checkpoint' not in previous:
            raise ValueError('predecessor validation-best checkpoint required')
        return previous['best_checkpoint']
    if predecessor != 'torch.p2.f0':
        raise ValueError('unexpected PyTorch weight predecessor')
    cycles = previous.get('cycle_checkpoints', {})
    if {str(k) for k in cycles} != {'50', '100', '150'} or len(cycles) != 3:
        raise ValueError('complete pseudo-only cycle-end checkpoints required')
    return cycles[150] if 150 in cycles else cycles['150']


def fit_arrays(fit, population, pseudo):
    """Convert binary pseudo masks to the source branch's pre-resize format."""
    labeled = np.asarray(population['labeled_images'])
    masks = np.asarray(population['labeled_masks'])
    query = np.asarray(population['query_images'])
    keras = fit['branch'] == 'keras'
    expected = (101, 101, 3) if keras else (101, 101)
    maximum = 255 if keras else 1
    for images in (labeled, query):
        if (images.ndim != len(expected) + 1 or not len(images) or images.shape[1:] != expected
                or not np.isfinite(images).all() or (images < 0).any() or (images > maximum).any()):
            raise ValueError('invalid branch image representation')
    if masks.shape != (len(labeled), 101, 101) or not np.isin(masks, [0, 1]).all():
        raise ValueError('aligned binary labeled masks required')
    if len(population['folds']) != len(labeled) or len(population['query_nonconstant']) != len(query):
        raise ValueError('population metadata must align with images')
    options = {}
    if fit['pseudo_round'] is not None:
        if not isinstance(pseudo, dict) or pseudo.get('stage') != fit['pseudo_round']:
            raise ValueError('required completed pseudo-label round missing')
        pseudo_masks = np.asarray(pseudo['masks'])
        if pseudo_masks.shape != (len(query), 101, 101) or not np.isin(pseudo_masks, [0, 1]).all():
            raise ValueError('aligned binary pseudo masks required')
        options = dict(confidence=pseudo['confidence'], area=pseudo['area'])
    indices = select_fit_population(fit['key'], population['folds'], population['labeled_nonconstant'],
                                    population['query_nonconstant'],
                                    pseudo_validation=population.get('pseudo_validation'), **options)
    ti, qi, vi = (indices[k] for k in ('training_labeled', 'training_query', 'validation_labeled'))
    train_images, train_masks = labeled[ti], masks[ti]
    if len(qi):
        train_images = np.concatenate((train_images, query[qi]))
        train_masks = np.concatenate((train_masks, pseudo_masks[qi]))
    dtype, scale = (np.uint8, 255) if keras else (np.float32, 1)
    arrays = (train_images, train_masks.astype(dtype) * scale,
              labeled[vi], masks[vi].astype(dtype) * scale)
    counts = dict(training_labeled=len(ti), training_query=len(qi), validation_labeled=len(vi))
    return arrays, counts


def run_fit(key, plan, population, completed, pseudo_rounds, references, store, *, rng):
    """Return a persisted phase result; caller records completion atomically.

    No control overrides, automatic missing-dependency fallback, or shortened
    schedule. Early stopping follows the reviewed Keras callback policy.
    """
    entries = [entry for entry in plan['fits'] if entry['key'] == key]
    if len(entries) != 1 or key in completed:
        raise ValueError('unique uncompleted fit required')
    fit = entries[0]
    receipt = predecessor_receipt(fit, completed)
    pseudo = pseudo_rounds.get(fit['pseudo_round'])
    arrays, counts = fit_arrays(fit, population, pseudo)
    keras = fit['branch'] == 'keras'
    if keras and len(arrays[0]) < int(fit['controls']['batch_size']):
        raise ValueError('source Keras schedule requires a complete training batch')
    model = TGSResNeXt50() if keras else TGSResNet34({'res34v4': 4, 'res34v3': 3, 'res34v5': 5}[fit['controls']['model']])
    if receipt is not None:
        model.load_state_dict(store.get(receipt), strict=True)
        initialization = dict(predecessor=fit['weight_predecessor'], checkpoint=receipt)
    else:
        reference = references[fit['initial_reference']]
        loader = load_resnext50_reference if keras else load_resnet34_reference
        initialization = loader(model, reference['path'], reference['sha256'])
    result = (train_keras if keras else train_torch)(model, *arrays, controls=fit['controls'], rng=rng)
    persisted = persist_phase_result(store, result, branch=fit['branch'])
    persisted.update(fit=key, fit_contract_sha256=hashlib.sha256(json.dumps(fit, sort_keys=True).encode()).hexdigest(),
                     population_counts=counts, initialization=initialization)
    return persisted
