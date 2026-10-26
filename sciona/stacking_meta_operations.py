"""Draft decomposed Otto meta stage, with source policy kept in adapters.

Selection, full-reference refitting and probability aggregation are separate
calls. The general aggregation functions support other class/run counts; these
adapters deliberately retain nine classes, four folds and 250/600/250 bags.
Upstream feature provenance, class meanings and row alignment remain caller
responsibilities. Selection losses are not independent generalization estimates.
This module is outside the immutable historical implementation inventory.
"""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from sciona.otto_meta_tuning import tune, _learner
from sciona.probability_ensemble import (
    mean_probability_bag, geometric_arithmetic_probability_blend,
)

FAMILIES = ('xgboost', 'lasagne_neural', 'adaboost_extratrees')
RUNS = dict(zip(FAMILIES, (250, 600, 250)))


@dataclass(frozen=True, repr=False)
class MetaInputs:
    """Private in-process values; not a serializable model or public artifact."""
    pairs: dict = field(repr=False)
    labels: np.ndarray = field(repr=False)
    folds: np.ndarray = field(repr=False)
    identities: list = field(repr=False)
    seed: int
    candidates: dict = field(repr=False)


@dataclass(frozen=True, repr=False)
class SelectedControls:
    inputs: MetaInputs = field(repr=False)
    family: str
    index: int
    controls: dict = field(repr=False)
    losses: tuple
    tuning_model_fits: int


def prepare_meta_inputs(layout, labels, folds, identities, seed, candidates):
    """Check every family before any learner runs; detach caller-owned values."""
    if type(candidates) is not dict or set(candidates) != set(FAMILIES):
        raise ValueError('Candidates for all three source families required')
    if any(type(cs) is not list or not cs or any(type(c) is not dict for c in cs)
           for cs in candidates.values()):
        raise ValueError('Nonempty explicit candidate lists required')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Common signed integer seed required')
    pairs = {}
    for family in FAMILIES:
        prefix = 'neural' if family == 'lasagne_neural' else 'tree'
        x = np.array(layout[prefix+'_training'], dtype=float, copy=True)
        q = np.array(layout[prefix+'_query'], dtype=float, copy=True)
        if (x.ndim != 2 or q.ndim != 2 or not len(x) or not len(q)
                or x.shape[1] < 1 or x.shape[1] != q.shape[1]
                or not np.isfinite(x).all() or not np.isfinite(q).all()):
            raise ValueError('Aligned finite meta matrices required')
        x.setflags(write=False)
        q.setflags(write=False)
        pairs[family] = (x, q)
    if len({(len(x), len(q)) for x, q in pairs.values()}) != 1:
        raise ValueError('Meta families must share row counts')
    n = len(pairs[FAMILIES[0]][0])
    y, f = np.array(labels, copy=True), np.array(folds, copy=True)
    if y.shape != (n,) or y.dtype.kind not in 'iu' or set(y.tolist()) != set(range(9)):
        raise ValueError('Nine aligned integer classes required')
    if f.shape != (n,) or f.dtype.kind not in 'iu' or set(f.tolist()) != set(range(4)):
        raise ValueError('Four aligned folds required')
    if (type(identities) is not list or len(identities) != n
            or any(type(i) is not str or not i for i in identities)
            or len(set(identities)) != n):
        raise ValueError('Unique aligned opaque identities required')
    if any(set(y[f != fold].tolist()) != set(range(9)) for fold in range(4)):
        raise ValueError('Every fitting fold requires all classes')
    y.setflags(write=False)
    f.setflags(write=False)
    return MetaInputs(pairs, y, f, list(identities), seed, deepcopy(candidates))


def select_family(inputs, family):
    """Run complete four-fold candidate selection for one source model family."""
    if type(inputs) is not MetaInputs or family not in FAMILIES:
        raise ValueError('Prepared meta inputs and supported family required')
    result = tune(inputs.pairs[family][0], inputs.labels, inputs.folds,
                  inputs.identities, family=family, seed=inputs.seed,
                  candidates=inputs.candidates[family])
    index = result['selected_index']
    losses = tuple(result['log_losses'])
    if (type(index) is not int or not 0 <= index < len(inputs.candidates[family])
            or len(losses) != len(inputs.candidates[family])
            or not np.isfinite(losses).all() or any(x < 0 for x in losses)
            or index != int(np.argmin(losses))
            or result['models_per_candidate'] != 4 * RUNS[family]):
        raise ValueError('Invalid source selection result')
    return SelectedControls(inputs, family, index, deepcopy(inputs.candidates[family][index]),
                            losses, result['models_per_candidate'] * len(losses))


def refit_family(inputs, selection):
    """Fit the selected full bag on all training rows and predict query rows."""
    if (type(inputs) is not MetaInputs or type(selection) is not SelectedControls
            or selection.inputs is not inputs or selection.family not in FAMILIES):
        raise ValueError('Selection from these prepared inputs required')
    family = selection.family
    if (not 0 <= selection.index < len(inputs.candidates[family])
            or selection.controls != inputs.candidates[family][selection.index]):
        raise ValueError('Selected controls changed')
    learner, runs = _learner(family)
    if runs != RUNS[family]:
        raise ValueError('Complete source bag size required')
    x, q = inputs.pairs[family]
    bag = learner(x, inputs.labels, q, seed=inputs.seed, controls=deepcopy(selection.controls))
    array = np.asarray(bag)
    if array.shape != (runs, len(q), 9):
        raise ValueError('Full source bag with aligned nine-class query rows required')
    # Validate the numeric contract before the separate averaging operation.
    from sciona.probability_ensemble import _probabilities
    return _probabilities(array, 3)


def selection_summary(selection):
    """Return the historical public result fields, without private input values."""
    return dict(selected_index=selection.index, log_losses=list(selection.losses),
                tuning_model_fits=selection.tuning_model_fits,
                refit_models=RUNS[selection.family])


def fit_decomposed(layout, labels, folds, identities, *, seed, candidates, progress=None):
    """Sequential reference composition of the separately callable operations.

This convenience composition is not itself sufficient graph decomposition.
The future CDG must expose select/refit/mean/blend as distinct nodes.
"""
    inputs = prepare_meta_inputs(layout, labels, folds, identities, seed, candidates)
    means, summaries = {}, {}
    for family in FAMILIES:
        selected = select_family(inputs, family)
        bag = refit_family(inputs, selected)
        means[family] = mean_probability_bag(bag)
        summaries[family] = selection_summary(selected)
        if progress is not None:
            progress(family)
    blend = geometric_arithmetic_probability_blend(
        *(means[family] for family in FAMILIES), .85, .15, .65)
    return dict(blend=blend, selection=summaries)
