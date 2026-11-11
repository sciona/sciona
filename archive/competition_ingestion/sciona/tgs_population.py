"""Positional training/validation populations for the reviewed 63 TGS fits.

Caller supplies the branch's explicit fold assignment. Query and labeled
positions occupy separate namespaces; no source identifiers are persisted.
"""
import re
import numpy as np

from sciona.tgs_pseudo_selection import select_pseudo_labels


def select_fit_population(fit_key, folds, labeled_nonconstant, query_nonconstant,
                          *, confidence=None, area=None, pseudo_validation=None):
    folds, labeled_flags, query_flags = map(np.asarray, (folds, labeled_nonconstant, query_nonconstant))
    if (folds.ndim != 1 or folds.dtype.kind not in 'iu' or set(folds.tolist()) != set(range(5))
            or labeled_flags.shape != folds.shape or labeled_flags.dtype != np.bool_
            or query_flags.ndim != 1 or query_flags.dtype != np.bool_):
        raise ValueError('explicit five-fold assignment and aligned boolean eligibility required')
    keras = re.fullmatch(r'keras\.r([12])\.p([0-5])\.f([0-4])', fit_key) if isinstance(fit_key, str) else None
    torch = re.fullmatch(r'torch\.p([0-3])\.f([0-4])', fit_key) if isinstance(fit_key, str) else None
    empty = np.empty(0, dtype=np.int64)
    query = empty
    if keras:
        group, phase, fold = map(int, keras.groups())
        if (group == 1 and phase > 4) or (group == 2 and phase < 2 and fold != 0):
            raise ValueError('fit is not in the reviewed training inventory')
        train = np.flatnonzero((folds != fold) & labeled_flags)
        valid = np.flatnonzero((folds == fold) & labeled_flags)
        if group == 2 and phase < 2:
            selected = select_pseudo_labels(confidence, area, query_flags)['keras_indices']
            query = selected[np.random.RandomState(13).permutation(len(selected))]
            train = empty
    elif torch:
        phase, fold = map(int, torch.groups())
        if phase == 2 and fold != 0:
            raise ValueError('pseudo-only phase has one fit')
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        if phase == 1:
            query = select_pseudo_labels(confidence, area, query_flags)['torch_folds'][fold]
        elif phase == 2:
            # The source enumerates every generated pseudo-mask, not its
            # confident CSV. Explicit positional order replaces os.listdir.
            query = np.arange(len(query_flags), dtype=np.int64)
            train = empty
            valid = np.asarray(pseudo_validation)
            if (valid.ndim != 1 or valid.dtype.kind not in 'iu' or not len(valid)
                    or (valid < 0).any() or (valid >= len(folds)).any()
                    or len(np.unique(valid)) != len(valid)):
                raise ValueError('explicit unique in-range labeled validation positions required')
            valid = valid.astype(np.int64, copy=True)
    else:
        raise ValueError('unknown fit key')
    if not len(valid) or not (len(train) + len(query)):
        raise ValueError('nonempty training and validation populations required')
    if np.intersect1d(train, valid).size:
        raise ValueError('labeled training/validation overlap')
    return dict(training_labeled=train, training_query=query, validation_labeled=valid)
