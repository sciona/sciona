"""Supervised categorical embedding recipe, pending catalog qualification."""
from dataclasses import dataclass
from types import MappingProxyType
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from sciona.tabular_ensemble_training import binary_labels


def _rows(rows):
    if (type(rows) is not list or not rows or any(type(row) is not list or not row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(type(token) is not str for row in rows for token in row)):
        raise ValueError('Nonempty rectangular string category rows required')
    return rows


@dataclass(frozen=True)
class EntityEmbeddings:
    """Frozen per-field category vectors in field-major order."""
    maps: tuple
    dimension: int

    def transform(self, rows):
        rows = _rows(rows)
        if len(rows[0]) != len(self.maps):
            raise ValueError('Category width differs from fitted embedding')
        zero = (0.,) * self.dimension
        return np.asarray([[value for mapping, token in zip(self.maps, row)
            for value in mapping.get(token, zero)] for row in rows], dtype=np.float64)


def fit_entity_embeddings(rows, labels, *, dimension, iterations, regularization, tolerance, seed):
    """Learn lookup vectors as input weights of a one-hidden-layer binary MLP.

One-hot columns select per-field vectors which the network sums before its
tanh hidden activation and sigmoid output. Return the learned vectors before
summation, concatenated by field for downstream models. All training labels
may affect these supervised representations; the caller must fit independently
inside each outer validation fold. No query fitting, semantic similarity,
cross-fit coordinate alignment or universal generalization is claimed.
L-BFGS convergence warnings fail fitting rather than qualify unfinished fits.
"""
    rows = _rows(rows); y = binary_labels(labels, len(rows))
    if any(type(v) is not int or v < 1 for v in (dimension, iterations)):
        raise ValueError('Positive embedding dimension and iteration limit required')
    if (type(regularization) not in (int, float) or not np.isfinite(regularization) or regularization < 0
            or type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance <= 0
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Finite regularization, positive tolerance and valid seed required')
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
    x = encoder.fit_transform(rows)
    model = MLPClassifier(hidden_layer_sizes=(dimension,), activation='tanh', solver='lbfgs',
        alpha=regularization, max_iter=iterations, tol=tolerance, random_state=seed)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    weights = model.coefs_[0]
    if not all(np.isfinite(v).all() for v in [*model.coefs_, *model.intercepts_]):
        raise ValueError('Nonfinite embedding network parameters')
    maps = []; offset = 0
    for categories in encoder.categories_:
        maps.append(MappingProxyType({token: tuple(float(v) for v in weights[offset+i])
            for i, token in enumerate(categories)}))
        offset += len(categories)
    state = EntityEmbeddings(tuple(maps), dimension)
    return state, state.transform(rows)
