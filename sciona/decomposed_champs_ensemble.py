"""CHAMPS identity and model-selection policy, without numerical reduction.

Selection retains the Bosch MIT-licensed source policy; see docs/licenses/CHAMPS-MIT.txt.
The central rank mean is a separately bound generic graph operation.
"""
from collections.abc import Mapping
import numpy as np
from sciona.champs_ensemble import MODEL_ORDER, COUPLING_TYPES, MODEL_MASK


def ensemble_inputs(predictions, coupling_types):
    """Validate all thirteen outputs, then select nine predictions per record."""
    if not isinstance(predictions, Mapping) or set(predictions) != set(MODEL_ORDER):
        raise ValueError("Exactly thirteen configured model variants are required")
    if not isinstance(coupling_types, Mapping):
        raise ValueError("Coupling types must be a mapping")
    keys = tuple(coupling_types)
    if any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Record keys must be nonempty strings")
    if any(not isinstance(t, str) or t not in COUPLING_TYPES for t in coupling_types.values()):
        raise ValueError("Unknown coupling type")
    rows = []
    for model in MODEL_ORDER:
        values = predictions[model]
        if not isinstance(values, Mapping) or set(values) != set(keys):
            raise ValueError("Each model must predict exactly the requested records")
        row = [values[k] for k in keys]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
               for v in row):
            raise ValueError("Predictions must be real numbers")
        try:
            array = np.asarray(row, dtype=np.float64)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Predictions must fit finite float64") from exc
        if not np.isfinite(array).all():
            raise ValueError("Predictions must be finite")
        rows.append(array)
    matrix = np.stack(rows)
    mask = np.asarray(MODEL_MASK, dtype=bool)
    selected = np.empty((9, len(keys)), dtype=np.float64, order='F')
    for index, coupling_type in enumerate(COUPLING_TYPES):
        columns = np.asarray([coupling_types[k] == coupling_type for k in keys], dtype=bool)
        if columns.any():
            selected[:, columns] = matrix[mask[:, index]][:, columns]
    return keys, selected, 5


def result(keys, values):
    """Restore opaque record order after the generic central mean."""
    if (not isinstance(keys, tuple) or any(not isinstance(k, str) or not k for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError('Unique opaque record keys required')
    if (not isinstance(values, np.ndarray) or values.shape != (len(keys),)
            or values.dtype != np.dtype('float64') or not np.isfinite(values).all()):
        raise ValueError('Aligned finite float64 results required')
    return {key: float(value) for key, value in zip(keys, values)}
