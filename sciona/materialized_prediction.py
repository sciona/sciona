"""Domain-neutral native prediction and ordered ensemble arithmetic."""
import numpy as np
import pandas as pd


def predict_named_frame(model, frame):
    """Call a thread-count-aware predictor using its exact ordered named schema."""
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(model.feature_names_):
        raise ValueError('Exact ordered native feature schema required')
    result = np.asarray(model.predict(frame, thread_count=1), dtype=np.float64)
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError('Finite row-aligned scalar predictions required')
    return result.copy()


def average_prediction_vectors(vectors):
    """Sum equally sized finite vectors in caller order, then divide by count."""
    if not isinstance(vectors, (list, tuple)) or not vectors:
        raise ValueError('Nonempty ordered prediction vectors required')
    values = [np.asarray(value, dtype=np.float64) for value in vectors]
    if any(value.ndim != 1 or value.shape != values[0].shape or not np.isfinite(value).all() for value in values):
        raise ValueError('Finite equally sized prediction vectors required')
    result = np.zeros_like(values[0])
    with np.errstate(over='raise', invalid='raise'):
        for value in values:
            result += value
        result /= len(values)
    return result
