"""Independent numerical reconstruction of the TGS winning blend contract.

Method reference: ybabakhin/kaggle_salt_bes_phalanx at
2f81d4dd8d50a01579e5f7650259dde92c5c3b8d, bes/ensemble.py.
This component does not replace the required training, TTA, pseudo-label
selection, or mosaic postprocessing stages. No source code is vendored.
"""
import numpy as np


def blend_predictions(keras_predictions, torch_predictions, constant_images, stage):
    """Blend already aligned predictions before mosaic postprocessing.

    Keras input: four snapshots by folds by population by height by width,
    in [0, 1], supplied in reverse snapshot order as in the source ensemble.
    Torch input: population by height by width, already averaged over its
    selected snapshots, horizontal TTA and folds. Its scores are min-max
    normalized over the ENTIRE supplied population, preserving the source's
    population dependence. Constant ranges fail explicitly instead of NaN.
    Constant-image flags zero only the Keras branch, matching that source.
    """
    if isinstance(stage, bool) or stage not in (1, 2, 3):
        raise ValueError('stage must be 1, 2 or 3')
    keras = np.asarray(keras_predictions, dtype=np.float64)
    torch = np.asarray(torch_predictions, dtype=np.float64)
    flags = np.asarray(constant_images)
    if keras.ndim != 5 or keras.shape[0] != 4 or any(n == 0 for n in keras.shape):
        raise ValueError('Keras predictions require four snapshots and nonempty fold/population/spatial axes')
    if torch.shape != keras.shape[2:]:
        raise ValueError('Torch predictions must align with population and spatial axes')
    if flags.dtype != np.bool_ or flags.shape != (torch.shape[0],):
        raise ValueError('constant_images must be an aligned boolean vector')
    if not np.isfinite(keras).all() or not np.isfinite(torch).all():
        raise ValueError('predictions must be finite')
    if (keras < 0).any() or (keras > 1).any():
        raise ValueError('Keras predictions must be probabilities')
    low, high = torch.min(), torch.max()
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError('Torch population requires a finite positive score range')
    torch_normalized = (torch - low) / span
    weights = np.array([1., 1., 1., 1. if stage == 1 else 3.])
    keras_average = np.einsum('s,snhw->nhw', weights / weights.sum(), keras.mean(axis=1))
    keras_average[flags] = 0
    scores = (keras_average + torch_normalized) / 2
    return dict(scores=scores, masks=scores > 0.5,
                confidence=((scores < 0.2) | (scores > 0.8)).mean(axis=(1, 2)),
                normalization_min=float(low), normalization_max=float(high))
