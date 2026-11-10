"""Source-specific OSIC image adapters over caller-decoded integer planes.

The winning notebooks use different training and inference formulas. Neither
adapter applies a conventional medical intensity conversion. Decoding, slice
selection and subject identity remain external. No clinical meaning is inferred.
"""
import cv2
import numpy as np


def _plane(pixels: np.ndarray) -> None:
    if (not isinstance(pixels, np.ndarray) or pixels.ndim != 2
            or not pixels.size or min(pixels.shape) == 0
            or pixels.dtype.kind not in 'iu'):
        raise ValueError('A nonempty two-dimensional decoded integer plane is required')
    # GCD builds may ignore setNumThreads(1); setNumThreads(0) disables workers.
    # The caller configures this global setting before executing any adapters.
    if cv2.getNumThreads() != 1:
        raise ValueError('Caller must disable OpenCV parallel workers before execution')


def training_plane(pixels: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Subtract metadata offset, divide by metadata scale times 1000, resize.

Preserves source order and OpenCV's default linear interpolation at 512x512.
Supports finite scalar metadata and a nonzero scale. No clipping or additional
normalization is performed. This is the source training policy only.
"""
    _plane(pixels)
    if (not np.isscalar(intercept) or not np.isscalar(slope)
            or not np.isrealobj(intercept) or not np.isrealobj(slope)
            or not np.isfinite(intercept) or not np.isfinite(slope) or slope == 0):
        raise ValueError('Finite real offset and nonzero scale required')
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        normalized = (pixels - float(intercept)) / (float(slope) * 1000)
    if not np.all(np.isfinite(normalized)):
        raise ValueError('Normalization must remain finite')
    return cv2.resize(normalized, (512, 512))


def inference_plane(pixels: np.ndarray) -> np.ndarray:
    """Divide decoded values by 2048 and resize, without using image metadata.

This deliberately preserves the source inference policy even when it differs
from training normalization. Returns a fresh 512x512 floating-point plane.
"""
    _plane(pixels)
    return cv2.resize(pixels / 2**11, (512, 512))
