"""Pure-NumPy special functions and numerically stable linear-algebra helpers.

Keeps ``sciona.uncertainty`` free of a hard ``scipy`` runtime dependency while
maintaining full float64 precision for exponential-family log-partitions,
score functions, Fisher information matrices, and Woodbury inversions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_LGAMMA_UFUNC = np.frompyfunc(math.lgamma, 1, 1)


def loggamma(x: Any) -> FloatArray:
    """Vectorized log-gamma ``ln Gamma(x)`` for positive real arrays."""
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError("loggamma requires strictly positive arguments.")
    return np.asarray(_LGAMMA_UFUNC(arr), dtype=np.float64)


def digamma(x: Any) -> FloatArray:
    """Vectorized digamma ``psi(x) = d/dx ln Gamma(x)`` to float64 precision.

    Uses upward recurrence ``psi(x) = psi(x + 1) - 1/x`` until ``x >= 8``,
    followed by a 7-term Bernoulli asymptotic expansion (< 1e-15 error).
    """
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError("digamma requires strictly positive arguments.")
    orig_shape = arr.shape
    work = arr.astype(np.float64, copy=True).reshape(-1)
    acc = np.zeros_like(work)

    mask = work < 12.0
    while np.any(mask):
        acc[mask] -= 1.0 / work[mask]
        work[mask] += 1.0
        mask = work < 12.0

    inv = 1.0 / work
    inv2 = inv * inv
    # Bernoulli numbers B_2..B_14 divided by 2k
    asymp = (
        np.log(work)
        - 0.5 * inv
        - inv2
        * (
            (1.0 / 12.0)
            - inv2
            * (
                (1.0 / 120.0)
                - inv2
                * (
                    (1.0 / 252.0)
                    - inv2
                    * (
                        (1.0 / 240.0)
                        - inv2 * ((1.0 / 132.0) - inv2 * (691.0 / 32760.0))
                    )
                )
            )
        )
    )
    return np.asarray((acc + asymp).reshape(orig_shape), dtype=np.float64)


def trigamma(x: Any) -> FloatArray:
    """Vectorized trigamma ``psi_1(x) = d^2/dx^2 ln Gamma(x)`` to float64 precision.

    Uses upward recurrence ``psi_1(x) = psi_1(x + 1) + 1/x^2`` until ``x >= 8``,
    followed by Euler-Maclaurin asymptotic expansion (< 1e-15 error).
    """
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError("trigamma requires strictly positive arguments.")
    orig_shape = arr.shape
    work = arr.astype(np.float64, copy=True).reshape(-1)
    acc = np.zeros_like(work)

    mask = work < 12.0
    while np.any(mask):
        w_m = work[mask]
        acc[mask] += 1.0 / (w_m * w_m)
        work[mask] = w_m + 1.0
        mask = work < 12.0

    inv = 1.0 / work
    inv2 = inv * inv
    asymp = (
        inv
        + 0.5 * inv2
        + inv2
        * inv
        * (
            (1.0 / 6.0)
            - inv2
            * (
                (1.0 / 30.0)
                - inv2
                * (
                    (1.0 / 42.0)
                    - inv2 * ((1.0 / 30.0) - inv2 * (5.0 / 66.0))
                )
            )
        )
    )
    return np.asarray((acc + asymp).reshape(orig_shape), dtype=np.float64)


def logsumexp(a: FloatArray, axis: int = -1, keepdims: bool = False) -> FloatArray:
    """Numerically stable log-sum-exp handling ``-inf`` entries cleanly."""
    arr = np.asarray(a, dtype=np.float64)
    a_max = np.max(arr, axis=axis, keepdims=True)
    # Replace non-finite maxima (e.g., all -inf slice) with 0 during shifting
    safe_max = np.where(np.isfinite(a_max), a_max, 0.0)
    shifted = arr - safe_max
    sum_exp = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = safe_max + np.log(sum_exp)
    # Restore -inf where all entries were -inf
    out = np.where(a_max == -np.inf, -np.inf, out)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return np.asarray(out, dtype=np.float64)


def symmetrize(mat: FloatArray) -> FloatArray:
    """Return the symmetric part ``0.5 * (A + A.T)``."""
    return np.asarray(0.5 * (mat + mat.T), dtype=np.float64)


def spd_eigendecomp(
    mat: FloatArray,
    *,
    tol: float = 1e-12,
) -> tuple[FloatArray, FloatArray, bool]:
    """Symmetric eigendecomposition checking strict positive-definiteness.

    Returns ``(eigenvalues, eigenvectors, is_spd)`` where ``is_spd`` is True
    iff all eigenvalues strictly exceed ``tol``.
    """
    sym = symmetrize(np.asarray(mat, dtype=np.float64))
    eigvals, eigvecs = np.linalg.eigh(sym)
    is_spd = bool(np.all(eigvals > tol) and np.all(np.isfinite(eigvals)))
    return eigvals, eigvecs, is_spd


def woodbury_inverse_and_logdet(
    diag: FloatArray,
    factor: FloatArray,
) -> tuple[FloatArray, float]:
    """Compute ``(Sigma^{-1}, log|Sigma|)`` for ``Sigma = diag(d) + W W^T``.

    Uses the Woodbury matrix identity and matrix determinant lemma in
    ``O(d r^2 + r^3)`` time where ``W`` has shape ``(d, r)``.
    """
    d = np.asarray(diag, dtype=np.float64)
    if d.ndim != 1 or np.any(d <= 0.0):
        raise ValueError("Diagonal variance entries must be 1-D and strictly positive.")

    inv_d = 1.0 / d
    logdet_d = float(np.sum(np.log(d)))

    w = np.asarray(factor, dtype=np.float64)
    if w.size == 0 or w.shape[1] == 0:
        return np.diag(inv_d), logdet_d

    if w.ndim != 2 or w.shape[0] != d.shape[0]:
        raise ValueError(
            f"Factor shape {w.shape} incompatible with diagonal length {d.shape[0]}."
        )

    # Capacitance matrix C = I_r + W^T D^{-1} W  (SPD of size r x r)
    inv_d_w = inv_d[:, None] * w  # (d, r)
    r = w.shape[1]
    cap = np.eye(r, dtype=np.float64) + w.T @ inv_d_w
    cap = symmetrize(cap)
    chol = np.linalg.cholesky(cap)
    logdet_cap = 2.0 * float(np.sum(np.log(np.diag(chol))))

    # Solve L Z = (D^{-1} W)^T  =>  Z = L^{-1} W^T D^{-1}
    z = np.linalg.solve(chol, inv_d_w.T)  # (r, d)
    precision: FloatArray = np.asarray(np.diag(inv_d) - z.T @ z, dtype=np.float64)
    return symmetrize(precision), logdet_d + logdet_cap
