"""Uncertainty estimation backends for precision-gradient computation.

Provides structured per-atom error-expansion estimates and pluggable
backends (heuristic, catalog, analytic) for the ghost simulation pass.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sciona.uncertainty.models import AtomUncertaintyProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomUncertaintyEstimate:
    """Structured uncertainty estimate for a single atom.

    Attributes:
        mode: How the estimate was obtained.
            ``"heuristic"`` — hand-tuned or keyword-matched.
            ``"empirical"`` — measured via perturbation harness.
            ``"analytic"`` — derived from numerical Jacobian.
            ``"unknown"``  — no estimate available.
        scalar_factor: Multiplicative error-expansion factor (None = unmeasured).
        confidence: Trust in the estimate, 0.0 (none) to 1.0 (fully calibrated).
        n_trials: Number of perturbation trials (empirical mode).
        epsilon: Perturbation magnitude used (empirical mode).
        input_regime: Description of the input regime tested.
        notes: Free-form annotation.
    """

    mode: str = "unknown"
    scalar_factor: float | None = None
    confidence: float = 0.0
    n_trials: int = 0
    epsilon: float = 0.0
    input_regime: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class UncertaintyBackend(Protocol):
    """Protocol for uncertainty estimation backends."""

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate: ...


# ---------------------------------------------------------------------------
# Known per-atom error expansion factors (moved from ghost_sim.py)
# ---------------------------------------------------------------------------

_ATOM_ERROR_FACTORS: dict[str, float] = {
    "fft": 1.5,       # O(n log n) rounding accumulation
    "ifft": 1.5,
    "rfft": 1.3,
    "irfft": 1.3,
    "stft": 1.6,
    "istft": 1.6,
    "butter": 1.1,    # filter design — coefficient quantisation
    "cheby1": 1.3,
    "cheby2": 1.3,
    "ellip": 1.4,
    "lfilter": 2.0,   # IIR recursive application amplifies error
    "sosfilt": 1.4,   # SOS form is more stable than TF
    "firwin": 1.05,
    "convolve": 1.2,
    "correlate": 1.2,
    "resample": 1.3,
    "hilbert": 1.4,
    "welch": 1.2,
}

_KEYWORD_FALLBACKS: list[tuple[tuple[str, ...], float]] = [
    (("filter", "smooth", "denoise", "bandpass"), 1.2),
    (("detect", "segment", "peak", "extract"), 1.35),
    (("rate", "metric", "measure", "cadence"), 1.15),
]


# ---------------------------------------------------------------------------
# HeuristicBackend — wraps the legacy dict + keyword fallback
# ---------------------------------------------------------------------------


class HeuristicBackend:
    """Heuristic uncertainty backend using hand-tuned factors and keyword matching.

    Dict miss + keyword miss → ``AtomUncertaintyEstimate(mode="unknown")``.
    """

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        # Exact match (using suffix leaf name)
        short_name = atom_name.rsplit(".", 1)[-1]
        if short_name in _ATOM_ERROR_FACTORS:
            return AtomUncertaintyEstimate(
                mode="heuristic",
                scalar_factor=_ATOM_ERROR_FACTORS[short_name],
                confidence=0.2,
                notes="hand-tuned exact match",
            )

        # Keyword fallback
        lowered = atom_name.lower()
        for keywords, factor in _KEYWORD_FALLBACKS:
            if any(token in lowered for token in keywords):
                return AtomUncertaintyEstimate(
                    mode="heuristic",
                    scalar_factor=factor,
                    confidence=0.1,
                    notes=f"keyword match: {keywords}",
                )

        # Unknown — not 1.0
        return AtomUncertaintyEstimate(mode="unknown")


# ---------------------------------------------------------------------------
# CatalogBackend — looks up from pre-loaded uncertainty.json data
# ---------------------------------------------------------------------------


class CatalogBackend:
    """Catalog-based backend using pre-loaded ``uncertainty.json`` estimates."""

    def __init__(self, catalog: dict[str, AtomUncertaintyEstimate]) -> None:
        self._catalog = catalog

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        return self._catalog.get(atom_name, AtomUncertaintyEstimate(mode="unknown"))


# ---------------------------------------------------------------------------
# ChainBackend — try each backend, return first non-unknown
# ---------------------------------------------------------------------------


class ChainBackend:
    """Composite backend: returns the first non-unknown estimate from a chain."""

    def __init__(self, *backends: UncertaintyBackend) -> None:
        self._backends = backends

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        for backend in self._backends:
            est = backend.estimate(atom_name)
            if est.mode != "unknown":
                return est
        return AtomUncertaintyEstimate(mode="unknown")

    def estimate_profile(self, atom_name: str) -> AtomUncertaintyProfile:
        from sciona.uncertainty.models import AtomUncertaintyProfile

        for backend in self._backends:
            prof_fn = getattr(backend, "estimate_profile", None)
            if callable(prof_fn):
                prof = prof_fn(atom_name)
                if isinstance(prof, AtomUncertaintyProfile) and prof.mode != "unknown":
                    return prof
            est = backend.estimate(atom_name)
            if est.mode != "unknown":
                short = atom_name.rsplit(".", 1)[-1]
                return AtomUncertaintyProfile.from_v1_estimate(short, est)
        return AtomUncertaintyProfile(atom_name=atom_name, mode="unknown")


# ---------------------------------------------------------------------------
# AnalyticBackend — numerical Jacobian spectral norm
# ---------------------------------------------------------------------------


class AnalyticBackend:
    """Uncertainty estimation via numerical Jacobian spectral norm.

    Only suitable for atoms confirmed epsilon-stable in empirical testing.
    Input size is capped at 512 (Jacobian is O(n²) evaluations).
    """

    _MAX_INPUT_SIZE = 512

    def __init__(
        self,
        atom_registry: dict[str, object],
        base_inputs: dict[str, NDArray[Any]],
        *,
        eps: float = 1e-7,
        stable_atoms: set[str] | None = None,
    ) -> None:
        self._registry = atom_registry
        self._base_inputs = base_inputs
        self._eps = eps
        self._stable_atoms = stable_atoms or set()

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        if atom_name not in self._stable_atoms:
            return AtomUncertaintyEstimate(mode="unknown")

        fn = self._registry.get(atom_name)
        x0 = self._base_inputs.get(atom_name)
        if fn is None or x0 is None:
            return AtomUncertaintyEstimate(mode="unknown")

        if x0.size > self._MAX_INPUT_SIZE:
            return AtomUncertaintyEstimate(
                mode="unknown",
                notes=f"input size {x0.size} exceeds cap {self._MAX_INPUT_SIZE}",
            )

        try:
            jac = _numerical_jacobian(fn, x0, self._eps)
            # Spectral norm (largest singular value) as factor
            factor = float(np.linalg.norm(jac, ord=2))
            return AtomUncertaintyEstimate(
                mode="analytic",
                scalar_factor=factor,
                confidence=0.7,
                notes=f"Jacobian spectral norm, eps={self._eps}",
            )
        except Exception as exc:
            logger.debug("Analytic estimate failed for %s: %s", atom_name, exc)
            return AtomUncertaintyEstimate(mode="unknown")


def _numerical_jacobian(
    fn: object,
    x0: NDArray[Any],
    eps: float = 1e-7,
) -> NDArray[np.float64]:
    """Compute the numerical Jacobian via central finite differences.

    Args:
        fn: Callable ``np.ndarray -> np.ndarray``.
        x0: Input point (1-D flattened).
        eps: Perturbation magnitude.

    Returns:
        Jacobian matrix of shape ``(m, n)`` where ``m = len(f(x0))``
        and ``n = len(x0)``.
    """
    x_flat = x0.ravel().astype(np.float64)
    n = len(x_flat)
    y0 = np.asarray(fn(x_flat.reshape(x0.shape))).ravel()  # type: ignore[operator]
    m = len(y0)

    jac = np.empty((m, n), dtype=np.float64)
    for i in range(n):
        x_plus = x_flat.copy()
        x_minus = x_flat.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        y_plus = np.asarray(fn(x_plus.reshape(x0.shape))).ravel()  # type: ignore[operator]
        y_minus = np.asarray(fn(x_minus.reshape(x0.shape))).ravel()  # type: ignore[operator]
        jac[:, i] = (y_plus - y_minus) / (2.0 * eps)

    return jac


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def load_uncertainty_json(
    path: str | Path,
) -> tuple[str, AtomUncertaintyEstimate]:
    """Load an ``uncertainty.json`` file and return ``(atom_name, best_estimate)``.

    Picks the estimate with the highest confidence.  Returns mode="unknown"
    if the file is empty or malformed.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return ("", AtomUncertaintyEstimate(mode="unknown"))

    atom_name = data.get("atom", "")
    if str(data.get("schema_version", "")).startswith("2") or "input_regimes" in data:
        from sciona.uncertainty.models import AtomUncertaintyProfile

        profile = AtomUncertaintyProfile.from_dict(data)
        return (profile.atom_name, profile.to_v1_estimate())

    estimates = data.get("estimates", [])
    if not estimates:
        return (atom_name, AtomUncertaintyEstimate(mode="unknown"))

    best: dict[str, Any] | None = None
    best_confidence = -1.0
    for entry in estimates:
        conf = float(entry.get("confidence", 0.0))
        if conf > best_confidence:
            best_confidence = conf
            best = entry

    if best is None:
        return (atom_name, AtomUncertaintyEstimate(mode="unknown"))

    raw_sf = best.get("scalar_factor")
    return (
        atom_name,
        AtomUncertaintyEstimate(
            mode=str(best.get("mode", "unknown")),
            scalar_factor=float(raw_sf) if isinstance(raw_sf, (int, float)) else None,
            confidence=float(best.get("confidence", 0.0)),
            n_trials=int(best.get("n_trials", 0)),
            epsilon=float(best.get("epsilon", 0.0)),
            input_regime=str(best.get("input_regime", "")),
            notes=str(best.get("notes", "")),
        ),
    )


class ProfileCatalogBackend:
    """Backend backed by v2 ``AtomUncertaintyProfile`` records.

    Satisfies ``UncertaintyBackend`` for legacy scalar callers while exposing
    ``estimate_profile`` for directional mixture/Jacobian propagation.
    """

    def __init__(self, profiles: dict[str, AtomUncertaintyProfile]) -> None:
        self._profiles = dict(profiles)

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        prof = self._profiles.get(atom_name)
        if prof is None:
            short = atom_name.rsplit(".", 1)[-1]
            prof = self._profiles.get(short)
        if prof is None:
            return AtomUncertaintyEstimate(mode="unknown")
        return prof.to_v1_estimate()

    def estimate_profile(self, atom_name: str) -> AtomUncertaintyProfile:
        from sciona.uncertainty.models import AtomUncertaintyProfile

        prof = self._profiles.get(atom_name)
        if prof is None:
            short = atom_name.rsplit(".", 1)[-1]
            prof = self._profiles.get(short)
        if prof is None:
            return AtomUncertaintyProfile(atom_name=atom_name, mode="unknown")
        return prof

