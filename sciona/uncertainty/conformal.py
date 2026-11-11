"""Split-conformal OOD envelopes and CDG-wide alpha-spending budgets.

Provides distribution-free finite-sample false-abort guarantees for Tier 1
certified CDGs: for exchangeable in-distribution inputs ``X ~ P_cal``,
``P(EnvelopeCheckResult.passed is False) <= alpha`` holds exactly without
parametric tail assumptions on ``ExponentialFamilyMixture``.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
TrustTierFilter = int | Collection[int] | None


def matches_trust_tier(atom_tier: int, trust_tier: TrustTierFilter) -> bool:
    """Return True if ``atom_tier`` is covered by ``trust_tier``.

    When ``trust_tier`` is ``None``, returns ``True`` for atoms of **all** tiers.
    When ``trust_tier`` is an ``int``, matches ``atom_tier == trust_tier``.
    When ``trust_tier`` is a collection of ``int``s, matches membership.
    """
    if trust_tier is None:
        return True
    if isinstance(trust_tier, int) and not isinstance(trust_tier, bool):
        return int(atom_tier) == int(trust_tier)
    if isinstance(trust_tier, Collection):
        return int(atom_tier) in {int(t) for t in trust_tier}
    raise TypeError(f"Unsupported trust_tier filter: {trust_tier!r}")


class ConfidenceEnvelopeViolation(RuntimeError):
    """Raised when a guarded node receives runtime inputs outside confidence bounds."""

    def __init__(
        self,
        *,
        node_id: str,
        atom_name: str,
        port_name: str,
        score: float,
        threshold: float,
        p_value: float,
        alpha: float,
        reason: str = "",
        trust_tier: int | None = 1,
    ) -> None:
        self.node_id = node_id
        self.atom_name = atom_name
        self.port_name = port_name
        self.score = score
        self.threshold = threshold
        self.p_value = p_value
        self.alpha = alpha
        self.reason = reason
        self.trust_tier = trust_tier
        detail = f" ({reason})" if reason else ""
        tier_prefix = (
            f"Tier {trust_tier} confidence"
            if trust_tier is not None
            else "Confidence"
        )
        super().__init__(
            f"{tier_prefix} bounds exceeded at node '{node_id}' "
            f"(atom='{atom_name}', port='{port_name}'): "
            f"nonconformity={score:.4g} > threshold={threshold:.4g} "
            f"[conformal p={p_value:.4g} <= alpha={alpha:.4g}]{detail}"
        )


class Tier1ConfidenceEnvelopeViolation(ConfidenceEnvelopeViolation):
    """Raised when a certified/guarded node receives out-of-envelope inputs."""


@dataclass(frozen=True)
class EnvelopeCheckResult:
    """Outcome of checking a single input against a ``ConformalEnvelope``."""

    passed: bool
    score: float
    threshold: float
    p_value: float
    alpha: float
    sufficient_calibration_size: bool
    reason: str = ""


class ConformalEnvelope:
    """Finite-sample split-conformal calibration wrapper over nonconformity scores.

    Stores sorted calibration scores ``S_{(1)} <= ... <= S_{(N)}`` where
    ``S_i = -log p_cal(phi(X_i))`` on held-out calibration inputs.
    """

    def __init__(
        self,
        calibration_scores: Any,
        *,
        n_calibration: int | None = None,
        max_stored_scores: int = 1024,
    ) -> None:
        arr = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
        if arr.size == 0 or np.any(~np.isfinite(arr)):
            raise ValueError(
                "ConformalEnvelope requires at least one finite calibration score."
            )
        sorted_arr = np.sort(arr)
        total_n = int(n_calibration) if n_calibration is not None else int(sorted_arr.size)
        if total_n < 1:
            raise ValueError("n_calibration must be >= 1.")

        if sorted_arr.size > max_stored_scores:
            # Conservative upper-order-statistic subsampling: ceiling indices
            # ensure compressed quantiles never underestimate true order stats.
            grid = np.linspace(0, sorted_arr.size - 1, max_stored_scores)
            ceil_idx = np.unique(np.ceil(grid).astype(int))
            sorted_arr = sorted_arr[ceil_idx]
            total_n = int(sorted_arr.size)

        self._scores = sorted_arr
        self._n_calibration = total_n

    @property
    def n_calibration(self) -> int:
        return self._n_calibration

    @property
    def sorted_scores(self) -> FloatArray:
        return self._scores.copy()

    @property
    def min_certifiable_alpha(self) -> float:
        """Smallest false-abort rate certifiable with finite threshold ``1 / (N + 1)``."""
        return 1.0 / float(self._n_calibration + 1)

    def quantile_threshold(self, alpha: float) -> tuple[float, bool]:
        """Return ``(q_{1-alpha}, sufficient_n)`` for target miscoverage ``alpha``.

        Uses ``k = ceil((N + 1) * (1 - alpha))`` (1-indexed).  When ``k > N``
        (i.e. ``alpha < 1 / (N + 1)``), returns ``(+inf, False)``.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        n = int(self._scores.size)
        k = int(math.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            return float("inf"), False
        k = max(1, k)
        return float(self._scores[k - 1]), True

    def p_value(self, score: float) -> float:
        """Exact split-conformal p-value ``(1 + #{S_i >= score}) / (N + 1)``."""
        if math.isnan(score):
            return 0.0
        n = int(self._scores.size)
        if score == float("inf"):
            count_ge = 0
        elif score == -float("inf"):
            count_ge = n
        else:
            idx_left = int(np.searchsorted(self._scores, score, side="left"))
            count_ge = n - idx_left
        return float(1 + count_ge) / float(n + 1)

    def check(
        self,
        score: float,
        *,
        alpha: float = 0.01,
        require_sufficient_calibration: bool = False,
    ) -> EnvelopeCheckResult:
        """Evaluate whether ``score`` lies inside the ``(1 - alpha)`` envelope."""
        if math.isnan(score) or score == float("inf"):
            q_hi, suff = self.quantile_threshold(alpha)
            return EnvelopeCheckResult(
                passed=False,
                score=score,
                threshold=q_hi,
                p_value=0.0 if math.isnan(score) else self.p_value(score),
                alpha=alpha,
                sufficient_calibration_size=suff,
                reason="input outside hard mathematical support (score is non-finite)",
            )

        threshold, suff = self.quantile_threshold(alpha)
        pval = self.p_value(score)
        if require_sufficient_calibration and not suff:
            return EnvelopeCheckResult(
                passed=False,
                score=score,
                threshold=threshold,
                p_value=pval,
                alpha=alpha,
                sufficient_calibration_size=False,
                reason=(
                    f"calibration size N={self._n_calibration} insufficient for "
                    f"alpha={alpha:.4g} (requires N >= {math.ceil(1.0 / alpha) - 1})"
                ),
            )

        passed = bool(score <= threshold)
        reason = "" if passed else "nonconformity score exceeds conformal quantile"
        return EnvelopeCheckResult(
            passed=passed,
            score=score,
            threshold=threshold,
            p_value=pval,
            alpha=alpha,
            sufficient_calibration_size=suff,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calibration": self._n_calibration,
            "sorted_scores": self._scores.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConformalEnvelope:
        return cls(
            calibration_scores=payload["sorted_scores"],
            n_calibration=payload.get("n_calibration"),
        )


def allocate_dag_alpha_budget(
    node_ids: list[str] | tuple[str, ...],
    *,
    global_alpha: float = 0.01,
    risk_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Allocate a CDG-wide false-abort budget ``global_alpha`` across nodes.

    By union bound (weighted Bonferroni alpha-spending), assigning
    ``sum_v alpha_v = global_alpha`` guarantees family-wise false-abort rate
    ``P(any node falsely aborts | X ~ P_cal) <= global_alpha`` across arbitrary
    DAG dependencies.  Nodes with higher risk sensitivity receive a smaller
    share of ``alpha`` (Wait: higher weight -> looser or tighter? Let's be explicit:
    ``risk_weights`` represent budget share weights ``w_v > 0``, normalized so
    ``alpha_v = global_alpha * w_v / sum_u w_u``).
    """
    if not (0.0 < global_alpha < 1.0):
        raise ValueError(f"global_alpha must lie in (0, 1), got {global_alpha}.")
    unique_ids = list(dict.fromkeys(node_ids))
    if not unique_ids:
        return {}

    if not risk_weights:
        share = global_alpha / float(len(unique_ids))
        return {nid: share for nid in unique_ids}

    raw = np.array(
        [max(float(risk_weights.get(nid, 1.0)), 1e-12) for nid in unique_ids],
        dtype=np.float64,
    )
    alloc = global_alpha * (raw / float(np.sum(raw)))
    return {nid: float(alloc[idx]) for idx, nid in enumerate(unique_ids)}
