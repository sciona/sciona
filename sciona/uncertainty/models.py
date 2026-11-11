"""Atom-level uncertainty profile combining mixture regimes, channels, & conformal gates.

Unifies:
1. Per-input-port calibrated ``ExponentialFamilyMixture`` regimes over
   ``SufficientStatProjector`` embeddings + optional ``ConformalEnvelope``s.
2. Per-mixture-component ``ComponentChannel`` operators (``LowRankPlusTailOperator``
   Jacobians + intrinsic approximation noise covariances ``Sigma_eps``).
3. Lossless round-trip serialization to v2 ``uncertainty.json`` and full backward
   compatibility with v1 scalar ``AtomUncertaintyEstimate`` records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sciona.uncertainty._math import logsumexp, symmetrize
from sciona.uncertainty.conformal import (
    ConfidenceEnvelopeViolation,
    ConformalEnvelope,
    EnvelopeCheckResult,
    Tier1ConfidenceEnvelopeViolation,
    TrustTierFilter,
    matches_trust_tier,
)
from sciona.uncertainty.distributions import (
    ExponentialFamilyMixture,
    mixture_from_dict,
)
from sciona.uncertainty.operators import LowRankPlusTailOperator, PullbackResult
from sciona.uncertainty.projectors import (
    ProjectorSpec,
    SufficientStatProjector,
    projector_from_spec,
)

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import AtomUncertaintyEstimate

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortRegimeSpec:
    """Calibrated epistemic input regime for a single atom input port."""

    port_name: str
    projector_spec: ProjectorSpec
    mixture: ExponentialFamilyMixture
    conformal_envelope: ConformalEnvelope | None = None

    def __post_init__(self) -> None:
        if self.projector_spec.dim != self.mixture.dim:
            raise ValueError(
                f"Port '{self.port_name}': projector dim ({self.projector_spec.dim}) "
                f"!= mixture dim ({self.mixture.dim})."
            )

    def build_projector(self) -> SufficientStatProjector:
        return projector_from_spec(self.projector_spec)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "port_name": self.port_name,
            "projector": self.projector_spec.to_dict(),
            "mixture": self.mixture.to_dict(),
        }
        if self.conformal_envelope is not None:
            payload["conformal_envelope"] = self.conformal_envelope.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PortRegimeSpec:
        env_raw = payload.get("conformal_envelope")
        return cls(
            port_name=str(payload["port_name"]),
            projector_spec=ProjectorSpec.from_dict(payload["projector"]),
            mixture=mixture_from_dict(payload["mixture"]),
            conformal_envelope=(
                ConformalEnvelope.from_dict(env_raw) if env_raw is not None else None
            ),
        )


@dataclass(frozen=True)
class ComponentChannel:
    """Local linearized sensitivity & intrinsic noise inside mixture component ``k``."""

    component_index: int
    operators: dict[tuple[str, str], LowRankPlusTailOperator] = field(
        default_factory=dict
    )
    """Map ``(input_port, output_port) -> LowRankPlusTailOperator``."""
    intrinsic_output_cov: dict[str, FloatArray] = field(default_factory=dict)
    """Map ``output_port -> Sigma_{eps, k}`` of shape ``(d_out, d_out)``."""

    def to_dict(self) -> dict[str, Any]:
        ops_list = [
            {
                "input_port": in_p,
                "output_port": out_p,
                "operator": op.to_dict(),
            }
            for (in_p, out_p), op in sorted(self.operators.items())
        ]
        intrinsic_dict = {
            out_p: np.asarray(cov, dtype=np.float64).tolist()
            for out_p, cov in sorted(self.intrinsic_output_cov.items())
        }
        return {
            "component_index": self.component_index,
            "operators": ops_list,
            "intrinsic_output_cov": intrinsic_dict,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentChannel:
        ops: dict[tuple[str, str], LowRankPlusTailOperator] = {}
        for item in payload.get("operators", []):
            key = (str(item["input_port"]), str(item["output_port"]))
            ops[key] = LowRankPlusTailOperator.from_dict(item["operator"])
        intrinsic: dict[str, FloatArray] = {}
        for out_p, raw_cov in payload.get("intrinsic_output_cov", {}).items():
            arr = np.asarray(raw_cov, dtype=np.float64)
            if arr.ndim == 1:
                arr = np.diag(arr)
            intrinsic[str(out_p)] = symmetrize(arr)
        return cls(
            component_index=int(payload.get("component_index", 0)),
            operators=ops,
            intrinsic_output_cov=intrinsic,
        )


@dataclass(frozen=True)
class PortEvaluation:
    """Runtime evaluation of a single input port against its ``PortRegimeSpec``."""

    port_name: str
    features: FloatArray
    in_domain: bool
    nonconformity_score: float
    bregman_divergence: float
    responsibilities: FloatArray
    envelope_check: EnvelopeCheckResult | None = None
    violation_reason: str = ""


_UNSET_TIER: Any = object()


@dataclass(frozen=True)
class InputEvaluationReport:
    """Combined runtime regime & confidence evaluation across all calibrated ports."""

    port_evaluations: dict[str, PortEvaluation]
    combined_responsibilities: FloatArray
    active_component: int
    in_confidence_bounds: bool
    violation_reason: str = ""

    def raise_if_violated(
        self,
        *,
        node_id: str,
        atom_name: str,
        trust_tier: int | None = 1,
    ) -> None:
        """Raise ``Tier1ConfidenceEnvelopeViolation`` on the first failing port."""
        if self.in_confidence_bounds:
            return
        for port_name, pe in self.port_evaluations.items():
            if not pe.in_domain:
                raise Tier1ConfidenceEnvelopeViolation(
                    node_id=node_id,
                    atom_name=atom_name,
                    port_name=port_name,
                    score=float("inf"),
                    threshold=pe.envelope_check.threshold
                    if pe.envelope_check
                    else float("inf"),
                    p_value=0.0,
                    alpha=pe.envelope_check.alpha if pe.envelope_check else 0.0,
                    reason=pe.violation_reason or "hard domain support violation",
                    trust_tier=trust_tier,
                )
            if pe.envelope_check is not None and not pe.envelope_check.passed:
                ec = pe.envelope_check
                raise Tier1ConfidenceEnvelopeViolation(
                    node_id=node_id,
                    atom_name=atom_name,
                    port_name=port_name,
                    score=ec.score,
                    threshold=ec.threshold,
                    p_value=ec.p_value,
                    alpha=ec.alpha,
                    reason=ec.reason,
                    trust_tier=trust_tier,
                )
        raise Tier1ConfidenceEnvelopeViolation(
            node_id=node_id,
            atom_name=atom_name,
            port_name="<uncalibrated>",
            score=float("inf"),
            threshold=float("nan"),
            p_value=0.0,
            alpha=0.0,
            reason=self.violation_reason or "guarded atom lacks calibrated input regimes",
            trust_tier=trust_tier,
        )


@dataclass(frozen=True)
class AtomUncertaintyProfile:
    """Full v2 uncertainty characterization of an atom."""

    atom_name: str
    mode: str = "unknown"
    confidence: float = 0.0
    trust_tier: int = 3
    input_regimes: dict[str, PortRegimeSpec] = field(default_factory=dict)
    output_projectors: dict[str, ProjectorSpec] = field(default_factory=dict)
    components: tuple[ComponentChannel, ...] = ()
    legacy_scalar_factor: float | None = None
    n_trials: int = 0
    epsilon: float = 0.0
    input_regime_summary: str = ""
    notes: str = ""
    schema_version: str = "2.0"

    @property
    def n_components(self) -> int:
        if self.components:
            return len(self.components)
        for reg in self.input_regimes.values():
            return reg.mixture.n_components
        return 1

    @property
    def is_calibrated(self) -> bool:
        return self.mode != "unknown" and (
            bool(self.input_regimes)
            or bool(self.components)
            or self.legacy_scalar_factor is not None
        )

    def prior_weights(self) -> FloatArray:
        """Prior mixture weights across components (uniform if uncalibrated)."""
        k_count = self.n_components
        for reg in self.input_regimes.values():
            if reg.mixture.n_components == k_count:
                return reg.mixture.weights
        return np.full(k_count, 1.0 / float(max(1, k_count)), dtype=np.float64)

    def effective_scalar_factor(self) -> float | None:
        """Conservative scalar amplification factor for v1 callers."""
        if not self.components:
            return self.legacy_scalar_factor

        weights = self.prior_weights()
        max_per_comp: list[float] = []
        for idx, comp in enumerate(self.components):
            if not comp.operators:
                continue
            comp_norm = max(op.spectral_norm for op in comp.operators.values())
            w = float(weights[idx]) if idx < weights.size else 1.0
            if w > 0.0:
                max_per_comp.append(comp_norm)
        if max_per_comp:
            return float(max(max_per_comp))
        return self.legacy_scalar_factor

    def evaluate_inputs(
        self,
        inputs: dict[str, Any],
        *,
        alpha: float = 0.01,
        require_conformal_for_tier1: bool = False,
        require_conformal: bool | None = None,
        trust_tier: TrustTierFilter | object = _UNSET_TIER,
        require_all_calibrated_ports: bool = False,
    ) -> InputEvaluationReport:
        """Project runtime inputs, compute regime responsibilities, and check bounds.

        Args:
            inputs: Mapping from port name to runtime value.
            alpha: Node-level false-abort tolerance in ``(0, 1)``.
            require_conformal_for_tier1: Legacy alias for ``require_conformal=True``
                with ``trust_tier=1``.
            require_conformal: When True, fail closed if this atom matches
                ``trust_tier`` and lacks a calibrated ``ConformalEnvelope`` or
                sufficient calibration size ``N >= ceil(1 / port_alpha) - 1``.
            trust_tier: Target trust tier(s) for strict conformal enforcement.
                Defaults to ``1``; pass ``None`` to enforce across atoms of all tiers.
            require_all_calibrated_ports: Fail closed if any calibrated port in
                ``self.input_regimes`` is missing from ``inputs``.
        """
        port_evals: dict[str, PortEvaluation] = {}
        k_count = self.n_components
        log_resp_accum = np.log(self.prior_weights())
        has_shared_k = False
        all_ok = True
        top_reason = ""

        strict_conformal = (
            bool(require_conformal)
            if require_conformal is not None
            else bool(require_conformal_for_tier1)
        )
        eff_tier: TrustTierFilter = (
            1 if trust_tier is _UNSET_TIER else trust_tier  # type: ignore[assignment]
        )
        tier_enforced = strict_conformal and matches_trust_tier(
            self.trust_tier, eff_tier
        )

        if not self.input_regimes:
            if tier_enforced:
                all_ok = False
                top_reason = (
                    f"Tier {self.trust_tier} atom lacks calibrated input regimes"
                )
            probs = self.prior_weights()
            return InputEvaluationReport(
                port_evaluations={},
                combined_responsibilities=probs,
                active_component=int(np.argmax(probs)),
                in_confidence_bounds=all_ok,
                violation_reason=top_reason,
            )

        # Split port-level alpha across calibrated input ports cleanly
        port_alpha = alpha / float(max(1, len(self.input_regimes)))

        for port_name, reg in self.input_regimes.items():
            if port_name not in inputs:
                if require_all_calibrated_ports and tier_enforced:
                    all_ok = False
                    port_evals[port_name] = PortEvaluation(
                        port_name=port_name,
                        features=np.zeros(reg.projector_spec.dim, dtype=np.float64),
                        in_domain=False,
                        nonconformity_score=float("inf"),
                        bregman_divergence=float("inf"),
                        responsibilities=reg.mixture.weights,
                        envelope_check=None,
                        violation_reason=f"missing runtime value for calibrated port '{port_name}'",
                    )
                continue
            proj = reg.build_projector()
            proj_res = proj.project(inputs[port_name])
            if not proj_res.in_domain:
                all_ok = False
                resp = reg.mixture.weights
                port_evals[port_name] = PortEvaluation(
                    port_name=port_name,
                    features=proj_res.features,
                    in_domain=False,
                    nonconformity_score=float("inf"),
                    bregman_divergence=float("inf"),
                    responsibilities=resp,
                    envelope_check=None,
                    violation_reason=proj_res.violation_reason,
                )
                continue

            z = proj_res.features
            score = float(reg.mixture.nonconformity_score(z))
            breg = float(reg.mixture.min_bregman_divergence(z))
            resp = reg.mixture.responsibilities(z)

            if reg.mixture.n_components == k_count:
                comp_lp = reg.mixture.component_log_prob(z)
                if np.all(np.isfinite(comp_lp)):
                    if not has_shared_k:
                        log_resp_accum = np.log(reg.mixture.weights) + comp_lp
                        has_shared_k = True
                    else:
                        log_resp_accum = log_resp_accum + comp_lp

            env_check: EnvelopeCheckResult | None = None
            if reg.conformal_envelope is not None:
                env_check = reg.conformal_envelope.check(
                    score,
                    alpha=port_alpha,
                    require_sufficient_calibration=tier_enforced,
                )
                if not env_check.passed:
                    all_ok = False
            elif tier_enforced:
                all_ok = False
                env_check = EnvelopeCheckResult(
                    passed=False,
                    score=score,
                    threshold=float("nan"),
                    p_value=0.0,
                    alpha=port_alpha,
                    sufficient_calibration_size=False,
                    reason=(
                        f"Tier {self.trust_tier} port lacks a calibrated ConformalEnvelope"
                    ),
                )

            port_evals[port_name] = PortEvaluation(
                port_name=port_name,
                features=z,
                in_domain=True,
                nonconformity_score=score,
                bregman_divergence=breg,
                responsibilities=resp,
                envelope_check=env_check,
            )

        if has_shared_k:
            lse = float(logsumexp(log_resp_accum, axis=0))
            combined_resp = (
                np.exp(log_resp_accum - lse)
                if np.isfinite(lse)
                else self.prior_weights()
            )
        else:
            combined_resp = self.prior_weights()

        return InputEvaluationReport(
            port_evaluations=port_evals,
            combined_responsibilities=combined_resp,
            active_component=int(np.argmax(combined_resp)),
            in_confidence_bounds=all_ok,
            violation_reason=top_reason,
        )

    def pushforward_covariances(
        self,
        input_covariances: dict[str, FloatArray],
        *,
        responsibilities: FloatArray | None = None,
    ) -> dict[str, FloatArray]:
        """Push per-port input covariances forward to output port covariances.

        Computes ``Sigma_{y, q} = sum_k gamma_k (sum_p J_{k, p->q} Sigma_{x, p} J_{k, p->q}^T + Sigma_{eps, k, q})``.
        """
        if not self.components:
            factor = self.legacy_scalar_factor if self.legacy_scalar_factor is not None else 1.0
            return {
                port: symmetrize((factor ** 2) * np.asarray(cov, dtype=np.float64))
                for port, cov in input_covariances.items()
            }

        gamma = (
            np.asarray(responsibilities, dtype=np.float64)
            if responsibilities is not None
            else self.prior_weights()
        )
        out_ports: dict[str, int] = {}
        for comp in self.components:
            for (_in_p, out_p), op in comp.operators.items():
                out_ports[out_p] = op.dim_out
            for out_p, eps_c in comp.intrinsic_output_cov.items():
                out_ports.setdefault(out_p, eps_c.shape[0])

        results: dict[str, FloatArray] = {
            out_p: np.zeros((d_out, d_out), dtype=np.float64)
            for out_p, d_out in out_ports.items()
        }

        for idx, comp in enumerate(self.components):
            w_k = float(gamma[idx]) if idx < gamma.size else 0.0
            if w_k <= 0.0:
                continue
            comp_out: dict[str, FloatArray] = {
                out_p: np.zeros((d_out, d_out), dtype=np.float64)
                for out_p, d_out in out_ports.items()
            }
            for (in_p, out_p), op in comp.operators.items():
                if in_p not in input_covariances:
                    continue
                comp_out[out_p] = comp_out[out_p] + op.pushforward_covariance(
                    input_covariances[in_p]
                )
            for out_p, eps_c in comp.intrinsic_output_cov.items():
                comp_out[out_p] = comp_out[out_p] + eps_c

            for out_p, cov_k in comp_out.items():
                results[out_p] = symmetrize(results[out_p] + w_k * cov_k)

        return results

    def pullback_precisions(
        self,
        required_output_precisions: dict[str, FloatArray],
        *,
        responsibilities: FloatArray | None = None,
    ) -> tuple[dict[str, FloatArray], bool, list[str]]:
        """Pull required output precisions backward to input ports across components.

        Returns ``(required_input_precisions, all_feasible, violation_reasons)``.
        """
        if not self.components:
            factor = self.legacy_scalar_factor if self.legacy_scalar_factor is not None else 1.0
            return (
                {
                    port: symmetrize((factor ** 2) * np.asarray(prec, dtype=np.float64))
                    for port, prec in required_output_precisions.items()
                },
                True,
                [],
            )

        gamma = (
            np.asarray(responsibilities, dtype=np.float64)
            if responsibilities is not None
            else self.prior_weights()
        )
        in_ports: dict[str, int] = {}
        for comp in self.components:
            for (in_p, _out_p), op in comp.operators.items():
                in_ports[in_p] = op.dim_in

        req_in: dict[str, FloatArray] = {
            in_p: np.zeros((d_in, d_in), dtype=np.float64)
            for in_p, d_in in in_ports.items()
        }
        all_feasible = True
        reasons: list[str] = []

        for idx, comp in enumerate(self.components):
            w_k = float(gamma[idx]) if idx < gamma.size else 0.0
            if w_k <= 0.0:
                continue
            for (in_p, out_p), op in comp.operators.items():
                if out_p not in required_output_precisions:
                    continue
                pb: PullbackResult = op.pullback_precision(
                    required_output_precisions[out_p],
                    intrinsic_output_cov=comp.intrinsic_output_cov.get(out_p),
                )
                req_in[in_p] = symmetrize(
                    req_in[in_p] + w_k * pb.required_input_precision
                )
                if not pb.feasible:
                    all_feasible = False
                    reasons.append(
                        f"component {comp.component_index} ({in_p}->{out_p}): {pb.violation_reason}"
                    )

        return req_in, all_feasible, reasons

    # ------------------------------------------------------------------
    # v1 <-> v2 Compatibility & JSON Serialization
    # ------------------------------------------------------------------

    def to_v1_estimate(self) -> AtomUncertaintyEstimate:
        """Project this v2 profile onto a v1 ``AtomUncertaintyEstimate``."""
        from sciona.synthesizer.uncertainty import AtomUncertaintyEstimate

        regime_desc = self.input_regime_summary
        if not regime_desc and self.input_regimes:
            parts = [
                f"{p}:{r.mixture.family}(K={r.mixture.n_components},d={r.mixture.dim})"
                for p, r in sorted(self.input_regimes.items())
            ]
            regime_desc = ", ".join(parts)

        return AtomUncertaintyEstimate(
            mode=self.mode,
            scalar_factor=self.effective_scalar_factor(),
            confidence=self.confidence,
            n_trials=self.n_trials,
            epsilon=self.epsilon,
            input_regime=regime_desc,
            notes=self.notes,
        )

    @classmethod
    def from_v1_estimate(
        cls,
        atom_name: str,
        estimate: AtomUncertaintyEstimate,
        *,
        trust_tier: int = 3,
    ) -> AtomUncertaintyProfile:
        """Lift a legacy v1 ``AtomUncertaintyEstimate`` into an ``AtomUncertaintyProfile``."""
        components: tuple[ComponentChannel, ...] = ()
        if estimate.scalar_factor is not None:
            op = LowRankPlusTailOperator.from_scalar_factor(
                estimate.scalar_factor, dim_out=1, dim_in=1
            )
            components = (
                ComponentChannel(
                    component_index=0,
                    operators={("input", "output"): op},
                ),
            )
        return cls(
            atom_name=atom_name,
            mode=estimate.mode,
            confidence=estimate.confidence,
            trust_tier=trust_tier,
            components=components,
            legacy_scalar_factor=estimate.scalar_factor,
            n_trials=estimate.n_trials,
            epsilon=estimate.epsilon,
            input_regime_summary=estimate.input_regime,
            notes=estimate.notes,
            schema_version="2.0",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize as v2 ``uncertainty.json`` document preserving ``estimates`` for v1 readers."""
        v1 = self.to_v1_estimate()
        v1_entry = {
            "mode": v1.mode,
            "scalar_factor": v1.scalar_factor,
            "confidence": v1.confidence,
            "n_trials": v1.n_trials,
            "epsilon": v1.epsilon,
            "input_regime": v1.input_regime,
            "notes": v1.notes,
        }
        return {
            "schema_version": self.schema_version,
            "atom": self.atom_name,
            "mode": self.mode,
            "confidence": self.confidence,
            "trust_tier": self.trust_tier,
            "legacy_scalar_factor": self.legacy_scalar_factor,
            "n_trials": self.n_trials,
            "epsilon": self.epsilon,
            "input_regime_summary": self.input_regime_summary,
            "notes": self.notes,
            "input_regimes": {
                k: v.to_dict() for k, v in sorted(self.input_regimes.items())
            },
            "output_projectors": {
                k: v.to_dict() for k, v in sorted(self.output_projectors.items())
            },
            "components": [c.to_dict() for c in self.components],
            "estimates": [v1_entry] if self.mode != "unknown" else [],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AtomUncertaintyProfile:
        """Deserialize either a v2 profile payload or a legacy v1 ``uncertainty.json`` dict."""
        atom_name = str(payload.get("atom", ""))
        if str(payload.get("schema_version", "")).startswith("2") or "input_regimes" in payload:
            input_regimes = {
                str(k): PortRegimeSpec.from_dict(v)
                for k, v in payload.get("input_regimes", {}).items()
            }
            output_projectors = {
                str(k): ProjectorSpec.from_dict(v)
                for k, v in payload.get("output_projectors", {}).items()
            }
            components = tuple(
                ComponentChannel.from_dict(c) for c in payload.get("components", [])
            )
            raw_sf = payload.get("legacy_scalar_factor")
            return cls(
                atom_name=atom_name,
                mode=str(payload.get("mode", "unknown")),
                confidence=float(payload.get("confidence", 0.0)),
                trust_tier=int(payload.get("trust_tier", 3)),
                input_regimes=input_regimes,
                output_projectors=output_projectors,
                components=components,
                legacy_scalar_factor=float(raw_sf) if raw_sf is not None else None,
                n_trials=int(payload.get("n_trials", 0)),
                epsilon=float(payload.get("epsilon", 0.0)),
                input_regime_summary=str(payload.get("input_regime_summary", "")),
                notes=str(payload.get("notes", "")),
                schema_version=str(payload.get("schema_version", "2.0")),
            )

        # Legacy v1 uncertainty.json fallback
        from sciona.synthesizer.uncertainty import AtomUncertaintyEstimate

        estimates = payload.get("estimates", [])
        if not estimates:
            return cls(atom_name=atom_name, mode="unknown")

        best = max(estimates, key=lambda item: float(item.get("confidence", 0.0)))
        raw_sf = best.get("scalar_factor")
        v1 = AtomUncertaintyEstimate(
            mode=str(best.get("mode", "unknown")),
            scalar_factor=float(raw_sf) if raw_sf is not None else None,
            confidence=float(best.get("confidence", 0.0)),
            n_trials=int(best.get("n_trials", 0)),
            epsilon=float(best.get("epsilon", 0.0)),
            input_regime=str(best.get("input_regime", "")),
            notes=str(best.get("notes", "")),
        )
        return cls.from_v1_estimate(atom_name, v1)

    @classmethod
    def from_supabase_row(
        cls,
        atom_name: str,
        row: dict[str, Any],
        *,
        trust_tier: int = 3,
    ) -> AtomUncertaintyProfile:
        """Hydrate an ``AtomUncertaintyProfile`` from a ``public.atom_uncertainty_estimates`` row."""
        schema_ver = str(row.get("schema_version") or "1.0")
        raw_regimes = row.get("input_regimes")
        raw_comps = row.get("components")
        if schema_ver.startswith("2") or bool(raw_regimes) or bool(raw_comps):
            input_regimes = {
                str(k): PortRegimeSpec.from_dict(v)
                for k, v in dict(raw_regimes or {}).items()
            }
            output_projectors = {
                str(k): ProjectorSpec.from_dict(v)
                for k, v in dict(row.get("output_projectors") or {}).items()
            }
            components = tuple(
                ComponentChannel.from_dict(c) for c in list(raw_comps or [])
            )
            raw_sf = row.get("scalar_factor")
            return cls(
                atom_name=atom_name,
                mode=str(row.get("mode", "empirical")),
                confidence=float(row.get("confidence", 0.0)),
                trust_tier=trust_tier,
                input_regimes=input_regimes,
                output_projectors=output_projectors,
                components=components,
                legacy_scalar_factor=float(raw_sf) if raw_sf is not None else None,
                n_trials=int(row.get("n_trials", 0)),
                epsilon=float(row.get("epsilon", 0.0)),
                input_regime_summary=str(row.get("input_regime", "")),
                notes=str(row.get("notes", "")),
                schema_version="2.0",
            )

        from sciona.synthesizer.uncertainty import AtomUncertaintyEstimate

        raw_sf = row.get("scalar_factor")
        v1 = AtomUncertaintyEstimate(
            mode=str(row.get("mode", "empirical")),
            scalar_factor=float(raw_sf) if raw_sf is not None else None,
            confidence=float(row.get("confidence", 0.0)),
            n_trials=int(row.get("n_trials", 0)),
            epsilon=float(row.get("epsilon", 0.0)),
            input_regime=str(row.get("input_regime", "")),
            notes=str(row.get("notes", "")),
        )
        return cls.from_v1_estimate(atom_name, v1, trust_tier=trust_tier)


def load_uncertainty_profile_json(path: str | Path) -> AtomUncertaintyProfile:
    """Load a v1 or v2 ``uncertainty.json`` file into an ``AtomUncertaintyProfile``."""
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load uncertainty profile %s: %s", p, exc)
        return AtomUncertaintyProfile(atom_name="", mode="unknown")
    if not isinstance(data, dict):
        return AtomUncertaintyProfile(atom_name="", mode="unknown")
    return AtomUncertaintyProfile.from_dict(data)


def dump_uncertainty_profile_json(
    profile: AtomUncertaintyProfile,
    path: str | Path,
) -> Path:
    """Write ``profile`` to ``path`` in v2 JSON format (with v1 ``estimates`` compatibility)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    return p
