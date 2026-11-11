"""Uncertainty-aware atom candidate ranking & constructive CDG gap/bounty specs.

Supports two core synthesis decisions driven by ``CDGUncertaintyReport``:
1. **Candidate Selection**: Rank alternative atoms for a CDG slot by regime
   compatibility, adjoint pullback feasibility, and directional risk ratio
   ``lambda_max(Lambda_req^{1/2} Sigma_out Lambda_req^{1/2})``.
2. **Constructive Gap & Bounty Mining**: When no catalog atom satisfies a node's
   risk budget, derive explicit mathematical bounds on input mixture regimes,
   maximum Jacobian singular values, intrinsic noise floors ``Sigma_eps``, and
   conformal calibration sample sizes required for a newly developed atom.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sciona.architect.expansion_gap_mining import ExpansionGapCluster
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, NodeStatus
from sciona.uncertainty._math import spd_eigendecomp, symmetrize
from sciona.uncertainty.models import AtomUncertaintyProfile
from sciona.uncertainty.propagation import (
    CDGUncertaintyReport,
    NodeUncertaintyEvaluation,
    _adapt_precision_dim,
    _evaluate_belief_against_regime,
    _match_port_key,
    _pushforward_node_mixtures,
    compute_risk_ratio,
    resolve_atom_profile,
)

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import UncertaintyBackend

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CandidateUncertaintyScore:
    """Evaluation of a candidate atom against a CDG node's directional risk budget."""

    atom_name: str
    is_calibrated: bool
    regime_compatible: bool
    pullback_feasible: bool
    max_risk_ratio: float
    effective_scalar_factor: float
    fisher_trace: float
    confidence: float
    meets_budget: bool
    prior_reward: float
    """Bounded utility score in ``(0, 1]`` suitable for ``AtomLedger`` priors."""
    rationale: str = ""


def score_candidate_for_node(
    atom_name: str,
    profile: AtomUncertaintyProfile,
    node_eval: NodeUncertaintyEvaluation,
) -> CandidateUncertaintyScore:
    """Evaluate how well ``profile`` satisfies ``node_eval``'s regimes and precisions."""
    eff_factor = profile.effective_scalar_factor()
    eff_val = float(eff_factor) if eff_factor is not None else 1.0

    if not profile.is_calibrated:
        return CandidateUncertaintyScore(
            atom_name=atom_name,
            is_calibrated=False,
            regime_compatible=True,
            pullback_feasible=True,
            max_risk_ratio=float("inf"),
            effective_scalar_factor=eff_val,
            fisher_trace=0.0,
            confidence=0.0,
            meets_budget=False,
            prior_reward=0.25,
            rationale="uncalibrated candidate (mode='unknown')",
        )

    # 1. Check incoming belief compatibility against candidate's PortRegimeSpecs
    port_alpha = node_eval.allocated_alpha / float(max(1, len(profile.input_regimes)))
    k_cand = profile.n_components
    log_resp = np.log(np.maximum(profile.prior_weights(), 1e-12))
    regime_ok = True

    for pname, reg in profile.input_regimes.items():
        b_key = _match_port_key(pname, list(node_eval.incoming_mixtures.keys()))
        if b_key is None:
            continue
        p_resp, _breg, _env, p_ok = _evaluate_belief_against_regime(
            node_eval.incoming_mixtures[b_key],
            reg,
            port_alpha,
        )
        if not p_ok:
            regime_ok = False
        if p_resp.size == k_cand:
            log_resp = log_resp + np.log(np.maximum(p_resp, 1e-12))

    log_resp = log_resp - float(np.max(log_resp))
    resp = np.exp(log_resp)
    resp = resp / max(float(np.sum(resp)), 1e-15)

    # 2. Push node's incoming mixtures through candidate profile
    out_dims = {
        q: m.shape[0] for q, m in node_eval.required_output_precisions.items()
    }
    if not out_dims:
        out_dims = {"out": 1}
    pushed_out = _pushforward_node_mixtures(
        profile,
        node_eval.incoming_mixtures,
        resp,
        out_dims,
        max_components=4,
    )

    has_budget = any(
        float(np.max(np.abs(m))) > 1e-15
        for m in node_eval.required_output_precisions.values()
    )

    # 3. Check adjoint precision pullback feasibility on candidate
    pullback_ok = True
    if has_budget and profile.components:
        prof_out_ports = sorted(
            {out_p for comp in profile.components for (_in_p, out_p) in comp.operators}
            | {out_p for comp in profile.components for out_p in comp.intrinsic_output_cov}
        )
        mapped_out_prec: dict[str, FloatArray] = {}
        for q_node, lam_q in node_eval.required_output_precisions.items():
            if float(np.max(np.abs(lam_q))) <= 1e-15:
                continue
            q_prof = _match_port_key(q_node, prof_out_ports) or q_node
            target_d = lam_q.shape[0]
            for comp in profile.components:
                for (_ip, op_q), op in comp.operators.items():
                    if op_q == q_prof:
                        target_d = op.dim_out
                        break
            lam_ad = _adapt_precision_dim(lam_q, target_d)
            mapped_out_prec[q_prof] = symmetrize(
                mapped_out_prec.get(q_prof, np.zeros_like(lam_ad)) + lam_ad
            )
        _pulled, pullback_ok, _reasons = profile.pullback_precisions(
            mapped_out_prec,
            responsibilities=resp,
        )

    # 4. Compute directional risk ratio & Fisher observability trace
    ratios: list[float] = []
    for qname, out_mix in pushed_out.items():
        req_lam_q = node_eval.required_output_precisions.get(qname)
        if req_lam_q is not None and float(np.max(np.abs(req_lam_q))) > 1e-15:
            r_q, _ = compute_risk_ratio(out_mix.mixture_covariance(), req_lam_q)
            ratios.append(r_q)

    max_ratio = float(max(ratios)) if ratios else eff_val

    fisher_tr = 0.0
    for k_idx, comp in enumerate(profile.components):
        w_k = float(resp[k_idx]) if k_idx < resp.size else 0.0
        if w_k <= 1e-9:
            continue
        for (_in_p, out_p), op in comp.operators.items():
            eps_c = comp.intrinsic_output_cov.get(out_p)
            noise_prec = (
                np.linalg.inv(symmetrize(eps_c) + 1e-4 * np.eye(op.dim_out))
                if eps_c is not None
                else np.eye(op.dim_out)
            )
            fim = op.fisher_information(noise_prec)
            fisher_tr += w_k * float(np.trace(fim))

    meets = bool(
        regime_ok
        and pullback_ok
        and (max_ratio <= 1.0 + 1e-9 if has_budget else True)
    )

    # Compute smooth bounded utility reward in (0, 1]
    conf = max(0.0, min(1.0, float(profile.confidence)))
    if not regime_ok or not pullback_ok:
        reward = 0.02 + 0.15 * (1.0 / (1.0 + max(0.0, max_ratio)))
        why = "violates input regime" if not regime_ok else "intrinsic noise exceeds precision budget"
    elif has_budget and meets:
        margin_bonus = 1.0 / (1.0 + max_ratio)
        reward = 0.70 + 0.25 * margin_bonus + 0.05 * conf
        why = f"satisfies risk budget (risk_ratio={max_ratio:.3f} <= 1.0)"
    elif has_budget:
        reward = 0.35 + 0.30 * (1.0 / (1.0 + max_ratio)) + 0.04 * conf
        why = f"exceeds risk budget (risk_ratio={max_ratio:.3f} > 1.0)"
    else:
        gain_score = 1.0 / (1.0 + max(0.0, eff_val))
        reward = 0.45 + 0.40 * gain_score + 0.10 * conf
        why = f"calibrated amplification={eff_val:.3f}"

    return CandidateUncertaintyScore(
        atom_name=atom_name,
        is_calibrated=True,
        regime_compatible=regime_ok,
        pullback_feasible=pullback_ok,
        max_risk_ratio=max_ratio,
        effective_scalar_factor=eff_val,
        fisher_trace=fisher_tr,
        confidence=conf,
        meets_budget=meets,
        prior_reward=round(min(1.0, max(0.01, reward)), 6),
        rationale=why,
    )


def rank_candidates_by_uncertainty(
    candidate_names: Sequence[str],
    *,
    node_eval: NodeUncertaintyEvaluation | None = None,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
) -> list[CandidateUncertaintyScore]:
    """Score and sort candidate atoms from lowest directional risk to highest."""
    scores: list[CandidateUncertaintyScore] = []
    for cname in candidate_names:
        prof = resolve_atom_profile(cname, backend=backend, profiles=profiles)
        if node_eval is not None:
            scores.append(score_candidate_for_node(cname, prof, node_eval))
            continue

        # Standalone profile scoring when no CDG NodeUncertaintyEvaluation is provided
        eff_factor = prof.effective_scalar_factor()
        eff_val = float(eff_factor) if eff_factor is not None else 1.0
        if not prof.is_calibrated:
            scores.append(
                CandidateUncertaintyScore(
                    atom_name=cname,
                    is_calibrated=False,
                    regime_compatible=True,
                    pullback_feasible=True,
                    max_risk_ratio=float("inf"),
                    effective_scalar_factor=eff_val,
                    fisher_trace=0.0,
                    confidence=0.0,
                    meets_budget=False,
                    prior_reward=0.25,
                    rationale="uncalibrated candidate",
                )
            )
            continue

        conf = max(0.0, min(1.0, float(prof.confidence)))
        gain_term = 1.0 / (1.0 + max(0.0, eff_val))
        reward = round(min(1.0, max(0.05, 0.45 + 0.45 * gain_term + 0.10 * conf)), 6)
        scores.append(
            CandidateUncertaintyScore(
                atom_name=cname,
                is_calibrated=True,
                regime_compatible=True,
                pullback_feasible=True,
                max_risk_ratio=eff_val,
                effective_scalar_factor=eff_val,
                fisher_trace=0.0,
                confidence=conf,
                meets_budget=True,
                prior_reward=reward,
                rationale=f"standalone amplification={eff_val:.3f}",
            )
        )

    scores.sort(
        key=lambda s: (
            not s.meets_budget,
            not s.pullback_feasible,
            not s.regime_compatible,
            not s.is_calibrated,
            -s.prior_reward,
            s.atom_name,
        )
    )
    return scores


def uncertainty_priors_for_candidates(
    candidate_names: Sequence[str],
    *,
    node_eval: NodeUncertaintyEvaluation | None = None,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
) -> dict[str, float]:
    """Return ``{atom_name: prior_reward}`` in ``[0, 1]`` for ``AtomLedger``."""
    ranked = rank_candidates_by_uncertainty(
        candidate_names,
        node_eval=node_eval,
        backend=backend,
        profiles=profiles,
    )
    return {item.atom_name: item.prior_reward for item in ranked}


# ---------------------------------------------------------------------------
# Constructive Gap & Bounty Specification Synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyGapSpec:
    """Constructive mathematical specification for a missing low-risk atom."""

    node_id: str
    node_name: str
    concept_type: str
    current_atom: str
    failure_mode: str
    """One of: intrinsic_noise_floor | input_regime_uncovered | stiff_jacobian_amplification | uncalibrated_high_risk."""
    bottleneck_rank: int
    current_risk_ratio: float
    marginal_risk_score: float
    target_input_regimes: dict[str, dict[str, Any]]
    required_output_precisions: dict[str, list[list[float]]]
    max_intrinsic_noise_eigenvalues: dict[str, float]
    max_stiff_singular_value: float
    worst_output_directions: dict[str, list[float]]
    required_conformal_alpha: float
    min_calibration_trials: int
    evaluated_candidates: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "concept_type": self.concept_type,
            "current_atom": self.current_atom,
            "failure_mode": self.failure_mode,
            "bottleneck_rank": self.bottleneck_rank,
            "current_risk_ratio": self.current_risk_ratio,
            "marginal_risk_score": self.marginal_risk_score,
            "target_input_regimes": self.target_input_regimes,
            "required_output_precisions": self.required_output_precisions,
            "max_intrinsic_noise_eigenvalues": self.max_intrinsic_noise_eigenvalues,
            "max_stiff_singular_value": self.max_stiff_singular_value,
            "worst_output_directions": self.worst_output_directions,
            "required_conformal_alpha": self.required_conformal_alpha,
            "min_calibration_trials": self.min_calibration_trials,
            "evaluated_candidates": list(self.evaluated_candidates),
            "rationale": self.rationale,
        }

    def to_bounty_request_dict(
        self,
        *,
        escrow_amount: float = 500.0,
        tier: str = "standard",
        domain_tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Format as a ``BountyCreateRequest``-compatible payload with verifiable spec."""
        tags = list(dict.fromkeys([self.concept_type, "uncertainty_gap", *domain_tags]))
        return {
            "title": (
                f"Certify low-risk '{self.node_name}' ({self.concept_type}) atom "
                f"[sigma_max <= {self.max_stiff_singular_value:.4g}]"
            ),
            "escrow_amount": float(escrow_amount),
            "tier": tier,
            "domain_tags": tags,
            "config_yml": {
                "verification_kind": "uncertainty_profile_v2",
                "target_node_id": self.node_id,
                "concept_type": self.concept_type,
                "failure_mode": self.failure_mode,
                "acceptance_criteria": {
                    "schema_version": "2.0",
                    "max_stiff_singular_value": self.max_stiff_singular_value,
                    "max_intrinsic_noise_eigenvalues": self.max_intrinsic_noise_eigenvalues,
                    "required_output_precisions": self.required_output_precisions,
                    "target_input_regimes": self.target_input_regimes,
                    "required_conformal_alpha": self.required_conformal_alpha,
                    "min_calibration_trials": self.min_calibration_trials,
                },
                "rationale": self.rationale,
            },
        }

    def to_expansion_gap_cluster(self) -> ExpansionGapCluster:
        """Convert into an ``ExpansionGapCluster`` for unified architect gap reporting."""
        return ExpansionGapCluster(
            cluster_id=f"uncertainty_gap:{self.node_id}",
            representative_terms=(self.node_name, self.concept_type, self.failure_mode),
            support=1,
            competitions=(),
            families=(self.concept_type,),
            paradigms=(self.concept_type,),
            recommended_action="develop_certified_atom",
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class UncertaintyGapReport:
    """Collection of constructive atom gap specifications mined from a CDG."""

    gaps: tuple[UncertaintyGapSpec, ...] = ()
    rescued_by_existing_candidates: dict[str, str] = field(default_factory=dict)
    """Map ``node_id -> existing_candidate_atom_name`` where an existing atom satisfies budget."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_count": len(self.gaps),
            "rescued_by_existing_candidates": dict(self.rescued_by_existing_candidates),
            "gaps": [g.to_dict() for g in self.gaps],
        }


def mine_cdg_uncertainty_gaps(
    cdg: CDGExport,
    report: CDGUncertaintyReport,
    *,
    candidates_by_node: Mapping[str, Sequence[str]] | None = None,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
) -> UncertaintyGapReport:
    """Identify high-risk CDG nodes where no existing atom meets the risk budget.

    If an alternative candidate in ``candidates_by_node[nid]`` satisfies the node's
    regime and precision budget, records it in ``rescued_by_existing_candidates``;
    otherwise synthesizes a constructive ``UncertaintyGapSpec``.
    """
    node_map: dict[str, AlgorithmicNode] = {
        n.node_id: n for n in cdg.nodes if n.status == NodeStatus.ATOMIC
    }
    cand_pool = dict(candidates_by_node or {})
    rescued: dict[str, str] = {}
    gaps: list[UncertaintyGapSpec] = []

    bottlenecks = report.ranked_bottlenecks()
    for rank_idx, neval in enumerate(bottlenecks, start=1):
        nid = neval.node_id
        has_budget = any(
            float(np.max(np.abs(m))) > 1e-15
            for m in neval.required_output_precisions.values()
        )
        is_uncalibrated_risky = (not neval.profile.is_calibrated) and has_budget
        needs_intervention = (not neval.meets_budget) or is_uncalibrated_risky
        if not needs_intervention:
            continue

        # Check whether any existing candidate in the catalog already solves this slot
        slot_cands = list(cand_pool.get(nid, ()))
        scored_cands = (
            rank_candidates_by_uncertainty(
                slot_cands,
                node_eval=neval,
                backend=backend,
                profiles=profiles,
            )
            if slot_cands
            else []
        )
        passing = [c for c in scored_cands if c.meets_budget and c.is_calibrated]
        if passing:
            rescued[nid] = passing[0].atom_name
            continue

        # Determine mathematical root cause
        if not neval.profile.is_calibrated:
            failure_mode = "uncalibrated_high_risk"
        elif not neval.pullback_feasible:
            failure_mode = "intrinsic_noise_floor"
        elif not neval.in_regime_bounds:
            failure_mode = "input_regime_uncovered"
        else:
            failure_mode = "stiff_jacobian_amplification"

        # Derive constructive Loewner & spectral bounds
        max_noise_eigs: dict[str, float] = {}
        max_lam_req = 0.0
        for qname, lam_q in neval.required_output_precisions.items():
            eig_q, _, _ = spd_eigendecomp(lam_q)
            top_lam = float(np.max(eig_q)) if eig_q.size > 0 else 0.0
            max_lam_req = max(max_lam_req, top_lam)
            if top_lam > 1e-15:
                # Reserve <= 50% of variance budget (1 / top_lam) for intrinsic noise Sigma_eps
                max_noise_eigs[qname] = 0.5 / top_lam
            else:
                max_noise_eigs[qname] = float("inf")

        max_sigma_in = max(
            (
                float(np.max(spd_eigendecomp(m.mixture_covariance())[0]))
                for m in neval.incoming_mixtures.values()
            ),
            default=0.25,
        )
        if max_lam_req > 1e-15 and max_sigma_in > 1e-15:
            # Ensure J Sigma_in J^T <= 0.5 * Lambda_req^{-1} => sigma_max(J) <= sqrt(0.5 / (lam_max * sig_in_max))
            target_sigma_max = math.sqrt(0.5 / (max_lam_req * max_sigma_in))
        else:
            cur_sf = neval.profile.effective_scalar_factor() or 1.0
            target_sigma_max = 0.5 * float(cur_sf)

        n_ports = max(1, len(neval.incoming_mixtures))
        port_alpha = max(min(neval.allocated_alpha / float(n_ports), 0.5), 1e-6)
        min_n_cal = max(20, int(math.ceil(1.0 / port_alpha)) - 1)

        node_obj = node_map.get(nid)
        concept_str = (
            node_obj.concept_type.value
            if node_obj is not None and hasattr(node_obj.concept_type, "value")
            else str(getattr(node_obj, "concept_type", "custom"))
        )

        rationale = (
            f"Node '{neval.node_name}' (atom='{neval.atom_name}') failed CDG risk budget "
            f"via {failure_mode} (risk_ratio={neval.max_risk_ratio:.3f}); "
            f"requires sigma_max(J) <= {target_sigma_max:.4g} and N_cal >= {min_n_cal} "
            f"at alpha_port={port_alpha:.4g}."
        )

        gaps.append(
            UncertaintyGapSpec(
                node_id=nid,
                node_name=neval.node_name,
                concept_type=concept_str,
                current_atom=neval.atom_name,
                failure_mode=failure_mode,
                bottleneck_rank=rank_idx,
                current_risk_ratio=neval.max_risk_ratio,
                marginal_risk_score=neval.marginal_risk_score,
                target_input_regimes={
                    p: m.to_dict() for p, m in sorted(neval.incoming_mixtures.items())
                },
                required_output_precisions={
                    q: np.asarray(lam, dtype=np.float64).tolist()
                    for q, lam in sorted(neval.required_output_precisions.items())
                },
                max_intrinsic_noise_eigenvalues=max_noise_eigs,
                max_stiff_singular_value=target_sigma_max,
                worst_output_directions={
                    q: np.asarray(d, dtype=np.float64).tolist()
                    for q, d in sorted(neval.worst_output_directions.items())
                },
                required_conformal_alpha=port_alpha,
                min_calibration_trials=min_n_cal,
                evaluated_candidates=tuple(c.atom_name for c in scored_cands),
                rationale=rationale,
            )
        )

    return UncertaintyGapReport(
        gaps=tuple(gaps),
        rescued_by_existing_candidates=rescued,
    )
