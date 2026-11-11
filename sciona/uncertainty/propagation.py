"""Forward Mixture Pushforward & Adjoint Precision Pullback across CDGs (Phase 3).

Chains calibrated ``AtomUncertaintyProfile`` instances across a ``CDGExport``:
1. **Forward Pass**: Projects root source distributions, evaluates per-node regime
   responsibilities & conformal bounds (with DAG alpha-spending), pushes Gaussian
   mixture covariances through ``LowRankPlusTailOperator`` channels, and compresses
   multi-branch mixtures via Runnalls ``kl_reduce``.
2. **Backward Adjoint Pass**: Pulls sink ``RiskBudget`` precision matrices backward
   in reverse topological order via ``LowRankPlusTailOperator.pullback_precision``,
   identifying intrinsic-noise infeasibility and per-node marginal risk attribution.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    EdgeKind,
    IOSpec,
    NodeStatus,
)
from sciona.synthesizer.toposort import detect_cycle_partition, toposort_nodes
from sciona.uncertainty._math import spd_eigendecomp, symmetrize
from sciona.uncertainty.conformal import EnvelopeCheckResult, allocate_dag_alpha_budget
from sciona.uncertainty.distributions import (
    ExponentialFamilyMixture,
    GaussianMixtureRegime,
)
from sciona.uncertainty.models import AtomUncertaintyProfile, PortRegimeSpec
from sciona.uncertainty.operators import SpectralSummary

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import UncertaintyBackend
    from sciona.types import MatchResult

FloatArray = NDArray[np.float64]

_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("signal", "conditioned_signal", "filtered", "filtered_signal", "waveform", "x", "input"),
    ("events", "rpeaks", "peaks", "beats", "onsets", "event_counts", "counts"),
    ("rate", "heart_rate", "bpm"),
    ("out", "output", "result", "y"),
)


def _aliases_for(name: str) -> tuple[str, ...]:
    low = name.strip().lower()
    for grp in _ALIAS_GROUPS:
        if low in grp:
            return grp
    return (low,)


def _match_port_key(requested: str, available: Sequence[str]) -> str | None:
    if not available:
        return None
    if requested in available:
        return requested
    low_map = {k.lower(): k for k in available}
    if requested.lower() in low_map:
        return low_map[requested.lower()]
    for alias in _aliases_for(requested):
        if alias in low_map:
            return low_map[alias]
    if len(available) == 1:
        return available[0]
    return None


def _parse_constraint_variance(constraints: str, type_desc: str, default_var: float) -> float:
    m = re.search(r"error_bounds\s*=\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", constraints)
    if m:
        try:
            lo = float(m.group(1))
            hi = float(m.group(2))
            half_width = 0.5 * max(0.0, hi - lo)
            if half_width > 0.0:
                return half_width * half_width
        except ValueError:
            pass

    low = type_desc.strip().lower()
    if any(
        tok in low
        for tok in ("ndarray", "float", "double", "int", "list[float", "tuple[float")
    ):
        return max(default_var, 1e-12)
    return 0.0


def _mixture_to_gaussian_belief(
    mix: ExponentialFamilyMixture,
    *,
    factor_rank: int = 3,
    min_diag: float = 1e-8,
) -> GaussianMixtureRegime:
    """Moment-match any ``ExponentialFamilyMixture`` into a ``GaussianMixtureRegime``."""
    if isinstance(mix, GaussianMixtureRegime):
        return mix
    k_count = mix.n_components
    dim = mix.dim
    means = np.stack([mix.component_mean(k) for k in range(k_count)], axis=0)
    covs = np.stack(
        [
            symmetrize(mix.component_covariance(k)) + min_diag * np.eye(dim)
            for k in range(k_count)
        ],
        axis=0,
    )
    return GaussianMixtureRegime.from_full_covariances(
        weights=mix.weights,
        means=means,
        covariances=covs,
        factor_rank=min(factor_rank, dim),
        min_diag=min_diag,
    )


def _adapt_gaussian_dim(
    belief: GaussianMixtureRegime,
    target_dim: int,
    *,
    prior_mean: FloatArray | None = None,
    factor_rank: int = 3,
) -> GaussianMixtureRegime:
    """Adapt a Gaussian mixture belief across mismatched port feature dimensions."""
    if belief.dim == target_dim:
        return belief

    k_count = belief.n_components
    new_means = np.zeros((k_count, target_dim), dtype=np.float64)
    if prior_mean is not None and prior_mean.size == target_dim:
        new_means[:] = prior_mean[None, :]

    new_covs = np.empty((k_count, target_dim, target_dim), dtype=np.float64)
    for k in range(k_count):
        cov_k = belief.component_covariance(k)
        avg_var = max(float(np.trace(cov_k)) / float(max(1, belief.dim)), 1e-9)
        if target_dim == 1:
            new_means[k, 0] = float(np.mean(belief.component_mean(k)))
            new_covs[k] = np.array([[avg_var]], dtype=np.float64)
        else:
            shared = min(belief.dim, target_dim)
            if prior_mean is None:
                new_means[k, :shared] = belief.component_mean(k)[:shared]
            cov_target = avg_var * np.eye(target_dim, dtype=np.float64)
            if belief.dim > 1:
                cov_target[:shared, :shared] = symmetrize(cov_k[:shared, :shared])
            new_covs[k] = symmetrize(cov_target) + 1e-9 * np.eye(target_dim)

    return GaussianMixtureRegime.from_full_covariances(
        weights=belief.weights,
        means=new_means,
        covariances=new_covs,
        factor_rank=min(factor_rank, target_dim),
        min_diag=1e-9,
    )


def _adapt_precision_dim(prec: FloatArray, target_dim: int) -> FloatArray:
    """Adapt a required precision matrix across port projection dimensions."""
    arr = symmetrize(np.asarray(prec, dtype=np.float64))
    src_dim = arr.shape[0]
    if src_dim == target_dim:
        return arr
    avg_prec = max(float(np.trace(arr)) / float(max(1, src_dim)), 0.0)
    if target_dim == 1:
        return np.array([[avg_prec]], dtype=np.float64)
    out = avg_prec * np.eye(target_dim, dtype=np.float64)
    if src_dim > 1:
        shared = min(src_dim, target_dim)
        out[:shared, :shared] = symmetrize(arr[:shared, :shared])
    return symmetrize(out)


def _as_precision_matrix(spec: FloatArray | float, dim: int) -> FloatArray:
    if isinstance(spec, (int, float, np.floating, np.integer)):
        val = max(float(spec), 0.0)
        return val * np.eye(dim, dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 0:
        return max(float(arr), 0.0) * np.eye(dim, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == dim:
            return np.diag(np.maximum(arr, 0.0))
        avg = max(float(np.mean(arr)), 0.0)
        return avg * np.eye(dim, dtype=np.float64)
    return _adapt_precision_dim(arr, dim)


def compute_risk_ratio(
    covariance: FloatArray,
    required_precision: FloatArray,
) -> tuple[float, FloatArray]:
    """Compute directional risk ratio ``lambda_max(Lambda^{1/2} Sigma Lambda^{1/2})``.

    Returns ``(risk_ratio, worst_unit_direction)``.  ``risk_ratio <= 1.0`` iff
    ``Sigma <= Lambda^{-1}`` in the Loewner positive semi-definite order.
    """
    cov = symmetrize(np.asarray(covariance, dtype=np.float64))
    prec = symmetrize(np.asarray(required_precision, dtype=np.float64))
    dim = cov.shape[0]
    p_vals, p_vecs, _ = spd_eigendecomp(prec)
    if float(np.max(p_vals)) <= 1e-15:
        e0 = np.zeros(dim, dtype=np.float64)
        e0[0] = 1.0
        return 0.0, e0

    sqrt_prec = (p_vecs * np.sqrt(np.maximum(p_vals, 0.0))[None, :]) @ p_vecs.T
    whitened = symmetrize(sqrt_prec @ cov @ sqrt_prec)
    w_vals, w_vecs = np.linalg.eigh(whitened)
    top_idx = int(np.argmax(w_vals))
    ratio = max(0.0, float(w_vals[top_idx]))
    # Map worst whitened eigenvector back to port coordinates and normalize
    raw_dir = p_vecs @ (np.sqrt(np.maximum(p_vals, 0.0)) * w_vecs[:, top_idx])
    norm = float(np.linalg.norm(raw_dir))
    unit_dir = raw_dir / norm if norm > 1e-15 else w_vecs[:, top_idx]
    return ratio, np.asarray(unit_dir, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public Configuration & Report Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBudget:
    """End-to-end risk and confidence tolerance specification for a CDG."""

    sink_precisions: Mapping[str, FloatArray | float] = field(default_factory=dict)
    """Required precision matrices keyed by ``"node_id.port"``, ``"port"``, or ``"*"``."""
    max_output_variance: float | None = None
    """Uniform variance ceiling ``sigma_max^2`` on sink outputs (sets precision ``1/sigma_max^2``)."""
    global_alpha: float = 0.01
    """Family-wise CDG false-abort / regime miscoverage budget ``alpha`` in ``(0, 1)``."""
    max_mixture_components: int = 4
    """Maximum mixture components carried along edges via Runnalls ``kl_reduce``."""
    default_source_variance: float = 0.25
    """Fallback input variance on unconstrained root inputs when no regime is provided."""

    def resolve_sink_precision(
        self,
        node_id: str,
        port_name: str,
        dim: int,
    ) -> FloatArray | None:
        qualified = f"{node_id}.{port_name}"
        if qualified in self.sink_precisions:
            return _as_precision_matrix(self.sink_precisions[qualified], dim)
        if node_id in self.sink_precisions:
            return _as_precision_matrix(self.sink_precisions[node_id], dim)
        if port_name in self.sink_precisions:
            return _as_precision_matrix(self.sink_precisions[port_name], dim)
        if "*" in self.sink_precisions:
            return _as_precision_matrix(self.sink_precisions["*"], dim)
        if self.max_output_variance is not None and self.max_output_variance > 0.0:
            return (1.0 / float(self.max_output_variance)) * np.eye(dim, dtype=np.float64)
        return None


@dataclass(frozen=True)
class NodeUncertaintyEvaluation:
    """Forward pushforward & backward precision pullback diagnostics for one CDG node."""

    node_id: str
    node_name: str
    atom_name: str
    profile: AtomUncertaintyProfile
    allocated_alpha: float
    responsibilities: FloatArray
    active_component: int
    incoming_mixtures: dict[str, GaussianMixtureRegime]
    outgoing_mixtures: dict[str, GaussianMixtureRegime]
    required_output_precisions: dict[str, FloatArray]
    required_input_precisions: dict[str, FloatArray]
    output_risk_ratios: dict[str, float]
    worst_output_directions: dict[str, FloatArray]
    spectral_summaries: dict[tuple[str, str], SpectralSummary]
    envelope_checks: dict[str, EnvelopeCheckResult]
    regime_bregman_divergences: dict[str, float]
    in_regime_bounds: bool
    pullback_feasible: bool
    pullback_violation_reasons: tuple[str, ...]
    scalar_width_expansion: float
    marginal_risk_score: float

    @property
    def max_risk_ratio(self) -> float:
        if not self.output_risk_ratios:
            return 0.0
        return float(max(self.output_risk_ratios.values()))

    @property
    def meets_budget(self) -> bool:
        return (
            self.in_regime_bounds
            and self.pullback_feasible
            and self.max_risk_ratio <= 1.0 + 1e-9
        )


@dataclass(frozen=True)
class CDGUncertaintyReport:
    """Complete CDG uncertainty pushforward and adjoint pullback report."""

    nodes: dict[str, NodeUncertaintyEvaluation]
    topological_order: tuple[str, ...]
    sink_node_ids: tuple[str, ...]
    alpha_allocation: dict[str, float]
    risk_budget: RiskBudget | None = None
    uncalibrated_nodes: tuple[str, ...] = ()
    infeasible_pullback_nodes: tuple[str, ...] = ()
    regime_violated_nodes: tuple[str, ...] = ()
    budget_violated_nodes: tuple[str, ...] = ()

    @property
    def meets_budget(self) -> bool:
        if self.risk_budget is None:
            return not self.regime_violated_nodes and not self.infeasible_pullback_nodes
        return (
            not self.infeasible_pullback_nodes
            and not self.regime_violated_nodes
            and not self.budget_violated_nodes
        )

    @property
    def max_sink_risk_ratio(self) -> float:
        ratios = [
            self.nodes[nid].max_risk_ratio
            for nid in self.sink_node_ids
            if nid in self.nodes
        ]
        return float(max(ratios)) if ratios else 0.0

    def ranked_bottlenecks(self) -> list[NodeUncertaintyEvaluation]:
        """Return atomic node evaluations ordered by descending marginal risk."""
        evals = list(self.nodes.values())
        evals.sort(key=lambda e: (-e.marginal_risk_score, e.node_id))
        return evals


# ---------------------------------------------------------------------------
# Profile Resolution Helper
# ---------------------------------------------------------------------------


def resolve_atom_profile(
    atom_name: str,
    *,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
) -> AtomUncertaintyProfile:
    """Resolve ``atom_name`` to an ``AtomUncertaintyProfile`` across v2/v1 sources."""
    short = atom_name.rsplit(".", 1)[-1]
    if profiles is not None:
        if atom_name in profiles:
            return profiles[atom_name]
        if short in profiles:
            return profiles[short]

    if backend is not None:
        est_prof_fn = getattr(backend, "estimate_profile", None)
        if callable(est_prof_fn):
            prof = est_prof_fn(atom_name)
            if isinstance(prof, AtomUncertaintyProfile) and prof.mode != "unknown":
                return prof
        est = backend.estimate(atom_name)
        if est.mode != "unknown":
            return AtomUncertaintyProfile.from_v1_estimate(short, est)

    return AtomUncertaintyProfile(atom_name=short, mode="unknown")


# ---------------------------------------------------------------------------
# Core Forward + Backward Propagation Engine
# ---------------------------------------------------------------------------


def _infer_port_dims(
    node: AlgorithmicNode,
    profile: AtomUncertaintyProfile,
) -> tuple[dict[str, int], dict[str, int]]:
    """Determine projected dimensions for input and output ports of ``node``."""
    in_dims: dict[str, int] = {}
    out_dims: dict[str, int] = {}

    for pname, reg in profile.input_regimes.items():
        in_dims[pname] = reg.mixture.dim
    for qname, pspec in profile.output_projectors.items():
        out_dims[qname] = pspec.dim
    for comp in profile.components:
        for (in_p, out_p), op in comp.operators.items():
            in_dims.setdefault(in_p, op.dim_in)
            out_dims.setdefault(out_p, op.dim_out)
        for out_p, cov in comp.intrinsic_output_cov.items():
            out_dims.setdefault(out_p, int(cov.shape[0]))

    if not in_dims:
        if node.inputs:
            for io_in in node.inputs:
                in_dims[io_in.name] = 1
        else:
            in_dims["input"] = 1

    if not out_dims:
        if node.outputs:
            for io_out in node.outputs:
                out_dims[io_out.name] = 1
        else:
            out_dims["output"] = 1

    return in_dims, out_dims


def _evaluate_belief_against_regime(
    belief: GaussianMixtureRegime,
    reg: PortRegimeSpec,
    port_alpha: float,
) -> tuple[FloatArray, float, EnvelopeCheckResult | None, bool]:
    """Evaluate an incoming Gaussian mixture belief against ``PortRegimeSpec``."""
    mix = reg.mixture
    adapted = _adapt_gaussian_dim(
        belief,
        mix.dim,
        prior_mean=mix.mixture_mean(),
    )
    means = adapted.means  # (C, d)
    weights = adapted.weights

    # Expected regime responsibilities over incoming mixture modes
    resp_c = mix.responsibilities(means)
    if resp_c.ndim == 1:
        resp_c = resp_c.reshape(1, -1)
    expected_resp = np.asarray(weights @ resp_c, dtype=np.float64)
    expected_resp = expected_resp / max(float(np.sum(expected_resp)), 1e-15)

    # Expected Bregman divergence across incoming component means
    breg_vals = np.asarray(mix.min_bregman_divergence(means), dtype=np.float64).reshape(-1)
    finite_b = np.isfinite(breg_vals)
    expected_breg = (
        float(weights[finite_b] @ breg_vals[finite_b])
        if np.all(finite_b)
        else float("inf")
    )

    # Conformal check on the marginal incoming mean (and worst heavily-weighted mode)
    env_check: EnvelopeCheckResult | None = None
    in_bounds = bool(np.all(mix.in_support(means)))
    if reg.conformal_envelope is not None:
        scores = np.asarray(mix.nonconformity_score(means), dtype=np.float64).reshape(-1)
        heavy_mask = weights >= 0.05
        worst_score = float(np.max(scores[heavy_mask])) if np.any(heavy_mask) else float(scores[0])
        env_check = reg.conformal_envelope.check(
            worst_score,
            alpha=max(min(port_alpha, 0.99), 1e-9),
            require_sufficient_calibration=(
                False  # Runtime Tier 1 gate enforces strict N_cal; design-time checks quantile if finite
            ),
        )
        if math.isfinite(env_check.threshold) and not env_check.passed:
            in_bounds = False

    return expected_resp, expected_breg, env_check, in_bounds


def _pushforward_node_mixtures(
    profile: AtomUncertaintyProfile,
    incoming_beliefs: Mapping[str, GaussianMixtureRegime],
    responsibilities: FloatArray,
    out_dims: Mapping[str, int],
    *,
    max_components: int = 4,
    iteration_scale: float = 1.0,
) -> dict[str, GaussianMixtureRegime]:
    """Push per-port Gaussian mixtures through ``profile`` and reduce via ``kl_reduce``."""
    var_scale = max(iteration_scale, 1.0)

    if not profile.components:
        factor = profile.legacy_scalar_factor if profile.legacy_scalar_factor is not None else 1.0
        first_in = next(iter(incoming_beliefs.values()))
        out_map: dict[str, GaussianMixtureRegime] = {}
        for qname, d_out in out_dims.items():
            adapted = _adapt_gaussian_dim(first_in, d_out)
            k_in = adapted.n_components
            scaled_covs = np.stack(
                [
                    symmetrize(var_scale * (factor * factor) * adapted.component_covariance(c))
                    + 1e-12 * np.eye(d_out)
                    for c in range(k_in)
                ],
                axis=0,
            )
            out_map[qname] = GaussianMixtureRegime.from_full_covariances(
                weights=adapted.weights,
                means=factor * adapted.means,
                covariances=scaled_covs,
                factor_rank=min(3, d_out),
                min_diag=1e-12,
            ).kl_reduce(max_components=max_components)
        return out_map

    # Identify primary input port for multi-mode branching; other ports contribute marginal moments
    in_names = list(incoming_beliefs.keys())
    primary_p = in_names[0]
    primary_belief = incoming_beliefs[primary_p]
    other_means = {p: incoming_beliefs[p].mixture_mean() for p in in_names[1:]}
    other_covs = {p: incoming_beliefs[p].mixture_covariance() for p in in_names[1:]}

    results: dict[str, GaussianMixtureRegime] = {}
    for qname, d_out in out_dims.items():
        branch_weights: list[float] = []
        branch_means: list[FloatArray] = []
        branch_covs: list[FloatArray] = []

        for c_idx in range(primary_belief.n_components):
            w_c = float(primary_belief.weights[c_idx])
            mu_p0 = primary_belief.component_mean(c_idx)
            cov_p0 = primary_belief.component_covariance(c_idx)

            for k_idx, comp in enumerate(profile.components):
                g_k = float(responsibilities[k_idx]) if k_idx < responsibilities.size else 0.0
                w_ck = w_c * g_k
                if w_ck <= 1e-9:
                    continue

                mu_q = np.zeros(d_out, dtype=np.float64)
                cov_q = np.zeros((d_out, d_out), dtype=np.float64)
                matched_any_op = False

                for (op_in, op_out), op in comp.operators.items():
                    if _match_port_key(op_out, [qname]) != qname and _match_port_key(qname, [op_out]) != op_out:
                        continue
                    matched_in = _match_port_key(op_in, in_names)
                    if matched_in is None:
                        continue
                    matched_any_op = True
                    if matched_in == primary_p:
                        src_mu = mu_p0
                        src_cov = cov_p0
                    else:
                        src_mu = other_means[matched_in]
                        src_cov = other_covs[matched_in]

                    if src_cov.shape[0] != op.dim_in:
                        adapted_tmp = _adapt_gaussian_dim(
                            GaussianMixtureRegime.from_full_covariances(
                                weights=[1.0],
                                means=src_mu.reshape(1, -1),
                                covariances=src_cov.reshape(1, src_cov.shape[0], src_cov.shape[1]),
                                factor_rank=min(3, src_cov.shape[0]),
                                min_diag=1e-12,
                            ),
                            op.dim_in,
                        )
                        src_mu = adapted_tmp.component_mean(0)
                        src_cov = adapted_tmp.component_covariance(0)

                    pushed = op.pushforward_covariance(src_cov)
                    if pushed.shape[0] != d_out:
                        pushed = _adapt_precision_dim(pushed, d_out)
                    cov_q = cov_q + pushed

                    # Linear mean pushforward using explicit low-rank stiff singular directions
                    if op.rank > 0:
                        vt_mu = op.right_vectors.T @ src_mu
                        pushed_mu = op.left_vectors @ (op.singular_values * vt_mu)
                        if pushed_mu.size == d_out:
                            mu_q = mu_q + pushed_mu

                for eps_port, eps_cov in comp.intrinsic_output_cov.items():
                    if _match_port_key(eps_port, [qname]) == qname or _match_port_key(qname, [eps_port]) == eps_port:
                        eps_adapted = (
                            eps_cov
                            if eps_cov.shape[0] == d_out
                            else _adapt_precision_dim(eps_cov, d_out)
                        )
                        cov_q = cov_q + eps_adapted
                        matched_any_op = True

                if not matched_any_op:
                    factor = profile.effective_scalar_factor() or 1.0
                    adapted_fb = _adapt_gaussian_dim(primary_belief, d_out)
                    cov_q = (factor * factor) * adapted_fb.component_covariance(c_idx)

                cov_q = symmetrize(var_scale * cov_q) + 1e-12 * np.eye(d_out)
                branch_weights.append(w_ck)
                branch_means.append(mu_q)
                branch_covs.append(cov_q)

        if not branch_weights:
            branch_weights = [1.0]
            branch_means = [np.zeros(d_out, dtype=np.float64)]
            branch_covs = [1e-6 * np.eye(d_out, dtype=np.float64)]

        raw_mix = GaussianMixtureRegime.from_full_covariances(
            weights=np.asarray(branch_weights, dtype=np.float64),
            means=np.stack(branch_means, axis=0),
            covariances=np.stack(branch_covs, axis=0),
            factor_rank=min(3, d_out),
            min_diag=1e-12,
        )
        results[qname] = raw_mix.kl_reduce(max_components=max_components)

    return results


def propagate_cdg_uncertainty(
    cdg: CDGExport,
    *,
    match_results: Sequence[MatchResult] | None = None,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
    risk_budget: RiskBudget | None = None,
    source_regimes: Mapping[str, ExponentialFamilyMixture] | None = None,
    iterations_used: int = 0,
) -> CDGUncertaintyReport:
    """Run forward mixture pushforward and backward precision pullback across ``cdg``."""
    node_map: dict[str, AlgorithmicNode] = {n.node_id: n for n in cdg.nodes}
    atomic_ids = [n.node_id for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
    atomic_set = set(atomic_ids)

    if not atomic_ids:
        return CDGUncertaintyReport(
            nodes={},
            topological_order=(),
            sink_node_ids=(),
            alpha_allocation={},
            risk_budget=risk_budget,
        )

    # Topological order (with cycle fallback for FIXED_POINT / MESSAGE_PASSING)
    try:
        sorted_all = toposort_nodes(cdg.nodes, cdg.edges)
    except ValueError:
        acyclic_sorted, cycle_ids, _valid = detect_cycle_partition(cdg.nodes, cdg.edges)
        sorted_all = list(acyclic_sorted) + sorted(cycle_ids)

    topo_atomic = [nid for nid in sorted_all if nid in atomic_set]

    # Identify FIXED_POINT body nodes
    fp_child_ids: set[str] = set()
    for n in cdg.nodes:
        if n.concept_type == ConceptType.FIXED_POINT and n.children:
            fp_child_ids.update(n.children)

    # Match map from optional MatchResults
    match_map: dict[str, MatchResult] = {}
    if match_results:
        for mr in match_results:
            match_map[mr.pdg_node.predicate_id] = mr

    # Data-flow edge indices among atomic nodes
    incoming_edges: dict[str, list[DependencyEdge]] = {nid: [] for nid in atomic_set}
    outgoing_edges: dict[str, list[DependencyEdge]] = {nid: [] for nid in atomic_set}
    for edge in cdg.edges:
        if edge.edge_kind != EdgeKind.DATA_FLOW:
            continue
        if edge.target_id in incoming_edges:
            incoming_edges[edge.target_id].append(edge)
        if edge.source_id in outgoing_edges:
            outgoing_edges[edge.source_id].append(edge)

    sink_ids = tuple(
        nid
        for nid in topo_atomic
        if not any(e.target_id in atomic_set for e in outgoing_edges.get(nid, []))
    )

    budget = risk_budget
    global_alpha = budget.global_alpha if budget is not None else 0.01
    max_comp = budget.max_mixture_components if budget is not None else 4
    default_var = budget.default_source_variance if budget is not None else 0.25

    alpha_alloc = allocate_dag_alpha_budget(topo_atomic, global_alpha=global_alpha)

    # ------------------------------------------------------------------
    # Pass 1: Forward Mixture Pushforward
    # ------------------------------------------------------------------
    resolved_profiles: dict[str, tuple[str, AtomUncertaintyProfile]] = {}
    node_in_dims: dict[str, dict[str, int]] = {}
    node_out_dims: dict[str, dict[str, int]] = {}
    forward_in_mixtures: dict[str, dict[str, GaussianMixtureRegime]] = {}
    forward_out_mixtures: dict[str, dict[str, GaussianMixtureRegime]] = {}
    node_resps: dict[str, FloatArray] = {}
    node_env_checks: dict[str, dict[str, EnvelopeCheckResult]] = {}
    node_bregmans: dict[str, dict[str, float]] = {}
    node_in_bounds: dict[str, bool] = {}
    node_scalar_deltas: dict[str, float] = {}
    uncalibrated_list: list[str] = []

    src_regimes = dict(source_regimes or {})

    for nid in topo_atomic:
        node = node_map[nid]
        atom_name = ""
        node_match = match_map.get(nid)
        if node_match is not None and node_match.success and node_match.verified_match is not None:
            atom_name = node_match.verified_match.candidate.declaration.name
        elif node.matched_primitive:
            atom_name = node.matched_primitive

        profile = (
            resolve_atom_profile(atom_name, backend=backend, profiles=profiles)
            if atom_name
            else AtomUncertaintyProfile(atom_name=node.name, mode="unknown")
        )
        if profile.mode == "unknown":
            uncalibrated_list.append(nid)

        resolved_profiles[nid] = (atom_name or node.name, profile)
        in_dims, out_dims = _infer_port_dims(node, profile)
        node_in_dims[nid] = in_dims
        node_out_dims[nid] = out_dims

        # Build incoming GaussianMixtureRegime per input port
        io_by_name: dict[str, IOSpec] = {io.name: io for io in node.inputs}
        in_beliefs: dict[str, GaussianMixtureRegime] = {}

        for pname, d_in in in_dims.items():
            # 1) Explicit source_regimes override
            src_key = (
                f"{nid}.{pname}"
                if f"{nid}.{pname}" in src_regimes
                else (pname if pname in src_regimes else None)
            )
            if src_key is not None:
                in_beliefs[pname] = _adapt_gaussian_dim(
                    _mixture_to_gaussian_belief(src_regimes[src_key]),
                    d_in,
                )
                continue

            # 2) Upstream DATA_FLOW edge(s)
            matched_upstream: list[GaussianMixtureRegime] = []
            for edge in incoming_edges.get(nid, []):
                if edge.source_id not in forward_out_mixtures:
                    continue
                target_match = _match_port_key(edge.input_name, list(in_dims.keys()))
                if target_match != pname:
                    continue
                up_outs = forward_out_mixtures[edge.source_id]
                up_port = _match_port_key(edge.output_name, list(up_outs.keys()))
                if up_port is not None:
                    prior_mu = (
                        profile.input_regimes[pname].mixture.mixture_mean()
                        if pname in profile.input_regimes
                        else None
                    )
                    matched_upstream.append(
                        _adapt_gaussian_dim(up_outs[up_port], d_in, prior_mean=prior_mu)
                    )

            if matched_upstream:
                if len(matched_upstream) == 1:
                    in_beliefs[pname] = matched_upstream[0]
                else:
                    # Combine multiple upstream edges via covariance union / moment envelope
                    cov_union = np.zeros((d_in, d_in), dtype=np.float64)
                    mu_avg = np.zeros(d_in, dtype=np.float64)
                    for mb in matched_upstream:
                        mu_avg += mb.mixture_mean() / len(matched_upstream)
                        cov_union = np.maximum(cov_union, mb.mixture_covariance())
                    in_beliefs[pname] = GaussianMixtureRegime.from_full_covariances(
                        weights=[1.0],
                        means=mu_avg.reshape(1, -1),
                        covariances=(symmetrize(cov_union) + 1e-12 * np.eye(d_in)).reshape(1, d_in, d_in),
                        factor_rank=min(3, d_in),
                        min_diag=1e-12,
                    )
                continue

            # 3) Root input: use calibrated PortRegimeSpec prior or IOSpec bounds
            if pname in profile.input_regimes:
                in_beliefs[pname] = _mixture_to_gaussian_belief(
                    profile.input_regimes[pname].mixture
                )
            else:
                io_match_key = _match_port_key(pname, list(io_by_name.keys()))
                io_spec = io_by_name[io_match_key] if io_match_key else (node.inputs[0] if node.inputs else None)
                var0 = (
                    _parse_constraint_variance(io_spec.constraints, io_spec.type_desc, default_var)
                    if io_spec is not None
                    else default_var
                )
                eff_var = max(var0, 1e-12)
                in_beliefs[pname] = GaussianMixtureRegime(
                    weights=[1.0],
                    means=np.zeros((1, d_in), dtype=np.float64),
                    diag_vars=np.full((1, d_in), eff_var, dtype=np.float64),
                )

        forward_in_mixtures[nid] = in_beliefs

        # Evaluate regime responsibilities & conformal bounds
        node_alpha = alpha_alloc.get(nid, global_alpha)
        port_alpha = node_alpha / float(max(1, len(profile.input_regimes)))
        k_count = profile.n_components
        log_resp = np.log(np.maximum(profile.prior_weights(), 1e-12))
        env_checks_n: dict[str, EnvelopeCheckResult] = {}
        breg_n: dict[str, float] = {}
        in_ok_n = True

        for pname, reg in profile.input_regimes.items():
            b_key = _match_port_key(pname, list(in_beliefs.keys()))
            if b_key is None:
                continue
            p_resp, p_breg, p_env, p_ok = _evaluate_belief_against_regime(
                in_beliefs[b_key],
                reg,
                port_alpha,
            )
            breg_n[pname] = p_breg
            if p_env is not None:
                env_checks_n[pname] = p_env
            if not p_ok:
                in_ok_n = False
            if p_resp.size == k_count:
                log_resp = log_resp + np.log(np.maximum(p_resp, 1e-12))

        log_resp = log_resp - float(np.max(log_resp))
        resp_norm = np.exp(log_resp)
        resp_norm = resp_norm / max(float(np.sum(resp_norm)), 1e-15)

        node_resps[nid] = resp_norm
        node_env_checks[nid] = env_checks_n
        node_bregmans[nid] = breg_n
        node_in_bounds[nid] = in_ok_n

        # Push mixtures forward (applying fixed-point iteration variance scaling if in body)
        iter_var_scale = float(iterations_used) if (nid in fp_child_ids and iterations_used > 1) else 1.0
        out_beliefs = _pushforward_node_mixtures(
            profile,
            in_beliefs,
            resp_norm,
            out_dims,
            max_components=max_comp,
            iteration_scale=iter_var_scale,
        )
        forward_out_mixtures[nid] = out_beliefs

        # Scalar width expansion summary (2 * rms_std_out - 2 * rms_std_in)
        in_widths = [
            2.0 * math.sqrt(max(0.0, float(np.trace(b.mixture_covariance())) / float(max(1, b.dim))))
            for b in in_beliefs.values()
        ]
        out_widths = [
            2.0 * math.sqrt(max(0.0, float(np.trace(b.mixture_covariance())) / float(max(1, b.dim))))
            for b in out_beliefs.values()
        ]
        w_in = max(in_widths) if in_widths else 0.0
        w_out = max(out_widths) if out_widths else 0.0
        delta_w = w_out - w_in
        if nid in fp_child_ids and iterations_used > 1:
            delta_w /= math.sqrt(iterations_used)
        node_scalar_deltas[nid] = delta_w

    # ------------------------------------------------------------------
    # Pass 2: Backward Adjoint Precision Pullback
    # ------------------------------------------------------------------
    req_out_prec: dict[str, dict[str, FloatArray]] = {
        nid: {q: np.zeros((d_out, d_out), dtype=np.float64) for q, d_out in node_out_dims[nid].items()}
        for nid in topo_atomic
    }
    req_in_prec: dict[str, dict[str, FloatArray]] = {}
    pullback_ok: dict[str, bool] = {}
    pullback_reasons: dict[str, tuple[str, ...]] = {}

    # Seed sink precisions from RiskBudget
    if budget is not None:
        for nid in sink_ids:
            for qname, d_out in node_out_dims[nid].items():
                sink_lam = budget.resolve_sink_precision(nid, qname, d_out)
                if sink_lam is not None:
                    req_out_prec[nid][qname] = symmetrize(
                        req_out_prec[nid][qname] + sink_lam
                    )
        # Also allow explicit budget constraints on intermediate nodes ("node_id.port")
        for nid in topo_atomic:
            if nid in sink_ids:
                continue
            for qname, d_out in node_out_dims[nid].items():
                qual = f"{nid}.{qname}"
                if qual in budget.sink_precisions or nid in budget.sink_precisions:
                    mid_lam = budget.resolve_sink_precision(nid, qname, d_out)
                    if mid_lam is not None:
                        req_out_prec[nid][qname] = symmetrize(
                            req_out_prec[nid][qname] + mid_lam
                        )

    for nid in reversed(topo_atomic):
        _atom_name, profile = resolved_profiles[nid]
        out_prec_n = req_out_prec[nid]
        has_nonzero_budget = any(float(np.max(np.abs(m))) > 1e-15 for m in out_prec_n.values())

        if not has_nonzero_budget:
            req_in_prec[nid] = {
                p: np.zeros((d_in, d_in), dtype=np.float64)
                for p, d_in in node_in_dims[nid].items()
            }
            pullback_ok[nid] = True
            pullback_reasons[nid] = ()
            continue

        # Map node's out_prec_n port names to profile's operator output port names
        if profile.components:
            prof_out_ports = sorted(
                {
                    out_p
                    for comp in profile.components
                    for (_in_p, out_p) in comp.operators
                }
                | {
                    out_p
                    for comp in profile.components
                    for out_p in comp.intrinsic_output_cov
                }
            )
            mapped_out_prec: dict[str, FloatArray] = {}
            for q_node, lam_q in out_prec_n.items():
                if float(np.max(np.abs(lam_q))) <= 1e-15:
                    continue
                q_prof = _match_port_key(q_node, prof_out_ports) or q_node
                # Adapt dimension to profile's operator dim_out if needed
                target_d_out = lam_q.shape[0]
                for comp in profile.components:
                    for (_ip, op_q), op in comp.operators.items():
                        if op_q == q_prof:
                            target_d_out = op.dim_out
                            break
                lam_adapted = _adapt_precision_dim(lam_q, target_d_out)
                mapped_out_prec[q_prof] = symmetrize(
                    mapped_out_prec.get(q_prof, np.zeros_like(lam_adapted)) + lam_adapted
                )

            pulled_raw, feasible, reasons = profile.pullback_precisions(
                mapped_out_prec,
                responsibilities=node_resps[nid],
            )
            mapped_in_prec: dict[str, FloatArray] = {
                p: np.zeros((d_in, d_in), dtype=np.float64)
                for p, d_in in node_in_dims[nid].items()
            }
            for p_prof, lam_in in pulled_raw.items():
                p_node = _match_port_key(p_prof, list(node_in_dims[nid].keys()))
                if p_node is not None:
                    d_target = node_in_dims[nid][p_node]
                    mapped_in_prec[p_node] = symmetrize(
                        mapped_in_prec[p_node] + _adapt_precision_dim(lam_in, d_target)
                    )
            if feasible:
                upstream_in_prec = mapped_in_prec
            else:
                # Node's own intrinsic noise exhausted the budget: propagate pure
                # Fisher directional sensitivity J^T Lambda_out J upstream rather
                # than a 1/tol singularity that would falsely blame upstream nodes.
                upstream_in_prec = {
                    p: np.zeros((d_in, d_in), dtype=np.float64)
                    for p, d_in in node_in_dims[nid].items()
                }
                gamma_n = node_resps[nid]
                for c_idx, comp in enumerate(profile.components):
                    w_k = float(gamma_n[c_idx]) if c_idx < gamma_n.size else 0.0
                    if w_k <= 0.0:
                        continue
                    for (in_p, out_p), op in comp.operators.items():
                        if out_p not in mapped_out_prec:
                            continue
                        fim_in = op.fisher_information(mapped_out_prec[out_p])
                        p_node = _match_port_key(in_p, list(node_in_dims[nid].keys()))
                        if p_node is not None:
                            d_target = node_in_dims[nid][p_node]
                            upstream_in_prec[p_node] = symmetrize(
                                upstream_in_prec[p_node]
                                + w_k * _adapt_precision_dim(fim_in, d_target)
                            )
        else:
            factor = profile.legacy_scalar_factor if profile.legacy_scalar_factor is not None else 1.0
            first_lam = next(iter(out_prec_n.values()))
            mapped_in_prec = {
                p: symmetrize((factor * factor) * _adapt_precision_dim(first_lam, d_in))
                for p, d_in in node_in_dims[nid].items()
            }
            upstream_in_prec = mapped_in_prec
            feasible = True
            reasons = []

        req_in_prec[nid] = mapped_in_prec
        pullback_ok[nid] = feasible
        pullback_reasons[nid] = tuple(reasons)

        # Propagate required input precisions further upstream across incoming DATA_FLOW edges
        for edge in incoming_edges.get(nid, []):
            u_id = edge.source_id
            if u_id not in req_out_prec:
                continue
            p_node = _match_port_key(edge.input_name, list(upstream_in_prec.keys()))
            q_up = _match_port_key(edge.output_name, list(req_out_prec[u_id].keys()))
            if p_node is None or q_up is None:
                continue
            lam_edge = _adapt_precision_dim(
                upstream_in_prec[p_node],
                req_out_prec[u_id][q_up].shape[0],
            )
            req_out_prec[u_id][q_up] = symmetrize(req_out_prec[u_id][q_up] + lam_edge)

    # ------------------------------------------------------------------
    # Assemble Node Evaluations & Bottleneck Risk Attribution
    # ------------------------------------------------------------------
    evaluations: dict[str, NodeUncertaintyEvaluation] = {}
    infeasible_nodes: list[str] = []
    regime_violated: list[str] = []
    budget_violated: list[str] = []

    for nid in topo_atomic:
        node = node_map[nid]
        atom_name, profile = resolved_profiles[nid]
        act_k = int(np.argmax(node_resps[nid]))

        # Spectral summaries for the active mixture component
        spec_sums: dict[tuple[str, str], SpectralSummary] = {}
        if profile.components and act_k < len(profile.components):
            for port_pair, op in profile.components[act_k].operators.items():
                spec_sums[port_pair] = op.stiff_sloppy_spectrum()

        out_ratios: dict[str, float] = {}
        worst_dirs: dict[str, FloatArray] = {}
        out_risk_trace = 0.0
        for qname, out_mix in forward_out_mixtures[nid].items():
            cov_q = out_mix.mixture_covariance()
            lam_q = req_out_prec[nid][qname]
            ratio_q, dir_q = compute_risk_ratio(cov_q, lam_q)
            out_ratios[qname] = ratio_q
            worst_dirs[qname] = dir_q
            out_risk_trace += float(np.trace(lam_q @ cov_q))

        in_cov_traces = [
            float(np.trace(b.mixture_covariance())) / float(max(1, b.dim))
            for b in forward_in_mixtures[nid].values()
        ]
        out_cov_traces = [
            float(np.trace(b.mixture_covariance())) / float(max(1, b.dim))
            for b in forward_out_mixtures[nid].values()
        ]
        mean_in_var = max(in_cov_traces) if in_cov_traces else 0.0
        mean_out_var = max(out_cov_traces) if out_cov_traces else 0.0

        # Marginal risk contribution: combines directional budget excess, local variance
        # injection/amplification, pullback infeasibility, and regime OOD penalties.
        has_budget = any(float(np.max(np.abs(m))) > 1e-15 for m in req_out_prec[nid].values())
        if has_budget:
            # Compare output precision-weighted risk against unweighted baseline input variance
            # scaled by average required precision
            avg_lam = max(
                float(np.trace(m)) / float(max(1, m.shape[0]))
                for m in req_out_prec[nid].values()
            )
            baseline_in_risk = avg_lam * mean_in_var
            local_excess = max(0.0, out_risk_trace - baseline_in_risk)
            marginal_score = local_excess + 0.25 * max(out_ratios.values(), default=0.0)
            if not pullback_ok[nid]:
                marginal_score += 10.0 + max(out_ratios.values(), default=1.0)
            if not node_in_bounds[nid]:
                marginal_score += 5.0
        else:
            marginal_score = max(0.0, node_scalar_deltas[nid]) + max(0.0, mean_out_var - mean_in_var)
            if not node_in_bounds[nid]:
                marginal_score += 5.0

        if not pullback_ok[nid]:
            infeasible_nodes.append(nid)
        if not node_in_bounds[nid]:
            regime_violated.append(nid)
        if has_budget and max(out_ratios.values(), default=0.0) > 1.0 + 1e-9:
            budget_violated.append(nid)

        evaluations[nid] = NodeUncertaintyEvaluation(
            node_id=nid,
            node_name=node.name,
            atom_name=atom_name,
            profile=profile,
            allocated_alpha=alpha_alloc.get(nid, global_alpha),
            responsibilities=node_resps[nid],
            active_component=act_k,
            incoming_mixtures=forward_in_mixtures[nid],
            outgoing_mixtures=forward_out_mixtures[nid],
            required_output_precisions=req_out_prec[nid],
            required_input_precisions=req_in_prec[nid],
            output_risk_ratios=out_ratios,
            worst_output_directions=worst_dirs,
            spectral_summaries=spec_sums,
            envelope_checks=node_env_checks[nid],
            regime_bregman_divergences=node_bregmans[nid],
            in_regime_bounds=node_in_bounds[nid],
            pullback_feasible=pullback_ok[nid],
            pullback_violation_reasons=pullback_reasons[nid],
            scalar_width_expansion=node_scalar_deltas[nid],
            marginal_risk_score=marginal_score,
        )

    return CDGUncertaintyReport(
        nodes=evaluations,
        topological_order=tuple(topo_atomic),
        sink_node_ids=sink_ids,
        alpha_allocation=alpha_alloc,
        risk_budget=budget,
        uncalibrated_nodes=tuple(uncalibrated_list),
        infeasible_pullback_nodes=tuple(infeasible_nodes),
        regime_violated_nodes=tuple(regime_violated),
        budget_violated_nodes=tuple(budget_violated),
    )
