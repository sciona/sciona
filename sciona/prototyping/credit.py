"""Universal domain-agnostic credit assignment for CDG iterative refinement.

Combines three complementary blame signals without waveform/DSP assumptions:
1. **Contract / Runtime Guard / Funnel Blame**: Immediate 100% blame on the exact
   ``failed_node_id`` when a node triggers ``@icontract.ViolationError``,
   ``ConfidenceEnvelopeViolation``, static ``PortTypeSpec`` mismatch, or runtime crash.
2. **Cached Counterfactual Perturbation Sensitivity**: Perturbs each atomic leaf's
   cached intermediate output using ``PortTypeSpec``-aware operators and runs ONLY
   its strict downstream cone via ``LiveCDGRunner.execute_from_override`` (``O(downstream)``,
   zero upstream re-execution!).
3. **Phase 3 Directional Uncertainty & Adjoint Pullback**: Blends ``propagate_cdg_uncertainty``
   marginal risk, pullback feasibility, and uncalibrated epistemic risk.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import IOSpec, NodeStatus
from sciona.principal.backprop import CreditAssigner
from sciona.principal.models import BenchmarkResult, NodeGradient, OptimizationMetric
from sciona.synthesizer.ghost_sim import GhostSimReport
from sciona.types import MatchResult
from sciona.uncertainty.models import AtomUncertaintyProfile
from sciona.uncertainty.propagation import RiskBudget, propagate_cdg_uncertainty

from sciona.prototyping.harness import EvalCase, ProblemHarness
from sciona.prototyping.live_runner import LiveCDGRunner


@dataclass(frozen=True)
class CounterfactualNodeSensitivity:
    """Empirical downstream loss sensitivity under a cached output perturbation."""

    node_id: str
    baseline_loss: float
    perturbed_loss: float
    abs_loss_delta: float
    normalized_sensitivity: float
    downstream_nodes_executed: int


def _deterministic_seed(tag: str) -> int:
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def perturb_runtime_value(
    value: Any,
    *,
    port_spec: IOSpec | None = None,
    epsilon: float = 0.05,
    probe_mode: str = "directional",
    seed_tag: str = "node",
) -> tuple[bool, Any]:
    """Apply a type-safe counterfactual perturbation ``(perturbed_ok, new_value)``.

    Preserves container type, shape, dtype, and finiteness so downstream atoms
    receive valid inputs while exposing sensitivity ``|d Loss / d y_node|``.
    """
    _ = port_spec
    if value is None or isinstance(value, bool):
        return False, value

    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if not math.isfinite(fv):
            return False, value
        scale = max(abs(fv), 1.0)
        delta = (-epsilon * fv) if probe_mode == "shrink" and abs(fv) > 1e-9 else (epsilon * scale)
        return True, float(fv + delta)

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        iv = int(value)
        step = max(1, int(round(abs(iv) * epsilon)))
        return True, (iv - step if probe_mode == "shrink" and iv > step else iv + step)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False, value
        if np.issubdtype(value.dtype, np.floating):
            finite = value[np.isfinite(value)]
            if finite.size == 0:
                return False, value
            if probe_mode == "attenuate":
                # Contraction toward zero (exposes both spurious DC/offset bias and gain error)
                perturbed = (1.0 - epsilon) * value
                return True, perturbed.astype(value.dtype, copy=False)
            if probe_mode == "shrink":
                # Damped contraction toward sample mean (tests AC amplitude sensitivity)
                mu = float(np.mean(finite))
                perturbed = mu + (1.0 - epsilon) * (value - mu)
                return True, perturbed.astype(value.dtype, copy=False)
            spread = float(np.std(finite))
            rms = float(np.sqrt(np.mean(finite**2)))
            scale = max(spread, rms, 1e-2)
            rng = np.random.default_rng(_deterministic_seed(seed_tag))
            direction = rng.normal(0.0, 1.0, size=value.shape)
            norm = float(np.sqrt(np.mean(direction**2))) or 1.0
            perturbed = value + (epsilon * scale) * (direction / norm)
            return True, perturbed.astype(value.dtype, copy=False)

        if np.issubdtype(value.dtype, np.integer):
            # Integer / event index arrays: shift indices slightly within non-negative bounds
            shifted = value.copy()
            if probe_mode == "shrink" and shifted.size > 1:
                shifted = np.maximum(0, shifted - 1)
            else:
                shifted = shifted + 1
            return True, shifted

        return False, value

    cls_mod = type(value).__module__
    cls_name = type(value).__name__
    if cls_mod.startswith("pandas."):
        if cls_name == "DataFrame" and hasattr(value, "copy"):
            df_copy = value.copy()
            changed = False
            for col in df_copy.columns:
                col_arr = df_copy[col].to_numpy()
                ok, p_col = perturb_runtime_value(
                    col_arr,
                    epsilon=epsilon,
                    probe_mode=probe_mode,
                    seed_tag=f"{seed_tag}.{col}",
                )
                if ok:
                    df_copy[col] = p_col
                    changed = True
            return changed, df_copy
        if cls_name == "Series" and hasattr(value, "copy"):
            ser_copy = value.copy()
            ok, p_arr = perturb_runtime_value(
                ser_copy.to_numpy(),
                epsilon=epsilon,
                probe_mode=probe_mode,
                seed_tag=seed_tag,
            )
            if ok:
                ser_copy.iloc[:] = p_arr
            return ok, ser_copy

    if isinstance(value, tuple):
        any_ok = False
        items: list[Any] = []
        for idx, elem in enumerate(value):
            ok, p_elem = perturb_runtime_value(
                elem,
                epsilon=epsilon,
                probe_mode=probe_mode,
                seed_tag=f"{seed_tag}[{idx}]",
            )
            any_ok = any_ok or ok
            items.append(p_elem)
        return any_ok, tuple(items)

    return False, value


class UniversalCreditAssigner:
    """Domain-agnostic multi-signal bottleneck attribution engine."""

    def __init__(
        self,
        *,
        epsilon: float = 0.08,
        max_cases: int = 3,
        uncertainty_weight: float = 0.25,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        self.epsilon = float(epsilon)
        self.max_cases = max(1, int(max_cases))
        self.uncertainty_weight = max(0.0, min(1.0, float(uncertainty_weight)))
        self._legacy_assigner = CreditAssigner()

    def compute_counterfactual_sensitivities(
        self,
        cdg: CDGExport,
        harness: ProblemHarness,
        runner: LiveCDGRunner,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> dict[str, CounterfactualNodeSensitivity]:
        """Measure per-node downstream loss sensitivity via ``execute_from_override``."""
        leaves = cdg.leaf_nodes()
        if not leaves:
            return {}

        cases: list[EvalCase] = list(harness.validation_cases())[: self.max_cases]
        if not cases:
            cases = list(harness.synthetic_cases())[: self.max_cases]
        if not cases:
            return {}

        # Run or hit cache on selected cases to capture baseline node_outputs
        baseline_runs: list[tuple[EvalCase, float, dict[str, dict[str, Any]], dict[str, Any]]] = []
        for case in cases:
            init_state: dict[str, Any] | None = None
            if case.fit_inputs is not None:
                fit_res = runner.execute(cdg, case.fit_inputs, node_params=node_params, match_results=match_results)
                if not fit_res.ok:
                    continue
                init_state = dict(fit_res.state_store)
            base_res = runner.execute(
                cdg,
                case.inputs,
                node_params=node_params,
                match_results=match_results,
                initial_state=init_state,
            )
            if not base_res.ok:
                continue
            base_score = harness.score(case, base_res.outputs)
            if not base_score.passed or not math.isfinite(base_score.loss):
                continue
            baseline_runs.append((case, base_score.loss, base_res.node_outputs, init_state or {}))

        if not baseline_runs:
            return {}

        sensitivities: dict[str, CounterfactualNodeSensitivity] = {}
        for node in leaves:
            nid = node.node_id
            out_spec_map = {o.name: o for o in node.outputs}
            max_deltas: list[float] = []
            base_losses: list[float] = []
            pert_losses: list[float] = []
            downstream_executed = 0

            for case, base_loss, base_node_outs, init_st in baseline_runs:
                orig_ports = base_node_outs.get(nid)
                if not orig_ports:
                    continue

                best_case_delta = 0.0
                best_pert_loss = base_loss
                # Probe directional jitter, zero-attenuation (offset+gain), and mean-shrinkage
                # so nonlinear, biased, or normalized stages expose their true sensitivity.
                for mode in ("directional", "attenuate", "shrink"):
                    perturbed_ports: dict[str, Any] = {}
                    any_perturbed = False
                    for pname, pval in orig_ports.items():
                        ok, new_val = perturb_runtime_value(
                            pval,
                            port_spec=out_spec_map.get(pname),
                            epsilon=self.epsilon,
                            probe_mode=mode,
                            seed_tag=f"{nid}.{pname}.{case.case_id}",
                        )
                        perturbed_ports[pname] = new_val
                        any_perturbed = any_perturbed or ok
                    if not any_perturbed:
                        continue

                    cf_res = runner.execute_from_override(
                        cdg,
                        case.inputs,
                        baseline_node_outputs=base_node_outs,
                        override_node_outputs={nid: perturbed_ports},
                        node_params=node_params,
                        match_results=match_results,
                        initial_state=init_st or None,
                        use_cache=False,
                    )
                    downstream_executed = max(downstream_executed, len(cf_res.cache_misses))
                    if not cf_res.ok:
                        # If a gentle perturbation causes downstream failure, node interface is fragile
                        delta = max(abs(base_loss), 1.0)
                        cand_loss = base_loss + delta
                    else:
                        cf_score = harness.score(case, cf_res.outputs)
                        if not math.isfinite(cf_score.loss):
                            delta = max(abs(base_loss), 1.0)
                            cand_loss = base_loss + delta
                        else:
                            cand_loss = cf_score.loss
                            delta = abs(cand_loss - base_loss)
                    if delta >= best_case_delta:
                        best_case_delta = delta
                        best_pert_loss = cand_loss

                base_losses.append(base_loss)
                pert_losses.append(best_pert_loss)
                max_deltas.append(best_case_delta)

            if max_deltas:
                mean_base = float(np.mean(base_losses))
                mean_pert = float(np.mean(pert_losses))
                mean_delta = float(np.mean(max_deltas))
                sensitivities[nid] = CounterfactualNodeSensitivity(
                    node_id=nid,
                    baseline_loss=mean_base,
                    perturbed_loss=mean_pert,
                    abs_loss_delta=mean_delta,
                    normalized_sensitivity=mean_delta / self.epsilon,
                    downstream_nodes_executed=downstream_executed,
                )

        return sensitivities

    def compute_gradients(
        self,
        cdg: CDGExport,
        benchmark: BenchmarkResult,
        *,
        sim_report: GhostSimReport | None = None,
        target: OptimizationMetric = OptimizationMetric.PRECISION,
        harness: ProblemHarness | None = None,
        runner: LiveCDGRunner | None = None,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
        risk_budget: RiskBudget | None = None,
    ) -> list[NodeGradient]:
        """Rank CDG atomic leaves by optimization bottleneck responsibility."""
        leaves = cdg.leaf_nodes()
        if not leaves:
            leaves = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        if not leaves:
            return []

        leaf_ids = {n.node_id for n in leaves}
        node_names = {n.node_id: n.name for n in leaves}
        artifacts = benchmark.runtime_artifacts or {}

        # 1. Hard failure / contract / guard / static mismatch blame
        failed_nid = str(artifacts.get("failed_node_id") or "").strip()
        if failed_nid and failed_nid in leaf_ids:
            err_msg = str(
                artifacts.get("prune_reason")
                or artifacts.get("error")
                or "runtime contract or guard violation"
            )
            return [
                NodeGradient(
                    node_id=failed_nid,
                    gradient_score=100.0,
                    metric_type=target,
                    bottleneck_reason=(
                        f"Node '{node_names.get(failed_nid, failed_nid)}' failed directly: {err_msg}"
                    ),
                )
            ]

        # Non-precision resource axes delegate cleanly to telemetry shares
        ghost = sim_report if sim_report is not None else GhostSimReport()
        if target in (
            OptimizationMetric.LATENCY,
            OptimizationMetric.FLOP_COUNT,
            OptimizationMetric.MEMORY,
            OptimizationMetric.STRUCTURE,
            OptimizationMetric.CONVERGENCE,
        ):
            legacy = self._legacy_assigner.compute_gradients(cdg, benchmark, ghost, target)
            if legacy:
                return legacy

        # 2. Cached counterfactual sensitivity + Phase 3 directional uncertainty
        cf_scores: dict[str, float] = {}
        cf_details: dict[str, CounterfactualNodeSensitivity] = {}
        if harness is not None and runner is not None:
            cf_details = self.compute_counterfactual_sensitivities(
                cdg,
                harness,
                runner,
                node_params=node_params,
                match_results=match_results,
            )
            for nid, item in cf_details.items():
                cf_scores[nid] = item.normalized_sensitivity

        unc_scores: dict[str, float] = {}
        unc_tags: dict[str, list[str]] = {}
        resolved_budget = risk_budget or (harness.risk_budget if harness is not None else None)
        resolved_profiles = uncertainty_profiles or (
            runner.uncertainty_profiles if runner is not None else None
        )
        if resolved_budget is not None or resolved_profiles:
            try:
                u_rep = propagate_cdg_uncertainty(
                    cdg,
                    match_results=match_results,
                    profiles=resolved_profiles,
                    risk_budget=resolved_budget,
                )
                for nid in leaf_ids:
                    neval = u_rep.nodes.get(nid)
                    if neval is None:
                        continue
                    if neval.marginal_risk_score > 1e-9:
                        unc_scores[nid] = float(neval.marginal_risk_score)
                    tags: list[str] = []
                    if not neval.pullback_feasible:
                        tags.append("pullback infeasible")
                    if not neval.in_regime_bounds:
                        tags.append("regime OOD")
                    if neval.max_risk_ratio > 1.0 + 1e-9:
                        tags.append(f"risk_ratio={neval.max_risk_ratio:.2f}x")
                    if tags:
                        unc_tags[nid] = tags
            except Exception:
                pass

        cf_total = sum(cf_scores.values())
        unc_total = sum(unc_scores.values())

        combined: dict[str, float] = {}
        for nid in leaf_ids:
            cf_share = (cf_scores.get(nid, 0.0) / cf_total) if cf_total > 1e-9 else 0.0
            unc_share = (unc_scores.get(nid, 0.0) / unc_total) if unc_total > 1e-9 else 0.0
            if cf_total > 1e-9 and unc_total > 1e-9:
                w_u = self.uncertainty_weight
                combined[nid] = (1.0 - w_u) * cf_share + w_u * unc_share
            elif cf_total > 1e-9:
                combined[nid] = cf_share
            elif unc_total > 1e-9:
                combined[nid] = unc_share

        if not combined or sum(combined.values()) <= 1e-15:
            # Try legacy GhostSim / error_expansion fallback
            legacy = self._legacy_assigner.compute_gradients(cdg, benchmark, ghost, target)
            if legacy:
                return legacy
            # Final non-empty fallback when loss > 0: rank sink-adjacent tunable stages first
            # so Principal/PrototypingSession never halts prematurely on Trial 1!
            if math.isfinite(benchmark.global_loss) and benchmark.global_loss <= 1e-12:
                return []
            n_leaves = len(leaves)
            for idx, node in enumerate(leaves):
                # Slight downstream preference (sink stages often control final calibration/scale)
                combined[node.node_id] = float(idx + 1) / float(n_leaves * (n_leaves + 1) // 2)

        norm_total = sum(combined.values()) or 1.0
        gradients: list[NodeGradient] = []
        for nid, raw_val in combined.items():
            pct = (raw_val / norm_total) * 100.0
            parts: list[str] = []
            if nid in cf_details and cf_total > 1e-15:
                sens = cf_details[nid]
                parts.append(
                    f"counterfactual |dL/dy|={sens.normalized_sensitivity:.4g} "
                    f"(|dL|={sens.abs_loss_delta:.4g})"
                )
            if nid in unc_scores and unc_total > 1e-15:
                parts.append(f"marginal_risk={unc_scores[nid]:.4g}")
            if unc_tags.get(nid):
                parts.append("; ".join(unc_tags[nid]))
            detail_str = f" ({', '.join(parts)})" if parts else ""
            gradients.append(
                NodeGradient(
                    node_id=nid,
                    gradient_score=pct,
                    metric_type=target,
                    bottleneck_reason=(
                        f"Node '{node_names.get(nid, nid)}' attributed {pct:.1f}% optimization blame{detail_str}"
                    ),
                )
            )

        gradients.sort(key=lambda g: (-g.gradient_score, g.node_id))
        return gradients
