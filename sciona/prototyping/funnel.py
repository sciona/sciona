"""4-Tier Multi-Fidelity Evaluation Funnel and Principal Live Prototyping Sandbox.

Funnels candidate CDGs and HPO trials from sub-millisecond static checks down
to resident holdout verification with zero OS subprocess overhead:

- **Tier 0 (~1–5 ms, static/analytic)**:
  CDG leaf/completeness check + ``PortTypeSpec`` edge compatibility +
  ``run_ghost_simulation`` + ``OptunaManager.check_early_prune`` +
  optional Phase 3 ``propagate_cdg_uncertainty`` risk-budget gate.
- **Tier 1 (~5–50 ms, ``SyntheticSuiteHarness``)**:
  Executes ``LiveCDGRunner`` on deterministic nominal/stress synthetic fixtures
  plus ``@icontract`` & runtime confidence guards and ``MetamorphicRelation``s.
  Prunes candidates that crash, emit NaN/Inf, violate contracts, or blow latency.
- **Tier 2 (~20–250 ms, ``validation_cases`` + ``IntermediateCache``)**:
  Scores surviving candidates on in-RAM validation cases with prefix memoization,
  producing a ``Principal``-compatible ``BenchmarkResult`` with full node telemetry.
- **Tier 3 (Holdout & Promotion Gate)**:
  Evaluates winning candidates on held-out cases (``holdout_cases``) to measure
  generalization gap and catch overfitting prior to final package synthesis/export.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, EdgeKind, IOSpec
from sciona.principal.hpo import OptunaManager, TrialPrunedEarly
from sciona.principal.models import BenchmarkResult, NodeTelemetry, OptimizationMetric
from sciona.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from sciona.synthesizer.models import ExportBundle
from sciona.types import MatchResult
from sciona.uncertainty.propagation import CDGUncertaintyReport, propagate_cdg_uncertainty

from sciona.prototyping.harness import (
    CaseScore,
    EvalCase,
    MetamorphicRelation,
    ProblemHarness,
)
from sciona.prototyping.live_runner import LiveCDGRunner, LiveRunResult


@dataclass(frozen=True)
class FunnelConfig:
    """Knobs controlling which multi-fidelity tiers gate candidate evaluation."""

    enable_tier0_type_check: bool = True
    enable_tier0_ghost_sim: bool = True
    enable_tier0_uncertainty_gate: bool = True
    enable_tier1_synthetic: bool = True
    enable_tier1_metamorphic: bool = True
    tier1_max_case_latency_ms: float = 2000.0
    holdout_overfit_tolerance: float = 0.25


@dataclass
class FunnelTierResult:
    """Outcome of a single funnel tier (`0`, `1`, `2`, or `3`)."""

    tier: int
    passed: bool
    loss: float = float("inf")
    elapsed_ms: float = 0.0
    case_scores: list[CaseScore] = field(default_factory=list)
    failed_node_id: str = ""
    prune_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunnelEvaluationReport:
    """Complete multi-fidelity evaluation record for a candidate CDG."""

    passed: bool
    pruned_at_tier: int | None
    global_loss: float
    benchmark: BenchmarkResult
    tier_results: dict[int, FunnelTierResult] = field(default_factory=dict)
    ghost_report: GhostSimReport | None = None
    uncertainty_report: CDGUncertaintyReport | None = None
    failed_node_id: str = ""
    prune_reason: str = ""


def check_static_edge_port_types(cdg: CDGExport) -> tuple[bool, str, str]:
    """Verify static ``PortTypeSpec`` compatibility across all leaf ``DATA_FLOW`` edges.

    Returns ``(ok, failed_node_id, reason)``.
    """
    leaves = cdg.leaf_nodes()
    if not leaves:
        return False, "", "CDG contains no atomic leaf nodes"
    non_atomic = cdg.non_atomic_leaves()
    if non_atomic:
        first_bad = non_atomic[0]
        return (
            False,
            first_bad.node_id,
            f"CDG contains {len(non_atomic)} non-atomic leaf node(s), starting with '{first_bad.node_id}' ({first_bad.name})",
        )

    node_map: dict[str, AlgorithmicNode] = {n.node_id: n for n in leaves}
    for edge in cdg.edges:
        if edge.edge_kind == EdgeKind.CALLABLE_INJECTION:
            continue
        src = node_map.get(edge.source_id)
        tgt = node_map.get(edge.target_id)
        if src is None or tgt is None:
            continue

        src_spec = _find_port_spec(src.outputs, edge.output_name)
        tgt_spec = _find_port_spec(tgt.inputs, edge.input_name)
        if src_spec is None or tgt_spec is None:
            continue

        src_pt = src_spec.resolved_port_type()
        tgt_pt = tgt_spec.resolved_port_type()
        if not src_pt.is_compatible_with(tgt_pt):
            return (
                False,
                tgt.node_id,
                f"Static PortTypeSpec mismatch on edge {src.node_id}.{src_spec.name} ({src_pt.raw} -> kind={src_pt.kind}, ndim={src_pt.ndim}, dtype={src_pt.dtype}) "
                f"-> {tgt.node_id}.{tgt_spec.name} ({tgt_pt.raw} -> kind={tgt_pt.kind}, ndim={tgt_pt.ndim}, dtype={tgt_pt.dtype})",
            )
    return True, "", ""


def _find_port_spec(specs: Sequence[IOSpec], name: str) -> IOSpec | None:
    for s in specs:
        if s.name == name:
            return s
    if len(specs) == 1:
        return specs[0]
    low = name.lower()
    for s in specs:
        if s.name.lower() == low:
            return s
    return None


class EvaluationFunnel:
    """Orchestrates Tier 0 -> Tier 1 -> Tier 2 (-> Tier 3 holdout) evaluation."""

    def __init__(
        self,
        harness: ProblemHarness,
        *,
        runner: LiveCDGRunner | None = None,
        config: FunnelConfig | None = None,
    ) -> None:
        self.harness = harness
        self.runner = runner if runner is not None else LiveCDGRunner()
        self.config = config if config is not None else FunnelConfig()

    def _execute_case(
        self,
        cdg: CDGExport,
        case: EvalCase,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> LiveRunResult:
        """Execute a single ``EvalCase``, threading ``fit_inputs`` state if stateful."""
        init_state: dict[str, Any] | None = None
        fit_time_ms = 0.0
        if case.fit_inputs is not None:
            fit_run = self.runner.execute(
                cdg,
                case.fit_inputs,
                node_params=node_params,
                match_results=match_results,
            )
            fit_time_ms = fit_run.total_time_ms
            if not fit_run.ok:
                return fit_run
            init_state = dict(fit_run.state_store)

        pred_run = self.runner.execute(
            cdg,
            case.inputs,
            node_params=node_params,
            match_results=match_results,
            initial_state=init_state,
        )
        pred_run.total_time_ms += fit_time_ms
        return pred_run

    def evaluate_tier0(
        self,
        cdg: CDGExport,
        *,
        match_results: Sequence[MatchResult] | None = None,
    ) -> tuple[FunnelTierResult, GhostSimReport | None, CDGUncertaintyReport | None]:
        """Tier 0: Static PortTypeSpec check + Ghost Simulation + Analytic Uncertainty Gate."""
        t0 = time.perf_counter()
        ghost_report: GhostSimReport | None = None
        unc_report: CDGUncertaintyReport | None = None

        if self.config.enable_tier0_type_check:
            ok_types, bad_nid, reason = check_static_edge_port_types(cdg)
            if not ok_types:
                return (
                    FunnelTierResult(
                        tier=0,
                        passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        failed_node_id=bad_nid,
                        prune_reason=reason,
                    ),
                    None,
                    None,
                )

        if self.config.enable_tier0_ghost_sim:
            ghost_report = run_ghost_simulation(cdg, list(match_results or []))
            try:
                OptunaManager.check_early_prune(ghost_report)
            except TrialPrunedEarly as exc:
                bad_nid = ghost_report.error_node or (
                    ghost_report.deadlock_nodes[0] if ghost_report.deadlock_nodes else ""
                )
                return (
                    FunnelTierResult(
                        tier=0,
                        passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        failed_node_id=bad_nid,
                        prune_reason=f"GhostSim pruned candidate: {exc}",
                    ),
                    ghost_report,
                    None,
                )

        budget = self.harness.risk_budget
        if self.config.enable_tier0_uncertainty_gate and budget is not None:
            unc_report = propagate_cdg_uncertainty(
                cdg,
                match_results=match_results,
                profiles=self.runner.uncertainty_profiles,
                risk_budget=budget,
            )
            if not unc_report.meets_budget:
                ranked = unc_report.ranked_bottlenecks()
                worst_nid = (
                    (unc_report.budget_violated_nodes or unc_report.infeasible_pullback_nodes or unc_report.regime_violated_nodes or ("",))[0]
                    or (ranked[0].node_id if ranked else "")
                )
                return (
                    FunnelTierResult(
                        tier=0,
                        passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        failed_node_id=worst_nid,
                        prune_reason=(
                            f"Analytic RiskBudget violated (max_sink_risk_ratio={unc_report.max_sink_risk_ratio:.3f}, "
                            f"budget_violated={list(unc_report.budget_violated_nodes)}, "
                            f"regime_violated={list(unc_report.regime_violated_nodes)})"
                        ),
                        details={"max_sink_risk_ratio": unc_report.max_sink_risk_ratio},
                    ),
                    ghost_report,
                    unc_report,
                )

        return (
            FunnelTierResult(
                tier=0,
                passed=True,
                loss=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            ),
            ghost_report,
            unc_report,
        )

    def evaluate_tier1(
        self,
        cdg: CDGExport,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> FunnelTierResult:
        """Tier 1: Fast synthetic/stress cases + contract/guard checks + metamorphic relations."""
        t0 = time.perf_counter()
        if not self.config.enable_tier1_synthetic:
            return FunnelTierResult(tier=1, passed=True, loss=0.0, elapsed_ms=0.0)

        synth_cases = list(self.harness.synthetic_cases())
        if not synth_cases:
            return FunnelTierResult(tier=1, passed=True, loss=0.0, elapsed_ms=0.0)

        scores: list[CaseScore] = []
        nominal_case: EvalCase | None = None
        nominal_outputs: dict[str, Any] | None = None

        for case in synth_cases:
            run_res = self._execute_case(
                cdg,
                case,
                node_params=node_params,
                match_results=match_results,
            )
            if not run_res.ok:
                cs = CaseScore(
                    case_id=case.case_id,
                    loss=float("inf"),
                    passed=False,
                    execution_time_ms=run_res.total_time_ms,
                    error=run_res.error,
                    failed_node_id=run_res.failed_node_id,
                )
                scores.append(cs)
                return FunnelTierResult(
                    tier=1,
                    passed=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    case_scores=scores,
                    failed_node_id=run_res.failed_node_id,
                    prune_reason=f"Tier 1 runtime failure on '{case.case_id}' ({case.regime}): {run_res.error}",
                )

            if run_res.total_time_ms > self.config.tier1_max_case_latency_ms:
                slowest_nid = (
                    max(
                        run_res.node_telemetry.values(),
                        key=lambda nt: nt.execution_time_ms,
                    ).node_id
                    if run_res.node_telemetry
                    else ""
                )
                return FunnelTierResult(
                    tier=1,
                    passed=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    case_scores=scores,
                    failed_node_id=slowest_nid,
                    prune_reason=(
                        f"Tier 1 latency budget exceeded on '{case.case_id}': "
                        f"{run_res.total_time_ms:.1f} ms > {self.config.tier1_max_case_latency_ms:.1f} ms"
                    ),
                )

            # Check output finiteness & PortTypeSpec contracts (without requiring ground-truth labels)
            from sciona.prototyping.harness import check_output_invariants

            allow_nf = bool(getattr(self.harness, "allow_nonfinite_outputs", False))
            inv_issues = check_output_invariants(
                run_res.outputs,
                self.harness.root_outputs,
                allow_nonfinite=allow_nf,
            )
            if inv_issues:
                cs = CaseScore(
                    case_id=case.case_id,
                    loss=float("inf"),
                    passed=False,
                    invariant_violations=inv_issues,
                    execution_time_ms=run_res.total_time_ms,
                    error="; ".join(inv_issues),
                )
                scores.append(cs)
                return FunnelTierResult(
                    tier=1,
                    passed=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    case_scores=scores,
                    prune_reason=f"Tier 1 output invariant violation on '{case.case_id}': {'; '.join(inv_issues)}",
                )

            scores.append(
                CaseScore(
                    case_id=case.case_id,
                    loss=0.0,
                    passed=True,
                    execution_time_ms=run_res.total_time_ms,
                )
            )
            if nominal_case is None and case.regime == "nominal":
                nominal_case = case
                nominal_outputs = dict(run_res.outputs)

        # Metamorphic invariant checks on the nominal synthetic case
        relations: Sequence[MetamorphicRelation] = getattr(self.harness, "metamorphic_relations", ()) or ()
        if self.config.enable_tier1_metamorphic and relations and nominal_case is not None and nominal_outputs is not None:
            for rel in relations:
                transformed_inputs = rel.transform_inputs(nominal_case.inputs)
                tx_case = EvalCase(
                    case_id=f"{nominal_case.case_id}__{rel.name}",
                    inputs=transformed_inputs,
                    fit_inputs=nominal_case.fit_inputs,
                    regime="metamorphic",
                    tier=1,
                )
                tx_run = self._execute_case(
                    cdg,
                    tx_case,
                    node_params=node_params,
                    match_results=match_results,
                )
                if not tx_run.ok:
                    return FunnelTierResult(
                        tier=1,
                        passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        case_scores=scores,
                        failed_node_id=tx_run.failed_node_id,
                        prune_reason=f"Metamorphic relation '{rel.name}' crashed: {tx_run.error}",
                    )
                rel_ok, rel_msg = rel.verify_outputs(nominal_outputs, tx_run.outputs)
                if not rel_ok:
                    return FunnelTierResult(
                        tier=1,
                        passed=False,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        case_scores=scores,
                        prune_reason=f"Metamorphic relation '{rel.name}' failed: {rel_msg}",
                    )

        return FunnelTierResult(
            tier=1,
            passed=True,
            loss=0.0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            case_scores=scores,
        )

    def _evaluate_case_list(
        self,
        tier: int,
        cases: Sequence[EvalCase],
        cdg: CDGExport,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> tuple[FunnelTierResult, BenchmarkResult]:
        t0 = time.perf_counter()
        if not cases:
            empty_bm = BenchmarkResult(global_loss=0.0, runtime_artifacts={"cache_stats": self.runner.cache.stats()})
            return (
                FunnelTierResult(tier=tier, passed=True, loss=0.0, elapsed_ms=0.0),
                empty_bm,
            )

        agg_telemetry_time: dict[str, float] = {}
        agg_telemetry_mem: dict[str, int] = {}
        agg_telemetry_err: dict[str, float] = {}
        last_run: LiveRunResult | None = None
        first_inputs: dict[str, Any] = dict(cases[0].inputs)
        case_scores: list[CaseScore] = []
        weighted_loss_sum = 0.0
        weight_sum = 0.0

        for case in cases:
            run_res = self._execute_case(
                cdg,
                case,
                node_params=node_params,
                match_results=match_results,
            )
            last_run = run_res
            for nid, nt in run_res.node_telemetry.items():
                agg_telemetry_time[nid] = agg_telemetry_time.get(nid, 0.0) + nt.execution_time_ms
                agg_telemetry_mem[nid] = max(agg_telemetry_mem.get(nid, 0), nt.peak_memory_bytes)
                agg_telemetry_err[nid] = max(agg_telemetry_err.get(nid, 0.0), nt.error_expansion)

            if not run_res.ok:
                cs = CaseScore(
                    case_id=case.case_id,
                    loss=float("inf"),
                    passed=False,
                    execution_time_ms=run_res.total_time_ms,
                    error=run_res.error,
                    failed_node_id=run_res.failed_node_id,
                )
                case_scores.append(cs)
                bm = run_res.to_benchmark_result(global_loss=float("inf"))
                bm.runtime_artifacts["cache_stats"] = self.runner.cache.stats()
                bm.runtime_artifacts["runtime_inputs"] = first_inputs
                return (
                    FunnelTierResult(
                        tier=tier,
                        passed=False,
                        loss=float("inf"),
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        case_scores=case_scores,
                        failed_node_id=run_res.failed_node_id,
                        prune_reason=f"Tier {tier} execution failed on '{case.case_id}': {run_res.error}",
                    ),
                    bm,
                )

            cscore = self.harness.score(case, run_res.outputs)
            cscore.execution_time_ms = run_res.total_time_ms
            case_scores.append(cscore)
            if not cscore.passed or not math.isfinite(cscore.loss):
                bm = run_res.to_benchmark_result(global_loss=float("inf"))
                bm.runtime_artifacts["cache_stats"] = self.runner.cache.stats()
                bm.runtime_artifacts["runtime_inputs"] = first_inputs
                return (
                    FunnelTierResult(
                        tier=tier,
                        passed=False,
                        loss=float("inf"),
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        case_scores=case_scores,
                        failed_node_id=cscore.failed_node_id,
                        prune_reason=cscore.error or f"Tier {tier} score invalid on '{case.case_id}'",
                    ),
                    bm,
                )

            w = max(1e-9, float(case.weight))
            weighted_loss_sum += w * cscore.loss
            weight_sum += w

        global_loss = weighted_loss_sum / weight_sum if weight_sum > 0 else 0.0
        merged_telemetry = {
            nid: NodeTelemetry(
                node_id=nid,
                execution_time_ms=agg_telemetry_time[nid],
                peak_memory_bytes=agg_telemetry_mem.get(nid, 0),
                error_expansion=agg_telemetry_err.get(nid, 0.0),
            )
            for nid in agg_telemetry_time
        }
        assert last_run is not None
        base_bm = last_run.to_benchmark_result(global_loss=global_loss)
        artifacts = dict(base_bm.runtime_artifacts)
        artifacts["runtime_inputs"] = first_inputs
        artifacts["cache_stats"] = self.runner.cache.stats()
        artifacts["case_scores"] = [
            {
                "case_id": cs.case_id,
                "loss": cs.loss,
                "passed": cs.passed,
                "metrics": cs.metrics,
                "execution_time_ms": cs.execution_time_ms,
            }
            for cs in case_scores
        ]
        artifacts["telemetry_summary"] = {
            "case_count": len(cases),
            "global_loss": global_loss,
            "intermediates": dict(last_run.intermediate_summaries),
        }
        bm = BenchmarkResult(
            global_loss=global_loss,
            node_telemetry=merged_telemetry,
            runtime_artifacts=artifacts,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return (
            FunnelTierResult(
                tier=tier,
                passed=True,
                loss=global_loss,
                elapsed_ms=elapsed,
                case_scores=case_scores,
            ),
            bm,
        )

    def evaluate_tier2(
        self,
        cdg: CDGExport,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> tuple[FunnelTierResult, BenchmarkResult]:
        """Tier 2: Resident validation-slice evaluation with ``IntermediateCache``."""
        return self._evaluate_case_list(
            2,
            self.harness.validation_cases(),
            cdg,
            node_params=node_params,
            match_results=match_results,
        )

    def evaluate_tier3_holdout(
        self,
        cdg: CDGExport,
        *,
        validation_loss: float | None = None,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> FunnelTierResult:
        """Tier 3: Holdout evaluation and overfitting generalization-gap check."""
        t3_res, _bm = self._evaluate_case_list(
            3,
            self.harness.holdout_cases(),
            cdg,
            node_params=node_params,
            match_results=match_results,
        )
        if not t3_res.passed:
            return t3_res

        overfit = False
        gap = 0.0
        if validation_loss is not None and math.isfinite(validation_loss):
            gap = float(t3_res.loss) - float(validation_loss)
            denom = max(abs(float(validation_loss)), 1.0)
            rel_gap = gap / denom
            if rel_gap > self.config.holdout_overfit_tolerance:
                overfit = True

        t3_res.details["validation_loss"] = validation_loss
        t3_res.details["holdout_loss"] = t3_res.loss
        t3_res.details["generalization_gap"] = gap
        t3_res.details["overfit_detected"] = overfit
        if overfit:
            t3_res.passed = False
            t3_res.prune_reason = (
                f"Tier 3 holdout overfitting detected: holdout_loss={t3_res.loss:.5f} vs "
                f"validation_loss={validation_loss:.5f} (gap={gap:.5f})"
            )
        return t3_res

    def evaluate_candidate(
        self,
        cdg: CDGExport,
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> FunnelEvaluationReport:
        """Run Tier 0 -> Tier 1 -> Tier 2 sequentially, pruning immediately on failure."""
        tier_results: dict[int, FunnelTierResult] = {}

        t0_res, ghost_report, unc_report = self.evaluate_tier0(cdg, match_results=match_results)
        tier_results[0] = t0_res
        if not t0_res.passed:
            bm = BenchmarkResult(
                global_loss=float("inf"),
                runtime_artifacts={
                    "pruned_at_tier": 0,
                    "prune_reason": t0_res.prune_reason,
                    "failed_node_id": t0_res.failed_node_id,
                    "cache_stats": self.runner.cache.stats(),
                },
            )
            return FunnelEvaluationReport(
                passed=False,
                pruned_at_tier=0,
                global_loss=float("inf"),
                benchmark=bm,
                tier_results=tier_results,
                ghost_report=ghost_report,
                uncertainty_report=unc_report,
                failed_node_id=t0_res.failed_node_id,
                prune_reason=t0_res.prune_reason,
            )

        t1_res = self.evaluate_tier1(
            cdg,
            node_params=node_params,
            match_results=match_results,
        )
        tier_results[1] = t1_res
        if not t1_res.passed:
            bm = BenchmarkResult(
                global_loss=float("inf"),
                runtime_artifacts={
                    "pruned_at_tier": 1,
                    "prune_reason": t1_res.prune_reason,
                    "failed_node_id": t1_res.failed_node_id,
                    "cache_stats": self.runner.cache.stats(),
                },
            )
            return FunnelEvaluationReport(
                passed=False,
                pruned_at_tier=1,
                global_loss=float("inf"),
                benchmark=bm,
                tier_results=tier_results,
                ghost_report=ghost_report,
                uncertainty_report=unc_report,
                failed_node_id=t1_res.failed_node_id,
                prune_reason=t1_res.prune_reason,
            )

        t2_res, bm = self.evaluate_tier2(
            cdg,
            node_params=node_params,
            match_results=match_results,
        )
        tier_results[2] = t2_res
        bm.runtime_artifacts["funnel_tier_timings_ms"] = {
            t: round(r.elapsed_ms, 4) for t, r in tier_results.items()
        }
        if not t2_res.passed:
            bm.runtime_artifacts["pruned_at_tier"] = 2
            bm.runtime_artifacts["prune_reason"] = t2_res.prune_reason
            return FunnelEvaluationReport(
                passed=False,
                pruned_at_tier=2,
                global_loss=float("inf"),
                benchmark=bm,
                tier_results=tier_results,
                ghost_report=ghost_report,
                uncertainty_report=unc_report,
                failed_node_id=t2_res.failed_node_id,
                prune_reason=t2_res.prune_reason,
            )

        return FunnelEvaluationReport(
            passed=True,
            pruned_at_tier=None,
            global_loss=t2_res.loss,
            benchmark=bm,
            tier_results=tier_results,
            ghost_report=ghost_report,
            uncertainty_report=unc_report,
        )


class LivePrototypingSandbox:
    """Zero-subprocess drop-in ``sandbox`` + ``synthesize_fn`` adapter for ``PrincipalDeps``.

    Wire into ``PrincipalDeps`` via:
    ``PrincipalDeps(architect=..., sandbox=live_sandbox, synthesize_fn=live_sandbox.synthesize_virtual_bundle)``
    """

    def __init__(
        self,
        funnel: EvaluationFunnel,
        *,
        virtual_root: Path | None = None,
    ) -> None:
        self.funnel = funnel
        self.virtual_root = virtual_root or Path("/virtual/sciona_live_prototyping")
        self._virtual_bundles: dict[str, tuple[CDGExport, list[Any]]] = {}
        self.evaluation_history: list[FunnelEvaluationReport] = []

    async def synthesize_virtual_bundle(
        self,
        cdg: CDGExport,
        match_results: list[Any],
    ) -> ExportBundle:
        """Register ``(cdg, match_results)`` in RAM without touching disk or spawning compilers."""
        token = uuid.uuid4().hex
        source_path = self.virtual_root / f"{token}.py"
        self._virtual_bundles[str(source_path)] = (cdg, list(match_results or []))
        return ExportBundle(
            target="live-in-process",
            output_dir=self.virtual_root,
            source_path=source_path,
            executable_artifact=source_path,
        )

    def _resolve_bundle(self, bundle: ExportBundle) -> tuple[CDGExport, list[Any]]:
        key = str(bundle.source_path)
        entry = self._virtual_bundles.get(key)
        if entry is None:
            src_p = Path(bundle.source_path)
            if src_p.is_file() and src_p.suffix.lower() == ".json":
                from sciona.architect.handoff import load_json

                loaded_cdg = load_json(src_p)
                self._virtual_bundles[key] = (loaded_cdg, [])
                return loaded_cdg, []
            raise KeyError(
                f"Virtual bundle '{key}' is not registered in this LivePrototypingSandbox"
            )
        return entry

    async def evaluate(
        self,
        bundle: ExportBundle,
        dataset_path: str,
        metric: OptimizationMetric,
        *,
        evaluation_spec: dict[str, Any] | str | None = None,
    ) -> BenchmarkResult:
        """Evaluate a virtual bundle through the in-process ``EvaluationFunnel``."""
        _ = (dataset_path, evaluation_spec)
        cdg, match_results = self._resolve_bundle(bundle)
        report = self.funnel.evaluate_candidate(
            cdg,
            node_params=bundle.parameter_assignments or None,
            match_results=match_results,
        )
        self.evaluation_history.append(report)
        bm = report.benchmark
        if not report.passed:
            return bm

        if metric == OptimizationMetric.LATENCY:
            total_lat = sum(nt.execution_time_ms for nt in bm.node_telemetry.values())
            bm = bm.model_copy(update={"global_loss": float(total_lat)})
        elif metric == OptimizationMetric.MEMORY:
            peak_mem = max((nt.peak_memory_bytes for nt in bm.node_telemetry.values()), default=0)
            bm = bm.model_copy(update={"global_loss": float(peak_mem)})
        return bm

    async def evaluate_adapter(
        self,
        bundle: ExportBundle,
        adapter_path: str,
        metric: OptimizationMetric,
        *,
        varset: dict[str, str] | None = None,
        evaluation_spec: dict[str, Any] | str | None = None,
        slice_start_s: float | None = None,
        slice_stop_s: float | None = None,
    ) -> BenchmarkResult:
        """Compatibility shim when ``PrincipalState.dataset_path`` ends in ``.yml``/``.yaml``."""
        _ = (varset, slice_start_s, slice_stop_s)
        return await self.evaluate(
            bundle,
            adapter_path,
            metric,
            evaluation_spec=evaluation_spec,
        )
