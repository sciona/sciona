"""Unified stateful ``PrototypingSession`` backing CLI (``sciona proto``), SDK, and Visualizer.

Drives iterative test-diagnose-revise cycles in memory via:
``EvaluationFunnel -> UniversalCreditAssigner -> RevisionLadder (Rung 1..4b) -> Tier 3 Holdout -> Export``.
"""

from __future__ import annotations

import ast
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.compiled_catalog import ManifestFTSIndex
from sciona.architect.handoff import CDGExport
from sciona.architect.models import EdgeKind
from sciona.principal.atom_ledger import AtomLedger
from sciona.principal.expansion import ExpansionEngine
from sciona.principal.models import NodeGradient, OptimizationMetric
from sciona.synthesizer.toposort import toposort_nodes
from sciona.types import MatchResult

from sciona.prototyping.credit import UniversalCreditAssigner
from sciona.prototyping.funnel import (
    EvaluationFunnel,
    FunnelConfig,
    FunnelEvaluationReport,
    FunnelTierResult,
)
from sciona.prototyping.harness import ProblemHarness, SyntheticSuiteHarness
from sciona.prototyping.ladder import (
    MicroAtomSynthesizerCallable,
    RevisionLadder,
    SubdecomposeCallable,
)
from sciona.prototyping.live_runner import LiveCDGRunner, resolve_atom_callable


@dataclass
class PrototypingStepRecord:
    """Telemetry and provenance record for a single prototyping iteration."""

    iteration: int
    rung: str
    improved: bool
    loss_before: float
    loss_after: float
    bottleneck_node_id: str
    bottleneck_reason: str
    trials_evaluated: int
    elapsed_ms: float
    cache_hit_rate: float
    rationale: str
    gradients: list[NodeGradient] = field(default_factory=list)
    synthesized_atom_fqn: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "rung": self.rung,
            "improved": self.improved,
            "loss_before": self.loss_before if math.isfinite(self.loss_before) else None,
            "loss_after": self.loss_after if math.isfinite(self.loss_after) else None,
            "bottleneck_node_id": self.bottleneck_node_id,
            "bottleneck_reason": self.bottleneck_reason,
            "trials_evaluated": self.trials_evaluated,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "rationale": self.rationale,
            "synthesized_atom_fqn": self.synthesized_atom_fqn,
            "gradients": [
                {
                    "node_id": g.node_id,
                    "gradient_score": round(g.gradient_score, 3),
                    "reason": g.bottleneck_reason,
                }
                for g in self.gradients
            ],
        }


@dataclass
class PrototypingSessionSummary:
    """Final outcome of a ``PrototypingSession.run()`` execution."""

    session_id: str
    goal: str
    initial_loss: float
    best_loss: float
    iterations_completed: int
    total_trials_evaluated: int
    total_elapsed_ms: float
    best_cdg: CDGExport
    best_node_params: dict[str, dict[str, Any]]
    holdout_result: FunnelTierResult | None
    history: list[PrototypingStepRecord]
    cache_stats: dict[str, Any]
    synthesized_micro_atoms: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "initial_loss": self.initial_loss if math.isfinite(self.initial_loss) else None,
            "best_loss": self.best_loss if math.isfinite(self.best_loss) else None,
            "iterations_completed": self.iterations_completed,
            "total_trials_evaluated": self.total_trials_evaluated,
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "best_node_params": self.best_node_params,
            "holdout": (
                {
                    "passed": self.holdout_result.passed,
                    "loss": self.holdout_result.loss if math.isfinite(self.holdout_result.loss) else None,
                    "details": self.holdout_result.details,
                    "prune_reason": self.holdout_result.prune_reason,
                }
                if self.holdout_result is not None
                else None
            ),
            "history": [h.to_dict() for h in self.history],
            "cache_stats": self.cache_stats,
            "synthesized_micro_atom_count": len(self.synthesized_micro_atoms),
        }


class PrototypingSession:
    """Interactive and batch in-memory CDG prototyping & refinement session."""

    def __init__(
        self,
        cdg: CDGExport,
        *,
        goal: str = "",
        session_id: str = "proto",
        harness: ProblemHarness | None = None,
        runner: LiveCDGRunner | None = None,
        funnel_config: FunnelConfig | None = None,
        catalog: PrimitiveCatalog | None = None,
        manifest_index: ManifestFTSIndex | None = None,
        manifest_sqlite_path: Path | None = None,
        atom_ledger: AtomLedger | None = None,
        expansion_engine: ExpansionEngine | None = None,
        subdecompose_fn: SubdecomposeCallable | None = None,
        micro_atom_synthesizer: MicroAtomSynthesizerCallable | None = None,
        metric: OptimizationMetric = OptimizationMetric.PRECISION,
        match_results: Sequence[MatchResult] | None = None,
        initial_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        hpo_trials_per_rung: int = 8,
        max_sibling_evals: int = 5,
    ) -> None:
        self.session_id = session_id
        self.goal = goal or str(cdg.metadata.get("goal") or "Prototyping pipeline")
        self.initial_cdg = cdg.model_copy(deep=True)
        self.best_cdg = cdg.model_copy(deep=True)
        self.best_node_params: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in (initial_node_params or {}).items()
        }
        self.metric = metric
        self.match_results = list(match_results or [])

        self.harness: ProblemHarness = (
            harness if harness is not None else SyntheticSuiteHarness.from_cdg(cdg)
        )
        self.runner = runner if runner is not None else LiveCDGRunner()
        self.funnel = EvaluationFunnel(
            self.harness,
            runner=self.runner,
            config=funnel_config,
        )
        self.credit_assigner = UniversalCreditAssigner()
        self.ladder = RevisionLadder(
            self.funnel,
            catalog=catalog,
            manifest_index=manifest_index,
            manifest_sqlite_path=manifest_sqlite_path,
            atom_ledger=atom_ledger,
            expansion_engine=expansion_engine,
            subdecompose_fn=subdecompose_fn,
            micro_atom_synthesizer=micro_atom_synthesizer,
            hpo_trials_per_rung=hpo_trials_per_rung,
            max_sibling_evals=max_sibling_evals,
        )

        self.initial_loss: float = float("inf")
        self.best_loss: float = float("inf")
        self.best_report: FunnelEvaluationReport | None = None
        self.last_gradients: list[NodeGradient] = []
        self.history: list[PrototypingStepRecord] = []
        self.synthesized_micro_atoms: dict[str, str] = {}
        self._initialized = False

    def evaluate_initial(self) -> PrototypingStepRecord:
        """Run initial Tier 0..2 funnel evaluation and universal credit assignment."""
        t0 = time.perf_counter()
        rep = self.funnel.evaluate_candidate(
            self.best_cdg,
            node_params=self.best_node_params or None,
            match_results=self.match_results,
        )
        self.initial_loss = rep.global_loss
        self.best_loss = rep.global_loss
        self.best_report = rep
        self.last_gradients = self.credit_assigner.compute_gradients(
            self.best_cdg,
            rep.benchmark,
            sim_report=rep.ghost_report,
            target=self.metric,
            harness=self.harness,
            runner=self.runner,
            node_params=self.best_node_params or None,
            match_results=self.match_results,
        )
        self._initialized = True

        top_g = self.last_gradients[0] if self.last_gradients else None
        cstats = self.runner.cache.stats()
        rec = PrototypingStepRecord(
            iteration=0,
            rung="initial_evaluation",
            improved=rep.passed,
            loss_before=float("inf"),
            loss_after=rep.global_loss,
            bottleneck_node_id=top_g.node_id if top_g is not None else rep.failed_node_id,
            bottleneck_reason=(
                top_g.bottleneck_reason if top_g is not None else rep.prune_reason
            ),
            trials_evaluated=1,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            cache_hit_rate=float(cstats.get("hit_rate", 0.0)),
            rationale=(
                f"Initial candidate loss={rep.global_loss:.5f}"
                if rep.passed
                else f"Initial candidate pruned at Tier {rep.pruned_at_tier}: {rep.prune_reason}"
            ),
            gradients=list(self.last_gradients),
        )
        self.history.append(rec)
        return rec

    async def step(self, *, max_rung: int = 4) -> PrototypingStepRecord:
        """Execute one iterative diagnose -> climb ladder -> update cycle."""
        if not self._initialized or self.best_report is None:
            self.evaluate_initial()
        assert self.best_report is not None

        iter_idx = len(self.history)
        loss_before = self.best_loss

        # Refresh gradients on current best state
        self.last_gradients = self.credit_assigner.compute_gradients(
            self.best_cdg,
            self.best_report.benchmark,
            sim_report=self.best_report.ghost_report,
            target=self.metric,
            harness=self.harness,
            runner=self.runner,
            node_params=self.best_node_params or None,
            match_results=self.match_results,
        )

        outcome = await self.ladder.climb(
            self.best_cdg,
            self.last_gradients,
            goal=self.goal,
            baseline_loss=loss_before,
            baseline_benchmark=self.best_report.benchmark,
            current_node_params=self.best_node_params,
            match_results=self.match_results,
            uncertainty_report=self.best_report.uncertainty_report,
            max_rung=max_rung,
            trial_index=iter_idx,
        )

        if outcome.improved and outcome.accepted_report is not None:
            self.best_cdg = outcome.accepted_cdg
            self.best_node_params = dict(outcome.accepted_node_params)
            self.best_loss = outcome.accepted_loss
            self.best_report = outcome.accepted_report
            if outcome.synthesized_atom_fqn and outcome.synthesized_source_code:
                self.synthesized_micro_atoms[outcome.synthesized_atom_fqn] = outcome.synthesized_source_code

        top_g = self.last_gradients[0] if self.last_gradients else None
        cstats = self.runner.cache.stats()
        rec = PrototypingStepRecord(
            iteration=iter_idx,
            rung=outcome.rung,
            improved=outcome.improved,
            loss_before=loss_before,
            loss_after=self.best_loss,
            bottleneck_node_id=outcome.bottleneck_node_id or (top_g.node_id if top_g else ""),
            bottleneck_reason=top_g.bottleneck_reason if top_g else "",
            trials_evaluated=outcome.trials_evaluated,
            elapsed_ms=outcome.elapsed_ms,
            cache_hit_rate=float(cstats.get("hit_rate", 0.0)),
            rationale=outcome.rationale,
            gradients=list(self.last_gradients),
            synthesized_atom_fqn=outcome.synthesized_atom_fqn,
        )
        self.history.append(rec)
        return rec

    async def run(
        self,
        *,
        max_iterations: int = 6,
        time_budget_s: float = 15.0,
        target_loss: float | None = None,
        max_rung: int = 4,
        verify_holdout: bool = True,
    ) -> PrototypingSessionSummary:
        """Run iterative refinement until convergence, target loss, or budget expiry."""
        t_start = time.perf_counter()
        if not self._initialized:
            self.evaluate_initial()

        stop_loss = target_loss if target_loss is not None else 1e-12
        for _ in range(max(0, int(max_iterations))):
            if math.isfinite(self.best_loss) and self.best_loss <= stop_loss:
                break
            if (time.perf_counter() - t_start) >= time_budget_s:
                break
            step_rec = await self.step(max_rung=max_rung)
            if not step_rec.improved:
                break

        holdout_res: FunnelTierResult | None = None
        if verify_holdout and math.isfinite(self.best_loss):
            holdout_res = self.funnel.evaluate_tier3_holdout(
                self.best_cdg,
                validation_loss=self.best_loss,
                node_params=self.best_node_params or None,
                match_results=self.match_results,
            )

        total_trials = sum(r.trials_evaluated for r in self.history)
        return PrototypingSessionSummary(
            session_id=self.session_id,
            goal=self.goal,
            initial_loss=self.initial_loss,
            best_loss=self.best_loss,
            iterations_completed=max(0, len(self.history) - 1),
            total_trials_evaluated=total_trials,
            total_elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
            best_cdg=self.best_cdg,
            best_node_params=dict(self.best_node_params),
            holdout_result=holdout_res,
            history=list(self.history),
            cache_stats=self.runner.cache.stats(),
            synthesized_micro_atoms=dict(self.synthesized_micro_atoms),
        )

    def export_cdg(self, output_path: str | Path) -> Path:
        """Persist the winning ``CDGExport`` + tuned ``node_params`` to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        exported = self.best_cdg.model_copy(deep=True)
        exported.metadata = dict(exported.metadata or {})
        exported.metadata["prototyping_best_loss"] = (
            self.best_loss if math.isfinite(self.best_loss) else None
        )
        exported.metadata["best_node_params"] = dict(self.best_node_params)
        if self.synthesized_micro_atoms:
            exported.metadata["synthesized_micro_atoms"] = dict(self.synthesized_micro_atoms)
        out.write_text(
            json.dumps(exported.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    def export_python_module(self, output_path: str | Path) -> Path:
        """Export the winning CDG + tuned parameters + synthesized micro-atoms as a standalone ``.py`` file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        leaves = self.best_cdg.leaf_nodes()
        leaf_ids = {n.node_id for n in leaves}
        leaf_map = {n.node_id: n for n in leaves}
        leaf_edges = [
            e for e in self.best_cdg.edges if e.source_id in leaf_ids and e.target_id in leaf_ids
        ]
        order = toposort_nodes(leaves, leaf_edges)

        lines: list[str] = [
            f'"""Standalone pipeline exported by Sciona PrototypingSession ({self.goal})."""',
            "",
            "from __future__ import annotations",
            "",
            "import importlib",
            "import inspect",
            "from typing import Any",
            "",
            "import numpy as np",
            "",
        ]

        # Embed any Rung 4b synthesized micro-atoms directly into the exported module!
        local_fn_aliases: dict[str, str] = {}
        if self.synthesized_micro_atoms:
            lines.append("# --- Synthesized Micro-Atoms (Rung 4b) ---")
            for fqn, code in self.synthesized_micro_atoms.items():
                cleaned_code = code.strip()
                lines.append(cleaned_code)
                lines.append("")
                fn_short = fqn.rsplit(".", 1)[-1].rsplit("_", 1)[0]
                try:
                    tree = ast.parse(cleaned_code, mode="exec")
                    defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
                    if defs:
                        fn_short = defs[-1]
                except Exception:
                    pass
                local_fn_aliases[fqn] = fn_short

        # Resolve importable module.symbol paths for standard catalog/Python atoms
        lines.extend(
            [
                "def _load_callable(mod_name: str, sym_name: str) -> Any:",
                "    return getattr(importlib.import_module(mod_name), sym_name)",
                "",
                "def _invoke_stage(fn: Any, port_order: tuple[str, ...], port_values: dict[str, Any]) -> Any:",
                "    try:",
                "        sig = inspect.signature(fn)",
                "    except (TypeError, ValueError):",
                "        return fn(**port_values)",
                "    params = list(sig.parameters.values())",
                "    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)",
                "    used: set[str] = set()",
                "    pos_args: list[Any] = []",
                "    kw_args: dict[str, Any] = {}",
                "    ordered_keys = [k for k in port_order if k in port_values] + [",
                "        k for k in port_values if k not in port_order",
                "    ]",
                "    for p in params:",
                "        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):",
                "            continue",
                "        val_found = False",
                "        val: Any = None",
                "        if p.name in port_values and p.name not in used:",
                "            used.add(p.name)",
                "            val_found, val = True, port_values[p.name]",
                "        elif p.default is inspect.Parameter.empty:",
                "            for ok in ordered_keys:",
                "                if ok not in used:",
                "                    used.add(ok)",
                "                    val_found, val = True, port_values[ok]",
                "                    break",
                "        if not val_found:",
                "            continue",
                "        if p.kind == inspect.Parameter.POSITIONAL_ONLY:",
                "            pos_args.append(val)",
                "        else:",
                "            kw_args[p.name] = val",
                "    if accepts_kw:",
                "        for k, v in port_values.items():",
                "            if k not in used and k not in kw_args:",
                "                kw_args[k] = v",
                "    return fn(*pos_args, **kw_args)",
                "",
            ]
        )

        node_fn_vars: dict[str, str] = {}
        for nid in order:
            node = leaf_map[nid]
            prim = str(node.matched_primitive or "")
            safe_var = f"_fn_{nid}".replace("-", "_").replace(".", "_")
            node_fn_vars[nid] = safe_var
            if prim in local_fn_aliases:
                lines.append(f"{safe_var} = {local_fn_aliases[prim]}")
                continue
            fn_obj: Callable[..., Any] | None = None
            resolved_fqn = prim
            try:
                fn_obj, resolved_fqn = resolve_atom_callable(
                    prim,
                    node=node,
                    extra_callables=self.runner.extra_callables,
                )
            except Exception:
                fn_obj = None
            mod_attr = str(getattr(fn_obj, "__module__", None) or "") if fn_obj is not None else ""
            sym_attr = str(getattr(fn_obj, "__name__", None) or "") if fn_obj is not None else ""
            if mod_attr and sym_attr and not mod_attr.startswith("sciona_synthesized_"):
                lines.append(f'{safe_var} = _load_callable({mod_attr!r}, {sym_attr!r})')
            elif "." in resolved_fqn:
                mod_n, sym_n = resolved_fqn.rsplit(".", 1)
                lines.append(f'{safe_var} = _load_callable({mod_n!r}, {sym_n!r})')
            else:
                lines.append(f"# Fallback lookup for {prim!r}")
                lines.append("from sciona.ghost.registry import REGISTRY as _REG")
                lines.append(f"{safe_var} = _REG[{prim!r}]['impl']")

        lines.append("")
        lines.append(f"TUNED_NODE_PARAMS: dict[str, dict[str, Any]] = {self.best_node_params!r}")
        lines.append("")
        lines.append("def run_pipeline(**inputs: Any) -> dict[str, Any]:")
        lines.append('    """Execute the refined CDG pipeline on keyword inputs."""')
        lines.append("    node_outs: dict[str, dict[str, Any]] = {}")

        incoming_by_tgt: dict[str, list[Any]] = {nid: [] for nid in leaf_ids}
        outgoing_data_sources: set[str] = set()
        outgoing_callable_sources: set[str] = set()
        for e in leaf_edges:
            incoming_by_tgt[e.target_id].append(e)
            if e.edge_kind == EdgeKind.CALLABLE_INJECTION:
                outgoing_callable_sources.add(e.source_id)
            else:
                outgoing_data_sources.add(e.source_id)
        callable_only_providers = outgoing_callable_sources - outgoing_data_sources

        total_unwired_required = 0
        for nid in order:
            if nid in callable_only_providers:
                continue
            wired_in = {e.input_name for e in incoming_by_tgt[nid]}
            for p_spec in leaf_map[nid].inputs:
                if p_spec.required and p_spec.name not in wired_in:
                    total_unwired_required += 1

        for nid in order:
            if nid in callable_only_providers:
                continue
            node = leaf_map[nid]
            wired_in = {e.input_name for e in incoming_by_tgt[nid]}
            port_order_tuple = tuple(p.name for p in node.inputs)
            lines.append(f"    # Stage: {nid} ({node.matched_primitive})")
            lines.append("    _kwargs: dict[str, Any] = {}")
            for p_spec in node.inputs:
                if p_spec.name in wired_in:
                    continue
                lines.append(f"    if {p_spec.name!r} in inputs:")
                lines.append(f"        _kwargs[{p_spec.name!r}] = inputs[{p_spec.name!r}]")
                if p_spec.required and total_unwired_required == 1:
                    lines.append("    elif len(inputs) == 1:")
                    lines.append(f"        _kwargs[{p_spec.name!r}] = next(iter(inputs.values()))")
            for edge in incoming_by_tgt[nid]:
                if edge.edge_kind == EdgeKind.CALLABLE_INJECTION:
                    lines.append(f"    _kwargs[{edge.input_name!r}] = {node_fn_vars[edge.source_id]}")
                else:
                    lines.append(
                        f"    _kwargs[{edge.input_name!r}] = node_outs[{edge.source_id!r}][{edge.output_name!r}]"
                    )
            lines.append(f"    _kwargs.update(TUNED_NODE_PARAMS.get({nid!r}, {{}}))")
            lines.append(f"    _res = _invoke_stage({node_fn_vars[nid]}, {port_order_tuple!r}, _kwargs)")
            if len(node.outputs) <= 1:
                oname = node.outputs[0].name if node.outputs else "out"
                lines.append(f"    node_outs[{nid!r}] = {{{oname!r}: _res}}")
            else:
                out_pairs = ", ".join(
                    f"{o.name!r}: _res[{i}]" for i, o in enumerate(node.outputs)
                )
                lines.append(f"    node_outs[{nid!r}] = {{{out_pairs}}}")

        lines.append("    outputs: dict[str, Any] = {}")
        for nid in order:
            if nid not in outgoing_data_sources and nid not in callable_only_providers:
                lines.append(f"    outputs.update(node_outs[{nid!r}])")
        lines.append("    return outputs")
        lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        return out
