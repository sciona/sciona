"""Canonical Frontier-LLM Convergence Reference Loader & Trajectory Comparator.

Provides zero-cost, offline access to the canonical 12-iteration locked-holdout
convergence curves (`eval_01`..`eval_05`) recorded by isolated blind frontier-LLM
agents across five mathematical disciplines, allowing future Sciona revisions to
benchmark speed, token economy, feasibility, and holdout loss without re-running
multi-million-token LLM baselines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FRONTIER_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "references" / "frontier_llm_reference_v1.json"
)


@dataclass(frozen=True)
class FrontierCheckpoint:
    """Single checkpoint along a canonical frontier-agent convergence curve."""

    iteration: int
    stage_category: str
    elapsed_s: float
    planner_turns: int
    cum_input_tokens: int
    cum_cache_read_tokens: int
    cum_output_tokens: int
    cum_fresh_tokens: int
    cum_total_tokens: int
    val_metric: float
    val_loss: float
    val_feasible: bool
    incumbent_val_loss: float
    holdout_metric: float
    holdout_loss: float
    holdout_feasible: bool
    incumbent_holdout_metric: float
    incumbent_holdout_loss: float
    incumbent_holdout_feasible: bool


@dataclass(frozen=True)
class FrontierTaskReference:
    """Canonical frontier-LLM convergence trajectory for one opaque benchmark task."""

    task_id: str
    discipline: str
    title: str
    metric_name: str
    loss_formula: str
    higher_is_better: bool
    has_hard_constraints: bool
    isolation_verified: bool
    total_planner_turns: int
    total_tool_calls: int
    total_input_tokens: int
    total_cache_read_tokens: int
    total_output_tokens: int
    total_fresh_tokens: int
    total_billed_context_tokens: int
    total_wall_clock_s: float
    initial_holdout_loss: float
    iter5_incumbent_holdout_loss: float
    final_incumbent_holdout_loss: float
    checkpoints: tuple[FrontierCheckpoint, ...]


@dataclass(frozen=True)
class FrontierReferenceSuite:
    """Suite-wide canonical frontier-LLM convergence reference."""

    schema_version: str
    reference_id: str
    description: str
    sha256_digest: str
    total_wall_clock_s: float
    total_planner_turns: int
    total_tool_calls: int
    total_fresh_tokens: int
    total_billed_context_tokens: int
    tasks: Mapping[str, FrontierTaskReference]


def load_frontier_llm_reference(
    path: Path | str | None = None,
    *,
    verify_digest: bool = True,
) -> FrontierReferenceSuite:
    """Load and integrity-check the canonical frontier-LLM convergence reference."""
    ref_path = Path(path) if path is not None else DEFAULT_FRONTIER_REFERENCE_PATH
    raw_obj: dict[str, Any] = json.loads(ref_path.read_text(encoding="utf-8"))
    recorded_digest = str(raw_obj.get("sha256_digest") or "")

    if verify_digest and recorded_digest:
        check_copy = {k: v for k, v in raw_obj.items() if k != "sha256_digest"}
        computed = hashlib.sha256(
            json.dumps(check_copy, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if computed != recorded_digest:
            raise ValueError(
                f"Canonical frontier reference digest mismatch: expected {recorded_digest}, got {computed}"
            )

    tasks: dict[str, FrontierTaskReference] = {}
    for tid, tdata in (raw_obj.get("tasks") or {}).items():
        ckpts = tuple(
            FrontierCheckpoint(
                iteration=int(c["iteration"]),
                stage_category=str(c["stage_category"]),
                elapsed_s=float(c["elapsed_s"]),
                planner_turns=int(c["planner_turns"]),
                cum_input_tokens=int(c["cum_input_tokens"]),
                cum_cache_read_tokens=int(c["cum_cache_read_tokens"]),
                cum_output_tokens=int(c["cum_output_tokens"]),
                cum_fresh_tokens=int(c["cum_fresh_tokens"]),
                cum_total_tokens=int(c["cum_total_tokens"]),
                val_metric=float(c["val_metric"]),
                val_loss=float(c["val_loss"]),
                val_feasible=bool(c["val_feasible"]),
                incumbent_val_loss=float(c["incumbent_val_loss"]),
                holdout_metric=float(c["holdout_metric"]),
                holdout_loss=float(c["holdout_loss"]),
                holdout_feasible=bool(c["holdout_feasible"]),
                incumbent_holdout_metric=float(c["incumbent_holdout_metric"]),
                incumbent_holdout_loss=float(c["incumbent_holdout_loss"]),
                incumbent_holdout_feasible=bool(c["incumbent_holdout_feasible"]),
            )
            for c in tdata.get("checkpoints", [])
        )
        tasks[str(tid)] = FrontierTaskReference(
            task_id=str(tid),
            discipline=str(tdata["discipline"]),
            title=str(tdata["title"]),
            metric_name=str(tdata["metric_name"]),
            loss_formula=str(tdata["loss_formula"]),
            higher_is_better=bool(tdata["higher_is_better"]),
            has_hard_constraints=bool(tdata["has_hard_constraints"]),
            isolation_verified=bool(tdata["isolation_verified"]),
            total_planner_turns=int(tdata["total_planner_turns"]),
            total_tool_calls=int(tdata["total_tool_calls"]),
            total_input_tokens=int(tdata["total_input_tokens"]),
            total_cache_read_tokens=int(tdata["total_cache_read_tokens"]),
            total_output_tokens=int(tdata["total_output_tokens"]),
            total_fresh_tokens=int(tdata["total_fresh_tokens"]),
            total_billed_context_tokens=int(tdata["total_billed_context_tokens"]),
            total_wall_clock_s=float(tdata["total_wall_clock_s"]),
            initial_holdout_loss=float(tdata["initial_holdout_loss"]),
            iter5_incumbent_holdout_loss=float(tdata["iter5_incumbent_holdout_loss"]),
            final_incumbent_holdout_loss=float(tdata["final_incumbent_holdout_loss"]),
            checkpoints=ckpts,
        )

    totals = raw_obj.get("suite_totals") or {}
    return FrontierReferenceSuite(
        schema_version=str(raw_obj.get("schema_version", "1.0")),
        reference_id=str(raw_obj.get("reference_id", "")),
        description=str(raw_obj.get("description", "")),
        sha256_digest=recorded_digest,
        total_wall_clock_s=float(totals.get("total_wall_clock_s", 0.0)),
        total_planner_turns=int(totals.get("total_planner_turns", 0)),
        total_tool_calls=int(totals.get("total_tool_calls", 0)),
        total_fresh_tokens=int(totals.get("total_fresh_tokens", 0)),
        total_billed_context_tokens=int(totals.get("total_billed_context_tokens", 0)),
        tasks=tasks,
    )


def compare_sciona_to_frontier(
    task_id: str,
    sciona_holdout_curve: Sequence[float],
    *,
    sciona_elapsed_s: float,
    sciona_tokens: int = 0,
    reference: FrontierReferenceSuite | None = None,
) -> dict[str, Any]:
    """Compare a Sciona holdout-loss trajectory against the canonical frontier reference."""
    suite = reference if reference is not None else load_frontier_llm_reference()
    if task_id not in suite.tasks:
        raise KeyError(f"Unknown benchmark task_id '{task_id}'")
    ref_task = suite.tasks[task_id]
    if not sciona_holdout_curve:
        raise ValueError("sciona_holdout_curve must contain at least one checkpoint")

    sc_init = float(sciona_holdout_curve[0])
    sc_best = float(sciona_holdout_curve[-1])

    # Find highest frontier iteration whose incumbent holdout loss Sciona matches or beats
    frontier_equiv_iter = 0
    for ck in ref_task.checkpoints:
        if sc_best <= ck.incumbent_holdout_loss + 1e-9:
            frontier_equiv_iter = ck.iteration

    speedup_x = ref_task.total_wall_clock_s / max(float(sciona_elapsed_s), 1e-3)
    token_savings = ref_task.total_billed_context_tokens - int(sciona_tokens)
    return {
        "task_id": task_id,
        "discipline": ref_task.discipline,
        "sciona_initial_holdout_loss": round(sc_init, 6),
        "sciona_final_holdout_loss": round(sc_best, 6),
        "frontier_initial_holdout_loss": ref_task.initial_holdout_loss,
        "frontier_iter5_holdout_loss": ref_task.iter5_incumbent_holdout_loss,
        "frontier_final_holdout_loss": ref_task.final_incumbent_holdout_loss,
        "beats_frontier_initial": sc_init < ref_task.initial_holdout_loss,
        "beats_frontier_iter5": sc_best < ref_task.iter5_incumbent_holdout_loss,
        "beats_frontier_final": sc_best < ref_task.final_incumbent_holdout_loss,
        "frontier_iterations_matched_or_beaten": frontier_equiv_iter,
        "wall_clock_speedup_x": round(speedup_x, 2),
        "billed_context_tokens_saved": token_savings,
    }
