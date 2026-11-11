"""Shared state and dependency types for the Principal graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.graph import DecompositionAgent
from sciona.architect.handoff import CDGExport
from sciona.principal.atom_ledger import AtomLedger
from sciona.principal.admissibility import AdmissibilityEvaluator
from sciona.principal.evaluator import ExecutionSandbox
from sciona.principal.expansion import ExpansionEngine
from sciona.principal.hpo import OptunaManager
from sciona.principal.models import BenchmarkResult, NodeGradient, OptimizationMetric
from sciona.synthesizer.ghost_sim import GhostSimReport
from sciona.synthesizer.models import ExportBundle


@dataclass
class PrincipalState:
    """Mutable state threaded through the Principal graph."""

    goal: str = ""
    metric: OptimizationMetric = OptimizationMetric.LATENCY
    dataset_path: str = ""
    max_trials: int = 50
    current_trial: int = 0
    best_loss: float = float("inf")

    # Pipeline artefacts
    thread_id: str = ""
    cdg: CDGExport | None = None
    planning_artifact: dict[str, Any] | None = None
    export_bundle: ExportBundle | None = None
    ghost_report: GhostSimReport = field(default_factory=GhostSimReport)
    benchmark: BenchmarkResult | None = None
    match_results: list[Any] = field(default_factory=list)

    # Gradient
    top_gradient: NodeGradient | None = None
    bottleneck_node_id: str = ""
    bottleneck_reason: str = ""

    # Hyperparameter assignments (node_id -> {param_name: value})
    node_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_node_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    param_signature: str = ""
    hpo_trial_number: int | None = None
    pending_param_search: bool = False
    param_trials_remaining: int = 0
    expansion_applied: bool = False
    expansion_rules_applied: list[str] = field(default_factory=list)
    selected_proposal: str = ""
    selected_proposal_reason: str = ""
    proposal_selection_summary: dict[str, Any] = field(default_factory=dict)
    reuse_cached_evaluation: bool = False
    admissibility_summary: dict[str, Any] = field(default_factory=dict)
    admissibility_requires_refinement: bool = False
    admissibility_hard_rejected: bool = False
    pending_redecompose_cdg: CDGExport | None = None
    pending_redecompose_thread_id: str = ""

    # Bookkeeping
    done: bool = False
    error: str = ""
    trial_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PrincipalDeps:
    """Injected dependencies for the Principal graph."""

    architect: DecompositionAgent
    sandbox: ExecutionSandbox | Any
    match_results_fn: Any = None  # Callable[[CDGExport], list[MatchResult]]
    synthesize_fn: Any = None  # Callable[[CDGExport, list], Awaitable[ExportBundle]]
    evaluation_spec: Any = None
    problem_harness: Any = None  # Optional ProblemHarness for live prototyping
    dataset_varset: dict[str, str] | None = None
    dataset_slice_start_s: float | None = None
    dataset_slice_stop_s: float | None = None
    atom_ledger: AtomLedger | None = None
    catalog: PrimitiveCatalog | None = None
    hpo_manager: OptunaManager | None = None
    param_trials_per_structure: int = 1
    heuristic_cohort_size: int = 1
    heuristic_cohort_concurrency: int = 1
    expansion_engine: ExpansionEngine | None = None
    admissibility_evaluator: AdmissibilityEvaluator | None = None
