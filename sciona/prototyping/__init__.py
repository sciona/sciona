"""Fast in-process CDG prototyping runtime, credit assignment, 5-rung ladder, and session."""

from __future__ import annotations

from sciona.prototyping.cache import (
    CacheEntry,
    IntermediateCache,
    callable_impl_digest,
    compute_node_cache_key,
    estimate_value_bytes,
    hash_runtime_value,
)
from sciona.prototyping.credit import (
    CounterfactualNodeSensitivity,
    UniversalCreditAssigner,
    perturb_runtime_value,
)
from sciona.prototyping.funnel import (
    EvaluationFunnel,
    FunnelConfig,
    FunnelEvaluationReport,
    FunnelTierResult,
    LivePrototypingSandbox,
    check_static_edge_port_types,
)
from sciona.prototyping.harness import (
    DEFAULT_SYNTHETIC_REGIMES,
    CaseScore,
    EvalCase,
    MetamorphicRelation,
    ProblemHarness,
    ResidentSliceHarness,
    SyntheticSuiteHarness,
    check_output_invariants,
    compute_objective_metric,
    generate_synthetic_value,
    infer_cdg_root_ports,
)
from sciona.prototyping.ladder import (
    LadderStepOutcome,
    MicroAtomSynthesisSpec,
    MicroAtomSynthesizerCallable,
    RevisionLadder,
    SubdecomposeCallable,
    SubgraphSpliceRequest,
    compile_micro_atom_source,
    infer_tunable_specs_for_node,
)
from sciona.prototyping.live_runner import (
    AtomResolutionError,
    LiveCDGRunner,
    LiveExecutionError,
    LiveRunResult,
    resolve_atom_callable,
    summarize_runtime_value,
)
from sciona.prototyping.session import (
    PrototypingSession,
    PrototypingSessionSummary,
    PrototypingStepRecord,
)

__all__ = [
    "AtomResolutionError",
    "CacheEntry",
    "CaseScore",
    "CounterfactualNodeSensitivity",
    "DEFAULT_SYNTHETIC_REGIMES",
    "EvalCase",
    "EvaluationFunnel",
    "FunnelConfig",
    "FunnelEvaluationReport",
    "FunnelTierResult",
    "IntermediateCache",
    "LadderStepOutcome",
    "LiveCDGRunner",
    "LiveExecutionError",
    "LivePrototypingSandbox",
    "LiveRunResult",
    "MetamorphicRelation",
    "MicroAtomSynthesisSpec",
    "MicroAtomSynthesizerCallable",
    "ProblemHarness",
    "PrototypingSession",
    "PrototypingSessionSummary",
    "PrototypingStepRecord",
    "ResidentSliceHarness",
    "RevisionLadder",
    "SubdecomposeCallable",
    "SubgraphSpliceRequest",
    "SyntheticSuiteHarness",
    "UniversalCreditAssigner",
    "callable_impl_digest",
    "check_output_invariants",
    "check_static_edge_port_types",
    "compile_micro_atom_source",
    "compute_node_cache_key",
    "compute_objective_metric",
    "estimate_value_bytes",
    "generate_synthetic_value",
    "hash_runtime_value",
    "infer_cdg_root_ports",
    "infer_tunable_specs_for_node",
    "perturb_runtime_value",
    "resolve_atom_callable",
    "summarize_runtime_value",
]
