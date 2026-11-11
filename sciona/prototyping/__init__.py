"""Fast in-process CDG prototyping runtime, content-addressed cache, harnesses, and evaluation funnel."""

from __future__ import annotations

from sciona.prototyping.cache import (
    CacheEntry,
    IntermediateCache,
    callable_impl_digest,
    compute_node_cache_key,
    estimate_value_bytes,
    hash_runtime_value,
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
from sciona.prototyping.live_runner import (
    AtomResolutionError,
    LiveCDGRunner,
    LiveExecutionError,
    LiveRunResult,
    resolve_atom_callable,
    summarize_runtime_value,
)

__all__ = [
    "AtomResolutionError",
    "CacheEntry",
    "CaseScore",
    "DEFAULT_SYNTHETIC_REGIMES",
    "EvalCase",
    "EvaluationFunnel",
    "FunnelConfig",
    "FunnelEvaluationReport",
    "FunnelTierResult",
    "IntermediateCache",
    "LiveCDGRunner",
    "LiveExecutionError",
    "LivePrototypingSandbox",
    "LiveRunResult",
    "MetamorphicRelation",
    "ProblemHarness",
    "ResidentSliceHarness",
    "SyntheticSuiteHarness",
    "callable_impl_digest",
    "check_output_invariants",
    "check_static_edge_port_types",
    "compute_node_cache_key",
    "compute_objective_metric",
    "estimate_value_bytes",
    "generate_synthetic_value",
    "hash_runtime_value",
    "infer_cdg_root_ports",
    "resolve_atom_callable",
    "summarize_runtime_value",
]
