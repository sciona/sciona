"""Deterministic-first 5-rung iterative CDG revision ladder.

Orders candidate revisions strictly by marginal compute/LLM cost so >80% of
refinement iterations complete in milliseconds with zero LLM calls:

- **Rung 1 — Bounded Node HPO (0 LLM calls)**:
  Tunes the bottleneck node's ``PrimitiveParamSpec``s (or auto-inferred Python
  default parameters) via Optuna while reusing cached upstream intermediates.
- **Rung 2 — Type-Compatible Sibling Swap (0 LLM calls, 0 Hunter re-matches)**:
  Queries ``PrimitiveCatalog`` and Phase B ``ManifestFTSIndex`` by ``PortTypeSpec``
  compatibility, ranks via ``AtomLedger`` UCB1 + ``benchmark_priors`` + Phase 3
  ``uncertainty_priors_for_candidates`` (and ``mine_cdg_uncertainty_gaps`` rescues),
  and binds directly onto the bottleneck slot.
- **Rung 3 — Delta-Planned DPO Structural Expansion (0 LLM calls)**:
  Builds ``ExpansionContext`` from cached live intermediates, runs
  ``plan_expansion_delta`` + ``ExpansionEngine.expand``, and evaluates expanded CDGs.
- **Rung 4a — Localized Subgraph Splice (1 scoped call)**:
  Decomposes ONLY the bottleneck node's ``IOSpec`` boundary contract and splices
  the child CDG via ``compose_cdg(..., operation="replacement")``.
- **Rung 4b — Guided Micro-Atom Synthesis on True Catalog Gaps (1 scoped call)**:
  Synthesizes a self-contained leaf Python function for an unrescued slot/gap,
  verifies it against Tier 1 synthetic/metamorphic checks, registers it in
  ``LiveCDGRunner``, and scores it on Tier 2 validation cases.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import time
import types
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.compiled_catalog import ManifestFTSIndex
from sciona.architect.composition import compose_cdg
from sciona.architect.graph_rewriter import GraphState, Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    EdgeKind,
    IOSpec,
    NodeStatus,
    PortTypeSpec,
    PrimitiveParamSpec,
    format_canonical_type_signature,
)
from sciona.principal.atom_ledger import AtomLedger, compute_slot_signature
from sciona.principal.benchmark_priors import load_benchmark_priors
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
    ExpansionRuleSet,
)
from sciona.principal.expansion_delta_planner import (
    DeltaPlan,
    DeltaPlanningQuery,
    plan_expansion_delta,
)
from sciona.principal.models import BenchmarkResult, NodeGradient, OptimizationMetric
from sciona.principal.proposal_helpers import build_expansion_context
from sciona.principal.graph_types import PrincipalState
from sciona.types import MatchResult
from sciona.uncertainty.gap_spec import (
    UncertaintyGapSpec,
    mine_cdg_uncertainty_gaps,
    uncertainty_priors_for_candidates,
)
from sciona.uncertainty.propagation import CDGUncertaintyReport, propagate_cdg_uncertainty

from sciona.prototyping.funnel import EvaluationFunnel, FunnelEvaluationReport
from sciona.prototyping.live_runner import resolve_atom_callable

logger = logging.getLogger(__name__)

optuna: Any = None
try:
    import optuna as _optuna

    optuna = _optuna
except ImportError:  # pragma: no cover
    optuna = None


@dataclass(frozen=True)
class SubgraphSpliceRequest:
    """Scoped contract passed to Rung 4a localized sub-decomposition."""

    goal: str
    bottleneck_node: AlgorithmicNode
    bottleneck_reason: str
    disfavored_atoms: tuple[str, ...] = ()
    gap_spec: UncertaintyGapSpec | None = None


@dataclass(frozen=True)
class MicroAtomSynthesisSpec:
    """Structured leaf specification passed to Rung 4b micro-atom synthesis."""

    slot_node_id: str
    slot_name: str
    description: str
    inputs: tuple[IOSpec, ...]
    outputs: tuple[IOSpec, ...]
    failure_reason: str = ""
    gap_spec: UncertaintyGapSpec | None = None
    suggested_function_name: str = ""


@dataclass
class LadderStepOutcome:
    """Result of executing a single revision rung (or full ladder climb)."""

    rung: str
    improved: bool
    accepted_cdg: CDGExport
    accepted_node_params: dict[str, dict[str, Any]]
    accepted_loss: float
    accepted_report: FunnelEvaluationReport | None
    baseline_loss: float
    bottleneck_node_id: str
    trials_evaluated: int = 0
    elapsed_ms: float = 0.0
    rationale: str = ""
    synthesized_atom_fqn: str = ""
    synthesized_source_code: str = ""
    delta_plan: DeltaPlan | None = None
    details: dict[str, Any] = field(default_factory=dict)


SubdecomposeCallable = Callable[
    [SubgraphSpliceRequest],
    CDGExport | Awaitable[CDGExport | None] | None,
]
MicroAtomSynthesizerCallable = Callable[
    [MicroAtomSynthesisSpec],
    str | tuple[str, str] | Awaitable[str | tuple[str, str] | None] | None,
]


def infer_tunable_specs_for_node(
    node: AlgorithmicNode,
    *,
    catalog: PrimitiveCatalog | None = None,
    callable_fn: Callable[..., Any] | None = None,
    wired_input_names: set[str] | None = None,
) -> list[PrimitiveParamSpec]:
    """Return tunable ``PrimitiveParamSpec``s from catalog metadata or Python signature defaults."""
    prim_name = str(node.matched_primitive or "").strip()
    if catalog is not None and prim_name:
        prim = catalog.get(prim_name)
        if prim is not None and prim.tunable_params:
            approved = [p for p in prim.tunable_params if p.safe_to_optimize]
            if approved:
                return approved

    wired = wired_input_names or set()
    specs: list[PrimitiveParamSpec] = []
    seen_names: set[str] = set()

    # Inspect resolved Python callable signature defaults when available
    if callable_fn is not None:
        try:
            sig = inspect.signature(callable_fn)
            for pname, param in sig.parameters.items():
                if pname in wired or param.default is inspect.Parameter.empty:
                    continue
                spec = _param_spec_from_default(pname, param.default)
                if spec is not None:
                    seen_names.add(pname)
                    specs.append(spec)
        except (TypeError, ValueError):
            pass

    # Also inspect optional IOSpec ports with literal default_value_repr
    for inp in node.inputs:
        if inp.name in wired or inp.name in seen_names or inp.required:
            continue
        if not inp.default_value_repr:
            continue
        try:
            default_val = ast.literal_eval(inp.default_value_repr)
        except Exception:
            continue
        spec = _param_spec_from_default(inp.name, default_val)
        if spec is not None:
            seen_names.add(inp.name)
            specs.append(spec)

    return specs


def _param_spec_from_default(name: str, default_val: Any) -> PrimitiveParamSpec | None:
    low = name.lower()
    if low in {"seed", "random_state", "n_jobs", "verbose", "axis", "dtype", "copy"}:
        return None
    if isinstance(default_val, bool):
        return PrimitiveParamSpec(
            name=name,
            kind="bool",
            default=bool(default_val),
            choices=[True, False],
        )
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        iv = int(default_val)
        if iv > 0:
            lo = max(1, iv // 3)
            hi = max(iv + 6, iv * 3)
        else:
            lo = iv - 5
            hi = iv + 5
        return PrimitiveParamSpec(
            name=name,
            kind="int",
            default=iv,
            min_value=lo,
            max_value=hi,
        )
    if isinstance(default_val, float) and math.isfinite(default_val):
        fv = float(default_val)
        if fv > 0.0:
            lo_f = max(1e-6, fv * 0.2)
            hi_f = fv * 5.0
            use_log = (hi_f / lo_f) >= 20.0
        elif fv < 0.0:
            lo_f = fv * 3.0
            hi_f = fv * 0.2
            use_log = False
        else:
            lo_f = -1.0
            hi_f = 1.0
            use_log = False
        return PrimitiveParamSpec(
            name=name,
            kind="float",
            default=fv,
            min_value=lo_f,
            max_value=hi_f,
            log_scale=use_log,
        )
    return None


def _ports_compatible(
    required_slot_inputs: Sequence[PortTypeSpec],
    slot_outputs: Sequence[PortTypeSpec],
    cand_inputs: Sequence[IOSpec],
    cand_outputs: Sequence[IOSpec],
) -> bool:
    """Check whether a candidate primitive's ports satisfy a CDG node's wired boundary."""
    req_cand_in = [p for p in cand_inputs if p.required]
    if not (len(req_cand_in) <= len(required_slot_inputs) <= len(cand_inputs)):
        return False
    if slot_outputs and len(cand_outputs) != len(slot_outputs):
        return False

    for slot_in_pt, cand_in in zip(required_slot_inputs, cand_inputs):
        if not slot_in_pt.is_compatible_with(cand_in.resolved_port_type()):
            return False
    for cand_out, slot_out_pt in zip(cand_outputs, slot_outputs):
        if not cand_out.resolved_port_type().is_compatible_with(slot_out_pt):
            return False
    return True


def compile_micro_atom_source(
    source_code: str,
    *,
    preferred_name: str = "",
) -> tuple[str, Callable[..., Any]]:
    """Compile synthesized Python leaf code in an isolated namespace and extract its callable."""
    cleaned = source_code.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    tree = ast.parse(cleaned, mode="exec")
    fn_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("Synthesized micro-atom source contains no top-level 'def' function")

    target_fn_name = (
        preferred_name
        if preferred_name and preferred_name in fn_names
        else fn_names[-1]
    )
    mod_name = f"sciona_synthesized_micro_atom_{uuid.uuid4().hex[:10]}"
    mod = types.ModuleType(mod_name)
    mod.__dict__["__builtins__"] = __builtins__
    exec(compile(tree, filename=f"<{mod_name}>", mode="exec"), mod.__dict__)
    fn = getattr(mod, target_fn_name, None)
    if not callable(fn):
        raise ValueError(f"Compiled micro-atom '{target_fn_name}' is not callable")
    return target_fn_name, fn


class GroupConditionalNormalizationRuleSet:
    """DPO rule set proposing group/unit/subject normalization when group structure exists.

    Addresses cross-unit / cross-subject covariate shift by discovering single-port
    ``(T -> T)`` group-conditional standardization or covariance alignment primitives
    in ``PrimitiveCatalog`` and splicing them onto the corresponding conditioning edge.
    """

    _GROUP_SIGNAL_TOKENS = ("group", "unit", "subject", "cohort", "unseen")
    _NORMALIZER_TOKENS = (
        "group_norm",
        "group_standardiz",
        "unit_norm",
        "unit_standardiz",
        "unit_align",
        "subject_norm",
        "coral",
        "per_unit",
        "per_group",
    )

    def __init__(self, catalog: PrimitiveCatalog | None = None) -> None:
        self._catalog = catalog

    @property
    def name(self) -> str:
        return "group_conditional_normalization"

    @property
    def domain(self) -> str:
        return "cross_group_generalization"

    def _has_group_context(self, cdg: CDGExport, context: ExpansionContext | None = None) -> bool:
        meta_blob = " ".join(
            [
                str((cdg.metadata or {}).get("goal") or ""),
                *(f"{k} {v}" for k, v in (cdg.metadata or {}).items()),
                *(p.name for n in cdg.nodes for p in (*n.inputs, *n.outputs)),
            ]
        ).lower()
        if any(tok in meta_blob for tok in self._GROUP_SIGNAL_TOKENS):
            return True
        if context is not None:
            ctx_keys = " ".join(
                [
                    *((context.runtime_inputs or {}).keys()),
                    *(context.intermediates.keys()),
                ]
            ).lower()
            if any(tok in ctx_keys for tok in self._GROUP_SIGNAL_TOKENS):
                return True
        return False

    def _find_candidate_primitive(self, cdg: CDGExport) -> tuple[AlgorithmicPrimitive, DependencyEdge] | None:
        if self._catalog is None:
            return None
        used_prims = {str(n.matched_primitive or "").strip() for n in cdg.nodes}
        for prim in self._catalog.all_primitives():
            if prim.name in used_prims:
                continue
            blob = f"{prim.name} {prim.description}".lower()
            if not any(tok in blob for tok in self._NORMALIZER_TOKENS):
                continue
            req_ins = [inp for inp in prim.inputs if inp.required]
            if len(req_ins) != 1 or len(prim.outputs) != 1:
                continue
            p_in = req_ins[0].name
            p_out = prim.outputs[0].name
            if p_in != p_out:
                continue
            for edge in cdg.edges:
                if (
                    edge.edge_kind != EdgeKind.CALLABLE_INJECTION
                    and edge.output_name == p_in
                    and edge.input_name == p_out
                ):
                    return prim, edge
        return None

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        if not self._has_group_context(cdg, context):
            return []
        found = self._find_candidate_primitive(cdg)
        if found is None:
            return []
        prim, edge = found
        return [
            ExpansionDiagnostic(
                rule_name="insert_group_conditional_normalizer",
                severity=0.88,
                evidence=(
                    f"Group/unit shift context detected; splicing '{prim.name}' on "
                    f"edge {edge.source_id} -> {edge.target_id}"
                ),
                metric_name="group_shift_exposure",
                metric_value=1.0,
                threshold=0.5,
                source_domain=self.domain,
            )
        ]

    def rules(self) -> list[RewriteRule]:
        sentinel = AlgorithmicNode(
            node_id="semantic_only_group_cond_norm",
            name="semantic_only_group_cond_norm",
            description="semantic_only_group_cond_norm",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            matched_primitive="__semantic_only__.insert_group_conditional_normalizer",
        )
        empty = CDGExport(nodes=[], edges=[])
        graph = CDGExport(nodes=[sentinel], edges=[])

        def _apply(target_cdg: CDGExport) -> GraphState[CDGExport]:
            found = self._find_candidate_primitive(target_cdg)
            if found is None:
                return GraphState.failure("No matching group-conditional normalizer candidate")
            prim, target_edge = found
            mutated = target_cdg.model_copy(deep=True)
            existing_ids = {n.node_id for n in mutated.nodes}
            base_id = "n_group_cond_norm"
            ins_id = base_id
            idx = 2
            while ins_id in existing_ids:
                ins_id = f"{base_id}_{idx}"
                idx += 1

            ins_node = AlgorithmicNode(
                node_id=ins_id,
                name="Group-Conditional Normalization",
                description=prim.description or prim.name,
                concept_type=prim.category,
                status=NodeStatus.ATOMIC,
                matched_primitive=prim.name,
                inputs=[inp.model_copy(deep=True) for inp in prim.inputs],
                outputs=[outp.model_copy(deep=True) for outp in prim.outputs],
                type_signature=format_canonical_type_signature(prim.inputs, prim.outputs),
            )
            new_edges: list[DependencyEdge] = []
            spliced = False
            for e in mutated.edges:
                if (
                    not spliced
                    and e.source_id == target_edge.source_id
                    and e.target_id == target_edge.target_id
                    and e.output_name == target_edge.output_name
                    and e.input_name == target_edge.input_name
                ):
                    spliced = True
                    new_edges.append(
                        DependencyEdge(
                            source_id=e.source_id,
                            target_id=ins_id,
                            output_name=e.output_name,
                            input_name=e.input_name,
                            source_type=e.source_type,
                            target_type=e.target_type,
                        )
                    )
                    new_edges.append(
                        DependencyEdge(
                            source_id=ins_id,
                            target_id=e.target_id,
                            output_name=e.output_name,
                            input_name=e.input_name,
                            source_type=e.source_type,
                            target_type=e.target_type,
                        )
                    )
                else:
                    new_edges.append(e)
            if not spliced:
                return GraphState.failure("Target edge for group normalizer not found")
            mutated.nodes.append(ins_node)
            mutated.edges = new_edges
            return GraphState.success(mutated)

        return [
            RewriteRule(
                name="insert_group_conditional_normalizer",
                lhs=graph,
                rhs=graph,
                interface=empty,
                l_morphism=Morphism(node_map={}, edge_map={}),
                r_morphism=Morphism(node_map={}, edge_map={}),
                priority=4,
                semantic_apply=_apply,
            )
        ]


class RevisionLadder:
    """Executes Rungs 1 -> 4b in strict cost order against an ``EvaluationFunnel``."""

    def __init__(
        self,
        funnel: EvaluationFunnel,
        *,
        catalog: PrimitiveCatalog | None = None,
        manifest_index: ManifestFTSIndex | None = None,
        manifest_sqlite_path: Path | None = None,
        atom_ledger: AtomLedger | None = None,
        expansion_engine: ExpansionEngine | None = None,
        subdecompose_fn: SubdecomposeCallable | None = None,
        micro_atom_synthesizer: MicroAtomSynthesizerCallable | None = None,
        hpo_trials_per_rung: int = 8,
        max_sibling_evals: int = 5,
        improvement_tol: float = 1e-7,
        min_rung1_rel_gain: float = 0.0025,
    ) -> None:
        self.funnel = funnel
        self.catalog = catalog
        self.manifest_index = manifest_index
        self.manifest_sqlite_path = manifest_sqlite_path
        if self.manifest_index is None and manifest_sqlite_path is not None and manifest_sqlite_path.is_file():
            self.manifest_index = ManifestFTSIndex(manifest_sqlite_path)
        self.atom_ledger = atom_ledger if atom_ledger is not None else AtomLedger()
        if catalog is not None:
            group_rs: ExpansionRuleSet = GroupConditionalNormalizationRuleSet(catalog=catalog)
            if expansion_engine is not None:
                existing_rs = list(getattr(expansion_engine, "_rule_sets", []))
                if not any(getattr(r, "name", "") == group_rs.name for r in existing_rs):
                    self.expansion_engine: ExpansionEngine | None = ExpansionEngine(
                        [*existing_rs, group_rs]
                    )
                else:
                    self.expansion_engine = expansion_engine
            else:
                self.expansion_engine = ExpansionEngine([group_rs])
        else:
            self.expansion_engine = expansion_engine
        self.subdecompose_fn = subdecompose_fn
        self.micro_atom_synthesizer = micro_atom_synthesizer
        self.hpo_trials_per_rung = max(1, int(hpo_trials_per_rung))
        self.max_sibling_evals = max(1, int(max_sibling_evals))
        self.improvement_tol = float(improvement_tol)
        self.min_rung1_rel_gain = max(0.0, float(min_rung1_rel_gain))
        self.llm_preferred_siblings: dict[str, list[str]] = {}
        self.llm_param_priors: dict[str, dict[str, Any]] = {}
        self.llm_synthesis_nodes: list[str] = []

        self._benchmark_priors: dict[str, float] = {}
        if manifest_sqlite_path is not None and manifest_sqlite_path.is_file():
            try:
                self._benchmark_priors = load_benchmark_priors(manifest_sqlite_path)
            except Exception:
                self._benchmark_priors = {}

    def _is_better(self, new_loss: float, baseline_loss: float) -> bool:
        if not math.isfinite(new_loss):
            return False
        if not math.isfinite(baseline_loss):
            return True
        return new_loss < (baseline_loss - self.improvement_tol)

    def _is_substantial_gain(self, new_loss: float, baseline_loss: float) -> bool:
        """Return True when ``new_loss`` improves over ``baseline_loss`` by at least ``min_rung1_rel_gain``."""
        if not self._is_better(new_loss, baseline_loss):
            return False
        if not math.isfinite(baseline_loss):
            return True
        if self.min_rung1_rel_gain <= 0.0:
            return True
        rel_gain = (baseline_loss - new_loss) / max(abs(baseline_loss), 1e-9)
        return rel_gain >= self.min_rung1_rel_gain

    def has_any_tunable_params(self, cdg: CDGExport) -> bool:
        """Return True if at least one atomic leaf exposes bounded tunable parameters."""
        for node in cdg.leaf_nodes():
            wired_in = {e.input_name for e in cdg.edges if e.target_id == node.node_id}
            fn: Callable[..., Any] | None = None
            try:
                fn, _ = resolve_atom_callable(
                    node.matched_primitive or "",
                    node=node,
                    extra_callables=self.funnel.runner.extra_callables,
                )
            except Exception:
                fn = None
            specs = infer_tunable_specs_for_node(
                node,
                catalog=self.catalog,
                callable_fn=fn,
                wired_input_names=wired_in,
            )
            if specs:
                return True
        return False

    # ------------------------------------------------------------------
    # Rung 1: Bounded Node-Scoped HPO (0 LLM calls)
    # ------------------------------------------------------------------

    def run_rung1_node_hpo(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
        *,
        baseline_loss: float,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        trial_index: int = 1,
    ) -> LadderStepOutcome:
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        target_node = next((n for n in cdg.nodes if n.node_id == bottleneck_node_id), None)
        if target_node is None:
            return LadderStepOutcome(
                rung="rung1_node_hpo",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Bottleneck node '{bottleneck_node_id}' not found",
            )

        wired_inputs = {
            e.input_name
            for e in cdg.edges
            if e.target_id == bottleneck_node_id
        }
        fn: Callable[..., Any] | None = None
        try:
            fn, _ = resolve_atom_callable(
                target_node.matched_primitive or "",
                node=target_node,
                extra_callables=self.funnel.runner.extra_callables,
            )
        except Exception:
            fn = None

        specs = infer_tunable_specs_for_node(
            target_node,
            catalog=self.catalog,
            callable_fn=fn,
            wired_input_names=wired_inputs,
        )
        if not specs:
            return LadderStepOutcome(
                rung="rung1_node_hpo",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Node '{bottleneck_node_id}' exposes no tunable parameters",
            )

        best_loss = baseline_loss
        best_params = {k: dict(v) for k, v in base_params.items()}
        best_report: FunnelEvaluationReport | None = None
        trials_run = 0

        if optuna is not None:
            hpo_seed = 42 + max(0, int(trial_index) - 1) * 101
            sampler = optuna.samplers.TPESampler(seed=hpo_seed)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            existing_node_p = base_params.get(bottleneck_node_id, {})
            enq: dict[str, Any] = {}
            for sp in specs:
                cur_v = existing_node_p.get(sp.name, sp.default)
                if sp.kind in {"int", "float"} and sp.min_value is not None and sp.max_value is not None:
                    if float(sp.min_value) <= float(cur_v) <= float(sp.max_value):
                        enq[sp.name] = cur_v
                else:
                    enq[sp.name] = cur_v
            if enq:
                study.enqueue_trial(enq)

            prior_raw = self.llm_param_priors.get(bottleneck_node_id, {})
            if prior_raw:
                prior_enq: dict[str, Any] = {}
                for sp in specs:
                    pv = prior_raw.get(sp.name, enq.get(sp.name, sp.default))
                    try:
                        if sp.kind == "bool":
                            prior_enq[sp.name] = bool(pv)
                        elif sp.kind == "categorical":
                            choices = list(sp.choices or [sp.default])
                            prior_enq[sp.name] = pv if pv in choices else choices[0]
                        elif sp.kind == "int" and sp.min_value is not None and sp.max_value is not None:
                            clamped_i = max(
                                int(sp.min_value),
                                min(int(sp.max_value), int(round(float(pv)))),
                            )
                            prior_enq[sp.name] = clamped_i
                        elif sp.kind == "float" and sp.min_value is not None and sp.max_value is not None:
                            clamped_f = max(
                                float(sp.min_value),
                                min(float(sp.max_value), float(pv)),
                            )
                            prior_enq[sp.name] = clamped_f
                    except (TypeError, ValueError):
                        if sp.name in enq:
                            prior_enq[sp.name] = enq[sp.name]
                if prior_enq and prior_enq != enq:
                    study.enqueue_trial(prior_enq)

            for _ in range(self.hpo_trials_per_rung):
                trial = study.ask()
                sampled: dict[str, Any] = {}
                for sp in specs:
                    if sp.kind == "bool":
                        sampled[sp.name] = trial.suggest_categorical(sp.name, list(sp.choices or [True, False]))
                    elif sp.kind == "categorical":
                        sampled[sp.name] = trial.suggest_categorical(sp.name, list(sp.choices or [sp.default]))
                    elif sp.kind == "int" and sp.min_value is not None and sp.max_value is not None:
                        sampled[sp.name] = trial.suggest_int(sp.name, int(sp.min_value), int(sp.max_value))
                    elif sp.kind == "float" and sp.min_value is not None and sp.max_value is not None:
                        sampled[sp.name] = trial.suggest_float(
                            sp.name,
                            float(sp.min_value),
                            float(sp.max_value),
                            log=bool(sp.log_scale) and float(sp.min_value) > 0,
                        )
                    else:
                        sampled[sp.name] = sp.default

                cand_params = {k: dict(v) for k, v in base_params.items()}
                cand_params[bottleneck_node_id] = {
                    **cand_params.get(bottleneck_node_id, {}),
                    **sampled,
                }
                rep = self.funnel.evaluate_candidate(
                    cdg,
                    node_params=cand_params,
                    match_results=match_results,
                )
                trials_run += 1
                if rep.passed and math.isfinite(rep.global_loss):
                    study.tell(trial, rep.global_loss)
                    if self._is_better(rep.global_loss, best_loss):
                        best_loss = rep.global_loss
                        best_params = cand_params
                        best_report = rep
                else:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        improved = best_report is not None and self._is_better(best_loss, baseline_loss)
        return LadderStepOutcome(
            rung="rung1_node_hpo",
            improved=improved,
            accepted_cdg=cdg,
            accepted_node_params=best_params if improved else base_params,
            accepted_loss=best_loss if improved else baseline_loss,
            accepted_report=best_report if improved else None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=bottleneck_node_id,
            trials_evaluated=trials_run,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(
                f"Tuned '{bottleneck_node_id}' params {best_params.get(bottleneck_node_id)} "
                f"({baseline_loss:.5f} -> {best_loss:.5f})"
                if improved
                else f"No HPO gain across {trials_run} trials on '{bottleneck_node_id}'"
            ),
            details={"tunable_names": [s.name for s in specs]},
        )

    # ------------------------------------------------------------------
    # Rung 2: Catalog / Manifest Type-Compatible Sibling Swap (0 LLM, 0 Hunter)
    # ------------------------------------------------------------------

    def discover_compatible_siblings(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
    ) -> list[str]:
        """Find type-compatible replacement primitives from ``PrimitiveCatalog`` & ``ManifestFTSIndex``."""
        target = next((n for n in cdg.nodes if n.node_id == bottleneck_node_id), None)
        if target is None:
            return []

        current_prim = str(target.matched_primitive or "").strip()
        wired_in_names = {
            e.input_name
            for e in cdg.edges
            if e.target_id == bottleneck_node_id and e.edge_kind != EdgeKind.CALLABLE_INJECTION
        }
        required_in_specs = [
            inp.resolved_port_type()
            for inp in target.inputs
            if inp.required or inp.name in wired_in_names
        ]
        out_specs = [outp.resolved_port_type() for outp in target.outputs]

        candidates: list[str] = []
        seen: set[str] = {current_prim} if current_prim else set()

        def _add_cand(name: str) -> None:
            clean = name.strip()
            if clean and clean not in seen:
                seen.add(clean)
                candidates.append(clean)

        # 1. PrimitiveCatalog candidates (same ConceptType first, then general compatible)
        if self.catalog is not None:
            cat_same = self.catalog.search_by_category(target.concept_type)
            for prim in cat_same:
                if _ports_compatible(required_in_specs, out_specs, prim.inputs, prim.outputs):
                    _add_cand(prim.name)
            for prim in self.catalog.all_primitives():
                if _ports_compatible(required_in_specs, out_specs, prim.inputs, prim.outputs):
                    _add_cand(prim.name)

        # 2. Phase B Compiled SQLite Manifest FTS5 + PortTypeSpec index (<1 ms)
        if self.manifest_index is not None:
            query_txt = f"{target.name} {target.description}"
            req_ios = [
                inp for inp in target.inputs if inp.required or inp.name in wired_in_names
            ]
            for decl, _score in self.manifest_index.search_compatible_ports(
                req_ios,
                target.outputs,
                query_text=query_txt,
                k=self.max_sibling_evals * 3,
            ):
                _add_cand(decl.name)

        for pref_name in reversed(self.llm_preferred_siblings.get(bottleneck_node_id, [])):
            if pref_name in candidates:
                candidates.remove(pref_name)
                candidates.insert(0, pref_name)

        return candidates

    def run_rung2_sibling_swap(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
        *,
        baseline_loss: float,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        uncertainty_report: CDGUncertaintyReport | None = None,
        trial_index: int = 1,
    ) -> LadderStepOutcome:
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        target = next((n for n in cdg.nodes if n.node_id == bottleneck_node_id), None)
        if target is None:
            return LadderStepOutcome(
                rung="rung2_sibling_swap",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Node '{bottleneck_node_id}' not found",
            )

        raw_candidates = self.discover_compatible_siblings(cdg, bottleneck_node_id)
        if not raw_candidates:
            return LadderStepOutcome(
                rung="rung2_sibling_swap",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"No type-compatible catalog/manifest siblings found for '{bottleneck_node_id}'",
            )

        # Combine benchmark priors + Phase 3 directional uncertainty priors inside AtomLedger UCB1
        node_map = {n.node_id: n for n in cdg.nodes}
        parent_node = node_map.get(target.parent_id) if target.parent_id else None
        slot = compute_slot_signature(target, parent_node)

        u_rep = uncertainty_report
        if u_rep is None and self.funnel.harness.risk_budget is not None:
            try:
                u_rep = propagate_cdg_uncertainty(
                    cdg,
                    match_results=match_results,
                    profiles=self.funnel.runner.uncertainty_profiles,
                    risk_budget=self.funnel.harness.risk_budget,
                )
            except Exception:
                u_rep = None

        neval = u_rep.nodes.get(bottleneck_node_id) if u_rep is not None else None
        unc_priors = uncertainty_priors_for_candidates(
            raw_candidates,
            node_eval=neval,
            profiles=self.funnel.runner.uncertainty_profiles,
        )
        merged_priors: dict[str, float] = {}
        for cname in raw_candidates:
            bp = self._benchmark_priors.get(cname)
            up = unc_priors.get(cname)
            if bp is not None and up is not None:
                merged_priors[cname] = 0.5 * bp + 0.5 * up
            elif bp is not None:
                merged_priors[cname] = bp
            elif up is not None:
                merged_priors[cname] = up

        ranked = [
            cname
            for cname, _score in self.atom_ledger.rank_candidates(
                slot,
                raw_candidates,
                benchmark_priors=merged_priors,
            )
        ]

        # Promote any Phase 3 uncertainty-rescue candidate to index 0
        if u_rep is not None:
            gap_rep = mine_cdg_uncertainty_gaps(
                cdg,
                u_rep,
                candidates_by_node={bottleneck_node_id: ranked},
                profiles=self.funnel.runner.uncertainty_profiles,
            )
            rescued_atom = gap_rep.rescued_by_existing_candidates.get(bottleneck_node_id)
            if rescued_atom and rescued_atom in ranked:
                ranked.remove(rescued_atom)
                ranked.insert(0, rescued_atom)

        for pref_name in reversed(self.llm_preferred_siblings.get(bottleneck_node_id, [])):
            if pref_name in ranked:
                ranked.remove(pref_name)
                ranked.insert(0, pref_name)

        best_cdg = cdg
        best_loss = baseline_loss
        best_params = base_params
        best_report: FunnelEvaluationReport | None = None
        best_sibling = ""
        trials_run = 0

        for cand_name in ranked[: self.max_sibling_evals]:
            mutated_cdg = _bind_sibling_onto_node(
                cdg,
                bottleneck_node_id,
                cand_name,
                catalog=self.catalog,
            )
            cand_params = {k: dict(v) for k, v in base_params.items() if k != bottleneck_node_id}
            rep = self.funnel.evaluate_candidate(
                mutated_cdg,
                node_params=cand_params,
                match_results=match_results,
            )
            trials_run += 1
            obs_grad = 0.0 if (rep.passed and self._is_better(rep.global_loss, baseline_loss)) else 85.0
            self.atom_ledger.record(slot, cand_name, obs_grad, trial_index)

            if rep.passed and self._is_better(rep.global_loss, best_loss):
                best_cdg = mutated_cdg
                best_loss = rep.global_loss
                best_params = cand_params
                best_report = rep
                best_sibling = cand_name

        improved = best_report is not None and self._is_better(best_loss, baseline_loss)
        return LadderStepOutcome(
            rung="rung2_sibling_swap",
            improved=improved,
            accepted_cdg=best_cdg if improved else cdg,
            accepted_node_params=best_params if improved else base_params,
            accepted_loss=best_loss if improved else baseline_loss,
            accepted_report=best_report if improved else None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=bottleneck_node_id,
            trials_evaluated=trials_run,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(
                f"Swapped '{bottleneck_node_id}' -> '{best_sibling}' without Hunter re-match "
                f"({baseline_loss:.5f} -> {best_loss:.5f})"
                if improved
                else f"Evaluated {trials_run} compatible sibling(s) for '{bottleneck_node_id}' without improvement"
            ),
            details={"evaluated_siblings": ranked[: self.max_sibling_evals], "selected_sibling": best_sibling},
        )

    # ------------------------------------------------------------------
    # Rung 3: Delta-Planned DPO Structural Expansion (0 LLM calls)
    # ------------------------------------------------------------------

    def run_rung3_dpo_expansion(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
        *,
        baseline_loss: float,
        baseline_benchmark: BenchmarkResult | None,
        bottleneck_reason: str = "",
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
    ) -> LadderStepOutcome:
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        if self.expansion_engine is None:
            return LadderStepOutcome(
                rung="rung3_dpo_expansion",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="No ExpansionEngine configured",
            )

        state_stub = PrincipalState(
            cdg=cdg,
            benchmark=baseline_benchmark,
            planning_artifact=cdg.planning_artifact,
        )
        exp_ctx: ExpansionContext = build_expansion_context(state_stub)

        exp_res = self.expansion_engine.expand(cdg, exp_ctx)
        if not exp_res.expanded:
            return LadderStepOutcome(
                rung="rung3_dpo_expansion",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="No DPO expansion rules triggered on current intermediates",
                delta_plan=None,
            )

        # Consult delta planner for structured coverage provenance when expansion triggers
        delta_plan: DeltaPlan | None = None
        try:
            leaves = cdg.leaf_nodes()
            d_query = DeltaPlanningQuery(
                missing_techniques=(bottleneck_reason,) if bottleneck_reason else (),
                stage_names=tuple(n.name for n in leaves),
                input_names=tuple(p.name for n in leaves for p in n.inputs),
                output_names=tuple(p.name for n in leaves for p in n.outputs),
                runtime_keys=tuple((exp_ctx.runtime_inputs or {}).keys()),
                intermediate_keys=tuple(exp_ctx.intermediates.keys()),
                base_coverage=0.5,
            )
            delta_plan = plan_expansion_delta(d_query, cdg=cdg)
        except Exception:
            delta_plan = None

        rep = self.funnel.evaluate_candidate(
            exp_res.cdg,
            node_params=base_params,
            match_results=match_results,
        )
        improved = rep.passed and self._is_better(rep.global_loss, baseline_loss)
        return LadderStepOutcome(
            rung="rung3_dpo_expansion",
            improved=improved,
            accepted_cdg=exp_res.cdg if improved else cdg,
            accepted_node_params=base_params,
            accepted_loss=rep.global_loss if improved else baseline_loss,
            accepted_report=rep if improved else None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=bottleneck_node_id,
            trials_evaluated=1,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(
                f"Applied DPO rule(s) {list(exp_res.applied_rules)} "
                f"({baseline_loss:.5f} -> {rep.global_loss:.5f})"
                if improved
                else f"Expanded CDG via {list(exp_res.applied_rules)} did not improve loss (loss={rep.global_loss})"
            ),
            delta_plan=delta_plan,
            details={
                "applied_rules": list(exp_res.applied_rules),
                "expanded_cdg": exp_res.cdg,
            },
        )

    def run_coupled_dpo_and_sibling_swap(
        self,
        expanded_cdg: CDGExport,
        applied_rules: Sequence[str],
        candidate_node_ids: Sequence[str],
        *,
        baseline_loss: float,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        uncertainty_report: CDGUncertaintyReport | None = None,
        trial_index: int = 1,
    ) -> LadderStepOutcome:
        """Evaluate coupled 2-node ``(Rung 3 DPO + Rung 2 Sibling Swap)`` transitions.

        Resolves multi-node Pareto traps under hard constraints or domain shift where
        upstream conditioning and downstream representation/calibration must change together.
        """
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        existing_ids = {n.node_id for n in expanded_cdg.nodes}
        best_outcome: LadderStepOutcome | None = None
        best_loss = baseline_loss
        trials_run = 0

        for nid in candidate_node_ids:
            if nid not in existing_ids:
                continue
            r2 = self.run_rung2_sibling_swap(
                expanded_cdg,
                nid,
                baseline_loss=best_loss,
                current_node_params=base_params,
                match_results=match_results,
                uncertainty_report=uncertainty_report,
                trial_index=trial_index,
            )
            trials_run += r2.trials_evaluated
            if r2.improved and self._is_better(r2.accepted_loss, best_loss):
                best_loss = r2.accepted_loss
                best_outcome = r2

        # If 1-node swap on expanded_cdg did not break a multi-stage constraint trap,
        # evaluate coordinated 2-stage compatible sibling pairs (top-1 sibling per node pair first,
        # excluding primitives already present in expanded_cdg to avoid redundant double stages).
        if best_outcome is None:
            already_bound = {
                str(n.matched_primitive or "").strip()
                for n in expanded_cdg.nodes
                if n.matched_primitive
            }
            valid_nodes = [nid for nid in dict.fromkeys(candidate_node_ids) if nid in existing_ids]
            sib_map = {
                nid: [
                    s
                    for s in self.discover_compatible_siblings(expanded_cdg, nid)
                    if s not in already_bound
                ][:2]
                for nid in valid_nodes
            }
            sib_nodes = [nid for nid in valid_nodes if sib_map.get(nid)]
            ordered_pair_specs: list[tuple[str, str, str, str]] = []
            # Pass 1: top-1 preferred sibling across every node pair
            for idx_a, nid_a in enumerate(sib_nodes):
                for nid_b in sib_nodes[idx_a + 1 :]:
                    ordered_pair_specs.append(
                        (nid_a, nid_b, sib_map[nid_a][0], sib_map[nid_b][0])
                    )
            # Pass 2: secondary sibling combinations if budget remains
            for idx_a, nid_a in enumerate(sib_nodes):
                for nid_b in sib_nodes[idx_a + 1 :]:
                    for i_a, sib_a in enumerate(sib_map[nid_a]):
                        for i_b, sib_b in enumerate(sib_map[nid_b]):
                            if i_a == 0 and i_b == 0:
                                continue
                            ordered_pair_specs.append((nid_a, nid_b, sib_a, sib_b))

            for nid_a, nid_b, sib_a, sib_b in ordered_pair_specs[:6]:
                pair_cdg = _bind_sibling_onto_node(
                    expanded_cdg, nid_a, sib_a, catalog=self.catalog
                )
                pair_cdg = _bind_sibling_onto_node(
                    pair_cdg, nid_b, sib_b, catalog=self.catalog
                )
                clean_params = {
                    k: dict(v)
                    for k, v in base_params.items()
                    if k not in {nid_a, nid_b}
                }
                rep = self.funnel.evaluate_candidate(
                    pair_cdg,
                    node_params=clean_params or None,
                    match_results=match_results,
                )
                trials_run += 1
                if rep.passed and self._is_better(rep.global_loss, best_loss):
                    best_loss = rep.global_loss
                    best_outcome = LadderStepOutcome(
                        rung="rung3_dpo_expansion",
                        improved=True,
                        accepted_cdg=pair_cdg,
                        accepted_node_params=clean_params,
                        accepted_loss=rep.global_loss,
                        accepted_report=rep,
                        baseline_loss=baseline_loss,
                        bottleneck_node_id=f"{nid_a}+{nid_b}",
                        trials_evaluated=1,
                        details={"selected_sibling": f"{sib_a}, {sib_b}"},
                    )

        if best_outcome is not None and best_outcome.accepted_report is not None:
            sel_sib = best_outcome.details.get("selected_sibling", "")
            return LadderStepOutcome(
                rung="rung3_dpo_expansion",
                improved=True,
                accepted_cdg=best_outcome.accepted_cdg,
                accepted_node_params=best_outcome.accepted_node_params,
                accepted_loss=best_outcome.accepted_loss,
                accepted_report=best_outcome.accepted_report,
                baseline_loss=baseline_loss,
                bottleneck_node_id=best_outcome.bottleneck_node_id,
                trials_evaluated=trials_run,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=(
                    f"Applied coupled DPO {list(applied_rules)} + sibling swap "
                    f"'{best_outcome.bottleneck_node_id}' -> '{sel_sib}' "
                    f"({baseline_loss:.5f} -> {best_outcome.accepted_loss:.5f})"
                ),
                details={
                    "applied_rules": list(applied_rules),
                    "coupled_sibling_node": best_outcome.bottleneck_node_id,
                    "selected_sibling": sel_sib,
                },
            )

        return LadderStepOutcome(
            rung="rung3_dpo_expansion",
            improved=False,
            accepted_cdg=expanded_cdg,
            accepted_node_params=base_params,
            accepted_loss=baseline_loss,
            accepted_report=None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=candidate_node_ids[0] if candidate_node_ids else "",
            trials_evaluated=trials_run,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale="Coupled DPO + sibling swap did not improve loss",
        )

    # ------------------------------------------------------------------
    # Rung 4a: Localized Subgraph Splice via compose_cdg (1 scoped call)
    # ------------------------------------------------------------------

    async def run_rung4a_localized_splice(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
        *,
        goal: str,
        bottleneck_reason: str,
        baseline_loss: float,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        gap_spec: UncertaintyGapSpec | None = None,
    ) -> LadderStepOutcome:
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        if self.subdecompose_fn is None:
            return LadderStepOutcome(
                rung="rung4a_localized_splice",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="No subdecompose_fn configured for Rung 4a",
            )

        target = next((n for n in cdg.nodes if n.node_id == bottleneck_node_id), None)
        if target is None:
            return LadderStepOutcome(
                rung="rung4a_localized_splice",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Node '{bottleneck_node_id}' not found",
            )

        disfavored = (target.matched_primitive,) if target.matched_primitive else ()
        req = SubgraphSpliceRequest(
            goal=goal,
            bottleneck_node=target,
            bottleneck_reason=bottleneck_reason,
            disfavored_atoms=disfavored,
            gap_spec=gap_spec,
        )
        maybe_child = self.subdecompose_fn(req)
        child_cdg = await maybe_child if inspect.isawaitable(maybe_child) else maybe_child
        if child_cdg is None:
            return LadderStepOutcome(
                rung="rung4a_localized_splice",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="subdecompose_fn returned None",
            )

        try:
            spliced_cdg = compose_cdg(
                cdg,
                child_cdg,
                operation="replacement",
                target_node_id=bottleneck_node_id,
                workspace_id="proto",
                version_id=uuid.uuid4().hex[:6],
            )
        except Exception as exc:
            return LadderStepOutcome(
                rung="rung4a_localized_splice",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"compose_cdg splice failed: {exc}",
            )

        spliced_params = {k: dict(v) for k, v in base_params.items() if k != bottleneck_node_id}
        rep = self.funnel.evaluate_candidate(
            spliced_cdg,
            node_params=spliced_params,
            match_results=match_results,
        )
        improved = rep.passed and self._is_better(rep.global_loss, baseline_loss)
        return LadderStepOutcome(
            rung="rung4a_localized_splice",
            improved=improved,
            accepted_cdg=spliced_cdg if improved else cdg,
            accepted_node_params=spliced_params if improved else base_params,
            accepted_loss=rep.global_loss if improved else baseline_loss,
            accepted_report=rep if improved else None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=bottleneck_node_id,
            trials_evaluated=1,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(
                f"Locally spliced '{bottleneck_node_id}' with {len(child_cdg.nodes)}-node subgraph "
                f"({baseline_loss:.5f} -> {rep.global_loss:.5f})"
                if improved
                else f"Localized splice on '{bottleneck_node_id}' did not improve loss (loss={rep.global_loss})"
            ),
        )

    # ------------------------------------------------------------------
    # Rung 4b: Guided Micro-Atom Synthesis on True Catalog Gaps
    # ------------------------------------------------------------------

    async def run_rung4b_micro_atom_synthesis(
        self,
        cdg: CDGExport,
        bottleneck_node_id: str,
        *,
        failure_reason: str,
        baseline_loss: float,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        gap_spec: UncertaintyGapSpec | None = None,
    ) -> LadderStepOutcome:
        t0 = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        if self.micro_atom_synthesizer is None:
            return LadderStepOutcome(
                rung="rung4b_micro_atom_synthesis",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="No micro_atom_synthesizer configured for Rung 4b",
            )

        target = next((n for n in cdg.nodes if n.node_id == bottleneck_node_id), None)
        if target is None:
            return LadderStepOutcome(
                rung="rung4b_micro_atom_synthesis",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Node '{bottleneck_node_id}' not found",
            )

        suggested_fn = f"micro_{target.node_id}".replace("-", "_").replace(".", "_")
        spec = MicroAtomSynthesisSpec(
            slot_node_id=target.node_id,
            slot_name=target.name,
            description=target.description,
            inputs=tuple(target.inputs),
            outputs=tuple(target.outputs),
            failure_reason=failure_reason,
            gap_spec=gap_spec,
            suggested_function_name=suggested_fn,
        )

        raw_synth = self.micro_atom_synthesizer(spec)
        synth_payload = await raw_synth if inspect.isawaitable(raw_synth) else raw_synth
        if not synth_payload:
            return LadderStepOutcome(
                rung="rung4b_micro_atom_synthesis",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale="micro_atom_synthesizer returned empty output",
            )

        if isinstance(synth_payload, tuple):
            pref_name, src_code = synth_payload
        else:
            pref_name, src_code = suggested_fn, str(synth_payload)

        try:
            fn_name, fn_callable = compile_micro_atom_source(src_code, preferred_name=pref_name)
        except Exception as exc:
            return LadderStepOutcome(
                rung="rung4b_micro_atom_synthesis",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id=bottleneck_node_id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                rationale=f"Synthesized micro-atom failed compilation: {exc}",
            )

        fqn = f"sciona.synthesized.{fn_name}_{uuid.uuid4().hex[:6]}"
        self.funnel.runner.extra_callables[fqn] = fn_callable
        mutated_cdg = _bind_sibling_onto_node(cdg, bottleneck_node_id, fqn, catalog=None)
        wired_in = {
            e.input_name
            for e in mutated_cdg.edges
            if e.target_id == bottleneck_node_id and e.edge_kind != EdgeKind.CALLABLE_INJECTION
        }
        for idx, n_mut in enumerate(mutated_cdg.nodes):
            if n_mut.node_id == bottleneck_node_id:
                mutated_cdg.nodes[idx] = sync_node_optional_ports_to_callable(
                    n_mut, fn_callable, wired_in
                )
                break
        cand_params = {k: dict(v) for k, v in base_params.items() if k != bottleneck_node_id}

        rep = self.funnel.evaluate_candidate(
            mutated_cdg,
            node_params=cand_params,
            match_results=match_results,
        )
        improved = rep.passed and self._is_better(rep.global_loss, baseline_loss)
        if not improved:
            # Remove rejected micro-atom from live callables so session stays clean
            self.funnel.runner.extra_callables.pop(fqn, None)

        return LadderStepOutcome(
            rung="rung4b_micro_atom_synthesis",
            improved=improved,
            accepted_cdg=mutated_cdg if improved else cdg,
            accepted_node_params=cand_params if improved else base_params,
            accepted_loss=rep.global_loss if improved else baseline_loss,
            accepted_report=rep if improved else None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=bottleneck_node_id,
            trials_evaluated=1,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rationale=(
                f"Synthesized & Tier-1/2 verified micro-atom '{fqn}' for '{bottleneck_node_id}' "
                f"({baseline_loss:.5f} -> {rep.global_loss:.5f})"
                if improved
                else f"Synthesized micro-atom '{fqn}' rejected by funnel ({rep.prune_reason or rep.global_loss})"
            ),
            synthesized_atom_fqn=fqn if improved else "",
            synthesized_source_code=src_code if improved else "",
        )

    # ------------------------------------------------------------------
    # Full Cost-Ordered Climb Across Bottleneck Candidates
    # ------------------------------------------------------------------

    async def climb(
        self,
        cdg: CDGExport,
        gradients: Sequence[NodeGradient],
        *,
        goal: str = "",
        baseline_loss: float,
        baseline_benchmark: BenchmarkResult | None = None,
        current_node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        uncertainty_report: CDGUncertaintyReport | None = None,
        max_rung: int = 4,
        trial_index: int = 1,
    ) -> LadderStepOutcome:
        """Climb Rung 1 -> Rung 2 -> Rung 3 -> Rung 4a -> Rung 4b, accepting first strict gain."""
        t_start = time.perf_counter()
        base_params = {k: dict(v) for k, v in (current_node_params or {}).items()}
        leaves = cdg.leaf_nodes()
        ordered_bottlenecks = [g for g in gradients]
        if not ordered_bottlenecks and leaves:
            ordered_bottlenecks = [
                NodeGradient(
                    node_id=leaves[-1].node_id,
                    gradient_score=100.0,
                    metric_type=OptimizationMetric.PRECISION,
                    bottleneck_reason="fallback leaf target",
                )
            ]
        if not ordered_bottlenecks:
            return LadderStepOutcome(
                rung="none",
                improved=False,
                accepted_cdg=cdg,
                accepted_node_params=base_params,
                accepted_loss=baseline_loss,
                accepted_report=None,
                baseline_loss=baseline_loss,
                bottleneck_node_id="",
                rationale="CDG has no atomic leaves to refine",
            )

        primary = ordered_bottlenecks[0]
        total_trials = 0
        attempted_rungs: list[str] = []
        deferred_micro_r1: LadderStepOutcome | None = None
        structural_target = baseline_loss

        # Rung 1: Bounded Node HPO (require >= min_rung1_rel_gain to preempt Rung 2/3;
        # smaller micro-tweaks are held as a zero-LLM fallback if Rung 2/3 do not beat them)
        if max_rung >= 1:
            for grad in ordered_bottlenecks[:2]:
                r1 = self.run_rung1_node_hpo(
                    cdg,
                    grad.node_id,
                    baseline_loss=structural_target,
                    current_node_params=base_params,
                    match_results=match_results,
                    trial_index=trial_index,
                )
                total_trials += r1.trials_evaluated
                attempted_rungs.append(r1.rung)
                if r1.improved:
                    if self._is_substantial_gain(r1.accepted_loss, baseline_loss):
                        r1.baseline_loss = baseline_loss
                        r1.trials_evaluated = total_trials
                        r1.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        r1.details["attempted_rungs"] = attempted_rungs
                        return r1
                    deferred_micro_r1 = r1
                    structural_target = r1.accepted_loss

        # Rung 2: Catalog / Manifest Type-Compatible Sibling Swap (0 LLM calls, 0 Hunter calls)
        if max_rung >= 2:
            for grad in ordered_bottlenecks[:2]:
                r2 = self.run_rung2_sibling_swap(
                    cdg,
                    grad.node_id,
                    baseline_loss=structural_target,
                    current_node_params=base_params,
                    match_results=match_results,
                    uncertainty_report=uncertainty_report,
                    trial_index=trial_index,
                )
                total_trials += r2.trials_evaluated
                attempted_rungs.append(r2.rung)
                if r2.improved:
                    r2.baseline_loss = baseline_loss
                    r2.trials_evaluated = total_trials
                    r2.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    r2.details["attempted_rungs"] = attempted_rungs
                    return r2

        # Rung 3: Delta-Planned DPO Structural Expansion (+ Coupled 2-Node DPO + Sibling Swap)
        expanded_candidate_cdg: CDGExport | None = None
        expanded_applied_rules: list[str] = []
        if max_rung >= 3:
            r3 = self.run_rung3_dpo_expansion(
                cdg,
                primary.node_id,
                baseline_loss=structural_target,
                baseline_benchmark=baseline_benchmark,
                bottleneck_reason=primary.bottleneck_reason,
                current_node_params=base_params,
                match_results=match_results,
            )
            maybe_exp = r3.details.pop("expanded_cdg", None)
            if isinstance(maybe_exp, CDGExport):
                expanded_candidate_cdg = maybe_exp
                expanded_applied_rules = list(r3.details.get("applied_rules", []))
            total_trials += r3.trials_evaluated
            attempted_rungs.append(r3.rung)
            if r3.improved:
                r3.baseline_loss = baseline_loss
                r3.trials_evaluated = total_trials
                r3.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                r3.details["attempted_rungs"] = attempted_rungs
                return r3

            # Coupled 2-node lookahead: if DPO expanded the topology without standalone gain,
            # evaluate compatible Rung 2 sibling swaps jointly on the expanded CDG!
            if expanded_candidate_cdg is not None and max_rung >= 2:
                coupled = self.run_coupled_dpo_and_sibling_swap(
                    expanded_candidate_cdg,
                    expanded_applied_rules,
                    [g.node_id for g in ordered_bottlenecks],
                    baseline_loss=structural_target,
                    current_node_params=base_params,
                    match_results=match_results,
                    uncertainty_report=uncertainty_report,
                    trial_index=trial_index,
                )
                total_trials += coupled.trials_evaluated
                if coupled.trials_evaluated > 0:
                    attempted_rungs.append("rung3_coupled_dpo_swap")
                if coupled.improved:
                    coupled.baseline_loss = baseline_loss
                    coupled.trials_evaluated = total_trials
                    coupled.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    coupled.details["attempted_rungs"] = attempted_rungs
                    return coupled

        # Extended deterministic sweep over secondary CDG leaves (ranks 2..N-1)
        # before escalating to LLM Rung 4a/4b on multi-stage competition graphs.
        if len(ordered_bottlenecks) > 2:
            if max_rung >= 2:
                for grad in ordered_bottlenecks[2:]:
                    r2_ext = self.run_rung2_sibling_swap(
                        cdg,
                        grad.node_id,
                        baseline_loss=structural_target,
                        current_node_params=base_params,
                        match_results=match_results,
                        uncertainty_report=uncertainty_report,
                        trial_index=trial_index,
                    )
                    total_trials += r2_ext.trials_evaluated
                    attempted_rungs.append(r2_ext.rung)
                    if r2_ext.improved:
                        r2_ext.baseline_loss = baseline_loss
                        r2_ext.trials_evaluated = total_trials
                        r2_ext.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        r2_ext.details["attempted_rungs"] = attempted_rungs
                        return r2_ext
            if max_rung >= 1:
                for grad in ordered_bottlenecks[2:]:
                    r1_ext = self.run_rung1_node_hpo(
                        cdg,
                        grad.node_id,
                        baseline_loss=structural_target,
                        current_node_params=base_params,
                        match_results=match_results,
                        trial_index=trial_index,
                    )
                    total_trials += r1_ext.trials_evaluated
                    attempted_rungs.append(r1_ext.rung)
                    if r1_ext.improved:
                        if self._is_substantial_gain(r1_ext.accepted_loss, baseline_loss):
                            r1_ext.baseline_loss = baseline_loss
                            r1_ext.trials_evaluated = total_trials
                            r1_ext.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                            r1_ext.details["attempted_rungs"] = attempted_rungs
                            return r1_ext
                        deferred_micro_r1 = r1_ext
                        structural_target = r1_ext.accepted_loss

        # Mine Phase 3 uncertainty gap specs if available for Rung 4a / 4b guidance
        gaps_by_node: dict[str, UncertaintyGapSpec] = {}
        if uncertainty_report is not None:
            gap_rep = mine_cdg_uncertainty_gaps(
                cdg,
                uncertainty_report,
                profiles=self.funnel.runner.uncertainty_profiles,
            )
            gaps_by_node = {g.node_id: g for g in gap_rep.gaps}

        # Rung 4a: Localized Subgraph Splice via compose_cdg
        if max_rung >= 4 and self.subdecompose_fn is not None:
            for grad in ordered_bottlenecks[:2]:
                r4a = await self.run_rung4a_localized_splice(
                    cdg,
                    grad.node_id,
                    goal=goal,
                    bottleneck_reason=grad.bottleneck_reason,
                    baseline_loss=structural_target,
                    current_node_params=base_params,
                    match_results=match_results,
                    gap_spec=gaps_by_node.get(grad.node_id),
                )
                total_trials += r4a.trials_evaluated
                attempted_rungs.append(r4a.rung)
                if r4a.improved:
                    r4a.baseline_loss = baseline_loss
                    r4a.trials_evaluated = total_trials
                    r4a.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    r4a.details["attempted_rungs"] = attempted_rungs
                    return r4a

        # Rung 4b: Guided Micro-Atom Synthesis on True Catalog Gaps
        if max_rung >= 4 and self.micro_atom_synthesizer is not None:
            grad_reason_map = {g.node_id: g.bottleneck_reason for g in ordered_bottlenecks}
            existing_leaf_ids = {n.node_id for n in leaves}
            r4b_candidates: list[tuple[str, str]] = []
            seen_r4b: set[str] = set()
            for synth_nid in self.llm_synthesis_nodes:
                if synth_nid in existing_leaf_ids and synth_nid not in seen_r4b:
                    seen_r4b.add(synth_nid)
                    r4b_candidates.append(
                        (
                            synth_nid,
                            grad_reason_map.get(synth_nid, "LLM-diagnosed accuracy bottleneck"),
                        )
                    )
            for grad in ordered_bottlenecks[:2]:
                if grad.node_id not in seen_r4b:
                    seen_r4b.add(grad.node_id)
                    r4b_candidates.append((grad.node_id, grad.bottleneck_reason))

            for cand_nid, cand_reason in r4b_candidates[:2]:
                r4b = await self.run_rung4b_micro_atom_synthesis(
                    cdg,
                    cand_nid,
                    failure_reason=cand_reason,
                    baseline_loss=structural_target,
                    current_node_params=base_params,
                    match_results=match_results,
                    gap_spec=gaps_by_node.get(cand_nid),
                )
                total_trials += r4b.trials_evaluated
                attempted_rungs.append(r4b.rung)
                if r4b.improved:
                    r4b.baseline_loss = baseline_loss
                    r4b.trials_evaluated = total_trials
                    r4b.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    r4b.details["attempted_rungs"] = attempted_rungs
                    return r4b

        # Accept deferred micro Rung 1 gain when no structural or Rung 4a/4b move beat it
        if deferred_micro_r1 is not None:
            deferred_micro_r1.baseline_loss = baseline_loss
            deferred_micro_r1.trials_evaluated = total_trials
            deferred_micro_r1.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            deferred_micro_r1.details["attempted_rungs"] = attempted_rungs
            return deferred_micro_r1

        return LadderStepOutcome(
            rung="exhausted",
            improved=False,
            accepted_cdg=cdg,
            accepted_node_params=base_params,
            accepted_loss=baseline_loss,
            accepted_report=None,
            baseline_loss=baseline_loss,
            bottleneck_node_id=primary.node_id,
            trials_evaluated=total_trials,
            elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
            rationale=f"No improvement across rungs {attempted_rungs} on '{primary.node_id}'",
            details={"attempted_rungs": attempted_rungs},
        )


def _bind_sibling_onto_node(
    cdg: CDGExport,
    node_id: str,
    candidate_primitive: str,
    *,
    catalog: PrimitiveCatalog | None = None,
) -> CDGExport:
    """Bind *candidate_primitive* directly onto *node_id* while keeping existing edges intact."""
    mutated = cdg.model_copy(deep=True)
    prim_meta: AlgorithmicPrimitive | None = (
        catalog.get(candidate_primitive) if catalog is not None else None
    )

    for idx, node in enumerate(mutated.nodes):
        if node.node_id != node_id:
            continue
        updates: dict[str, Any] = {
            "matched_primitive": candidate_primitive,
            "status": NodeStatus.ATOMIC,
        }
        if prim_meta is not None and prim_meta.inputs:
            # Preserve wired input/output port names on existing edges while appending
            # any new optional hyperparameter IOSpecs exposed by the replacement primitive!
            wired_in = {e.input_name for e in mutated.edges if e.target_id == node_id}
            kept_required = [
                inp for inp in node.inputs if inp.required or inp.name in wired_in
            ]
            existing_names = {inp.name for inp in kept_required}
            extra_optional = [
                inp.model_copy(deep=True)
                for inp in prim_meta.inputs
                if not inp.required and inp.name not in existing_names
            ]
            new_inputs = kept_required + extra_optional
            updates["inputs"] = new_inputs
            updates["type_signature"] = format_canonical_type_signature(new_inputs, node.outputs)
        mutated.nodes[idx] = node.model_copy(update=updates)
        break
    return mutated


def sync_node_optional_ports_to_callable(
    node: AlgorithmicNode,
    fn: Callable[..., Any],
    wired_in_names: set[str],
) -> AlgorithmicNode:
    """Rebuild optional ``IOSpec`` ports on *node* from *fn*'s Python default parameters."""
    kept_req = [inp for inp in node.inputs if inp.required or inp.name in wired_in_names]
    existing_names = {inp.name for inp in kept_req}
    extra_opt: list[IOSpec] = []
    try:
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            if (
                pname in existing_names
                or param.kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                or param.default is inspect.Parameter.empty
            ):
                continue
            dv = param.default
            if isinstance(dv, bool):
                td = "bool"
            elif isinstance(dv, int):
                td = "int"
            elif isinstance(dv, float):
                td = "float"
            else:
                continue
            extra_opt.append(
                IOSpec(
                    name=pname,
                    type_desc=td,
                    required=False,
                    default_value_repr=repr(dv),
                    port_type=PortTypeSpec(raw=td, kind="scalar", dtype=td),
                )
            )
    except Exception:
        pass

    new_inputs = kept_req + extra_opt
    return node.model_copy(
        update={
            "inputs": new_inputs,
            "type_signature": format_canonical_type_signature(new_inputs, node.outputs),
        }
    )
