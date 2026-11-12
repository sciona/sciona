"""Telemetry-grounded LLM Navigator and Residual-Informed Leaf Synthesizer for Prototyping.

Implements the neuro-symbolic control plane over Sciona's deterministic prototyping tools:
1. ``TrackedLLMClient`` & ``LLMUsageLedger`` — wraps any ``sciona.hunter.llm.LLMClient`` with
   exact per-call token and wall-clock accounting.
2. ``build_compact_telemetry_digest`` — extracts a bounded, dataset-confidentiality-safe JSON
   telemetry card summarizing CDG topology, runtime intermediate schemas/quantiles, validation
   residual slices, compatible catalog siblings, and triggerable DPO rules.
3. ``LLMPrototypingNavigator`` — invokes the LLM in 1–3 surgical, stateless bounded-context
   calls per session to:
   - re-prioritize bottleneck nodes by mathematical error leverage rather than runtime alone,
   - rank ``Rung 2`` catalog siblings and warm-start ``Rung 1`` Optuna HPO priors, and
   - drive ``Rung 4b`` leaf micro-atom synthesis from empirical residual diagnostics with
     in-memory Tier-1 pre-verification and single-turn self-repair.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import textwrap
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, EdgeKind
from sciona.hunter.llm import LLMClient
from sciona.llm_router import (
    LLMRouter,
    PROTOTYPING_NAVIGATE,
    PROTOTYPING_SYNTHESIZE,
)
from sciona.principal.expansion import ExpansionContext
from sciona.principal.graph_types import PrincipalState
from sciona.principal.models import NodeGradient
from sciona.principal.proposal_helpers import build_expansion_context
from sciona.synthesizer.toposort import toposort_nodes

from sciona.prototyping.ladder import (
    MicroAtomSynthesisSpec,
    _bind_sibling_onto_node,
    compile_micro_atom_source,
    infer_tunable_specs_for_node,
    sync_node_optional_ports_to_callable,
)
from sciona.prototyping.live_runner import resolve_atom_callable

if TYPE_CHECKING:
    from sciona.prototyping.funnel import FunnelEvaluationReport
    from sciona.prototyping.harness import EvalCase, ProblemHarness
    from sciona.prototyping.ladder import RevisionLadder
    from sciona.prototyping.live_runner import LiveCDGRunner
    from sciona.prototyping.session import PrototypingSession


@dataclass
class LLMCallRecord:
    """Single bounded-context LLM invocation record."""

    call_index: int
    role: str
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    billed_tokens: int
    fresh_tokens: int
    latency_ms: float


@dataclass
class LLMUsageLedger:
    """Accumulates token and latency metrics across a ``PrototypingSession``."""

    calls: list[LLMCallRecord] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(c.cache_read_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_fresh_tokens(self) -> int:
        return sum(c.fresh_tokens for c in self.calls)

    @property
    def total_billed_tokens(self) -> int:
        return sum(c.billed_tokens for c in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)

    def snapshot_counts(self) -> tuple[int, int, float]:
        """Return ``(call_count, total_billed_tokens, total_latency_ms)``."""
        return self.call_count, self.total_billed_tokens, self.total_latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_fresh_tokens": self.total_fresh_tokens,
            "total_billed_tokens": self.total_billed_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "calls": [
                {
                    "call_index": c.call_index,
                    "role": c.role,
                    "input_tokens": c.input_tokens,
                    "cache_read_tokens": c.cache_read_tokens,
                    "output_tokens": c.output_tokens,
                    "fresh_tokens": c.fresh_tokens,
                    "billed_tokens": c.billed_tokens,
                    "latency_ms": round(c.latency_ms, 2),
                }
                for c in self.calls
            ],
        }


class TrackedLLMClient:
    """Wraps any ``LLMClient`` (or ``LLMRouter``) and records token & latency usage."""

    def __init__(
        self,
        client: LLMClient,
        ledger: LLMUsageLedger | None = None,
    ) -> None:
        self._client = client
        self.ledger = ledger if ledger is not None else LLMUsageLedger()

    def _resolve_target(self, prompt_key: str) -> LLMClient:
        if isinstance(self._client, LLMRouter) and prompt_key:
            return self._client.for_prompt(prompt_key)
        return self._client

    async def complete_tracked(
        self,
        system: str,
        user: str,
        *,
        role: str = "navigate",
        prompt_key: str = "",
    ) -> str:
        target = self._resolve_target(prompt_key)
        t0 = time.perf_counter()
        text = await target.complete(system, user)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta: dict[str, Any] = {}
        getter = getattr(target, "get_last_completion_metadata", None)
        if callable(getter) and not inspect.iscoroutinefunction(getter):
            try:
                raw_meta = getter()
                if isinstance(raw_meta, dict):
                    meta = raw_meta
            except Exception:
                meta = {}

        est_in = max(1, (len(system) + len(user)) // 4)
        est_out = max(1, len(text) // 4)
        in_tok = int(meta.get("input_tokens") or meta.get("prompt_tokens") or est_in)
        cr_tok = int(meta.get("cache_read_tokens") or 0)
        out_tok = int(meta.get("output_tokens") or meta.get("completion_tokens") or est_out)
        fresh = in_tok + out_tok
        billed = int(meta.get("total_tokens") or (in_tok + cr_tok + out_tok))

        self.ledger.calls.append(
            LLMCallRecord(
                call_index=len(self.ledger.calls) + 1,
                role=role,
                input_tokens=in_tok,
                cache_read_tokens=cr_tok,
                output_tokens=out_tok,
                billed_tokens=billed,
                fresh_tokens=fresh,
                latency_ms=elapsed_ms,
            )
        )
        return text


@dataclass
class NavigationPlan:
    """Structured strategic guidance emitted by ``LLMPrototypingNavigator``."""

    bottleneck_priority: list[str] = field(default_factory=list)
    preferred_siblings: dict[str, list[str]] = field(default_factory=dict)
    param_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    synthesis_targets: list[str] = field(default_factory=list)
    synthesis_hypotheses: dict[str, str] = field(default_factory=dict)
    reasoning_summary: str = ""


def _summarize_value_schema(val: Any, *, max_keys: int = 16) -> dict[str, Any]:
    """Summarize runtime port value schema & aggregate quantiles without exposing raw records."""
    if isinstance(val, np.ndarray):
        if val.dtype == object and val.ndim == 0:
            inner = val.item()
            if isinstance(inner, Mapping):
                sub: dict[str, Any] = {}
                for idx, (k, v) in enumerate(inner.items()):
                    if idx >= max_keys:
                        break
                    sub[str(k)] = _summarize_value_schema(v, max_keys=16)
                return {
                    "container": "np_object_dict",
                    "keys": sub,
                }
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
        }
        if val.size > 0 and np.issubdtype(val.dtype, np.number):
            arr_f = val.astype(np.float64, copy=False)
            finite_mask = np.isfinite(arr_f)
            nan_frac = 1.0 - float(np.mean(finite_mask))
            summary["nan_frac"] = round(nan_frac, 4)
            if np.any(finite_mask):
                fin = arr_f[finite_mask]
                q05, q50, q95 = np.quantile(fin, [0.05, 0.50, 0.95])
                summary["quantiles_p05_p50_p95"] = [
                    round(float(q05), 5),
                    round(float(q50), 5),
                    round(float(q95), 5),
                ]
        return summary

    if isinstance(val, Mapping):
        return {
            "container": "dict",
            "keys": {
                str(k): _summarize_value_schema(v, max_keys=16)
                for idx, (k, v) in enumerate(val.items())
                if idx < max_keys
            },
        }
    if isinstance(val, (list, tuple)):
        out: dict[str, Any] = {"container": type(val).__name__, "len": len(val)}
        if val:
            out["elem0"] = _summarize_value_schema(val[0], max_keys=4)
        return out
    if isinstance(val, (int, float, np.number, bool)):
        return {"type": type(val).__name__}
    return {"type": type(val).__name__}


def _compute_residual_diagnostics(
    expected: Any,
    outputs: Mapping[str, Any],
    case_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute aggregate, privacy-safe validation residual diagnostics."""
    diag: dict[str, Any] = {"metrics": {str(k): round(float(v), 6) for k, v in case_metrics.items() if isinstance(v, (int, float, np.number))}}
    unqualified = [v for k, v in outputs.items() if "." not in str(k) and isinstance(v, np.ndarray)]
    if not unqualified:
        all_arrs = [v for v in outputs.values() if isinstance(v, np.ndarray)]
        if len(all_arrs) == 1:
            unqualified = all_arrs
    pred = unqualified[0] if len(unqualified) == 1 else None
    if pred is None:
        return diag

    diag["pred_schema"] = _summarize_value_schema(pred)
    if not isinstance(expected, np.ndarray):
        if isinstance(expected, Mapping):
            diag["expected_keys"] = list(expected.keys())
        return diag

    y = np.asarray(expected)
    p = np.asarray(pred, dtype=np.float64)
    diag["target_schema"] = _summarize_value_schema(y)

    # Case 1: Same-shape continuous regression / multi-horizon forecasting
    if y.shape == p.shape and np.issubdtype(y.dtype, np.number) and y.size > 0:
        yf = y.astype(np.float64)
        mask = np.isfinite(yf) & np.isfinite(p)
        if np.any(mask):
            yt = yf[mask]
            yp = p[mask]
            abs_err = np.abs(yt - yp)
            denom = np.maximum(np.abs(yt) + np.abs(yp), 1e-12)
            smape_elem = np.where((np.abs(yt) + np.abs(yp)) > 1e-12, 200.0 * abs_err / denom, 0.0)
            diag["mean_signed_bias"] = round(float(np.mean(yp - yt)), 5)
            med_t = float(np.median(np.abs(yt)))
            if med_t > 1e-9:
                diag["median_pred_to_target_ratio"] = round(float(np.median(np.abs(yp)) / med_t), 4)
            # Tercile error breakdown by target magnitude
            q33, q67 = np.quantile(yt, [0.33, 0.67])
            low_m = yt <= q33
            mid_m = (yt > q33) & (yt <= q67)
            hi_m = yt > q67
            diag["smape_by_target_tercile"] = {
                "low": round(float(np.mean(smape_elem[low_m])), 4) if np.any(low_m) else 0.0,
                "mid": round(float(np.mean(smape_elem[mid_m])), 4) if np.any(mid_m) else 0.0,
                "high": round(float(np.mean(smape_elem[hi_m])), 4) if np.any(hi_m) else 0.0,
            }
            if y.ndim == 2 and y.shape[1] <= 16:
                col_smape = [
                    round(
                        float(
                            np.mean(
                                200.0
                                * np.abs(yf[:, j] - p[:, j])
                                / np.maximum(np.abs(yf[:, j]) + np.abs(p[:, j]), 1e-12)
                            )
                        ),
                        3,
                    )
                    for j in range(y.shape[1])
                ]
                diag["horizon_step_smape"] = col_smape

    # Case 2: Multiclass probability matrix (N, C) vs integer class labels (N,)
    elif (
        p.ndim == 2
        and y.ndim == 1
        and p.shape[0] == y.shape[0]
        and np.issubdtype(y.dtype, np.integer)
    ):
        yi = y.astype(np.int64)
        n_cls = p.shape[1]
        p_clip = np.clip(p, 1e-12, 1.0)
        p_norm = p_clip / np.sum(p_clip, axis=1, keepdims=True)
        cls_losses: list[tuple[int, float, int]] = []
        for c in range(n_cls):
            mc = yi == c
            cnt = int(np.sum(mc))
            if cnt > 0:
                ll_c = float(-np.mean(np.log(p_norm[mc, c])))
                cls_losses.append((c, round(ll_c, 4), cnt))
        cls_losses.sort(key=lambda t: t[1], reverse=True)
        diag["multiclass_n_classes"] = n_cls
        diag["worst_classes_logloss"] = [
            {"class_idx": c, "logloss": ll, "support": cnt} for c, ll, cnt in cls_losses[:4]
        ]
        # Check temperature calibration sensitivity
        logits = np.log(p_norm)
        temp_probe: dict[str, float] = {}
        for temp in (0.7, 1.0, 1.35):
            scaled = logits / temp
            scaled -= np.max(scaled, axis=1, keepdims=True)
            probs_t = np.exp(scaled)
            probs_t /= np.sum(probs_t, axis=1, keepdims=True)
            per_c = [
                float(-np.mean(np.log(np.clip(probs_t[yi == c, c], 1e-12, 1.0))))
                for c in range(n_cls)
                if np.any(yi == c)
            ]
            temp_probe[f"T={temp}"] = round(float(np.mean(per_c)), 5)
        diag["temperature_logloss_probe"] = temp_probe

    return diag


def _extract_clean_callable_snippet(fn: Callable[..., Any], max_lines: int = 120) -> str:
    """Return a compact, comment-and-docstring-scrubbed function snippet when inspectable."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        try:
            return f"def incumbent_leaf{inspect.signature(fn)}: ..."
        except Exception:
            return ""
    try:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.name = "incumbent_leaf"
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:] or [ast.Pass()]
        src = ast.unparse(tree)
    except Exception:
        fn_name = getattr(fn, "__name__", "")
        if fn_name:
            src = src.replace(f"def {fn_name}(", "def incumbent_leaf(", 1)
    lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    if len(lines) > max_lines:
        head = max(10, max_lines - 12)
        lines = [*lines[:head], "    # ... (body truncated) ...", *lines[-12:]]
    return "\n".join(lines)


def build_compact_telemetry_digest(
    *,
    cdg: CDGExport,
    goal: str,
    best_loss: float,
    report: FunnelEvaluationReport | None,
    gradients: Sequence[NodeGradient],
    harness: ProblemHarness,
    runner: LiveCDGRunner,
    ladder: RevisionLadder,
    node_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct a bounded, privacy-safe JSON telemetry digest for LLM navigation."""
    leaves = cdg.leaf_nodes() or [n for n in cdg.nodes if not n.children]
    leaf_map = {n.node_id: n for n in leaves}
    try:
        order_ids = toposort_nodes(leaves, cdg.edges)
        ordered_leaves = [leaf_map[nid] for nid in order_ids if nid in leaf_map]
    except Exception:
        ordered_leaves = list(leaves)

    grad_map = {g.node_id: g for g in gradients}

    # Execute (mostly from IntermediateCache!) on first validation or synthetic case for port schemas
    val_cases = list(harness.validation_cases()) or list(harness.synthetic_cases())
    port_schemas: dict[str, Any] = {}
    residual_diag: dict[str, Any] = {}
    if val_cases:
        sample_case: EvalCase = val_cases[0]
        run_res = runner.execute(
            cdg,
            sample_case.inputs,
            node_params=node_params or None,
        )
        if run_res.ok:
            for nid, port_dict in run_res.node_outputs.items():
                port_schemas[nid] = {
                    str(pk): _summarize_value_schema(pv)
                    for pk, pv in port_dict.items()
                }
            case_metrics: dict[str, Any] = {}
            if report is not None and report.tier_results:
                for tr in reversed(list(report.tier_results.values())):
                    if tr.case_scores:
                        case_metrics = dict(tr.case_scores[0].metrics)
                        break
            residual_diag = _compute_residual_diagnostics(
                sample_case.expected, run_res.outputs, case_metrics
            )

    nodes_digest: list[dict[str, Any]] = []
    for node in ordered_leaves:
        prim_name = str(node.matched_primitive or "")
        fn_callable: Callable[..., Any] | None = None
        sig_str = ""
        if prim_name:
            try:
                fn_callable, _ = resolve_atom_callable(
                    prim_name,
                    node=node,
                    extra_callables=runner.extra_callables,
                )
                sig_str = str(inspect.signature(fn_callable))
            except Exception:
                fn_callable = None

        wired_in = {
            e.input_name
            for e in cdg.edges
            if e.target_id == node.node_id and e.edge_kind != EdgeKind.CALLABLE_INJECTION
        }
        specs = infer_tunable_specs_for_node(
            node,
            catalog=ladder.catalog,
            callable_fn=fn_callable,
            wired_input_names=wired_in,
        )
        siblings = ladder.discover_compatible_siblings(cdg, node.node_id)
        sib_info = []
        for s_fqn in siblings[:5]:
            p_meta = ladder.catalog.get(s_fqn) if ladder.catalog is not None else None
            sib_info.append(
                {
                    "fqn": s_fqn,
                    "description": p_meta.description if p_meta is not None else "",
                }
            )

        g = grad_map.get(node.node_id)
        nodes_digest.append(
            {
                "node_id": node.node_id,
                "name": node.name,
                "description": node.description,
                "concept_type": node.concept_type.value,
                "matched_primitive": prim_name,
                "callable_signature": sig_str,
                "inputs": [p.name for p in node.inputs if p.required or p.name in wired_in],
                "outputs": [p.name for p in node.outputs],
                "tunable_params": [
                    {
                        "name": sp.name,
                        "kind": sp.kind,
                        "default": sp.default,
                        "bounds": [sp.min_value, sp.max_value] if sp.min_value is not None else sp.choices,
                        "current": (node_params.get(node.node_id) or {}).get(sp.name, sp.default),
                    }
                    for sp in specs
                ],
                "compatible_siblings": sib_info,
                "credit_score": round(g.gradient_score, 4) if g is not None else 0.0,
                "credit_reason": g.bottleneck_reason if g is not None else "",
                "output_port_schema": port_schemas.get(node.node_id, {}),
            }
        )

    dpo_rules: list[dict[str, Any]] = []
    if ladder.expansion_engine is not None:
        try:
            st = PrincipalState(
                cdg=cdg,
                benchmark=report.benchmark if report is not None else None,
                planning_artifact=cdg.planning_artifact,
            )
            ctx: ExpansionContext = build_expansion_context(st)
            for rs in getattr(ladder.expansion_engine, "_rule_sets", []):
                for d in rs.diagnose(cdg, ctx):
                    dpo_rules.append(
                        {
                            "rule_name": d.rule_name,
                            "severity": round(float(d.severity), 3),
                            "evidence": d.evidence,
                        }
                    )
        except Exception:
            pass

    return {
        "goal": goal,
        "current_validation_loss": round(float(best_loss), 6) if math.isfinite(best_loss) else None,
        "nodes": nodes_digest,
        "dpo_expansion_candidates": dpo_rules,
        "validation_residual_diagnostics": residual_diag,
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


def _extract_python_code(raw_text: str) -> str:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned


NAVIGATOR_SYSTEM_PROMPT = """You are the Strategic Mathematical Navigator for Sciona's deterministic CDG prototyping engine.
You receive a compact JSON telemetry card of an executable Computation Dependency Graph (CDG), its node port schemas, tunable parameters, compatible Rung-2 catalog siblings, triggerable Rung-3 DPO graph-rewrite rules, and empirical validation residual diagnostics.

Your job is NOT to write monolithic pipelines. Sciona executes Rung 1 (Optuna HPO), Rung 2 (sibling swaps), and Rung 3 (DPO expansions) deterministically in memory with intermediate caching.
Your job is to return a single JSON object steering Sciona's search order and identifying the highest-leverage leaf node for Rung 4b residual-driven micro-atom synthesis when deterministic catalog primitives leave an accuracy gap.

Pay special attention to:
1. Fast closed-form nodes (e.g., L2 Ridge on asymmetric/SMAPE loss, Euclidean features under group shift, uncalibrated probabilities under weighted log-loss) that have low execution time so naive runtime heuristics under-rank them, even though they dominate validation loss!
2. Objective-loss alignment (e.g., L1/Huber/log1p median forecasting + day-of-week seasonal ratios for SMAPE; rich domain-specific time-series decay/skew/SNR/color-ratio feature extraction + regularized LightGBM+CatBoost temperature-calibrated blending for imbalanced multiclass log-loss; group/cohort invariance under constraints).
3. Token economy: if `compatible_siblings` and/or `dpo_expansion_candidates` already directly resolve the mathematical bottleneck (e.g., covariance/Mahalanobis whitening + constrained/nullspace projection or calibrated GBDT siblings), set `"synthesis_targets": []` so Sciona solves the task deterministically at zero synthesis cost! Reserve `"synthesis_targets"` (at most 1–2 highest-leverage leaf nodes) for genuine catalog gaps where existing siblings miss critical domain physics, seasonal/asymmetric structure, or objective-aligned modeling.

Respond ONLY with valid JSON matching this exact schema:
{
  "bottleneck_priority": ["node_id_most_impactful_on_loss", "second_node_id", "..."],
  "preferred_siblings": {"node_id": ["best_sibling_fqn"]},
  "param_priors": {"node_id": {"param_name": numeric_or_categorical_value}},
  "synthesis_targets": ["best_node_id_for_custom_leaf_upgrade", "optional_second_node_id"],
  "synthesis_hypotheses": {
    "node_id": "Concise mathematical specification of what the upgraded Python leaf function should compute to slash validation loss while generalizing out-of-sample."
  },
  "reasoning_summary": "1-2 sentence mathematical diagnosis."
}"""


SYNTHESIZER_SYSTEM_PROMPT = """You are Sciona's Residual-Informed Leaf Micro-Atom Synthesizer (Rung 4b).
You write a SINGLE, self-contained, drop-in Python leaf function for ONE bottleneck node in an executable Computation Dependency Graph (CDG).
All upstream and downstream CDG nodes remain frozen and cached in memory!

STRICT ENGINEERING RULES:
1. Output ONLY valid Python code (a single ```python ... ``` block) defining the requested top-level function name and any private `_helper` functions inside the same snippet.
2. Allowed imports ONLY from standard scientific Python installed in-process: `math`, `numpy as np`, `pandas as pd`, `scipy` (`scipy.stats`, `scipy.signal`, `scipy.special`, `scipy.optimize`), `sklearn` (`sklearn.linear_model`, `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.decomposition`), `lightgbm as lgb`, `catboost` (`CatBoostClassifier`, `CatBoostRegressor`).
3. Keep `n_jobs=1` / `thread_count=1` on any LightGBM/CatBoost/sklearn estimator (avoid OpenMP thread contention on macOS) and keep per-call runtime fast (< 15 seconds on validation).
4. Match the EXACT required input parameter names and EXACT output type/shape/dict-keys shown in the runtime port schema! If an input or output port is an `np.ndarray` of `dtype=object` wrapping a `dict` (`d = port_arg.item()`, returning `np.array(out_dict, dtype=object)`), preserve all passthrough dict keys expected by downstream nodes!
5. Expose 1 to 3 sensible optional scalar hyperparameter arguments with strong default values (e.g., `blend_weight: float = 0.5, temperature: float = 1.0`) AFTER the required port input(s), so Sciona's deterministic Rung 1 Optuna HPO can fine-tune your leaf automatically on the next iteration!
6. Prevent validation overfitting: use out-of-fold / regularized cross-validation or shrinkage when blending learned transformations, handle NaNs/Infs safely, and optimize directly for the task's evaluation objective."""


class LLMPrototypingNavigator:
    """Coordinates bounded-context LLM direction planning and leaf synthesis over ``PrototypingSession``."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_synthesis_calls: int = 2,
        enable_self_repair: bool = True,
        repropose_on_exhaustion: bool = True,
    ) -> None:
        self.tracked_llm = (
            llm_client
            if isinstance(llm_client, TrackedLLMClient)
            else TrackedLLMClient(llm_client)
        )
        self.max_synthesis_calls = max(0, int(max_synthesis_calls))
        self.enable_self_repair = enable_self_repair
        self.repropose_on_exhaustion = repropose_on_exhaustion
        self.last_plan: NavigationPlan | None = None
        self._synthesis_calls_used = 0
        self._attempted_synthesis_nodes: set[str] = set()
        self._exhaustion_reproposed = False

    @property
    def usage_ledger(self) -> LLMUsageLedger:
        return self.tracked_llm.ledger

    async def propose_direction(
        self,
        session: PrototypingSession,
        *,
        post_deterministic_exhaustion: bool = False,
    ) -> NavigationPlan:
        """Execute 1 compact LLM call to diagnose bottlenecks and set search priors."""
        digest = build_compact_telemetry_digest(
            cdg=session.best_cdg,
            goal=session.goal,
            best_loss=session.best_loss,
            report=session.best_report,
            gradients=session.last_gradients,
            harness=session.harness,
            runner=session.runner,
            ladder=session.ladder,
            node_params=session.best_node_params,
        )
        header = "Analyze this executable CDG telemetry digest and return your JSON navigation plan:\n"
        if post_deterministic_exhaustion:
            self._exhaustion_reproposed = True
            header = (
                "Deterministic Rungs 1-3 (catalog sibling swaps, DPO expansions, and Optuna HPO) have now "
                "CONVERGED and EXHAUSTED on this CDG. Inspect `current_validation_loss`, bound primitives, "
                "and `validation_residual_diagnostics`: if validation metrics are already near ceiling / "
                "constraints satisfied (or further custom modeling risks overfitting small N), return "
                '`"synthesis_targets": []` to stop immediately; otherwise select the single highest-leverage '
                'leaf node in `"synthesis_targets": ["<node_id>"]` with a concrete `"synthesis_hypotheses"` '
                "specification for Rung 4b Python leaf synthesis:\n"
            )
        user_prompt = header + json.dumps(digest, separators=(",", ":"))
        raw = await self.tracked_llm.complete_tracked(
            NAVIGATOR_SYSTEM_PROMPT,
            user_prompt,
            role="navigate",
            prompt_key=PROTOTYPING_NAVIGATE,
        )
        try:
            obj = _extract_json_object(raw)
        except Exception:
            obj = {}

        valid_ids = {n.node_id for n in session.best_cdg.nodes}
        prio = [str(x) for x in (obj.get("bottleneck_priority") or []) if str(x) in valid_ids]
        pref_sibs: dict[str, list[str]] = {}
        raw_sibs = obj.get("preferred_siblings")
        if isinstance(raw_sibs, Mapping):
            for k, v in raw_sibs.items():
                if str(k) in valid_ids and isinstance(v, list):
                    pref_sibs[str(k)] = [str(item) for item in v if item]

        priors: dict[str, dict[str, Any]] = {}
        raw_priors = obj.get("param_priors")
        if isinstance(raw_priors, Mapping):
            for k, v in raw_priors.items():
                if str(k) in valid_ids and isinstance(v, Mapping):
                    priors[str(k)] = dict(v)

        synth_targets = [
            str(x) for x in (obj.get("synthesis_targets") or []) if str(x) in valid_ids
        ]
        synth_hyps: dict[str, str] = {}
        raw_hyps = obj.get("synthesis_hypotheses")
        if isinstance(raw_hyps, Mapping):
            for k, v in raw_hyps.items():
                if str(v).strip():
                    synth_hyps[str(k)] = str(v).strip()

        plan = NavigationPlan(
            bottleneck_priority=prio,
            preferred_siblings=pref_sibs,
            param_priors=priors,
            synthesis_targets=synth_targets,
            synthesis_hypotheses=synth_hyps,
            reasoning_summary=str(obj.get("reasoning_summary") or ""),
        )
        self.last_plan = plan
        self.apply_plan_to_session(session, plan)
        return plan

    def apply_plan_to_session(self, session: PrototypingSession, plan: NavigationPlan) -> None:
        """Wire LLM strategic priors into ``RevisionLadder`` and ``session.last_gradients``."""
        session.ladder.llm_preferred_siblings.update(plan.preferred_siblings)
        session.ladder.llm_param_priors.update(plan.param_priors)
        if plan.synthesis_targets:
            session.ladder.llm_synthesis_nodes = list(dict.fromkeys(plan.synthesis_targets))

        if plan.bottleneck_priority and session.last_gradients:
            rank_Lookup = {nid: idx for idx, nid in enumerate(plan.bottleneck_priority)}
            ordered = sorted(
                session.last_gradients,
                key=lambda g: (rank_Lookup.get(g.node_id, 999), -g.gradient_score),
            )
            # Assign monotonically decreasing boosted gradient scores so top LLM-diagnosed
            # accuracy bottlenecks lead Rung 1/2/3/4b sweeps.
            n_tot = max(1, len(ordered))
            boosted: list[NodeGradient] = []
            for idx, g in enumerate(ordered):
                hyp = plan.synthesis_hypotheses.get(g.node_id, "")
                reason = f"{g.bottleneck_reason}; LLM diagnosis: {hyp}" if hyp else g.bottleneck_reason
                boosted.append(
                    g.model_copy(
                        update={
                            "gradient_score": max(
                                float(g.gradient_score), 100.0 - 12.0 * (idx / n_tot)
                            ),
                            "bottleneck_reason": reason[:280],
                        }
                    )
                )
            session.last_gradients = boosted

    def build_micro_atom_synthesizer(
        self,
        session: PrototypingSession,
    ) -> Callable[[MicroAtomSynthesisSpec], Any]:
        """Return an async ``MicroAtomSynthesizerCallable`` grounded in live session residuals."""

        async def _synthesize(spec: MicroAtomSynthesisSpec) -> tuple[str, str] | None:
            if self._synthesis_calls_used >= self.max_synthesis_calls:
                return None
            if spec.slot_node_id in self._attempted_synthesis_nodes:
                return None
            if self.last_plan is not None and spec.slot_node_id not in self.last_plan.synthesis_targets:
                return None

            cdg = session.best_cdg
            target = next((n for n in cdg.nodes if n.node_id == spec.slot_node_id), None)
            if target is None:
                return None

            self._attempted_synthesis_nodes.add(spec.slot_node_id)
            self._synthesis_calls_used += 1

            # Capture exact runtime input & output schemas from cached validation/synthetic execution
            cases = list(session.harness.validation_cases()) or list(session.harness.synthetic_cases())
            in_port_schemas: dict[str, Any] = {}
            out_port_schemas: dict[str, Any] = {}
            residual_diag: dict[str, Any] = {}

            wired_edges = [
                e
                for e in cdg.edges
                if e.target_id == target.node_id and e.edge_kind != EdgeKind.CALLABLE_INJECTION
            ]
            wired_in_names = {e.input_name for e in wired_edges}
            req_in_ports = [
                p.name for p in target.inputs if p.required or p.name in wired_in_names
            ]

            if cases:
                c0 = cases[0]
                run_res = session.runner.execute(
                    cdg,
                    c0.inputs,
                    node_params=session.best_node_params or None,
                )
                if run_res.ok:
                    for edge in wired_edges:
                        src_outs = run_res.node_outputs.get(edge.source_id, {})
                        if edge.output_name in src_outs:
                            in_port_schemas[edge.input_name] = _summarize_value_schema(
                                src_outs[edge.output_name]
                            )
                    for p_name in req_in_ports:
                        if p_name not in in_port_schemas and p_name in c0.inputs:
                            in_port_schemas[p_name] = _summarize_value_schema(c0.inputs[p_name])
                    tgt_outs = run_res.node_outputs.get(target.node_id, {})
                    for out_p in target.outputs:
                        if out_p.name in tgt_outs:
                            out_port_schemas[out_p.name] = _summarize_value_schema(
                                tgt_outs[out_p.name]
                            )
                    case_metrics: dict[str, Any] = {}
                    if session.best_report is not None and session.best_report.tier_results:
                        for tr in reversed(list(session.best_report.tier_results.values())):
                            if tr.case_scores:
                                case_metrics = dict(tr.case_scores[0].metrics)
                                break
                    residual_diag = _compute_residual_diagnostics(
                        c0.expected, run_res.outputs, case_metrics
                    )

            incumbent_snippet = ""
            prim_name = str(target.matched_primitive or "")
            if prim_name:
                try:
                    inc_fn, _ = resolve_atom_callable(
                        prim_name,
                        node=target,
                        extra_callables=session.runner.extra_callables,
                    )
                    incumbent_snippet = _extract_clean_callable_snippet(inc_fn, max_lines=125)
                except Exception:
                    incumbent_snippet = ""

            hyp = ""
            if self.last_plan is not None:
                hyp = self.last_plan.synthesis_hypotheses.get(spec.slot_node_id, "")

            synth_context = {
                "goal": session.goal,
                "current_validation_loss_to_beat": round(float(session.best_loss), 6),
                "target_node": {
                    "node_id": target.node_id,
                    "name": target.name,
                    "description": target.description,
                    "required_input_arg_names": req_in_ports,
                    "output_port_names": [p.name for p in target.outputs],
                    "return_convention": (
                        "Return single output value directly"
                        if len(target.outputs) == 1
                        else f"Return tuple of {len(target.outputs)} outputs in exact order {[p.name for p in target.outputs]}"
                    ),
                },
                "runtime_input_port_schemas": in_port_schemas,
                "runtime_expected_output_schemas": out_port_schemas,
                "validation_residual_diagnostics": residual_diag,
                "navigation_hypothesis": hyp or spec.failure_reason,
                "downstream_nodes": [
                    {"node_id": n.node_id, "name": n.name, "matched_primitive": n.matched_primitive}
                    for n in cdg.nodes
                    if any(e.source_id == target.node_id and e.target_id == n.node_id for e in cdg.edges)
                ],
            }

            user_prompt = (
                f"Write a standalone upgraded Python leaf function named `{spec.suggested_function_name}` "
                f"for node `{target.node_id}` that strictly improves validation loss below "
                f"`{session.best_loss:.6f}`.\n\n"
                f"Context JSON:\n{json.dumps(synth_context, indent=2)}\n"
            )
            if incumbent_snippet:
                user_prompt += (
                    f"\nCurrent incumbent leaf implementation (upgrade upon its mathematical weaknesses; "
                    f"do NOT import internal helper modules—write self-contained numpy/scipy/sklearn/lightgbm/catboost code):\n"
                    f"```python\n{incumbent_snippet}\n```\n"
                )

            raw_resp = await self.tracked_llm.complete_tracked(
                SYNTHESIZER_SYSTEM_PROMPT,
                user_prompt,
                role="synthesize_leaf",
                prompt_key=PROTOTYPING_SYNTHESIZE,
            )
            code = _extract_python_code(raw_resp)

            # Pre-verify compilation + Tier-1 synthetic execution; allow 1 compact self-repair turn
            err_msg = self._dry_run_tier1_check(session, target, spec.suggested_function_name, code)
            if err_msg and self.enable_self_repair:
                repair_prompt = (
                    f"Your synthesized function `{spec.suggested_function_name}` failed pre-verification with error:\n"
                    f"{err_msg}\n\n"
                    f"Required input arg names: {req_in_ports}\n"
                    f"Expected output schemas: {json.dumps(out_port_schemas)}\n\n"
                    f"Previous code:\n```python\n{code}\n```\n\n"
                    f"Return ONLY the corrected standalone ```python ... ``` block."
                )
                repaired_raw = await self.tracked_llm.complete_tracked(
                    SYNTHESIZER_SYSTEM_PROMPT,
                    repair_prompt,
                    role="repair_leaf",
                    prompt_key=PROTOTYPING_SYNTHESIZE,
                )
                code = _extract_python_code(repaired_raw)

            return spec.suggested_function_name, code

        return _synthesize

    @staticmethod
    def _dry_run_tier1_check(
        session: PrototypingSession,
        target: AlgorithmicNode,
        preferred_fn_name: str,
        source_code: str,
    ) -> str:
        """Return empty string if *source_code* compiles and passes Tier 1 synthetic execution, else error."""
        try:
            fn_name, fn_callable = compile_micro_atom_source(
                source_code, preferred_name=preferred_fn_name
            )
        except Exception as exc:
            return f"Compilation error: {exc}"

        synth_cases = list(session.harness.synthetic_cases())
        if not synth_cases:
            return ""

        temp_fqn = f"sciona.synthesized._dryrun_{fn_name}_{uuid.uuid4().hex[:6]}"
        session.runner.extra_callables[temp_fqn] = fn_callable
        try:
            mutated = _bind_sibling_onto_node(
                session.best_cdg, target.node_id, temp_fqn, catalog=None
            )
            wired_in = {
                e.input_name
                for e in mutated.edges
                if e.target_id == target.node_id and e.edge_kind != EdgeKind.CALLABLE_INJECTION
            }
            for idx, n in enumerate(mutated.nodes):
                if n.node_id == target.node_id:
                    mutated.nodes[idx] = sync_node_optional_ports_to_callable(
                        n, fn_callable, wired_in
                    )
                    break
            cand_params = {
                k: dict(v)
                for k, v in session.best_node_params.items()
                if k != target.node_id
            }
            res = session.runner.execute(
                mutated,
                synth_cases[0].inputs,
                node_params=cand_params or None,
            )
            if not res.ok:
                return f"Runtime error on synthetic check: {res.error}"
            sc = session.harness.score(synth_cases[0], res.outputs)
            if not sc.passed or not math.isfinite(sc.loss):
                return f"Synthetic score check failed: {sc.error or sc.loss}"
            return ""
        except Exception as exc:
            return f"Dry-run exception: {exc}"
        finally:
            session.runner.extra_callables.pop(temp_fqn, None)
