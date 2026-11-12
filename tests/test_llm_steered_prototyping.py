"""Unit tests for LLM-steered neuro-symbolic prototyping (Navigator + Rung 4b leaf synthesis)."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    PortTypeSpec,
    PrimitiveParamSpec,
)
from sciona.prototyping import (
    EvalCase,
    LLMPrototypingNavigator,
    LiveCDGRunner,
    ProblemHarness,
    PrototypingSession,
    TrackedLLMClient,
    build_compact_telemetry_digest,
)
from sciona.prototyping.harness import CaseScore


class _MockSteeredLLM:
    """Deterministic async mock ``LLMClient`` with optional completion token metadata."""

    def __init__(self, responses: list[tuple[str, dict[str, int]]]) -> None:
        self._responses = list(responses)
        self.prompts: list[tuple[str, str]] = []
        self._last_meta: dict[str, int] = {}

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if not self._responses:
            raise AssertionError("Unexpected extra LLM call in _MockSteeredLLM")
        text, meta = self._responses.pop(0)
        self._last_meta = dict(meta)
        return text

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)

    def get_last_completion_metadata(self) -> dict[str, int]:
        return dict(self._last_meta)


class _QuadraticGapHarness(ProblemHarness):
    """Synthetic regression harness y = x + 0.8 * x^2 where linear leaf leaves a catalog gap."""

    def __init__(self) -> None:
        arr_pt = PortTypeSpec(raw="np.ndarray", kind="array", dtype="float64", ndim=2)
        super().__init__(
            root_inputs=[IOSpec(name="x", type_desc="np.ndarray", required=True, port_type=arr_pt)],
            root_outputs=[IOSpec(name="y", type_desc="np.ndarray", required=True, port_type=arr_pt)],
        )
        rng = np.random.default_rng(7)
        x_s = rng.uniform(-2.0, 2.0, size=(32, 1))
        x_v = rng.uniform(-2.0, 2.0, size=(96, 1))
        x_h = rng.uniform(-2.0, 2.0, size=(96, 1))
        self._synth = [EvalCase("s0", {"x": x_s}, expected=x_s + 0.8 * (x_s**2))]
        self._val = [EvalCase("v0", {"x": x_v}, expected=x_v + 0.8 * (x_v**2))]
        self._hold = [EvalCase("h0", {"x": x_h}, expected=x_h + 0.8 * (x_h**2))]

    def synthetic_cases(self) -> list[EvalCase]:
        return self._synth

    def validation_cases(self) -> list[EvalCase]:
        return self._val

    def holdout_cases(self) -> list[EvalCase]:
        return self._hold

    def score(self, case: EvalCase, outputs: dict[str, Any]) -> CaseScore:
        raw_y = outputs["y"] if "y" in outputs else outputs["n2_head.y"]
        pred = np.asarray(raw_y, dtype=np.float64)
        exp = np.asarray(case.expected, dtype=np.float64)
        if pred.shape != exp.shape or not np.all(np.isfinite(pred)):
            return CaseScore(case_id=case.case_id, loss=float("inf"), passed=False, error="bad_shape_or_nan")
        rmse = float(np.sqrt(np.mean((pred - exp) ** 2)))
        return CaseScore(case_id=case.case_id, loss=rmse, passed=True, metrics={"rmse": rmse})


def _build_two_stage_synthetic_cdg() -> tuple[CDGExport, LiveCDGRunner, PrimitiveCatalog]:
    def slow_upstream_identity(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        time.sleep(0.012)
        return np.asarray(x, dtype=np.float64) * float(scale)

    def fast_linear_head(h: np.ndarray, slope: float = 1.0) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) * float(slope)

    def alt_linear_sibling(h: np.ndarray, slope: float = 0.95) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) * float(slope)

    arr_pt = PortTypeSpec(raw="np.ndarray", kind="array", dtype="float64", ndim=2)
    n1 = AlgorithmicNode(
        node_id="n1_prep",
        name="Slow Upstream Conditioning",
        description="Upstream feature pass",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive="synth_pkg.slow_upstream_identity",
        inputs=[
            IOSpec(name="x", type_desc="np.ndarray", required=True, port_type=arr_pt),
            IOSpec(name="scale", type_desc="float", required=False, default_value_repr="1.0"),
        ],
        outputs=[IOSpec(name="h", type_desc="np.ndarray", required=True, port_type=arr_pt)],
    )
    n2 = AlgorithmicNode(
        node_id="n2_head",
        name="Fast Linear Predictor",
        description="Downstream prediction leaf",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive="synth_pkg.fast_linear_head",
        inputs=[
            IOSpec(name="h", type_desc="np.ndarray", required=True, port_type=arr_pt),
            IOSpec(name="slope", type_desc="float", required=False, default_value_repr="1.0"),
        ],
        outputs=[IOSpec(name="y", type_desc="np.ndarray", required=True, port_type=arr_pt)],
    )
    cdg = CDGExport(
        nodes=[n1, n2],
        edges=[
            DependencyEdge(
                source_id="n1_prep",
                target_id="n2_head",
                output_name="h",
                input_name="h",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
        metadata={"goal": "Minimize RMSE on nonlinear target"},
    )
    runner = LiveCDGRunner(
        extra_callables={
            "synth_pkg.slow_upstream_identity": slow_upstream_identity,
            "synth_pkg.fast_linear_head": fast_linear_head,
            "synth_pkg.alt_linear_sibling": alt_linear_sibling,
        }
    )
    catalog = PrimitiveCatalog()
    catalog.add(
        AlgorithmicPrimitive(
            name="synth_pkg.fast_linear_head",
            source="synth_pkg",
            category=ConceptType.CUSTOM,
            description="Linear slope predictor",
            inputs=list(n2.inputs),
            outputs=list(n2.outputs),
            tunable_params=[
                PrimitiveParamSpec(name="slope", kind="float", default=1.0, min_value=0.5, max_value=1.5)
            ],
        )
    )
    catalog.add(
        AlgorithmicPrimitive(
            name="synth_pkg.alt_linear_sibling",
            source="synth_pkg",
            category=ConceptType.CUSTOM,
            description="Alternative linear slope predictor",
            inputs=list(n2.inputs),
            outputs=list(n2.outputs),
        )
    )
    return cdg, runner, catalog


@pytest.mark.asyncio
async def test_tracked_llm_client_and_compact_digest_privacy() -> None:
    cdg, runner, catalog = _build_two_stage_synthetic_cdg()
    harness = _QuadraticGapHarness()
    session = PrototypingSession(cdg, harness=harness, runner=runner, catalog=catalog)
    session.evaluate_initial()

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
    assert len(digest["nodes"]) == 2
    assert "smape_by_target_tercile" in digest["validation_residual_diagnostics"]
    # Digest summarizes shape & quantiles, never raw 96-element arrays
    serialized = json.dumps(digest)
    assert len(serialized) < 4000

    mock_llm = _MockSteeredLLM(
        [
            (
                json.dumps(
                    {
                        "bottleneck_priority": ["n2_head", "n1_prep"],
                        "preferred_siblings": {"n2_head": ["synth_pkg.alt_linear_sibling"]},
                        "param_priors": {"n2_head": {"slope": 1.15}},
                        "synthesis_targets": [],
                        "synthesis_hypotheses": {},
                        "reasoning_summary": "Linear head cannot model quadratic curvature.",
                    }
                ),
                {"input_tokens": 620, "cache_read_tokens": 100, "output_tokens": 85},
            )
        ]
    )
    tracked = TrackedLLMClient(mock_llm)
    nav = LLMPrototypingNavigator(tracked)
    plan = await nav.propose_direction(session)

    assert plan.bottleneck_priority == ["n2_head", "n1_prep"]
    assert session.last_gradients[0].node_id == "n2_head"
    assert session.ladder.llm_param_priors["n2_head"]["slope"] == 1.15
    assert nav.usage_ledger.call_count == 1
    assert nav.usage_ledger.total_billed_tokens == 805
    assert nav.usage_ledger.total_fresh_tokens == 705


@pytest.mark.asyncio
async def test_llm_steered_rung4b_self_repair_and_subsequent_zero_llm_hpo() -> None:
    cdg, runner, catalog = _build_two_stage_synthetic_cdg()
    harness = _QuadraticGapHarness()

    nav_json = json.dumps(
        {
            "bottleneck_priority": ["n2_head", "n1_prep"],
            "preferred_siblings": {},
            "param_priors": {"n2_head": {"slope": 1.02}},
            "synthesis_targets": ["n2_head"],
            "synthesis_hypotheses": {
                "n2_head": "Include quadratic basis term h + quad_coef * h**2 with tunable quad_coef."
            },
            "reasoning_summary": "Fast linear head lacks quadratic term.",
        }
    )
    # First synthesis response intentionally has a shape bug (flattens to 1D) to exercise Tier-1 self-repair!
    broken_leaf = """```python
import numpy as np

def micro_n2_head(h: np.ndarray, quad_coef: float = 0.5) -> np.ndarray:
    return (h + quad_coef * (h ** 2)).ravel()
```"""
    # Repaired leaf preserves 2D shape AND exposes suboptimal default quad_coef=0.45 (true optimum is 0.80)
    # so subsequent Rung 1 Optuna HPO tunes quad_coef -> ~0.80 at 0 extra LLM tokens!
    repaired_leaf = """```python
import numpy as np

def micro_n2_head(h: np.ndarray, quad_coef: float = 0.45) -> np.ndarray:
    arr = np.asarray(h, dtype=np.float64)
    return arr + float(quad_coef) * (arr ** 2)
```"""

    mock_llm = _MockSteeredLLM(
        [
            (nav_json, {"input_tokens": 700, "output_tokens": 90}),
            (broken_leaf, {"input_tokens": 850, "output_tokens": 75}),
            (repaired_leaf, {"input_tokens": 500, "output_tokens": 80}),
        ]
    )

    session = PrototypingSession(
        cdg,
        harness=harness,
        runner=runner,
        catalog=catalog,
        hpo_trials_per_rung=10,
        llm_client=mock_llm,
    )
    summary = await session.run(max_iterations=6, exhausted_patience=2, time_budget_s=20.0)

    # Initial linear loss is ~1.1; repaired Rung 4b leaf drops it to ~0.5, and Rung 1 HPO tunes quad_coef -> ~0.8 (< 0.12)!
    assert summary.best_loss < 0.15
    assert summary.holdout_result is not None and summary.holdout_result.passed
    assert summary.holdout_result.loss < 0.15

    rungs_accepted = [r.rung for r in summary.history if r.improved and r.iteration > 0]
    assert "rung4b_micro_atom_synthesis" in rungs_accepted
    # Verify Rung 1 HPO tuned the newly synthesized leaf's `quad_coef` parameter AFTER Rung 4b with 0 extra LLM calls!
    r4b_iter = next(r.iteration for r in summary.history if r.rung == "rung4b_micro_atom_synthesis" and r.improved)
    post_r4b_hpo = [
        r for r in summary.history if r.iteration > r4b_iter and r.rung == "rung1_node_hpo" and r.improved
    ]
    assert len(post_r4b_hpo) >= 1
    assert all(r.llm_calls == 0 for r in post_r4b_hpo)
    assert "quad_coef" in summary.best_node_params.get("n2_head", {})

    # Exactly 3 bounded LLM calls total (1 navigate + 1 synthesize + 1 repair = 2,295 tokens)
    assert summary.llm_usage["call_count"] == 3
    assert summary.llm_usage["total_billed_tokens"] == 2295
