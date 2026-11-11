"""Phase C tests: Zero-Subprocess LiveCDGRunner, IntermediateCache, ProblemHarness, and 4-Tier Funnel.

All fixtures in this test module are 100% synthetic mathematical constructs.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any

import icontract
import numpy as np
import pandas as pd
import pytest
from langchain_core.runnables import RunnableConfig

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    EdgeKind,
    IOSpec,
    NodeStatus,
    StatePortSpec,
)
from sciona.principal.graph import evaluate_run, execute_forward, suggest_params
from sciona.principal.graph_types import PrincipalDeps, PrincipalState
from sciona.principal.hpo import OptunaManager
from sciona.principal.models import OptimizationMetric
from sciona.principal.proposal_helpers import (
    build_expansion_context,
    evaluate_proposal_candidate,
)
from sciona.prototyping import (
    EvalCase,
    EvaluationFunnel,
    FunnelConfig,
    IntermediateCache,
    LiveCDGRunner,
    LivePrototypingSandbox,
    MetamorphicRelation,
    ProblemHarness,
    ResidentSliceHarness,
    SyntheticSuiteHarness,
    hash_runtime_value,
)
from sciona.types import CandidateMatch, Declaration, MatchResult, PDGNode, Prover, VerificationResult
from sciona.uncertainty.conformal import ConformalEnvelope
from sciona.uncertainty.distributions import GaussianMixtureRegime
from sciona.uncertainty.models import (
    AtomUncertaintyProfile,
    ComponentChannel,
    PortRegimeSpec,
)
from sciona.uncertainty.operators import LowRankPlusTailOperator
from sciona.uncertainty.projectors import IdentityProjector
from sciona.uncertainty.propagation import RiskBudget
from sciona.uncertainty.runtime_guard import ConfidenceGuardConfig


def _make_node(
    node_id: str,
    primitive: str,
    inputs: list[IOSpec],
    outputs: list[IOSpec],
    *,
    concept_type: ConceptType = ConceptType.SIGNAL_FILTER,
    state_ports: list[StatePortSpec] | None = None,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=node_id,
        description=f"Synthetic stage {node_id}",
        concept_type=concept_type,
        inputs=inputs,
        outputs=outputs,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        state_ports=state_ports or [],
    )


def _edge(
    source_id: str,
    target_id: str,
    output_name: str,
    input_name: str,
    *,
    source_type: str = "np.ndarray",
    target_type: str = "np.ndarray",
    kind: EdgeKind = EdgeKind.DATA_FLOW,
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=source_type,
        target_type=target_type,
        edge_kind=kind,
    )


# ---------------------------------------------------------------------------
# 1. Content-Addressed IntermediateCache & Deterministic Hashing
# ---------------------------------------------------------------------------


def test_hash_runtime_value_determinism_and_lru_eviction() -> None:
    base_2d = np.arange(48, dtype=np.float64).reshape(4, 12)
    arr_sliced = base_2d[:, ::2]
    arr_contig = arr_sliced.copy()
    assert not arr_sliced.flags["C_CONTIGUOUS"]
    assert np.array_equal(arr_contig, arr_sliced)
    assert hash_runtime_value(arr_contig) == hash_runtime_value(arr_sliced)

    nan_a = np.array([1.0, np.nan, -0.0], dtype=np.float64)
    nan_b = np.array([1.0, float("nan"), 0.0], dtype=np.float64)
    assert hash_runtime_value(nan_a[:2]) == hash_runtime_value(nan_b[:2])
    assert hash_runtime_value(-0.0) == hash_runtime_value(0.0)
    assert hash_runtime_value((1, 2)) != hash_runtime_value([1, 2])

    df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    df3 = pd.DataFrame({"a": [1.0, 2.5], "b": [3, 4]})
    assert hash_runtime_value(df1) == hash_runtime_value(df2)
    assert hash_runtime_value(df1) != hash_runtime_value(df3)

    @dataclass
    class StateObj:
        mu: float
        scale: float

    assert hash_runtime_value(StateObj(1.0, 2.0)) == hash_runtime_value(StateObj(1.0, 2.0))
    assert hash_runtime_value(StateObj(1.0, 2.0)) != hash_runtime_value(StateObj(1.0, 3.0))

    cache = IntermediateCache(max_entries=2, max_bytes=1024 * 1024)
    cache.put("k1", atom_fqn="pkg.a", outputs={"y": np.ones(4)}, compute_time_ms=12.5)
    cache.put("k2", atom_fqn="pkg.b", outputs={"y": np.zeros(4)}, compute_time_ms=5.0)
    assert cache.get("k1") is not None  # promotes k1 to MRU
    cache.put("k3", atom_fqn="pkg.c", outputs={"y": np.full(4, 2.0)}, compute_time_ms=3.0)
    assert cache.get("k2") is None  # k2 was LRU and evicted
    assert cache.get("k1") is not None
    assert cache.evictions == 1
    assert cache.saved_time_ms >= 25.0


# ---------------------------------------------------------------------------
# 2. Prefix Memoization on HPO Sweeps, Sibling Swaps & Counterfactual Overrides
# ---------------------------------------------------------------------------


def test_prefix_memoization_across_hpo_sweep_and_sibling_swap() -> None:
    counts = {"a": 0, "b1": 0, "b2": 0, "c": 0}

    def atom_a(x: np.ndarray) -> np.ndarray:
        counts["a"] += 1
        return x * 2.0 + 1.0

    def atom_b1(h: np.ndarray, gain: float = 1.0) -> np.ndarray:
        counts["b1"] += 1
        return np.tanh(h * gain)

    def atom_b2(h: np.ndarray, gain: float = 1.0) -> np.ndarray:
        counts["b2"] += 1
        return h * gain

    def atom_c(z: np.ndarray) -> float:
        counts["c"] += 1
        return float(np.mean(z**2))

    cdg_1 = CDGExport(
        nodes=[
            _make_node("n_a", "synth.atom_a", [IOSpec(name="x", type_desc="np.ndarray")], [IOSpec(name="h", type_desc="np.ndarray")]),
            _make_node(
                "n_b",
                "synth.atom_b1",
                [IOSpec(name="h", type_desc="np.ndarray"), IOSpec(name="gain", type_desc="float", required=False, default_value_repr="1.0")],
                [IOSpec(name="z", type_desc="np.ndarray")],
            ),
            _make_node("n_c", "synth.atom_c", [IOSpec(name="z", type_desc="np.ndarray")], [IOSpec(name="energy", type_desc="float")]),
        ],
        edges=[
            _edge("n_a", "n_b", "h", "h"),
            _edge("n_b", "n_c", "z", "z"),
        ],
    )

    cache = IntermediateCache()
    runner = LiveCDGRunner(
        cache=cache,
        extra_callables={
            "synth.atom_a": atom_a,
            "synth.atom_b1": atom_b1,
            "synth.atom_b2": atom_b2,
            "synth.atom_c": atom_c,
        },
    )
    x_in = np.linspace(-1.0, 1.0, 64)

    # 10-trial HPO sweep over downstream node n_b's `gain` parameter:
    # upstream node n_a MUST execute only once across all 10 trials!
    for gain_val in np.linspace(0.2, 2.0, 10):
        res = runner.execute(cdg_1, {"x": x_in}, node_params={"n_b": {"gain": float(gain_val)}})
        assert res.ok
        assert "energy" in res.outputs

    assert counts["a"] == 1
    assert counts["b1"] == 10
    assert counts["c"] == 10
    assert cache.hits == 9

    # Repeat exact last parameter assignment -> all 3 nodes hit cache (0 executions)
    res_repeat = runner.execute(cdg_1, {"x": x_in}, node_params={"n_b": {"gain": 2.0}})
    assert res_repeat.ok
    assert res_repeat.cache_hits == ["n_a", "n_b", "n_c"]
    assert counts["a"] == 1
    assert counts["b1"] == 10
    assert counts["c"] == 10

    # Sibling swap at n_b (synth.atom_b1 -> synth.atom_b2): n_a still hits cache!
    cdg_2 = cdg_1.model_copy(deep=True)
    cdg_2.nodes[1].matched_primitive = "synth.atom_b2"
    res_swap = runner.execute(cdg_2, {"x": x_in}, node_params={"n_b": {"gain": 2.0}})
    assert res_swap.ok
    assert counts["a"] == 1  # upstream still memoized!
    assert counts["b2"] == 1
    assert counts["c"] == 11


def test_execute_from_override_runs_strict_downstream_cone_only() -> None:
    executed: list[str] = []

    def fn_root(x: np.ndarray) -> np.ndarray:
        executed.append("root")
        return x + 1.0

    def fn_left(u: np.ndarray) -> np.ndarray:
        executed.append("left")
        return u * 3.0

    def fn_right(u: np.ndarray) -> np.ndarray:
        executed.append("right")
        return u - 2.0

    def fn_sink(l_val: np.ndarray, r_val: np.ndarray) -> float:
        executed.append("sink")
        return float(np.mean(l_val + r_val))

    cdg = CDGExport(
        nodes=[
            _make_node("root", "p.root", [IOSpec(name="x", type_desc="np.ndarray")], [IOSpec(name="u", type_desc="np.ndarray")]),
            _make_node("left", "p.left", [IOSpec(name="u", type_desc="np.ndarray")], [IOSpec(name="l_val", type_desc="np.ndarray")]),
            _make_node("right", "p.right", [IOSpec(name="u", type_desc="np.ndarray")], [IOSpec(name="r_val", type_desc="np.ndarray")]),
            _make_node(
                "sink",
                "p.sink",
                [IOSpec(name="l_val", type_desc="np.ndarray"), IOSpec(name="r_val", type_desc="np.ndarray")],
                [IOSpec(name="score", type_desc="float")],
            ),
        ],
        edges=[
            _edge("root", "left", "u", "u"),
            _edge("root", "right", "u", "u"),
            _edge("left", "sink", "l_val", "l_val"),
            _edge("right", "sink", "r_val", "r_val"),
        ],
    )
    runner = LiveCDGRunner(
        extra_callables={
            "p.root": fn_root,
            "p.left": fn_left,
            "p.right": fn_right,
            "p.sink": fn_sink,
        }
    )
    x = np.array([1.0, 2.0, 3.0])
    base = runner.execute(cdg, {"x": x})
    assert base.ok
    assert executed == ["root", "left", "right", "sink"]

    # Override `left` output -> ONLY `sink` should execute (neither `root` nor `left` nor parallel `right`!)
    executed.clear()
    cf = runner.execute_from_override(
        cdg,
        {"x": x},
        baseline_node_outputs=base.node_outputs,
        override_node_outputs={"left": {"l_val": np.zeros_like(x)}},
        use_cache=False,
    )
    assert cf.ok
    assert executed == ["sink"]
    assert np.isclose(cf.outputs["score"], float(np.mean(0.0 + ((x + 1.0) - 2.0))))


# ---------------------------------------------------------------------------
# 3. Advanced Semantics: Multi-Output, Callable Injection, State Ports,
#    Fixed-Point Cycles, @icontract Blame, and Runtime Confidence Guards
# ---------------------------------------------------------------------------


def test_stateful_fit_predict_multi_output_callable_injection_and_contracts() -> None:
    def scaler_fit_or_transform(
        x: np.ndarray,
        scaler_state: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if scaler_state is None:
            st = {"mean": float(np.mean(x)), "std": float(np.std(x) + 1e-8)}
        else:
            st = scaler_state
        normed = (x - st["mean"]) / st["std"]
        return normed, st

    def kernel_fn(v: np.ndarray) -> np.ndarray:
        return v**2

    @icontract.require(lambda scale: scale > 0.0, "scale must be strictly positive")
    def apply_injected(
        normed: np.ndarray,
        op: Any,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        transformed = op(normed) * scale
        return transformed, float(np.mean(transformed))

    cdg = CDGExport(
        nodes=[
            _make_node(
                "scaler",
                "s.scaler",
                [IOSpec(name="x", type_desc="np.ndarray")],
                [IOSpec(name="normed", type_desc="np.ndarray")],
                state_ports=[
                    StatePortSpec(port_name="scaler_state", direction="update", required=False),
                ],
            ),
            _make_node(
                "kernel",
                "s.kernel",
                [IOSpec(name="v", type_desc="np.ndarray", required=False)],
                [IOSpec(name="k_out", type_desc="np.ndarray")],
            ),
            _make_node(
                "head",
                "s.head",
                [
                    IOSpec(name="normed", type_desc="np.ndarray"),
                    IOSpec(name="op", type_desc="Callable"),
                    IOSpec(name="scale", type_desc="float", required=False, default_value_repr="1.0"),
                ],
                [
                    IOSpec(name="features", type_desc="np.ndarray"),
                    IOSpec(name="summary", type_desc="float"),
                ],
            ),
        ],
        edges=[
            _edge("scaler", "head", "normed", "normed"),
            _edge("kernel", "head", "k_out", "op", kind=EdgeKind.CALLABLE_INJECTION),
        ],
    )

    runner = LiveCDGRunner(
        extra_callables={
            "s.scaler": scaler_fit_or_transform,
            "s.kernel": kernel_fn,
            "s.head": apply_injected,
        }
    )

    # Fit run populates state_store['scaler_state']
    fit_x = np.array([10.0, 20.0, 30.0])
    fit_res = runner.execute(cdg, {"x": fit_x})
    assert fit_res.ok
    assert "scaler_state" in fit_res.state_store
    assert np.isclose(fit_res.state_store["scaler_state"]["mean"], 20.0)

    # Predict run threads fitted scaler_state without refitting on test input
    test_x = np.array([20.0, 20.0, 20.0])
    pred_res = runner.execute(cdg, {"x": test_x}, initial_state=fit_res.state_store)
    assert pred_res.ok
    assert np.allclose(pred_res.outputs["features"], np.zeros(3))
    assert np.isclose(pred_res.outputs["summary"], 0.0)

    # @icontract violation pinpoints exact failed_node_id ('head')
    bad_res = runner.execute(cdg, {"x": test_x}, node_params={"head": {"scale": -2.0}})
    assert not bad_res.ok
    assert bad_res.failed_node_id == "head"
    assert "ViolationError" in bad_res.exception_type


def test_fixed_point_cycle_and_dotted_import_resolution() -> None:
    # Contractive fixed point x_{t+1} = 0.5 * (x_t + target / x_t) (Babylonian sqrt)
    def init_guess(target: float) -> float:
        return max(1.0, target * 0.5)

    def newton_step(est: float = 2.0, target: float = 9.0) -> float:
        return 0.5 * (est + target / est)

    cdg = CDGExport(
        nodes=[
            _make_node("seed", "fp.init", [IOSpec(name="target", type_desc="float")], [IOSpec(name="est0", type_desc="float")]),
            _make_node(
                "iter",
                "fp.step",
                [IOSpec(name="est", type_desc="float"), IOSpec(name="target", type_desc="float")],
                [IOSpec(name="est", type_desc="float")],
                concept_type=ConceptType.FIXED_POINT,
            ),
            # Dotted stdlib callable resolved directly via importlib (no extra_callables entry!)
            _make_node("post", "math.fabs", [IOSpec(name="x", type_desc="float")], [IOSpec(name="root", type_desc="float")]),
        ],
        edges=[
            _edge("seed", "iter", "est0", "est", source_type="float", target_type="float"),
            _edge("iter", "iter", "est", "est", source_type="float", target_type="float"),
            _edge("iter", "post", "est", "x", source_type="float", target_type="float"),
        ],
    )
    runner = LiveCDGRunner(
        extra_callables={"fp.init": init_guess, "fp.step": newton_step},
        fixed_point_tol=1e-9,
    )
    res = runner.execute(cdg, {"target": 16.0})
    assert res.ok
    assert np.isclose(res.outputs["root"], 4.0, atol=1e-6)


def test_live_runner_phase4_runtime_confidence_guard_integration() -> None:
    cal_scores = np.linspace(0.05, 1.5, 200)
    scalar_mix = GaussianMixtureRegime.from_full_covariances(
        weights=[1.0],
        means=[[0.0]],
        covariances=[[[1.0]]],
    )
    profile = AtomUncertaintyProfile(
        atom_name="guarded.atom",
        mode="calibrated",
        trust_tier=1,
        n_trials=200,
        input_regimes={
            "x": PortRegimeSpec(
                port_name="x",
                projector_spec=IdentityProjector(dim=1).to_spec(),
                mixture=scalar_mix,
                conformal_envelope=ConformalEnvelope(cal_scores),
            )
        },
    )
    cdg = CDGExport(
        nodes=[
            _make_node("g1", "guarded.atom", [IOSpec(name="x", type_desc="float")], [IOSpec(name="y", type_desc="float")]),
        ],
        edges=[],
    )
    runner = LiveCDGRunner(
        extra_callables={"guarded.atom": lambda x: float(x) * 2.0},
        confidence_guard=ConfidenceGuardConfig(enabled=True, trust_tier=1, global_alpha=0.05),
        uncertainty_profiles={"guarded.atom": profile},
    )

    ok_run = runner.execute(cdg, {"x": 0.25})
    assert ok_run.ok
    assert ok_run.confidence_guards["g1"]["in_confidence_bounds"] is True

    ood_run = runner.execute(cdg, {"x": 50.0})
    assert not ood_run.ok
    assert ood_run.failed_node_id == "g1"
    assert "ConfidenceEnvelopeViolation" in ood_run.exception_type


# ---------------------------------------------------------------------------
# 4. SyntheticSuiteHarness, MetamorphicRelation & 4-Tier EvaluationFunnel
# ---------------------------------------------------------------------------


def test_synthetic_suite_harness_and_4_tier_funnel_pruning() -> None:
    call_tracker = {"broken_type": 0, "fragile_nan": 0, "solid": 0}

    # Candidate A: Static PortTypeSpec mismatch (1D array -> pd.DataFrame) -> Pruned at Tier 0 in <10ms!
    def emit_vec(signal: np.ndarray) -> np.ndarray:
        call_tracker["broken_type"] += 1
        return signal

    def consume_df(table: pd.DataFrame) -> float:
        call_tracker["broken_type"] += 1
        return float(table.to_numpy().mean())

    cdg_bad_types = CDGExport(
        nodes=[
            _make_node("u1", "cand.emit_vec", [IOSpec(name="signal", type_desc="np.ndarray", constraints="1D")], [IOSpec(name="vec", type_desc="np.ndarray", constraints="1D")]),
            _make_node("u2", "cand.consume_df", [IOSpec(name="table", type_desc="pd.DataFrame")], [IOSpec(name="stat", type_desc="float")]),
        ],
        edges=[_edge("u1", "u2", "vec", "table", source_type="np.ndarray", target_type="pd.DataFrame")],
    )

    # Candidate B: Valid types, works on nominal inputs, but produces NaN on outlier spikes (>5.0) -> Pruned at Tier 1!
    def fragile_std(signal: np.ndarray) -> float:
        call_tracker["fragile_nan"] += 1
        if float(np.max(np.abs(signal))) > 5.0:
            return float("nan")
        return float(np.std(signal))

    cdg_fragile = CDGExport(
        nodes=[
            _make_node("f1", "cand.fragile_std", [IOSpec(name="signal", type_desc="np.ndarray", constraints="1D signal")], [IOSpec(name="spread", type_desc="float")]),
        ],
        edges=[],
    )

    # Candidate C: Shift-invariant demeaned RMS estimator -> Passes Tier 0, Tier 1 (including shift_invariance!), Tier 2, Tier 3!
    def solid_std(signal: np.ndarray, scale: float = 1.0) -> float:
        call_tracker["solid"] += 1
        centered = signal - np.mean(signal)
        return float(np.sqrt(np.mean(centered**2)) * scale)

    cdg_solid = CDGExport(
        nodes=[
            _make_node(
                "s1",
                "cand.solid_std",
                [
                    IOSpec(name="signal", type_desc="np.ndarray", constraints="1D signal"),
                    IOSpec(name="scale", type_desc="float", required=False, default_value_repr="1.0"),
                ],
                [IOSpec(name="spread", type_desc="float")],
            ),
        ],
        edges=[],
    )

    rng = np.random.default_rng(7)
    val_cases = [
        EvalCase(
            case_id=f"val_{i}",
            inputs={"signal": rng.normal(loc=3.0 * i, scale=2.0, size=128)},
            expected=2.0,
            tier=2,
        )
        for i in range(6)
    ]
    harness = ResidentSliceHarness(
        val_cases,
        root_inputs=[IOSpec(name="signal", type_desc="np.ndarray", constraints="1D signal")],
        root_outputs=[IOSpec(name="spread", type_desc="float")],
        metamorphic_relations=[MetamorphicRelation.shift_invariance("signal", shift=10.0)],
        metric="rmse",
    )
    assert isinstance(harness, ProblemHarness)

    runner = LiveCDGRunner(
        extra_callables={
            "cand.emit_vec": emit_vec,
            "cand.consume_df": consume_df,
            "cand.fragile_std": fragile_std,
            "cand.solid_std": solid_std,
        }
    )
    funnel = EvaluationFunnel(harness, runner=runner)

    # Tier 0 prune: zero python calls executed!
    rep_a = funnel.evaluate_candidate(cdg_bad_types)
    assert not rep_a.passed
    assert rep_a.pruned_at_tier == 0
    assert rep_a.failed_node_id == "u2"
    assert call_tracker["broken_type"] == 0

    # Tier 1 prune: caught NaN on synthetic noisy_outliers regime before Tier 2!
    rep_b = funnel.evaluate_candidate(cdg_fragile)
    assert not rep_b.passed
    assert rep_b.pruned_at_tier == 1
    assert "nan" in rep_b.prune_reason.lower()

    # Full pass through Tier 0 -> Tier 1 -> Tier 2 -> Tier 3
    rep_c = funnel.evaluate_candidate(cdg_solid)
    assert rep_c.passed
    assert rep_c.pruned_at_tier is None
    assert rep_c.global_loss < 0.35

    # Verify Principal ExpansionContext compatibility with live BenchmarkResult
    exp_ctx = build_expansion_context(PrincipalState(benchmark=rep_c.benchmark))
    assert exp_ctx.intermediates is not None
    assert "s1" in exp_ctx.intermediates
    assert exp_ctx.runtime_inputs is not None
    assert "signal" in exp_ctx.runtime_inputs

    # Tier 3 holdout verification & overfitting detection
    t3_ok = funnel.evaluate_tier3_holdout(cdg_solid, validation_loss=rep_c.global_loss)
    assert t3_ok.passed
    assert t3_ok.details["overfit_detected"] is False

    t3_overfit = funnel.evaluate_tier3_holdout(
        cdg_solid,
        validation_loss=0.0,
        node_params={"s1": {"scale": 5.0}},  # deliberately poor generalization vs validation_loss=0
    )
    assert not t3_overfit.passed
    assert t3_overfit.details["overfit_detected"] is True


def test_tier0_analytic_risk_budget_gate_prunes_excess_variance() -> None:
    proj_1d = IdentityProjector(dim=1).to_spec()
    in_mix = GaussianMixtureRegime.from_full_covariances(
        weights=[1.0],
        means=[[0.0]],
        covariances=[[[0.25]]],
    )
    high_noise_profile = AtomUncertaintyProfile(
        atom_name="noisy.op",
        mode="calibrated",
        trust_tier=1,
        input_regimes={
            "x": PortRegimeSpec(port_name="x", projector_spec=proj_1d, mixture=in_mix),
        },
        output_projectors={"y": proj_1d},
        components=[
            ComponentChannel(
                component_index=0,
                operators={
                    ("x", "y"): LowRankPlusTailOperator(
                        dim_out=1,
                        dim_in=1,
                        singular_values=[4.0],
                        left_vectors=[[1.0]],
                        right_vectors=[[1.0]],
                    )
                },
                intrinsic_output_cov={"y": np.array([[0.5]])},
            )
        ],
    )
    cdg = CDGExport(
        nodes=[_make_node("n1", "noisy.op", [IOSpec(name="x", type_desc="float")], [IOSpec(name="y", type_desc="float")])],
        edges=[],
    )
    harness = SyntheticSuiteHarness.from_cdg(
        cdg,
        risk_budget=RiskBudget(max_output_variance=0.1),  # strict variance ceiling
    )
    called = {"count": 0}
    runner = LiveCDGRunner(
        extra_callables={"noisy.op": lambda x: (called.__setitem__("count", called["count"] + 1) or float(x))},
        uncertainty_profiles={"noisy.op": high_noise_profile},
    )
    funnel = EvaluationFunnel(harness, runner=runner)
    rep = funnel.evaluate_candidate(cdg)
    assert not rep.passed
    assert rep.pruned_at_tier == 0
    assert "RiskBudget" in rep.prune_reason
    assert called["count"] == 0  # Pruned analytically at Tier 0 without executing!


# ---------------------------------------------------------------------------
# 5. End-to-End Principal Integration via LivePrototypingSandbox (Zero Subprocesses)
# ---------------------------------------------------------------------------


def test_principal_live_prototyping_sandbox_hpo_zero_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Guarantee ZERO OS subprocess spawns occur anywhere in Principal forward/eval/proposal paths
    def _forbid_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("OS subprocess spawn is forbidden in LivePrototypingSandbox!")

    monkeypatch.setattr(subprocess, "Popen", _forbid_subprocess)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _forbid_subprocess)

    upstream_calls = {"pre": 0, "gain_stage": 0}

    def pre_filter(x: np.ndarray) -> np.ndarray:
        upstream_calls["pre"] += 1
        return x - np.mean(x)

    def gain_stage(centered: np.ndarray, gain: float = 1.0) -> np.ndarray:
        upstream_calls["gain_stage"] += 1
        return centered * gain

    cdg = CDGExport(
        nodes=[
            _make_node("pre", "live.pre", [IOSpec(name="x", type_desc="np.ndarray")], [IOSpec(name="centered", type_desc="np.ndarray")]),
            _make_node(
                "amp",
                "live.amp",
                [
                    IOSpec(name="centered", type_desc="np.ndarray"),
                    IOSpec(name="gain", type_desc="float", required=False, default_value_repr="1.0"),
                ],
                [IOSpec(name="pred", type_desc="np.ndarray")],
            ),
        ],
        edges=[_edge("pre", "amp", "centered", "centered")],
    )

    t = np.linspace(0.0, 2.0 * np.pi, 128)
    raw_sig = 3.0 + np.sin(t)
    true_target = 2.5 * np.sin(t)  # optimal gain == 2.5

    harness = ResidentSliceHarness(
        [EvalCase(case_id="target_wave", inputs={"x": raw_sig}, expected=true_target, tier=2)],
        root_inputs=[IOSpec(name="x", type_desc="np.ndarray")],
        root_outputs=[IOSpec(name="pred", type_desc="np.ndarray")],
        metric="rmse",
    )
    runner = LiveCDGRunner(extra_callables={"live.pre": pre_filter, "live.amp": gain_stage})
    funnel = EvaluationFunnel(
        harness,
        runner=runner,
        config=FunnelConfig(enable_tier1_synthetic=False),  # focus directly on HPO loop caching
    )
    live_sandbox = LivePrototypingSandbox(funnel)

    class _StubCatalog:
        def get(self, name: str) -> Any:
            if name == "live.amp":
                from sciona.architect.models import AlgorithmicPrimitive, PrimitiveParamSpec

                return AlgorithmicPrimitive(
                    name="live.amp",
                    source="synthetic",
                    category=ConceptType.SIGNAL_FILTER,
                    description="Gain stage",
                    tunable_params=[
                        PrimitiveParamSpec(name="gain", kind="float", default=1.0, min_value=0.5, max_value=4.0),
                    ],
                )
            return None

    hpo = OptunaManager()
    deps = PrincipalDeps(
        architect=None,  # type: ignore[arg-type]
        sandbox=live_sandbox,
        synthesize_fn=live_sandbox.synthesize_virtual_bundle,
        problem_harness=harness,
        catalog=_StubCatalog(),  # type: ignore[arg-type]
        hpo_manager=hpo,
    )
    state = PrincipalState(
        goal="Recover scaled zero-mean sinusoid",
        metric=OptimizationMetric.PRECISION,
        dataset_path="resident://in_memory",
        cdg=cdg,
    )
    cfg: RunnableConfig = {"configurable": {"deps": deps}}

    async def _run_trials() -> list[float]:
        losses: list[float] = []
        for _ in range(8):
            await suggest_params(state, cfg)
            await execute_forward(state, cfg)
            await evaluate_run(state, cfg)
            assert state.benchmark is not None
            losses.append(state.benchmark.global_loss)
            state.trial_history.append({"loss": state.benchmark.global_loss})

        # Also verify evaluate_proposal_candidate works identically with zero subprocesses
        prop_loss, prop_bundle, prop_bm, _, _ = await evaluate_proposal_candidate(state, deps, cdg)
        assert prop_bundle is not None
        assert prop_bm is not None
        assert np.isfinite(prop_loss)
        return losses

    losses = asyncio.run(_run_trials())
    assert len(losses) == 8
    assert min(losses) < losses[0] or min(losses) < 0.5
    # Upstream `pre_filter` ran ONLY ONCE across all 8 HPO trials + 1 proposal eval!
    assert upstream_calls["pre"] == 1
    assert upstream_calls["gain_stage"] >= 8


def test_resident_slice_dataset_loader_and_match_result_resolution(
    tmp_path: Any,
) -> None:
    npz_path = tmp_path / "synthetic_slice.npz"
    x_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    expected_cumsum = np.cumsum(x_vec)
    np.savez(npz_path, a=x_vec, expected=expected_cumsum)

    harness = ResidentSliceHarness.from_dataset_file(
        npz_path,
        root_inputs=[IOSpec(name="a", type_desc="np.ndarray")],
        root_outputs=[IOSpec(name="out", type_desc="np.ndarray")],
        metric="rmse",
    )
    cdg = CDGExport(
        nodes=[
            _make_node(
                "cs",
                "unresolved_short_name",
                [IOSpec(name="a", type_desc="np.ndarray")],
                [IOSpec(name="out", type_desc="np.ndarray")],
            )
        ],
        edges=[],
    )
    decl = Declaration(
        name="cumsum",
        type_signature="(a: np.ndarray) -> np.ndarray",
        prover=Prover.PYTHON,
        source_lib="numpy",
    )
    cand = CandidateMatch(declaration=decl, score=1.0, retrieval_method="test")
    vr = VerificationResult(
        candidate=cand,
        verified=True,
        proof_term="@numpy.cumsum",
    )
    mr = MatchResult(
        pdg_node=PDGNode(predicate_id="cs", statement="(a: np.ndarray) -> np.ndarray", prover=Prover.PYTHON),
        verified_match=vr,
        all_candidates=[cand],
        all_verifications=[vr],
    )

    funnel = EvaluationFunnel(
        harness,
        config=FunnelConfig(enable_tier1_synthetic=False),
    )
    rep = funnel.evaluate_candidate(cdg, match_results=[mr])
    assert rep.passed
    assert np.isclose(rep.global_loss, 0.0, atol=1e-12)

