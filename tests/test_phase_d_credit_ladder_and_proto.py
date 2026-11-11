"""Phase D verification: UniversalCreditAssigner, 5-Rung RevisionLadder, PrototypingSession, SDK & CLI.

Uses 100% synthetic mathematical constructs (strictly no real dataset paths, records, or schemas).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import icontract
import numpy as np
import pytest

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.compiled_catalog import compile_local_manifest_sqlite
from sciona.sources import load_sources
from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    PortTypeSpec,
)
from sciona.commands.proto_cmds import _cmd_proto
from sciona.principal.backprop import CreditAssigner
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
)
from sciona.principal.graph import compute_gradients
from sciona.principal.graph_types import PrincipalDeps, PrincipalState
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.prototyping import (
    EvalCase,
    EvaluationFunnel,
    LiveCDGRunner,
    LivePrototypingSandbox,
    MetamorphicRelation,
    MicroAtomSynthesisSpec,
    PrototypingSession,
    ResidentSliceHarness,
    RevisionLadder,
    SubgraphSpliceRequest,
    SyntheticSuiteHarness,
    UniversalCreditAssigner,
)
from sciona.sdk import Sciona
from sciona.synthesizer.ghost_sim import GhostSimReport
from sciona.synthesizer.models import ExportBundle
from sciona.uncertainty.distributions import GaussianMixtureRegime
from sciona.uncertainty.models import (
    AtomUncertaintyProfile,
    ComponentChannel,
    PortRegimeSpec,
)
from sciona.uncertainty.operators import LowRankPlusTailOperator
from sciona.uncertainty.projectors import IdentityProjector
from sciona.uncertainty.propagation import RiskBudget


# ---------------------------------------------------------------------------
# Importable top-level synthetic helper functions (for export_python_module too)
# ---------------------------------------------------------------------------


def synth_center(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(x, dtype=np.float64) - float(np.mean(x))


def synth_scale_tunable(x: np.ndarray[Any, Any], gain: float = 1.0) -> np.ndarray[Any, Any]:
    return np.asarray(x, dtype=np.float64) * float(gain)


def synth_passthrough(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(x, dtype=np.float64)


def synth_biased_smoother(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    # Intentionally adds large constant offset + distortion
    return 0.35 * np.asarray(x, dtype=np.float64) + 4.5


def synth_accurate_smoother(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 3:
        return arr.copy()
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    out = np.convolve(arr, kernel, mode="same")
    out[0] = arr[0]
    out[-1] = arr[-1]
    return out


def synth_outlier_clip(x: np.ndarray[Any, Any], limit: float = 3.0) -> np.ndarray[Any, Any]:
    return np.clip(np.asarray(x, dtype=np.float64), -float(limit), float(limit))


def synth_linear_predict(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return 2.0 * np.asarray(x, dtype=np.float64)


def _vec_port(name: str, *, required: bool = True, default_repr: str = "") -> IOSpec:
    return IOSpec(
        name=name,
        type_desc="np.ndarray",
        required=required,
        default_value_repr=default_repr,
        port_type=PortTypeSpec(
            raw="np.ndarray",
            kind="array",
            dtype="float",
            ndim=1,
        ),
    )


def _build_3stage_cdg(
    *,
    mid_prim: str,
    post_prim: str = f"{__name__}.synth_passthrough",
) -> CDGExport:
    n_pre = AlgorithmicNode(
        node_id="pre",
        name="Center Vector",
        description="Zero-mean center input vector",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{__name__}.synth_center",
        inputs=[_vec_port("x")],
        outputs=[_vec_port("x")],
    )
    n_mid = AlgorithmicNode(
        node_id="mid",
        name="Scale Or Filter Stage",
        description="Core transform stage",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive=mid_prim,
        inputs=[_vec_port("x")],
        outputs=[_vec_port("x")],
    )
    n_post = AlgorithmicNode(
        node_id="post",
        name="Readout Stage",
        description="Final readout stage",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=post_prim,
        inputs=[_vec_port("x")],
        outputs=[_vec_port("y")],
    )
    edges = [
        DependencyEdge(
            source_id="pre",
            target_id="mid",
            output_name="x",
            input_name="x",
            source_type="np.ndarray",
            target_type="np.ndarray",
        ),
        DependencyEdge(
            source_id="mid",
            target_id="post",
            output_name="x",
            input_name="x",
            source_type="np.ndarray",
            target_type="np.ndarray",
        ),
    ]
    return CDGExport(nodes=[n_pre, n_mid, n_post], edges=edges)


# ---------------------------------------------------------------------------
# 1. UniversalCreditAssigner Tests
# ---------------------------------------------------------------------------


def test_universal_credit_contract_blame_and_counterfactual_o_downstream() -> None:
    """Direct contract failures get 100% blame; counterfactuals execute ONLY downstream nodes."""
    call_counts = {"pre": 0, "mid": 0, "post": 0}

    def counted_pre(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        call_counts["pre"] += 1
        return np.asarray(x - np.mean(x), dtype=np.float64)

    def counted_mid(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        call_counts["mid"] += 1
        # Saturated + biased bottleneck: upstream perturbations on 'pre' are crushed by
        # 0.01 * sech^2(x), whereas perturbing 'mid' output directly hits 10x 'post'!
        return np.asarray(0.01 * np.tanh(x) + 4.0, dtype=np.float64)

    def counted_post(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        call_counts["post"] += 1
        return np.asarray(10.0 * x, dtype=np.float64)

    cdg = _build_3stage_cdg(mid_prim="test.counted_mid", post_prim="test.counted_post")
    cdg.nodes[0].matched_primitive = "test.counted_pre"

    runner = LiveCDGRunner(
        extra_callables={
            "test.counted_pre": counted_pre,
            "test.counted_mid": counted_mid,
            "test.counted_post": counted_post,
        }
    )

    t = np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float64)
    x_val = np.sin(t)
    y_target = 2.5 * (x_val - np.mean(x_val))
    harness = ResidentSliceHarness(
        [EvalCase(case_id="c0", inputs={"x": x_val}, expected=y_target)],
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        holdout_fraction=0.0,
    )

    assigner = UniversalCreditAssigner(epsilon=0.1, max_cases=1)
    sens = assigner.compute_counterfactual_sensitivities(cdg, harness, runner)
    assert set(sens.keys()) == {"pre", "mid", "post"}

    # Perturbing 'mid' should execute ONLY 'post' (1 downstream node) and NEVER re-run 'pre'!
    assert sens["mid"].downstream_nodes_executed == 1
    assert sens["post"].downstream_nodes_executed == 0
    # 'pre' ran ONLY once during initial baseline evaluation!
    assert call_counts["pre"] == 1
    # 'mid' has 10x amplification downstream into 'post', whereas 'pre' is damped 0.02x by 'mid'
    assert sens["mid"].normalized_sensitivity > sens["pre"].normalized_sensitivity * 5.0

    # Also verify direct failed_node_id blame (100% score) on both UniversalCreditAssigner & CreditAssigner
    crashed_bm = BenchmarkResult(
        global_loss=1e12,
        runtime_artifacts={"failed_node_id": "mid", "prune_reason": "@icontract.require violated"},
    )
    u_grads = assigner.compute_gradients(cdg, crashed_bm)
    assert len(u_grads) == 1
    assert u_grads[0].node_id == "mid"
    assert u_grads[0].gradient_score == pytest.approx(100.0)

    legacy_grads = CreditAssigner().compute_gradients(
        cdg,
        crashed_bm,
        GhostSimReport(),
        OptimizationMetric.PRECISION,
    )
    assert len(legacy_grads) == 1
    assert legacy_grads[0].node_id == "mid"
    assert legacy_grads[0].gradient_score == pytest.approx(100.0)


def _make_calibrated_1d_profile(
    atom_name: str,
    *,
    singular_val: float,
    intrinsic_var: float,
) -> AtomUncertaintyProfile:
    mix = GaussianMixtureRegime.from_full_covariances(
        weights=[1.0],
        means=[[0.0]],
        covariances=[[[1.0]]],
    )
    proj_spec = IdentityProjector(dim=1).to_spec()
    op = LowRankPlusTailOperator(
        1,
        1,
        singular_values=[singular_val],
        left_vectors=[[1.0]],
        right_vectors=[[1.0]],
    )
    return AtomUncertaintyProfile(
        atom_name=atom_name,
        mode="calibrated",
        confidence=0.95,
        trust_tier=1,
        input_regimes={"x": PortRegimeSpec(port_name="x", projector_spec=proj_spec, mixture=mix)},
        output_projectors={"x": proj_spec},
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "x"): op},
                intrinsic_output_cov={"x": np.array([[intrinsic_var]], dtype=np.float64)},
            ),
        ),
    )


def test_universal_credit_blends_phase3_directional_uncertainty() -> None:
    """UniversalCreditAssigner incorporates Phase 3 marginal risk and pullback feasibility tags."""
    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_scale_tunable")
    profiles = {
        f"{__name__}.synth_scale_tunable": _make_calibrated_1d_profile(
            f"{__name__}.synth_scale_tunable",
            singular_val=1.0,
            intrinsic_var=2.5,
        )
    }
    budget = RiskBudget(sink_precisions={"y": 100.0})
    bm = BenchmarkResult(global_loss=1.5)
    grads = UniversalCreditAssigner(uncertainty_weight=0.8).compute_gradients(
        cdg,
        bm,
        uncertainty_profiles=profiles,
        risk_budget=budget,
    )
    assert grads
    assert grads[0].node_id == "mid"
    assert "pullback infeasible" in grads[0].bottleneck_reason


# ---------------------------------------------------------------------------
# 2. Rung 1 (Bounded Node HPO) & Cache Prefix Reuse
# ---------------------------------------------------------------------------


def test_rung1_bounded_node_hpo_infers_defaults_and_reuses_prefix_cache() -> None:
    """Rung 1 infers Python signature defaults, tunes bottleneck gain, and hits cache upstream."""
    pre_calls = {"n": 0}

    def tracked_center(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        pre_calls["n"] += 1
        return x - float(np.mean(x))

    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_scale_tunable")
    cdg.nodes[0].matched_primitive = "test.tracked_center"

    t = np.linspace(0.0, 4.0, 80, dtype=np.float64)
    x_in = np.cos(2.0 * np.pi * t) + 3.0
    # True target has gain = 2.6 (default gain is 1.0 -> auto-inferred range [0.2, 5.0])
    y_true = 2.6 * (x_in - float(np.mean(x_in)))

    harness = ResidentSliceHarness(
        [EvalCase(case_id="val_0", inputs={"x": x_in}, expected=y_true)],
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        holdout_fraction=0.0,
    )
    runner = LiveCDGRunner(extra_callables={"test.tracked_center": tracked_center})
    funnel = EvaluationFunnel(harness, runner=runner)
    base_rep = funnel.evaluate_candidate(cdg)
    assert base_rep.passed
    # Synthetic Tier 1 (5 regimes) + Tier 2 validation (1 case) = 6 unique inputs evaluated once
    calls_after_baseline = pre_calls["n"]

    ladder = RevisionLadder(funnel, hpo_trials_per_rung=14)
    out = ladder.run_rung1_node_hpo(
        cdg,
        "mid",
        baseline_loss=base_rep.global_loss,
    )
    assert out.improved
    assert out.accepted_loss < base_rep.global_loss * 0.4
    tuned_gain = out.accepted_node_params["mid"]["gain"]
    assert 1.8 < tuned_gain < 3.4
    # Across all 14 Optuna trials on 'mid', upstream 'pre' experienced ZERO cache misses!
    assert pre_calls["n"] == calls_after_baseline
    assert runner.cache.stats()["hits"] >= 14


# ---------------------------------------------------------------------------
# 3. Rung 2 (Catalog + SQLite Manifest Sibling Swap with Zero Hunter Calls)
# ---------------------------------------------------------------------------


def test_rung2_manifest_and_uncertainty_rescue_sibling_swap(tmp_path: Path) -> None:
    """Rung 2 discovers compatible siblings from ManifestFTSIndex + PrimitiveCatalog with 0 LLM/Hunter calls."""
    # Compile a tiny synthetic Plane-1 source package into manifest.sqlite
    pkg_dir = tmp_path / "synth_pkg" / "sciona" / "atoms" / "synth_smooth"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "atoms.py").write_text(
        "import numpy as np\n"
        "from sciona.ghost.registry import register_atom\n\n"
        "def w_vec(x):\n"
        "    return x\n\n"
        "@register_atom(w_vec)\n"
        "def synth_accurate_smoother(x: np.ndarray) -> np.ndarray:\n"
        '    """Accurate zero-bias symmetric smoother."""\n'
        "    return x\n",
        encoding="utf-8",
    )
    sources_yml = tmp_path / "sources.yml"
    sources_yml.write_text(
        "sources:\n"
        "  - name: synth-smooth\n"
        "    package: sciona.atoms.synth_smooth\n"
        "    path: synth_pkg\n",
        encoding="utf-8",
    )
    manifest_db = tmp_path / "manifest.sqlite"
    compile_local_manifest_sqlite(
        manifest_db,
        config=load_sources(sources_yml),
        base_dir=tmp_path,
        installed_only=False,
    )

    # Also register both biased & accurate smoothers in PrimitiveCatalog + LiveCDGRunner
    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name=f"{__name__}.synth_biased_smoother",
            source="synth",
            category=ConceptType.SIGNAL_FILTER,
            description="Biased smoother with offset",
            inputs=[_vec_port("x")],
            outputs=[_vec_port("x")],
        )
    )
    cat.add(
        AlgorithmicPrimitive(
            name=f"{__name__}.synth_accurate_smoother",
            source="synth",
            category=ConceptType.SIGNAL_FILTER,
            description="Accurate zero-bias symmetric smoother",
            inputs=[_vec_port("x")],
            outputs=[_vec_port("x")],
        )
    )

    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_biased_smoother")
    t = np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float64)
    clean = np.sin(t)
    noisy_in = clean + 0.05 * np.cos(13.0 * t)

    # Attach Phase 3 calibrated uncertainty profiles so mine_cdg_uncertainty_gaps rescues accurate smoother!
    profiles = {
        f"{__name__}.synth_biased_smoother": _make_calibrated_1d_profile(
            f"{__name__}.synth_biased_smoother",
            singular_val=1.5,
            intrinsic_var=9.0,
        ),
        f"{__name__}.synth_accurate_smoother": _make_calibrated_1d_profile(
            f"{__name__}.synth_accurate_smoother",
            singular_val=0.8,
            intrinsic_var=0.001,
        ),
        "sciona.atoms.synth_smooth.atoms.synth_accurate_smoother": _make_calibrated_1d_profile(
            "sciona.atoms.synth_smooth.atoms.synth_accurate_smoother",
            singular_val=0.8,
            intrinsic_var=0.001,
        ),
    }

    harness = ResidentSliceHarness(
        [EvalCase(case_id="v0", inputs={"x": noisy_in}, expected=clean)],
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        risk_budget=RiskBudget(sink_precisions={"y": 1.0}),
        holdout_fraction=0.0,
    )
    runner = LiveCDGRunner(
        extra_callables={"sciona.atoms.synth_smooth.atoms.synth_accurate_smoother": synth_accurate_smoother},
        uncertainty_profiles=profiles,
    )
    funnel = EvaluationFunnel(harness, runner=runner)
    base_rep = funnel.evaluate_candidate(cdg)
    assert base_rep.global_loss > 4.0

    ladder = RevisionLadder(
        funnel,
        catalog=cat,
        manifest_sqlite_path=manifest_db,
        max_sibling_evals=3,
    )
    siblings = ladder.discover_compatible_siblings(cdg, "mid")
    assert f"{__name__}.synth_accurate_smoother" in siblings
    assert "sciona.atoms.synth_smooth.atoms.synth_accurate_smoother" in siblings

    out = ladder.run_rung2_sibling_swap(
        cdg,
        "mid",
        baseline_loss=base_rep.global_loss,
        uncertainty_report=base_rep.uncertainty_report,
    )
    assert out.improved
    assert out.accepted_loss < 0.1
    assert "synth_accurate_smoother" in str(
        next(n for n in out.accepted_cdg.nodes if n.node_id == "mid").matched_primitive
    )


# ---------------------------------------------------------------------------
# 4. Rung 3 (Delta-Planned DPO Structural Expansion)
# ---------------------------------------------------------------------------


class _SyntheticOutlierClipRuleSet:
    """Domain-agnostic synthetic DPO rule inserting an outlier clipper before linear readout."""

    @property
    def name(self) -> str:
        return "synthetic_outlier_clip_rules"

    @property
    def domain(self) -> str:
        return "tabular_numeric"

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        already_clipped = any("synth_outlier_clip" in (n.matched_primitive or "") for n in cdg.nodes)
        if already_clipped:
            return []
        # Inspect live intermediate ndarray populated by EvaluationFunnel!
        src_val = context.intermediates.get("raw_stage")
        if isinstance(src_val, np.ndarray) and src_val.size > 0:
            max_val = float(np.max(np.abs(src_val)))
        elif isinstance(src_val, dict):
            max_val = abs(float(src_val.get("max", 0.0)))
        else:
            max_val = 0.0
        if max_val > 8.0:
            return [
                ExpansionDiagnostic(
                    rule_name="insert_outlier_clipper",
                    severity=0.9,
                    evidence=f"Observed extreme spike max={max_val:.1f} > 8.0",
                    metric_name="max_abs",
                    metric_value=max_val,
                    threshold=8.0,
                    source_domain=self.domain,
                )
            ]
        return []

    def rules(self) -> list[RewriteRule]:
        p_src = AlgorithmicNode(
            node_id="p_src",
            name="src",
            description="source pattern",
            concept_type=ConceptType.SIGNAL_TRANSFORM,
            status=NodeStatus.ATOMIC,
            inputs=[_vec_port("x")],
            outputs=[_vec_port("x")],
        )
        p_sink = AlgorithmicNode(
            node_id="p_sink",
            name="sink",
            description="sink pattern",
            concept_type=ConceptType.CUSTOM,
            matched_primitive=f"{__name__}.synth_linear_predict",
            status=NodeStatus.ATOMIC,
            inputs=[_vec_port("x")],
            outputs=[_vec_port("y")],
        )
        p_clip = AlgorithmicNode(
            node_id="outlier_clipper",
            name="Outlier Clipper",
            description="Clip extreme spikes",
            concept_type=ConceptType.SIGNAL_FILTER,
            matched_primitive=f"{__name__}.synth_outlier_clip",
            status=NodeStatus.ATOMIC,
            inputs=[_vec_port("x"), IOSpec(name="limit", type_desc="float", required=False, default_value_repr="3.0")],
            outputs=[_vec_port("x")],
        )
        lhs = CDGExport(
            nodes=[p_src, p_sink],
            edges=[
                DependencyEdge(
                    source_id="p_src",
                    target_id="p_sink",
                    output_name="x",
                    input_name="x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                )
            ],
        )
        interface = CDGExport(nodes=[p_src, p_sink], edges=[])
        rhs = CDGExport(
            nodes=[p_src, p_clip, p_sink],
            edges=[
                DependencyEdge(
                    source_id="p_src",
                    target_id="outlier_clipper",
                    output_name="x",
                    input_name="x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
                DependencyEdge(
                    source_id="outlier_clipper",
                    target_id="p_sink",
                    output_name="x",
                    input_name="x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
            ],
        )
        m = Morphism(node_map={"p_src": "p_src", "p_sink": "p_sink"}, edge_map={})
        return [
            RewriteRule(
                name="insert_outlier_clipper",
                lhs=lhs,
                rhs=rhs,
                interface=interface,
                l_morphism=m,
                r_morphism=m,
                priority=10,
            )
        ]


def test_rung3_delta_planned_dpo_structural_expansion() -> None:
    """Rung 3 inspects cached live intermediates via ExpansionContext and inserts a DPO stage."""
    n_raw = AlgorithmicNode(
        node_id="raw_stage",
        name="Pass Raw Input",
        description="Pass raw input vector",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{__name__}.synth_passthrough",
        inputs=[_vec_port("x")],
        outputs=[_vec_port("x")],
    )
    n_lin = AlgorithmicNode(
        node_id="lin_stage",
        name="Linear Predictor",
        description="Scale input linearly",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{__name__}.synth_linear_predict",
        inputs=[_vec_port("x")],
        outputs=[_vec_port("y")],
    )
    cdg = CDGExport(
        nodes=[n_raw, n_lin],
        edges=[
            DependencyEdge(
                source_id="raw_stage",
                target_id="lin_stage",
                output_name="x",
                input_name="x",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )

    clean_x = np.linspace(-1.5, 1.5, 50, dtype=np.float64)
    corrupted_x = clean_x.copy()
    corrupted_x[10] = 40.0  # massive synthetic spike!
    expected_y = 2.0 * np.clip(corrupted_x, -3.0, 3.0)

    harness = ResidentSliceHarness(
        [EvalCase(case_id="spike_case", inputs={"x": corrupted_x}, expected=expected_y)],
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        holdout_fraction=0.0,
    )
    funnel = EvaluationFunnel(harness)
    base_rep = funnel.evaluate_candidate(cdg)
    assert base_rep.global_loss > 5.0

    engine = ExpansionEngine([_SyntheticOutlierClipRuleSet()])
    ladder = RevisionLadder(funnel, expansion_engine=engine)
    out = ladder.run_rung3_dpo_expansion(
        cdg,
        "lin_stage",
        baseline_loss=base_rep.global_loss,
        baseline_benchmark=base_rep.benchmark,
        bottleneck_reason="Outlier sensitivity in linear stage",
    )
    assert out.improved
    assert out.accepted_loss < 1e-9
    assert out.delta_plan is not None
    assert len(out.accepted_cdg.leaf_nodes()) == 3


# ---------------------------------------------------------------------------
# 5. Rung 4a (Localized Subgraph Splice via compose_cdg)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung4a_localized_subgraph_splice() -> None:
    """Rung 4a decomposes ONLY the bottleneck node's IOSpec and splices via compose_cdg."""
    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_biased_smoother")
    x_val = np.linspace(-2.0, 2.0, 40, dtype=np.float64)
    x_val[5] = 25.0
    centered = x_val - np.mean(x_val)
    expected = 2.0 * np.clip(centered, -3.0, 3.0)

    harness = ResidentSliceHarness(
        [EvalCase(case_id="s0", inputs={"x": x_val}, expected=expected)],
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        holdout_fraction=0.0,
    )
    funnel = EvaluationFunnel(harness)
    base_rep = funnel.evaluate_candidate(cdg)

    seen_requests: list[SubgraphSpliceRequest] = []

    def scoped_subdecompose(req: SubgraphSpliceRequest) -> CDGExport:
        seen_requests.append(req)
        # Child subgraph replacing ONLY 'mid' (boundary x -> x): clip -> linear_predict
        c1 = AlgorithmicNode(
            node_id="clip_step",
            name="Clip",
            description="Clip spikes",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            matched_primitive=f"{__name__}.synth_outlier_clip",
            inputs=[_vec_port("x")],
            outputs=[_vec_port("x")],
        )
        c2 = AlgorithmicNode(
            node_id="scale2_step",
            name="Scale 2x",
            description="Double clipped values",
            concept_type=ConceptType.SIGNAL_TRANSFORM,
            status=NodeStatus.ATOMIC,
            matched_primitive=f"{__name__}.synth_linear_predict",
            inputs=[_vec_port("x")],
            outputs=[_vec_port("x")],
        )
        return CDGExport(
            nodes=[c1, c2],
            edges=[
                DependencyEdge(
                    source_id="clip_step",
                    target_id="scale2_step",
                    output_name="x",
                    input_name="x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                )
            ],
        )

    ladder = RevisionLadder(funnel, subdecompose_fn=scoped_subdecompose)
    out = await ladder.run_rung4a_localized_splice(
        cdg,
        "mid",
        goal="Clip spikes and double centered signal",
        bottleneck_reason="Biased smoother corrupts amplitude",
        baseline_loss=base_rep.global_loss,
    )
    assert len(seen_requests) == 1
    assert seen_requests[0].bottleneck_node.node_id == "mid"
    assert out.improved
    assert out.accepted_loss < 1e-9
    # Original 'pre' and 'post' stages remain intact while 'mid' became 2 leaf stages (total 4 leaves)
    leaf_ids = [n.node_id for n in out.accepted_cdg.leaf_nodes()]
    assert "pre" in leaf_ids and "post" in leaf_ids
    assert len(leaf_ids) == 4


# ---------------------------------------------------------------------------
# 6. Rung 4b (Guided Micro-Atom Synthesis) + Standalone Python Export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung4b_micro_atom_synthesis_rejects_broken_accepts_valid_and_exports_py(
    tmp_path: Path,
) -> None:
    """Rung 4b rejects metamorphic-violating code, accepts valid @icontract micro-atom, and exports runnable .py."""
    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_biased_smoother")
    # Target requires exact cubed centered vector: y = (x - mean(x)) ** 3 (shift-invariant wrt input x!)
    rng = np.random.default_rng(7)
    cases = []
    for i in range(6):
        xv = rng.normal(2.0, 1.0, size=48).astype(np.float64)
        yv = (xv - float(np.mean(xv))) ** 3
        cases.append(EvalCase(case_id=f"c_{i}", inputs={"x": xv}, expected=yv))

    synth_suite = SyntheticSuiteHarness(
        [_vec_port("x")],
        [_vec_port("y")],
        metamorphic_relations=[MetamorphicRelation.shift_invariance("x", shift=10.0)],
    )
    harness = ResidentSliceHarness(
        cases,
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
        synthetic_harness=synth_suite,
        metamorphic_relations=[MetamorphicRelation.shift_invariance("x", shift=10.0)],
        holdout_fraction=0.33,
    )

    # First test rejection of a broken micro-atom (produces NaN)
    def broken_synth(_spec: MicroAtomSynthesisSpec) -> str:
        return (
            "import numpy as np\n"
            "def micro_mid(x: np.ndarray) -> np.ndarray:\n"
            "    out = x.copy()\n"
            "    out[0] = np.nan\n"
            "    return out\n"
        )

    bad_session = PrototypingSession(
        cdg,
        goal="Cube centered signal",
        harness=harness,
        micro_atom_synthesizer=broken_synth,
    )
    bad_session.evaluate_initial()
    bad_step = await bad_session.ladder.run_rung4b_micro_atom_synthesis(
        cdg,
        "mid",
        failure_reason="test broken",
        baseline_loss=bad_session.best_loss,
    )
    assert not bad_step.improved
    assert len(bad_session.runner.extra_callables) == 0  # cleaned up!

    # Now run full PrototypingSession with a valid @icontract micro-atom synthesizer
    def valid_cubic_synth(spec: MicroAtomSynthesisSpec) -> str:
        assert spec.slot_node_id == "mid"
        return (
            "import icontract\n"
            "import numpy as np\n\n"
            "@icontract.require(lambda x: x.ndim == 1)\n"
            "@icontract.ensure(lambda result, x: result.shape == x.shape)\n"
            "def micro_mid_cubic(x: np.ndarray) -> np.ndarray:\n"
            "    return np.asarray(x, dtype=np.float64) ** 3\n"
        )

    session = PrototypingSession(
        cdg,
        goal="Cube centered signal",
        harness=harness,
        micro_atom_synthesizer=valid_cubic_synth,
        hpo_trials_per_rung=2,
    )
    summary = await session.run(max_iterations=3, verify_holdout=True)
    assert summary.best_loss < 1e-9
    assert summary.holdout_result is not None and summary.holdout_result.passed
    assert summary.holdout_result.loss < 1e-9
    assert len(summary.synthesized_micro_atoms) == 1

    # Export standalone Python module and execute run_pipeline(x=...)!
    out_py = session.export_python_module(tmp_path / "exported_cubic_pipeline.py")
    spec_mod = importlib.util.spec_from_file_location("exported_cubic_pipeline", out_py)
    assert spec_mod is not None and spec_mod.loader is not None
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    probe_x = np.linspace(-1.0, 3.0, 32, dtype=np.float64)
    exported_out = mod.run_pipeline(x=probe_x)
    live_out = session.runner.execute(
        session.best_cdg,
        {"x": probe_x},
        node_params=session.best_node_params,
    ).outputs
    np.testing.assert_allclose(exported_out["y"], live_out["y"], atol=1e-12)


# ---------------------------------------------------------------------------
# 7. CLI (`sciona proto`), SDK (`Sciona.proto`), and Principal Integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_sciona_proto_and_sdk_and_principal_compute_gradients(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `sciona proto` CLI, `Sciona.proto` SDK, and `Principal` `compute_gradients`."""
    cdg = _build_3stage_cdg(mid_prim=f"{__name__}.synth_scale_tunable")
    cdg_file = tmp_path / "seed_cdg.json"
    cdg_file.write_text(json.dumps(cdg.model_dump(mode="json"), indent=2), encoding="utf-8")

    x_vec = np.linspace(0.0, 5.0, 40, dtype=np.float64)
    expected_vec = 3.0 * (x_vec - float(np.mean(x_vec)))
    slice_file = tmp_path / "slice.json"
    slice_file.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "cli_0",
                        "inputs": {"x": x_vec.tolist()},
                        "expected": expected_vec.tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out_cdg_path = tmp_path / "refined_cdg.json"
    out_py_path = tmp_path / "refined_pipeline.py"
    args = argparse.Namespace(
        cdg=str(cdg_file),
        goal="Tune gain via CLI",
        data=str(slice_file),
        target_key="expected",
        metric="rmse",
        max_iterations=3,
        budget=10.0,
        target_loss=0.05,
        max_rung=2,
        hpo_trials=12,
        catalog=None,
        manifest=None,
        output_cdg=str(out_cdg_path),
        export=str(out_py_path),
        no_holdout=False,
        json=True,
    )
    await _cmd_proto(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["best_loss"] < payload["initial_loss"] * 0.5
    assert out_cdg_path.is_file()
    assert out_py_path.is_file()

    # Verify SDK Sciona.from_catalog(...).proto(...)
    sdk = Sciona.from_catalog(PrimitiveCatalog())
    sdk_session = sdk.proto(cdg, goal="SDK proto check")
    rec0 = sdk_session.evaluate_initial()
    assert rec0.iteration == 0

    # Verify Principal compute_gradients no longer exits with "No gradients computed" on Trial 1!
    harness = ResidentSliceHarness.from_dataset_file(
        slice_file,
        root_inputs=[_vec_port("x")],
        root_outputs=[_vec_port("y")],
        metric="rmse",
    )
    sandbox = LivePrototypingSandbox(EvaluationFunnel(harness))
    bundle = ExportBundle(
        target="python-pkg",
        output_dir=tmp_path,
        source_path=cdg_file,
        compiled_artifact=cdg_file,
        executable_artifact=cdg_file,
    )
    bm = await sandbox.evaluate(bundle, str(slice_file), OptimizationMetric.PRECISION)
    state = PrincipalState(
        goal="Tune gain",
        metric=OptimizationMetric.PRECISION,
        cdg=cdg,
        benchmark=bm,
        current_trial=1,
    )
    deps = PrincipalDeps(
        architect=None,  # type: ignore[arg-type]
        sandbox=sandbox,
        problem_harness=harness,
    )
    grad_update = await compute_gradients(state, {"configurable": {"deps": deps}})
    assert not grad_update.get("done", False)
    assert grad_update["bottleneck_node_id"] in {"pre", "mid", "post"}
    assert grad_update["top_gradient"].gradient_score > 0.0
