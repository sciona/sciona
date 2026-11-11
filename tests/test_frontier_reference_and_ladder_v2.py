"""Tests for canonical Frontier-LLM reference store and RevisionLadder v2 enhancements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from sciona.architect.catalog import PrimitiveCatalog
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
    PrimitiveParamSpec,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
)
from sciona.prototyping import (
    DEFAULT_FRONTIER_REFERENCE_PATH,
    CaseScore,
    EvalCase,
    GroupConditionalNormalizationRuleSet,
    ProblemHarness,
    PrototypingConfig,
    PrototypingSession,
    compare_sciona_to_frontier,
    load_frontier_llm_reference,
)


def _arr_port(name: str) -> IOSpec:
    return IOSpec(
        name=name,
        type_desc="np.ndarray",
        required=True,
        port_type=PortTypeSpec(raw="np.ndarray", kind="array", dtype="float64", ndim=1),
    )


def _opt_float_port(name: str, default_repr: str) -> IOSpec:
    return IOSpec(
        name=name,
        type_desc="float",
        required=False,
        default_value_repr=default_repr,
        port_type=PortTypeSpec(raw="float", kind="scalar", dtype="float"),
    )


class _SimpleScoreHarness:
    def __init__(self, x_in: np.ndarray, target: np.ndarray) -> None:
        self._case = EvalCase(case_id="val", inputs={"x": x_in}, expected=target, tier=2)

    @property
    def root_inputs(self) -> list[IOSpec]:
        return [_arr_port("x")]

    @property
    def root_outputs(self) -> list[IOSpec]:
        return [_arr_port("y")]

    @property
    def risk_budget(self) -> None:
        return None

    def synthetic_cases(self) -> list[EvalCase]:
        return []

    def validation_cases(self) -> list[EvalCase]:
        return [self._case]

    def holdout_cases(self) -> list[EvalCase]:
        return []

    def score(self, case: EvalCase, outputs: Mapping[str, Any]) -> CaseScore:
        pred = np.asarray(outputs["y"], dtype=np.float64)
        err = float(np.sqrt(np.mean((pred - np.asarray(case.expected, dtype=np.float64)) ** 2)))
        return CaseScore(case_id=case.case_id, loss=err, passed=True)


def test_frontier_llm_reference_integrity_and_confidentiality() -> None:
    raw_text = DEFAULT_FRONTIER_REFERENCE_PATH.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for generic_leak in ("http://", "https://", "file://", ".npz", ".csv", ".parquet", ".edf"):
        assert generic_leak not in lowered, f"Opaque reference must not embed '{generic_leak}'"

    suite = load_frontier_llm_reference(verify_digest=True)
    assert len(suite.tasks) == 5
    assert set(suite.tasks.keys()) == {"eval_01", "eval_02", "eval_03", "eval_04", "eval_05"}
    assert suite.total_billed_context_tokens > 20_000_000
    assert suite.total_fresh_tokens > 2_000_000

    for task in suite.tasks.values():
        assert task.isolation_verified is True
        assert len(task.checkpoints) == 12
        assert [c.iteration for c in task.checkpoints] == list(range(1, 13))

    cmp_eval02 = compare_sciona_to_frontier(
        "eval_02",
        [0.68728, 0.62593, 0.61223],
        sciona_elapsed_s=189.8,
        sciona_tokens=0,
        reference=suite,
    )
    assert cmp_eval02["beats_frontier_final"] is True
    assert cmp_eval02["wall_clock_speedup_x"] > 4.5


def test_prototyping_config_defaults() -> None:
    cfg = PrototypingConfig()
    assert cfg.max_iterations == 10
    assert cfg.exhausted_patience == 2
    assert cfg.min_rung1_rel_gain == pytest.approx(0.0025)
    assert cfg.escalate_rung4_on_exhaustion is True


@pytest.mark.asyncio
async def test_min_rung1_rel_gain_prevents_micro_tweak_from_starving_rung2_swap() -> None:
    x = np.linspace(1.0, 5.0, 32)
    target = 2.0 * x

    def pass_in(x: np.ndarray) -> np.ndarray:
        return x

    def micro_tunable_est(x: np.ndarray, tiny_bias: float = 0.0) -> np.ndarray:
        # Base loss ~10.0; tiny_bias in [-0.005, 0.005] yields <0.05% relative change
        return 2.0 * x + 10.0 + float(tiny_bias)

    def exact_sibling_est(x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 0.1

    n1 = AlgorithmicNode(
        node_id="n1",
        name="Pass",
        description="Pass",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        matched_primitive="test.pass_in",
        inputs=[_arr_port("x")],
        outputs=[_arr_port("x")],
    )
    n2 = AlgorithmicNode(
        node_id="n2",
        name="Estimator",
        description="Estimator",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive="test.micro_tunable_est",
        inputs=[_arr_port("x"), _opt_float_port("tiny_bias", "0.0")],
        outputs=[_arr_port("y")],
    )
    cdg = CDGExport(
        nodes=[n1, n2],
        edges=[
            DependencyEdge(
                source_id="n1",
                target_id="n2",
                output_name="x",
                input_name="x",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )
    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name="test.micro_tunable_est",
            source="test",
            category=ConceptType.CUSTOM,
            description="Micro tunable",
            inputs=n2.inputs,
            outputs=n2.outputs,
            tunable_params=[
                PrimitiveParamSpec(name="tiny_bias", kind="float", default=0.0, min_value=-0.005, max_value=0.005)
            ],
        )
    )
    cat.add(
        AlgorithmicPrimitive(
            name="test.exact_sibling_est",
            source="test",
            category=ConceptType.CUSTOM,
            description="Structural sibling",
            inputs=[_arr_port("x")],
            outputs=[_arr_port("y")],
        )
    )

    session = PrototypingSession(
        cdg,
        harness=_SimpleScoreHarness(x, target),
        catalog=cat,
        min_rung1_rel_gain=0.005,
    )
    session.runner.extra_callables.update(
        {
            "test.pass_in": pass_in,
            "test.micro_tunable_est": micro_tunable_est,
            "test.exact_sibling_est": exact_sibling_est,
        }
    )
    step = await session.step(max_rung=3)
    assert step.improved is True
    assert step.rung == "rung2_sibling_swap"
    assert step.loss_after < 0.2


@pytest.mark.asyncio
async def test_coupled_dpo_and_sibling_swap_escapes_pareto_trap() -> None:
    x = np.ones(16, dtype=np.float64) * 5.0
    target = np.zeros(16, dtype=np.float64)

    # Trap geometry:
    # - Seed: raw(5) -> noisy_post(5) => y = 1.0 (loss = 1.0)
    # - DPO alone: align(5 -> 0) -> noisy_post(0) => y = 2.0 (loss = 2.0, WORSE!)
    # - Swap alone: raw(5) -> clean_post(5) => y = 5.0 (loss = 5.0, WORSE!)
    # - Coupled DPO + Swap: align(5 -> 0) -> clean_post(0) => y = 0.05 (loss = 0.05, WINS!)
    def raw_pass(x: np.ndarray) -> np.ndarray:
        return x

    def align_zero(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def noisy_post(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) > 1.0, np.ones_like(x) * 1.0, np.ones_like(x) * 2.0)

    def clean_post(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) > 1.0, np.ones_like(x) * 5.0, np.ones_like(x) * 0.05)

    n1 = AlgorithmicNode(
        node_id="n1",
        name="Pass",
        description="Pass",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        matched_primitive="trap.raw_pass",
        inputs=[_arr_port("x")],
        outputs=[_arr_port("x")],
    )
    n2 = AlgorithmicNode(
        node_id="n2",
        name="Post",
        description="Post",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive="trap.noisy_post",
        inputs=[_arr_port("x")],
        outputs=[_arr_port("y")],
    )
    cdg = CDGExport(
        nodes=[n1, n2],
        edges=[
            DependencyEdge(
                source_id="n1",
                target_id="n2",
                output_name="x",
                input_name="x",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )

    class _AlignRuleSet:
        @property
        def name(self) -> str:
            return "align_rs"

        @property
        def domain(self) -> str:
            return "test"

        def diagnose(self, g: CDGExport, ctx: ExpansionContext) -> list[ExpansionDiagnostic]:
            _ = ctx
            if any(n.node_id == "n_align" for n in g.nodes):
                return []
            return [
                ExpansionDiagnostic(
                    rule_name="insert_align",
                    severity=0.9,
                    evidence="drift",
                    metric_name="drift",
                    metric_value=1.0,
                    threshold=0.5,
                )
            ]

        def rules(self) -> list[RewriteRule]:
            p1 = n1.model_copy(update={"node_id": "p1"})
            p2 = n2.model_copy(update={"node_id": "p2", "matched_primitive": None})
            ins = AlgorithmicNode(
                node_id="n_align",
                name="Align",
                description="Align",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="trap.align_zero",
                inputs=[_arr_port("x")],
                outputs=[_arr_port("x")],
            )
            lhs = CDGExport(
                nodes=[p1, p2],
                edges=[
                    DependencyEdge(
                        source_id="p1",
                        target_id="p2",
                        output_name="x",
                        input_name="x",
                        source_type="np.ndarray",
                        target_type="np.ndarray",
                    )
                ],
            )
            rhs = CDGExport(
                nodes=[p1, ins, p2],
                edges=[
                    DependencyEdge(
                        source_id="p1",
                        target_id="n_align",
                        output_name="x",
                        input_name="x",
                        source_type="np.ndarray",
                        target_type="np.ndarray",
                    ),
                    DependencyEdge(
                        source_id="n_align",
                        target_id="p2",
                        output_name="x",
                        input_name="x",
                        source_type="np.ndarray",
                        target_type="np.ndarray",
                    ),
                ],
            )
            iface = CDGExport(nodes=[p1, p2], edges=[])
            m = Morphism(node_map={"p1": "p1", "p2": "p2"}, edge_map={})
            return [RewriteRule(name="insert_align", lhs=lhs, rhs=rhs, interface=iface, l_morphism=m, r_morphism=m)]

    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name="trap.clean_post",
            source="test",
            category=ConceptType.CUSTOM,
            description="Clean zero-noise calibrator",
            inputs=[_arr_port("x")],
            outputs=[_arr_port("y")],
        )
    )

    session = PrototypingSession(
        cdg,
        harness=_SimpleScoreHarness(x, target),
        catalog=cat,
        expansion_engine=ExpansionEngine([_AlignRuleSet()]),
    )
    session.runner.extra_callables.update(
        {
            "trap.raw_pass": raw_pass,
            "trap.align_zero": align_zero,
            "trap.noisy_post": noisy_post,
            "trap.clean_post": clean_post,
        }
    )

    step = await session.step(max_rung=3)
    assert step.improved is True
    assert step.rung == "rung3_dpo_expansion"
    assert "coupled DPO" in step.rationale
    assert step.loss_after == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_group_conditional_normalization_ruleset_splices_normalizer() -> None:
    x = np.array([[0.1, 0.1], [0.25, 0.25], [0.4, 0.4], [0.55, 0.55]], dtype=np.float64)
    feat_raw = np.sum(x, axis=1)
    feat_norm = (feat_raw - np.mean(feat_raw)) / (np.std(feat_raw) + 1e-8)
    target = 10.0 * feat_norm + 50.0

    arr_2d_port = IOSpec(
        name="x",
        type_desc="np.ndarray",
        required=True,
        port_type=PortTypeSpec(raw="np.ndarray", kind="array", dtype="float64", ndim=2),
    )

    def extract_feat(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    def per_unit_standardizer(feat: np.ndarray) -> np.ndarray:
        return (feat - np.mean(feat)) / (np.std(feat) + 1e-8)

    def affine_readout(feat: np.ndarray) -> np.ndarray:
        return 10.0 * feat + 50.0

    n1 = AlgorithmicNode(
        node_id="n1",
        name="Ingest",
        description="Ingest",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive="grp.extract_feat",
        inputs=[arr_2d_port],
        outputs=[_arr_port("feat")],
    )
    n2 = AlgorithmicNode(
        node_id="n2",
        name="Readout",
        description="Readout",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive="grp.affine_readout",
        inputs=[_arr_port("feat")],
        outputs=[_arr_port("y")],
    )
    cdg = CDGExport(
        nodes=[n1, n2],
        edges=[
            DependencyEdge(
                source_id="n1",
                target_id="n2",
                output_name="feat",
                input_name="feat",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
        metadata={"goal": "Classify multichannel segments across unseen sensor units"},
    )
    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name="grp.per_unit_standardizer",
            source="test",
            category=ConceptType.SIGNAL_FILTER,
            description="Per-unit group standardization",
            inputs=[_arr_port("feat")],
            outputs=[_arr_port("feat")],
        )
    )
    rs = GroupConditionalNormalizationRuleSet(catalog=cat)
    assert len(rs.diagnose(cdg, ExpansionContext())) == 1

    class _Harness2D(_SimpleScoreHarness):
        @property
        def root_inputs(self) -> list[IOSpec]:
            return [arr_2d_port]

    session = PrototypingSession(
        cdg,
        goal=cdg.metadata["goal"],
        harness=_Harness2D(x, target),
        catalog=cat,
    )
    session.runner.extra_callables.update(
        {
            "grp.extract_feat": extract_feat,
            "grp.per_unit_standardizer": per_unit_standardizer,
            "grp.affine_readout": affine_readout,
        }
    )
    step = await session.step(max_rung=3)
    assert step.improved is True
    assert step.rung == "rung3_dpo_expansion"
    assert step.loss_after < 1e-4
