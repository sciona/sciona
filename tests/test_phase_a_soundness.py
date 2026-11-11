# mypy: disable-error-code=untyped-decorator
"""Regression tests for Phase A P0 soundness and pipeline fixes."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import (
    CDGExport,
    _validate_type_signature_syntax,
    validate_handoff,
)
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    PrimitiveParamSpec,
)
from sciona.architect.state import _merge_edges
from sciona.judge.models import CompilerFeedback
from sciona.orchestrator import _apply_split_subnodes, run_orchestration
from sciona.principal.admissibility import (
    AdmissibilityContext,
    AdmissibilityDecision,
    AdmissibilityDisposition,
    AdmissibilityReport,
)
from sciona.principal.graph import (
    check_admissibility,
    time_travel_update,
)
from sciona.principal.graph_types import PrincipalDeps, PrincipalState
from sciona.principal.hpo import OptunaManager
from sciona.principal.models import BenchmarkResult, NodeTelemetry
from sciona.provider_expansion_declarations import _load_module
from sciona.sdk import Sciona
from sciona.synthesizer.assembler import Assembler
from sciona.synthesizer.ghost_sim import PlanError, _FallbackPlanError, run_ghost_simulation
from sciona.synthesizer.models import SkeletonFile
from sciona.synthesizer.python_template import generate_pipeline_py
from sciona.synthesizer.repair import (
    CompileCheck,
    DeterministicFix,
    LLMRepair,
    RepairDeps,
    RepairState,
    SorryElimination,
    _find_header_insert_line,
    repair_graph,
)
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


def _match(node_id: str, decl_name: str, sig: str = "(x: float) -> float") -> MatchResult:
    decl = Declaration(
        name=decl_name,
        type_signature=sig,
        prover=Prover.PYTHON,
        provider_id="sciona-atoms",
        distribution_name="sciona-atoms",
        distribution_version="0.1.0",
        install_requirement="sciona-atoms==0.1.0",
        import_module="math",
        import_symbol=decl_name,
        wheel_url="https://example.invalid/sciona_atoms.whl",
        wheel_sha256="deadbeef",
    )
    cand = CandidateMatch(declaration=decl, score=0.95, retrieval_method="test")
    vr = VerificationResult(
        candidate=cand,
        verified=True,
        verification_level=VerificationLevel.TYPE_CHECKED,
    )
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=sig, prover=Prover.PYTHON),
        verified_match=vr,
        all_candidates=[cand],
        all_verifications=[vr],
    )


class TestHandoffLeafAndSignatureFixes:
    def test_upstream_dataflow_nodes_are_still_validated_as_leaves(self) -> None:
        root = AlgorithmicNode(
            node_id="root",
            name="Pipeline",
            description="Two-stage pipeline",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.DECOMPOSED,
            children=["stage_a", "stage_b"],
            depth=0,
        )
        # Upstream leaf is PENDING (not atomic) AND is a source_id in cdg.edges!
        stage_a = AlgorithmicNode(
            node_id="stage_a",
            parent_id="root",
            name="Upstream Stage",
            description="Unfinished stage",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=1,
        )
        stage_b = AlgorithmicNode(
            node_id="stage_b",
            parent_id="root",
            name="Downstream Stage",
            description="Valid sink",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="(x: float) -> float",
            depth=1,
        )
        edge = DependencyEdge(
            source_id="stage_a",
            target_id="stage_b",
            output_name="y",
            input_name="x",
            source_type="float",
            target_type="float",
        )
        cdg = CDGExport(nodes=[root, stage_a, stage_b], edges=[edge])

        non_atomic = cdg.non_atomic_leaves()
        assert [n.node_id for n in non_atomic] == ["stage_a"]
        issues = validate_handoff(cdg)
        assert any("stage_a" in issue for issue in issues)

    def test_python_arrow_signatures_validate_cleanly(self) -> None:
        assert _validate_type_signature_syntax("(x: int, y: float) -> bool", Prover.PYTHON) is None
        assert _validate_type_signature_syntax("list[int], int -> bool", Prover.PYTHON) is None
        assert _validate_type_signature_syntax("Callable[[int], str]", Prover.PYTHON) is None
        err = _validate_type_signature_syntax("list[int", Prover.PYTHON)
        assert err is not None and "Invalid Python type annotation" in err


class TestArchitectEdgeDeduplication:
    def test_merge_edges_deduplicates_retries(self) -> None:
        e1 = DependencyEdge(
            source_id="a",
            target_id="b",
            output_name="out",
            input_name="inp",
            source_type="int",
            target_type="int",
        )
        e2 = DependencyEdge(
            source_id="a",
            target_id="b",
            output_name="out",
            input_name="inp",
            source_type="float",
            target_type="float",
        )
        merged = _merge_edges([e1], [e2])
        assert len(merged) == 1
        assert merged[0].source_type == "float"


class TestOrchestratorSplitAndMatchPruning:
    def test_apply_split_subnodes_wires_ports_and_edges(self) -> None:
        n_in = AlgorithmicNode(
            node_id="src",
            name="Source",
            description="Source",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="float -> float",
            inputs=[IOSpec(name="raw", type_desc="float")],
            outputs=[IOSpec(name="mid", type_desc="float")],
        )
        target = AlgorithmicNode(
            node_id="mid_node",
            name="Complex Step",
            description="Step to split",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="float -> int",
            inputs=[IOSpec(name="mid", type_desc="float")],
            outputs=[IOSpec(name="idx", type_desc="int")],
        )
        cdg = CDGExport(
            nodes=[n_in, target],
            edges=[
                DependencyEdge(
                    source_id="src",
                    target_id="mid_node",
                    output_name="mid",
                    input_name="mid",
                    source_type="float",
                    target_type="float",
                )
            ],
        )

        _apply_split_subnodes(
            cdg,
            target,
            [
                {"name": "First Half", "description": "part 1", "type_signature": "float -> float"},
                {"name": "Second Half", "description": "part 2", "type_signature": "float -> int"},
            ],
        )

        assert target.status == NodeStatus.DECOMPOSED
        assert target.children == ["mid_node_sub0", "mid_node_sub1"]
        edge_pairs = [(e.source_id, e.target_id) for e in cdg.edges]
        assert ("src", "mid_node_sub0") in edge_pairs
        assert ("mid_node_sub0", "mid_node_sub1") in edge_pairs

    @pytest.mark.asyncio
    async def test_run_orchestration_marks_all_matched_after_split_succeeds(self) -> None:
        leaf = AlgorithmicNode(
            node_id="a",
            name="Composite Leaf",
            description="Composite step",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.ATOMIC,
            type_signature="int -> int",
            inputs=[IOSpec(name="x", type_desc="int")],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        cdg = CDGExport(nodes=[leaf], edges=[])

        async def mock_find_match(pdg_node: PDGNode) -> MatchResult:
            if pdg_node.predicate_id == "a":
                decl = Declaration(name="miss", type_signature="int -> int")
                cand = CandidateMatch(declaration=decl, score=0.1, retrieval_method="test")
                vr = VerificationResult(candidate=cand, verified=False, error_message="miss")
                return MatchResult(
                    pdg_node=pdg_node,
                    verified_match=None,
                    all_candidates=[cand],
                    all_verifications=[vr],
                )
            return _match(pdg_node.predicate_id, "hit", "int -> int")

        hunter = AsyncMock()
        hunter.find_match = AsyncMock(side_effect=mock_find_match)
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value='[{"name": "sub_a", "description": "step 1", "type_signature": "int -> int"}]'
        )

        res = await run_orchestration(cdg, hunter_agent=hunter, llm=llm, max_rounds=3)
        assert res.all_matched is True
        assert [mr.pdg_node.predicate_id for mr in res.match_results] == ["a_sub0"]


class TestGhostSimFixes:
    def test_plan_error_accepts_positional_and_keyword_args(self) -> None:
        fb_err = _FallbackPlanError("fn_name", "arg_x", detail="bad domain", extra=1)
        assert "bad domain" in str(fb_err)
        assert fb_err.function_name == "fn_name"
        real_err = PlanError("fn_name", "arg_x", detail="bad domain")
        assert "bad domain" in str(real_err)

    def test_unwitnessed_upstream_leaf_does_not_crash_downstream_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sciona.synthesizer.ghost_sim as gs

        monkeypatch.setattr(gs, "_load_ghost_backend", lambda: True)
        monkeypatch.setattr(gs, "list_registered", lambda: ["witnessed_sink"])

        def _fake_simulate(sim_nodes: list[object], initial_state: dict[str, object]) -> object:
            # Every wired input key for witnessed_sink must exist in initial_state!
            for sn in sim_nodes:
                for state_key in sn.inputs.values():  # type: ignore[attr-defined]
                    assert state_key in initial_state
            res = types.SimpleNamespace(node_count=len(sim_nodes), trace=["Sink"], final_state=initial_state)
            return res

        monkeypatch.setattr(gs, "_simulate_with_bayesian_checks", _fake_simulate)
        monkeypatch.setattr(
            gs,
            "SimNode",
            lambda **kw: types.SimpleNamespace(**kw),
        )

        n1 = AlgorithmicNode(
            node_id="n1",
            name="Unwitnessed Source",
            description="No ghost witness",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            inputs=[IOSpec(name="raw", type_desc="np.ndarray")],
            outputs=[IOSpec(name="clean", type_desc="np.ndarray")],
        )
        n2 = AlgorithmicNode(
            node_id="n2",
            name="Sink",
            description="Has ghost witness",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            inputs=[IOSpec(name="clean", type_desc="np.ndarray")],
            outputs=[IOSpec(name="out", type_desc="np.ndarray")],
        )
        cdg = CDGExport(
            nodes=[n1, n2],
            edges=[
                DependencyEdge(
                    source_id="n1",
                    target_id="n2",
                    output_name="clean",
                    input_name="clean",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                )
            ],
        )
        matches = [_match("n1", "unregistered_src"), _match("n2", "witnessed_sink")]
        report = run_ghost_simulation(cdg, matches)
        assert report.ran is True
        assert report.passed is True


class TestAssemblerAndTemplateFixes:
    def test_assembler_handles_duplicate_names_nested_roots_multi_out_and_parallel_sinks(self) -> None:
        root = AlgorithmicNode(
            node_id="root",
            name="Main Pipeline",
            description="Top",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.DECOMPOSED,
            children=["sub_group"],
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="left", type_desc="float"), IOSpec(name="right", type_desc="float")],
        )
        sub_group = AlgorithmicNode(
            node_id="sub_group",
            parent_id="root",
            name="Nested Group",
            description="Middle decomposed layer",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.DECOMPOSED,
            children=["splitter", "branch_1", "branch_2"],
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="left", type_desc="float"), IOSpec(name="right", type_desc="float")],
        )
        splitter = AlgorithmicNode(
            node_id="splitter",
            parent_id="sub_group",
            name="Split Pair",
            description="Returns 2 outputs",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="(x: float) -> tuple[float, float]",
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="low", type_desc="float"), IOSpec(name="high", type_desc="float")],
        )
        # Deliberately give branch_1 and branch_2 the EXACT same human-readable name!
        b1 = AlgorithmicNode(
            node_id="branch_1",
            parent_id="sub_group",
            name="Scale Value",
            description="Left sink",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="(val: float) -> float",
            inputs=[IOSpec(name="val", type_desc="float")],
            outputs=[IOSpec(name="left", type_desc="float")],
        )
        b2 = AlgorithmicNode(
            node_id="branch_2",
            parent_id="sub_group",
            name="Scale Value",
            description="Right sink",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            type_signature="(val: float) -> float",
            inputs=[IOSpec(name="val", type_desc="float")],
            outputs=[IOSpec(name="right", type_desc="float")],
        )
        edges = [
            DependencyEdge(
                source_id="splitter",
                target_id="branch_1",
                output_name="low",
                input_name="val",
                source_type="float",
                target_type="float",
            ),
            DependencyEdge(
                source_id="splitter",
                target_id="branch_2",
                output_name="high",
                input_name="val",
                source_type="float",
                target_type="float",
            ),
        ]
        cdg = CDGExport(nodes=[root, sub_group, splitter, b1, b2], edges=edges)
        matches = [
            _match("splitter", "math.frexp"),
            _match("branch_1", "math.fabs"),
            _match("branch_2", "math.ceil"),
        ]

        skeleton = Assembler(Prover.PYTHON).assemble(cdg, matches)
        src = skeleton.source_code
        assert "def scale_value_branch_1(" in src
        assert "def scale_value_branch_2(" in src
        assert "split_pair_result[0]" in src
        assert "split_pair_result[1]" in src
        assert "return (scale_value_branch_1_result, scale_value_branch_2_result)" in src
        assert skeleton.sorry_count == 0

    def test_generated_pipeline_standalone_and_raises_on_missing_required_arg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg = types.ModuleType("_test_exported_pkg")
        atoms_mod = types.ModuleType("_test_exported_pkg.atoms")

        def strict_fn(required_x: float, scale: float = 2.0) -> float:
            return required_x * scale

        atoms_mod.strict_fn = strict_fn  # type: ignore[attr-defined]
        pkg.atoms = atoms_mod  # type: ignore[attr-defined]

        code = generate_pipeline_py(
            [],
            entrypoint_names=["strict_fn"],
            default_entrypoint="strict_fn",
        )
        assert "'loss': None" in code
        assert "fallback_kwargs[name] = None" not in code

        # Simulate standalone environment without sciona.principal installed
        real_import = builtins.__import__

        def _guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("sciona.principal"):
                raise ImportError(f"Standalone environment blocks {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setitem(sys.modules, "_test_exported_pkg", pkg)
        monkeypatch.setitem(sys.modules, "_test_exported_pkg.atoms", atoms_mod)
        monkeypatch.setattr(builtins, "__import__", _guarded_import)

        pipe_mod = types.ModuleType("_test_exported_pkg.pipeline")
        pipe_mod.__package__ = "_test_exported_pkg"
        exec(compile(code, "<pipeline>", "exec"), pipe_mod.__dict__)

        assert pipe_mod._HAS_SCIONA_PRINCIPAL is False
        assert pipe_mod.run_pipeline(required_x=3.5) == pytest.approx(7.0)
        with pytest.raises(TypeError):
            pipe_mod.run_pipeline()


class TestRepairStateMachineFixes:
    def test_header_insert_line_skips_multiline_comments(self) -> None:
        lean_lines = ["/-!", "  Header comment", "-/", "import Mathlib", "", "def x := 1"]
        assert _find_header_insert_line(lean_lines, "lean4") == 5

        coq_lines = ["(*", "  Header", "*)", "Require Import List.", "Definition x := 1."]
        assert _find_header_insert_line(coq_lines, "coq") == 4

        py_lines = ['"""Module docstring.', 'Line 2."""', "import math", "def f(): pass"]
        assert _find_header_insert_line(py_lines, "python") == 3

    @pytest.mark.asyncio
    async def test_python_not_implemented_stub_is_not_marked_compiled_ok_prematurely(self) -> None:
        source = "def compose(x: float) -> float:\n    raise NotImplementedError('todo')\n"
        skeleton = SkeletonFile(prover="python", source_code=source, sorry_count=1)
        env = AsyncMock()
        env.prover_name = "python"
        env._run = AsyncMock(
            side_effect=[
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value="    return x * 2.0")

        state = RepairState(skeleton=skeleton, max_iterations=3, sorry_remaining=1)
        res = await repair_graph.run(CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm))
        assert state.compiled_ok is True
        assert "NotImplementedError" not in res.output.source_code

    @pytest.mark.asyncio
    async def test_unsolved_goal_without_sorry_token_does_not_infinite_loop(self) -> None:
        skeleton = SkeletonFile(prover="lean4", source_code="theorem t : True := by\n  trivial\n")
        env = AsyncMock()
        env.prover_name = "lean4"
        env._run = AsyncMock(
            return_value=CompilerFeedback(raw_output="⊢ True", errors=[], goals_remaining=["⊢ True"])
        )
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value="invalid json")

        state = RepairState(skeleton=skeleton, max_iterations=2)
        await repair_graph.run(CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm))
        assert state.iteration >= 2
        assert state.compiled_ok is False


class TestHPOAndPrincipalFixes:
    def test_optuna_trial0_enqueues_defaults_and_computes_scoped_importances(self) -> None:
        cat = PrimitiveCatalog()
        cat.add(
            AlgorithmicPrimitive(
                name="filter_op",
                source="test",
                category=ConceptType.SIGNAL_FILTER,
                description="Filter",
                type_signature="(x: np.ndarray) -> np.ndarray",
                tunable_params=[
                    PrimitiveParamSpec(
                        name="cutoff",
                        kind="float",
                        default=12.5,
                        min_value=1.0,
                        max_value=50.0,
                    )
                ],
            )
        )
        node = AlgorithmicNode(
            node_id="n1",
            name="Filter",
            description="Filter",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            matched_primitive="filter_op",
        )
        cdg = CDGExport(nodes=[node], edges=[])

        mgr = OptunaManager(study_name="phase_a_hpo")
        s0 = mgr.suggest_node_params(signature="sig_1", cdg=cdg, catalog=cat)
        assert s0.assignments["n1"]["cutoff"] == pytest.approx(12.5)
        # Verify Optuna's internal trial record ALSO matches 12.5!
        scoped_study = mgr._studies["sig_1"]
        assert scoped_study.trials[0].params["n1.cutoff"] == pytest.approx(12.5)
        mgr.complete_trial(signature="sig_1", trial_number=s0.trial_number, loss=0.5)

        s1 = mgr.suggest_node_params(signature="sig_1", cdg=cdg, catalog=cat)
        mgr.complete_trial(signature="sig_1", trial_number=s1.trial_number, loss=0.25)

        s2 = mgr.suggest_node_params(signature="sig_1", cdg=cdg, catalog=cat)
        mgr.complete_trial(signature="sig_1", trial_number=s2.trial_number, loss=0.75)

        importances = mgr.param_importances()
        assert "n1.cutoff" in importances

    @pytest.mark.asyncio
    async def test_trial1_hard_reject_populates_bottleneck_and_reuses_cached_redecompose(self) -> None:
        leaf = AlgorithmicNode(
            node_id="leaf_1",
            name="Bad Detector",
            description="Fails admissibility",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
        )
        cdg = CDGExport(nodes=[leaf], edges=[])

        class _HardRejectEvaluator:
            def evaluate(self, context: AdmissibilityContext) -> AdmissibilityReport:
                return AdmissibilityReport(
                    decisions=(
                        AdmissibilityDecision(
                            rule_id="degenerate_output",
                            disposition=AdmissibilityDisposition.HARD_REJECT,
                            summary="Output collapsed to constant",
                            severity=1.0,
                            family="generic",
                        ),
                    )
                )

        state = PrincipalState(
            goal="Detect events",
            cdg=cdg,
            benchmark=BenchmarkResult(
                global_loss=10.0,
                node_telemetry={
                    "leaf_1": NodeTelemetry(
                        node_id="leaf_1",
                        execution_time_ms=1.0,
                        peak_memory_bytes=1,
                        error_expansion=0.0,
                    )
                },
            ),
            trial_history=[{"trial": 1}],
        )
        architect = AsyncMock()
        deps = PrincipalDeps(
            architect=architect,
            sandbox=AsyncMock(),
            admissibility_evaluator=_HardRejectEvaluator(),
        )
        out = await check_admissibility(state, {"configurable": {"deps": deps}})
        assert out["admissibility_hard_rejected"] is True
        assert state.bottleneck_node_id == "leaf_1"
        assert "degenerate_output" in state.bottleneck_reason

        # Cached redecompose candidate is consumed without calling architect.fork/decompose again
        replacement_cdg = CDGExport(nodes=[leaf], edges=[])
        state.pending_redecompose_cdg = replacement_cdg
        state.pending_redecompose_thread_id = "forked_thread_99"
        tt_out = await time_travel_update(state, {"configurable": {"deps": deps}})
        assert tt_out["thread_id"] == "forked_thread_99"
        assert architect.fork.await_count == 0
        assert architect.decompose.await_count == 0


class TestTypesProviderExpansionAndSDKFixes:
    def test_match_result_roundtrip_preserves_provider_wheel_fields(self) -> None:
        mr = _match("n1", "cos")
        restored = MatchResult.from_dict(mr.to_dict())
        assert restored.verified_match is not None
        d = restored.verified_match.candidate.declaration
        assert d.provider_id == "sciona-atoms"
        assert d.distribution_name == "sciona-atoms"
        assert d.distribution_version == "0.1.0"
        assert d.install_requirement == "sciona-atoms==0.1.0"
        assert d.import_module == "math"
        assert d.import_symbol == "cos"
        assert d.wheel_url == "https://example.invalid/sciona_atoms.whl"
        assert d.wheel_sha256 == "deadbeef"

    def test_provider_expansion_loader_supports_relative_imports(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "src" / "sciona" / "atoms" / "expansion"
        exp_dir.mkdir(parents=True)
        (exp_dir / "helper_defs.py").write_text("VALUE = ('mod', 'fn', '(x) -> y')\n")
        reg_file = exp_dir / "demo_registry.py"
        reg_file.write_text(
            "from .helper_defs import VALUE\n"
            "DECLARATIONS = {'demo': VALUE}\n"
        )

        mod = _load_module(reg_file)
        assert mod is not None
        assert getattr(mod, "DECLARATIONS") == {"demo": ("mod", "fn", "(x) -> y")}

    def test_sdk_assemble_and_generate_code_produce_python_source(self) -> None:
        root = AlgorithmicNode(
            node_id="root",
            name="Math Pipe",
            description="Root",
            concept_type=ConceptType.ARITHMETIC,
            status=NodeStatus.DECOMPOSED,
            children=["step1"],
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="y", type_desc="float")],
        )
        leaf = AlgorithmicNode(
            node_id="step1",
            parent_id="root",
            name="Cosine",
            description="Compute cosine",
            concept_type=ConceptType.ARITHMETIC,
            status=NodeStatus.ATOMIC,
            matched_primitive="math.cos",
            type_signature="(x: float) -> float",
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="y", type_desc="float")],
        )
        cdg = CDGExport(nodes=[root, leaf], edges=[])
        sdk = Sciona(catalog=PrimitiveCatalog(), config=None)
        generated = sdk.generate_code(cdg)
        assert "def cosine(" in generated.source
        assert "math.cos" in generated.atom_fqdns_used
