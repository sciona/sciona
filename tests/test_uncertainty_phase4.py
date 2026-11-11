"""Phase 4 synthetic tests for runtime confidence guards & configurable trust_tier."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.synthesizer.assembler import Assembler
from sciona.synthesizer.contracts import ContractGenerator
from sciona.synthesizer.models import AssemblyUnit
from sciona.synthesizer.pipeline import assemble_and_check
from sciona.synthesizer.uncertainty import ProfileCatalogBackend
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)
from sciona.uncertainty import (
    AtomUncertaintyProfile,
    CalibrationConfig,
    ConfidenceEnvelopeViolation,
    ConformalEnvelope,
    GaussianMixtureRegime,
    IdentityProjector,
    PortCalibrationConfig,
    PortRegimeSpec,
    PositiveScaleProjector,
    Tier1ConfidenceEnvelopeViolation,
    calibrate_atom,
    clear_runtime_guard_cache,
    matches_trust_tier,
)


# ---------------------------------------------------------------------------
# Synthetic helpers & temporary dynamic test module
# ---------------------------------------------------------------------------


def _install_synthetic_atom_module(monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[str, int]]:
    """Register an in-memory synthetic module for compiled skeleton imports."""
    mod_name = "_sciona_phase4_synth_atoms"
    call_counts: dict[str, int] = {"stage1": 0, "stage2": 0, "stage3": 0}
    mod = types.ModuleType(mod_name)

    def stage1_normalize(signal: NDArray[np.float64]) -> NDArray[np.float64]:
        call_counts["stage1"] += 1
        arr = np.asarray(signal, dtype=np.float64)
        return 0.9 * arr + 0.05

    def stage2_nonlin(signal: NDArray[np.float64]) -> NDArray[np.float64]:
        call_counts["stage2"] += 1
        arr = np.asarray(signal, dtype=np.float64)
        # Mildly amplifies large values cubically so subtle shifts can blow up stage3
        return arr + 0.15 * (arr ** 3)

    def stage3_readout(signal: NDArray[np.float64]) -> NDArray[np.float64]:
        call_counts["stage3"] += 1
        arr = np.asarray(signal, dtype=np.float64)
        return np.array([float(np.mean(arr)), float(np.std(arr))], dtype=np.float64)

    mod.stage1_normalize = stage1_normalize  # type: ignore[attr-defined]
    mod.stage2_nonlin = stage2_nonlin  # type: ignore[attr-defined]
    mod.stage3_readout = stage3_readout  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, mod)
    return mod_name, call_counts


def _make_match(node_id: str, decl_name: str, sig: str = "(signal: np.ndarray) -> np.ndarray") -> MatchResult:
    source_lib = decl_name.rsplit(".", 1)[0] if "." in decl_name else ""
    decl = Declaration(
        name=decl_name,
        type_signature=sig,
        source_lib=source_lib,
        prover=Prover.PYTHON,
    )
    cand = CandidateMatch(declaration=decl, score=0.99, retrieval_method="test")
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


def _build_three_tier_cdg(mod_name: str) -> tuple[CDGExport, list[MatchResult]]:
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="root_pipe",
                name="Three Tier Pipeline",
                description="Synthetic 3-stage pipeline across trust tiers 1, 2, 3",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.DECOMPOSED,
                children=["n_tier1", "n_tier2", "n_tier3"],
                inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                outputs=[IOSpec(name="summary", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n_tier1",
                name="Stage1 Normalize",
                description="Tier 1 input stage",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="stage1_normalize",
                inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                outputs=[IOSpec(name="s1", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n_tier2",
                name="Stage2 Nonlin",
                description="Tier 2 middle stage",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="stage2_nonlin",
                inputs=[IOSpec(name="s1", type_desc="np.ndarray")],
                outputs=[IOSpec(name="s2", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n_tier3",
                name="Stage3 Readout",
                description="Tier 3 readout stage",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="stage3_readout",
                inputs=[IOSpec(name="s2", type_desc="np.ndarray")],
                outputs=[IOSpec(name="summary", type_desc="np.ndarray")],
            ),
        ],
        edges=[
            DependencyEdge(
                source_id="n_tier1",
                target_id="n_tier2",
                output_name="s1",
                input_name="s1",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="n_tier2",
                target_id="n_tier3",
                output_name="s2",
                input_name="s2",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ],
    )
    matches = [
        _make_match("n_tier1", f"{mod_name}.stage1_normalize"),
        _make_match("n_tier2", f"{mod_name}.stage2_nonlin"),
        _make_match("n_tier3", f"{mod_name}.stage3_readout"),
    ]
    return cdg, matches


def _exec_skeleton(source_code: str) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    exec(compile(source_code, "<phase4_skeleton>", "exec"), ns)
    return ns


def _make_calibrated_1d_profile(
    atom_name: str,
    *,
    port_name: str = "signal",
    trust_tier: int = 1,
    center: float = 0.0,
    scale: float = 1.0,
    n_cal: int = 249,
    seed: int = 7,
) -> AtomUncertaintyProfile:
    """Create a tightly controlled synthetic profile with exact conformal scores."""
    rng = np.random.default_rng(seed)
    proj = IdentityProjector(dim=4)
    mix = GaussianMixtureRegime(
        weights=np.array([1.0]),
        means=np.full((1, 4), center, dtype=np.float64),
        diag_vars=np.full((1, 4), scale * scale, dtype=np.float64),
    )
    cal_samples = rng.normal(loc=center, scale=scale, size=(n_cal, 4))
    cal_scores = np.array([mix.nonconformity_score(row) for row in cal_samples])
    env = ConformalEnvelope(cal_scores)
    return AtomUncertaintyProfile(
        atom_name=atom_name,
        mode="empirical",
        confidence=0.95,
        trust_tier=trust_tier,
        n_trials=n_cal,
        input_regimes={
            port_name: PortRegimeSpec(
                port_name=port_name,
                projector_spec=proj.to_spec(),
                mixture=mix,
                conformal_envelope=env,
            )
        },
    )


# ---------------------------------------------------------------------------
# 1. Unit tests for matches_trust_tier & profile-level tier configuration
# ---------------------------------------------------------------------------


def test_matches_trust_tier_and_evaluate_inputs_configurable_tiers() -> None:
    """trust_tier=None matches all tiers; int and collection filters match selectively."""
    assert matches_trust_tier(1, 1) is True
    assert matches_trust_tier(2, 1) is False
    assert matches_trust_tier(3, 1) is False

    assert matches_trust_tier(1, None) is True
    assert matches_trust_tier(2, None) is True
    assert matches_trust_tier(3, None) is True

    assert matches_trust_tier(2, (1, 2)) is True
    assert matches_trust_tier(3, (1, 2)) is False
    assert matches_trust_tier(3, {3}) is True

    # Uncalibrated Tier 3 atom: passes when trust_tier=1, fails closed when trust_tier=None or 3
    uncal_t3 = AtomUncertaintyProfile(atom_name="draft_atom", mode="empirical", trust_tier=3)
    rep_t1 = uncal_t3.evaluate_inputs({"x": np.ones(4)}, require_conformal=True, trust_tier=1)
    assert rep_t1.in_confidence_bounds is True

    rep_all = uncal_t3.evaluate_inputs({"x": np.ones(4)}, require_conformal=True, trust_tier=None)
    assert rep_all.in_confidence_bounds is False
    with pytest.raises(ConfidenceEnvelopeViolation) as exc_info:
        rep_all.raise_if_violated(node_id="n_draft", atom_name="draft_atom", trust_tier=3)
    assert isinstance(exc_info.value, Tier1ConfidenceEnvelopeViolation)
    assert exc_info.value.trust_tier == 3
    assert "Tier 3 confidence bounds exceeded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Assembler runtime guards across trust_tier=1, 2, (2, 3), and None (all)
# ---------------------------------------------------------------------------


def test_assembler_runtime_guards_tier1_vs_all_tiers_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assembler guards only Tier 1 by default and all tiers when trust_tier=None."""
    clear_runtime_guard_cache()
    mod_name, call_counts = _install_synthetic_atom_module(monkeypatch)
    cdg, matches = _build_three_tier_cdg(mod_name)

    # Calibrate stage1 (Tier 1) and stage2 (Tier 2) around ~N(0, 1),
    # while stage3 (Tier 3) is calibrated around tighter ~N(0, 0.25^2)
    profiles = {
        "stage1_normalize": _make_calibrated_1d_profile(
            "stage1_normalize", trust_tier=1, center=0.0, scale=1.0, n_cal=399, seed=11
        ),
        "stage2_nonlin": _make_calibrated_1d_profile(
            "stage2_nonlin", trust_tier=2, center=0.0, scale=1.2, n_cal=399, seed=12
        ),
        "stage3_readout": _make_calibrated_1d_profile(
            "stage3_readout", trust_tier=3, center=0.0, scale=0.25, n_cal=399, seed=13
        ),
    }

    # --- Case A: Default trust_tier=1 (only n_tier1 is guarded) ---
    assembler_t1 = Assembler(
        Prover.PYTHON,
        uncertainty_profiles=profiles,
        confidence_alpha=0.03,
    )
    skel_t1 = assembler_t1.assemble(cdg, matches)
    guard_meta_t1 = skel_t1.metadata["confidence_guards"]
    assert guard_meta_t1["trust_tier"] == 1
    assert list(guard_meta_t1["guarded_nodes"].keys()) == ["n_tier1"]
    assert guard_meta_t1["guarded_nodes"]["n_tier1"]["allocated_alpha"] == pytest.approx(0.03)

    ns_t1 = _exec_skeleton(skel_t1.source_code)
    x_nominal = np.array([0.05, -0.05, 0.02, -0.02], dtype=np.float64)
    out_ok = ns_t1["three_tier_pipeline_composition"](x_nominal)
    assert out_ok.shape == (2,)
    assert call_counts == {"stage1": 1, "stage2": 1, "stage3": 1}

    # Moderate input [0.9, -0.9, 0.8, -0.8] is well inside Stage 1's N(0, 1) envelope,
    # even though it exceeds Stage 3's tight N(0, 0.25^2) envelope -> passes under trust_tier=1!
    x_mid = np.array([0.9, -0.9, 0.8, -0.8], dtype=np.float64)
    out_mid = ns_t1["three_tier_pipeline_composition"](x_mid)
    assert out_mid.shape == (2,)
    assert call_counts == {"stage1": 2, "stage2": 2, "stage3": 2}

    # Extreme OOD input [25, 25, 25, 25] violates Stage 1 immediately; ZERO atoms execute!
    x_ood = np.full(4, 25.0, dtype=np.float64)
    with pytest.raises(Tier1ConfidenceEnvelopeViolation) as exc_t1:
        ns_t1["three_tier_pipeline_composition"](x_ood)
    assert exc_t1.value.node_id == "n_tier1"
    assert exc_t1.value.trust_tier == 1
    assert call_counts == {"stage1": 2, "stage2": 2, "stage3": 2}

    # --- Case B: Per-call override trust_tier=None -> guards ALL tiers (1, 2, 3)! ---
    skel_all = assembler_t1.assemble(cdg, matches, trust_tier=None, confidence_alpha=0.03)
    guard_meta_all = skel_all.metadata["confidence_guards"]
    assert guard_meta_all["trust_tier"] is None
    assert set(guard_meta_all["guarded_nodes"].keys()) == {"n_tier1", "n_tier2", "n_tier3"}
    for nid in ("n_tier1", "n_tier2", "n_tier3"):
        assert guard_meta_all["guarded_nodes"][nid]["allocated_alpha"] == pytest.approx(0.01)

    ns_all = _exec_skeleton(skel_all.source_code)
    # Nominal small input passes all 3 guarded tiers
    out_all_ok = ns_all["three_tier_pipeline_composition"](x_nominal)
    assert np.allclose(out_all_ok, out_ok)
    assert call_counts == {"stage1": 3, "stage2": 3, "stage3": 3}

    # x_mid passes Stage 1 and Stage 2, then aborts right before Stage 3 (Tier 3)!
    with pytest.raises(ConfidenceEnvelopeViolation) as exc_all:
        ns_all["three_tier_pipeline_composition"](x_mid)
    assert exc_all.value.node_id == "n_tier3"
    assert exc_all.value.trust_tier == 3
    # Stage 1 and Stage 2 ran, Stage 3 was prevented from executing!
    assert call_counts == {"stage1": 4, "stage2": 4, "stage3": 3}

    # --- Case C: Subset collection trust_tier=(2, 3) with weighted alpha budget ---
    skel_23 = Assembler(
        Prover.PYTHON,
        uncertainty_backend=ProfileCatalogBackend(profiles),
        trust_tier=(2, 3),
        confidence_alpha=0.03,
    ).assemble(cdg, matches, risk_weights={"n_tier2": 1.0, "n_tier3": 2.0})
    gm_23 = skel_23.metadata["confidence_guards"]
    assert gm_23["trust_tier"] == [2, 3]
    assert set(gm_23["guarded_nodes"].keys()) == {"n_tier2", "n_tier3"}
    assert gm_23["guarded_nodes"]["n_tier2"]["allocated_alpha"] == pytest.approx(0.01)
    assert gm_23["guarded_nodes"]["n_tier3"]["allocated_alpha"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 3. End-to-end empirical calibrate_atom + hard domain & fail-closed checks
# ---------------------------------------------------------------------------


def test_calibrated_positive_domain_and_fail_closed_conformal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empirically calibrated atom enforces both positive domain support and conformal N."""
    clear_runtime_guard_cache()
    mod_name = "_sciona_phase4_pos_mod"
    mod = types.ModuleType(mod_name)

    def log_scale_transform(scale: float) -> NDArray[np.float64]:
        if scale <= 0.0:
            raise ValueError("scale must be > 0")
        return np.array([ np.log(scale), np.sqrt(scale) ], dtype=np.float64)

    mod.log_scale_transform = log_scale_transform  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, mod)

    # Calibrate empirically via Phase 2 calibrate_atom on Tier 2
    prof = calibrate_atom(
        "log_scale_transform",
        log_scale_transform,
        config=CalibrationConfig(
            n_trials=200,
            fit_fraction=0.4,
            trust_tier=2,
            seed=21,
        ),
        port_configs={
            "scale": PortCalibrationConfig(
                port_name="scale",
                projector=PositiveScaleProjector(),
                constraints="> 0",
            )
        },
    )
    assert prof.trust_tier == 2

    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="root_s",
                name="Scale Root",
                description="Root",
                concept_type=ConceptType.ARITHMETIC,
                status=NodeStatus.DECOMPOSED,
                children=["n_scale"],
                inputs=[IOSpec(name="scale", type_desc="float")],
                outputs=[IOSpec(name="out", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n_scale",
                name="Scale Node",
                description="Positive scale transform",
                concept_type=ConceptType.ARITHMETIC,
                status=NodeStatus.ATOMIC,
                matched_primitive="log_scale_transform",
                inputs=[IOSpec(name="scale", type_desc="float")],
                outputs=[IOSpec(name="out", type_desc="np.ndarray")],
            ),
        ],
        edges=[],
    )
    matches = [_make_match("n_scale", f"{mod_name}.log_scale_transform", "(scale: float) -> np.ndarray")]

    skel = Assembler(
        Prover.PYTHON,
        uncertainty_profiles={"log_scale_transform": prof},
        trust_tier=2,
        confidence_alpha=0.05,
    ).assemble(cdg, matches)
    ns = _exec_skeleton(skel.source_code)

    # In-distribution positive scale succeeds
    res = ns["scale_root_composition"](1.5)
    assert np.allclose(res, [np.log(1.5), np.sqrt(1.5)])

    # Non-positive scale (-2.0) violates PositiveScaleProjector hard domain before calling atom
    with pytest.raises(ConfidenceEnvelopeViolation) as dom_exc:
        ns["scale_root_composition"](-2.0)
    assert dom_exc.value.trust_tier == 2
    assert dom_exc.value.port_name == "scale"

    # Requesting alpha=0.001 when N_cal=120 (< 999 required) fails closed under require_conformal=True
    skel_too_strict = Assembler(
        Prover.PYTHON,
        uncertainty_profiles={"log_scale_transform": prof},
        trust_tier=None,
        confidence_alpha=0.001,
        require_conformal=True,
    ).assemble(cdg, matches)
    ns_strict = _exec_skeleton(skel_too_strict.source_code)
    with pytest.raises(ConfidenceEnvelopeViolation, match="insufficient"):
        ns_strict["scale_root_composition"](1.5)

    # Same profile with require_conformal=False still blocks hard domain (-2.0) without failing N check
    skel_relaxed = Assembler(
        Prover.PYTHON,
        uncertainty_profiles={"log_scale_transform": prof},
        trust_tier=None,
        confidence_alpha=0.001,
        require_conformal=False,
    ).assemble(cdg, matches)
    ns_relaxed = _exec_skeleton(skel_relaxed.source_code)
    assert np.allclose(ns_relaxed["scale_root_composition"](1.5), [np.log(1.5), np.sqrt(1.5)])
    with pytest.raises(ConfidenceEnvelopeViolation):
        ns_relaxed["scale_root_composition"](-2.0)


# ---------------------------------------------------------------------------
# 4. Telemetry integration (_sciona_probe trace.jsonl records guard status)
# ---------------------------------------------------------------------------


def test_telemetry_probe_records_confidence_guard_pass_and_abort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Instrumented skeletons log confidence_guard diagnostics on both pass and abort."""
    clear_runtime_guard_cache()
    mod_name, _counts = _install_synthetic_atom_module(monkeypatch)
    cdg, matches = _build_three_tier_cdg(mod_name)
    profiles = {
        "stage1_normalize": _make_calibrated_1d_profile(
            "stage1_normalize", trust_tier=1, n_cal=299, seed=31
        ),
        "stage2_nonlin": _make_calibrated_1d_profile(
            "stage2_nonlin", trust_tier=2, n_cal=299, seed=32
        ),
        "stage3_readout": _make_calibrated_1d_profile(
            "stage3_readout", trust_tier=3, n_cal=299, seed=33
        ),
    }

    skel = Assembler(
        Prover.PYTHON,
        with_telemetry=True,
        uncertainty_profiles=profiles,
        trust_tier=None,
        confidence_alpha=0.03,
    ).assemble(cdg, matches)

    trace_file = tmp_path / "trace.jsonl"
    ns = _exec_skeleton(skel.source_code)
    ns["_SCIONA_TRACE_PATH"] = str(trace_file)

    # 1. Passing execution writes 3 trace records with in_confidence_bounds=True
    ns["three_tier_pipeline_composition"](np.array([0.1, -0.1, 0.05, -0.05]))
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert len(records) == 3
    assert [r["node_id"] for r in records] == ["n_tier1", "n_tier2", "n_tier3"]
    for rec in records:
        assert rec["confidence_guard"]["in_confidence_bounds"] is True
        assert "aborted_error" not in rec

    # 2. Aborting execution logs failing confidence_guard + aborted_error on n_tier1
    trace_file.write_text("")
    with pytest.raises(ConfidenceEnvelopeViolation):
        ns["three_tier_pipeline_composition"](np.full(4, 30.0))
    abort_records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert len(abort_records) == 1
    assert abort_records[0]["node_id"] == "n_tier1"
    assert abort_records[0]["confidence_guard"]["in_confidence_bounds"] is False
    assert "Tier1ConfidenceEnvelopeViolation" in abort_records[0]["aborted_error"]


# ---------------------------------------------------------------------------
# 5. ContractGenerator & assemble_and_check integration
# ---------------------------------------------------------------------------


def test_contract_generator_and_pipeline_configurable_trust_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ContractGenerator & assemble_and_check support trust_tier=1, 3, and None."""
    clear_runtime_guard_cache()
    mod_name, _counts = _install_synthetic_atom_module(monkeypatch)
    prof_t3 = _make_calibrated_1d_profile(
        "stage3_readout", trust_tier=3, center=0.0, scale=1.0, n_cal=249, seed=41
    )

    unit = AssemblyUnit(
        node_id="readout_step",
        name="Stage3 Readout",
        declaration_name=f"{mod_name}.stage3_readout",
        matched_primitive="stage3_readout",
        type_signature="(signal: np.ndarray) -> np.ndarray",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray", constraints="signal.ndim == 1")],
        outputs=[IOSpec(name="summary", type_desc="np.ndarray")],
    )
    decl = Declaration(
        name=f"{mod_name}.stage3_readout",
        type_signature="(signal: np.ndarray) -> np.ndarray",
        source_lib=mod_name,
        prover=Prover.PYTHON,
    )

    # Under trust_tier=1, a Tier 3 atom does NOT emit a guard
    gen_t1 = ContractGenerator(
        uncertainty_profiles={"stage3_readout": prof_t3},
        trust_tier=1,
        confidence_alpha=0.02,
    )
    w_t1 = gen_t1.generate_wrapper(unit, decl)
    assert w_t1.confidence_guard_spec is None

    # Under trust_tier=None (all tiers), the Tier 3 atom DOES emit a working guard!
    gen_all = ContractGenerator(
        uncertainty_profiles={"stage3_readout": prof_t3},
        trust_tier=None,
        confidence_alpha=0.02,
    )
    w_all = gen_all.generate_wrapper(unit, decl)
    assert w_all.confidence_guard_spec is not None
    rendered = "\n".join(w_all.imports) + "\nimport numpy as np\n\n" + gen_all.render_wrapper(w_all)
    ns_w: dict[str, Any] = {}
    exec(compile(rendered, "<contract_wrapper>", "exec"), ns_w)

    ok_val = ns_w["stage3_readout_wrapper"](np.array([0.1, -0.2, 0.05, 0.0]))
    assert ok_val.shape == (2,)
    with pytest.raises(ConfidenceEnvelopeViolation) as w_exc:
        ns_w["stage3_readout_wrapper"](np.full(4, 20.0))
    assert w_exc.value.trust_tier == 3

    # Also verify assemble_and_check forwards trust_tier=None cleanly
    class _DummyPythonEnv:
        prover_name = "python"

        async def check_proof(self, theorem_src: str, context_src: str = "") -> tuple[bool, str]:
            compile(theorem_src, "<env_check>", "exec")
            return True, ""

        async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
            return True, ""

        async def get_type(self, name: str) -> str | None:
            return None

        async def close(self) -> None:
            return None

    cdg, matches = _build_three_tier_cdg(mod_name)
    profiles = {
        "stage1_normalize": _make_calibrated_1d_profile("stage1_normalize", trust_tier=1, n_cal=249, seed=51),
        "stage2_nonlin": _make_calibrated_1d_profile("stage2_nonlin", trust_tier=2, n_cal=249, seed=52),
        "stage3_readout": prof_t3,
    }
    res = asyncio.run(
        assemble_and_check(
            cdg,
            matches,
            _DummyPythonEnv(),
            skip_ghost_sim=True,
            uncertainty_profiles=profiles,
            trust_tier=None,
            confidence_alpha=0.03,
        )
    )
    assert res.compiled_ok is True
    assert res.skeleton.metadata["confidence_guards"]["trust_tier"] is None
    assert set(res.skeleton.metadata["confidence_guards"]["guarded_nodes"]) == {
        "n_tier1",
        "n_tier2",
        "n_tier3",
    }
