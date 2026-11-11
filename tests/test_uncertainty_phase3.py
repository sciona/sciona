"""Phase 3 tests: CDG forward mixture pushforward, adjoint pullback, ranking, & gap specs.

All graphs, profiles, and test cases are strictly synthetic.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from sciona.api.models import BountyCreateRequest
from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicPrimitive,
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.hunter.candidate_ranker import HeuristicCandidateRanker
from sciona.principal.atom_ledger import AtomLedger
from sciona.principal.backprop import CreditAssigner
from sciona.principal.hpo import OptunaManager, TrialPrunedEarly
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.principal.variant_mutation import maybe_apply_bottleneck_variant
from sciona.synthesizer.ghost_sim import run_ghost_simulation
from sciona.synthesizer.uncertainty import ProfileCatalogBackend
from sciona.types import CandidateMatch, Declaration, MatchResult, PDGNode, VerificationResult
from sciona.uncertainty import (
    AtomUncertaintyProfile,
    ComponentChannel,
    ConformalEnvelope,
    GaussianMixtureRegime,
    IdentityProjector,
    LowRankPlusTailOperator,
    PortRegimeSpec,
    RiskBudget,
    mine_cdg_uncertainty_gaps,
    propagate_cdg_uncertainty,
    rank_candidates_by_uncertainty,
)


def _make_match(node_id: str, atom_name: str) -> MatchResult:
    decl = Declaration(name=atom_name, type_signature="(x: np.ndarray) -> np.ndarray")
    cand = CandidateMatch(declaration=decl, score=0.95, retrieval_method="test")
    ver = VerificationResult(candidate=cand, verified=True)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="(x: np.ndarray) -> np.ndarray"),
        verified_match=ver,
        all_candidates=[cand],
        all_verifications=[ver],
    )


def test_directional_orthogonality_vs_aligned_stiff_subspace() -> None:
    """Orthogonal stiff subspaces pass risk budget where aligned subspaces fail."""
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n_up",
                name="Upstream Stage",
                description="Amplifies e_0 only",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="up_atom",
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n_down",
                name="Downstream Stage",
                description="Sensitive along e_1 or e_0",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="down_orth",
                inputs=[IOSpec(name="y", type_desc="np.ndarray")],
                outputs=[IOSpec(name="z", type_desc="np.ndarray")],
            ),
        ],
        edges=[
            DependencyEdge(
                source_id="n_up",
                target_id="n_down",
                output_name="y",
                input_name="y",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )

    # Upstream amplifies e_0 by 8.0, attenuates e_1, e_2 to 0.1
    j_up = np.diag([8.0, 0.1, 0.1])
    prof_up = AtomUncertaintyProfile(
        atom_name="up_atom",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(j_up, max_rank=3)},
            ),
        ),
    )

    # Orthogonal downstream: stiff gain 2.5 along e_1, tiny tail 0.01 on e_0 & e_2
    j_down_orth = np.diag([0.01, 2.5, 0.01])
    prof_down_orth = AtomUncertaintyProfile(
        atom_name="down_orth",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("y", "z"): LowRankPlusTailOperator.from_dense(j_down_orth, max_rank=3)},
            ),
        ),
    )

    # Aligned downstream: stiff gain 2.5 along e_0 (collides with upstream amplification!)
    j_down_aligned = np.diag([2.5, 0.01, 0.01])
    prof_down_aligned = AtomUncertaintyProfile(
        atom_name="down_aligned",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("y", "z"): LowRankPlusTailOperator.from_dense(j_down_aligned, max_rank=3)},
            ),
        ),
    )

    budget = RiskBudget(max_output_variance=0.05, default_source_variance=0.25)

    rep_orth = propagate_cdg_uncertainty(
        cdg,
        profiles={"up_atom": prof_up, "down_orth": prof_down_orth},
        risk_budget=budget,
    )
    assert rep_orth.meets_budget
    assert rep_orth.max_sink_risk_ratio < 0.4

    cdg_aligned = cdg.model_copy(
        update={
            "nodes": [
                cdg.nodes[0],
                cdg.nodes[1].model_copy(update={"matched_primitive": "down_aligned"}),
            ]
        }
    )
    rep_aligned = propagate_cdg_uncertainty(
        cdg_aligned,
        profiles={"up_atom": prof_up, "down_aligned": prof_down_aligned},
        risk_budget=budget,
    )
    assert not rep_aligned.meets_budget
    assert rep_aligned.max_sink_risk_ratio > 100.0


def test_mixture_kl_reduction_and_regime_ood_detection() -> None:
    """Multi-mode mixtures compress via Runnalls kl_reduce and flag OOD upstream shifts."""
    proj_spec = IdentityProjector(dim=2).to_spec()
    cal_mix = GaussianMixtureRegime(
        weights=[0.5, 0.5],
        means=[[-2.0, 0.0], [2.0, 0.0]],
        diag_vars=[[0.25, 0.25], [0.25, 0.25]],
    )
    rng = np.random.default_rng(77)
    cal_draws = cal_mix.sample(150, rng=rng)
    cal_scores = cal_mix.nonconformity_score(cal_draws)
    env = ConformalEnvelope(cal_scores)

    bimodal_prof = AtomUncertaintyProfile(
        atom_name="bimodal_op",
        mode="empirical",
        confidence=0.92,
        input_regimes={
            "x": PortRegimeSpec(
                port_name="x",
                projector_spec=proj_spec,
                mixture=cal_mix,
                conformal_envelope=env,
            )
        },
        output_projectors={"y": proj_spec},
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(0.5 * np.eye(2))},
            ),
            ComponentChannel(
                component_index=1,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(2.0 * np.eye(2))},
            ),
        ),
    )

    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Bimodal Node",
                description="Two regimes",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="bimodal_op",
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            )
        ],
        edges=[],
    )

    # 3-component source x 2-component atom = 6 raw branches -> reduced to max_mixture_components=3
    src_3 = GaussianMixtureRegime(
        weights=[0.4, 0.4, 0.2],
        means=[[-2.1, 0.1], [1.9, -0.1], [-1.8, -0.1]],
        diag_vars=[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
    )
    rep_in = propagate_cdg_uncertainty(
        cdg,
        profiles={"bimodal_op": bimodal_prof},
        source_regimes={"x": src_3},
        risk_budget=RiskBudget(max_mixture_components=3, global_alpha=0.05),
    )
    assert rep_in.nodes["n1"].in_regime_bounds
    assert rep_in.nodes["n1"].outgoing_mixtures["y"].n_components == 3

    # OOD source far from both calibrated clusters triggers regime_violated_nodes
    src_ood = GaussianMixtureRegime(
        weights=[1.0],
        means=[[50.0, -50.0]],
        diag_vars=[[0.1, 0.1]],
    )
    rep_ood = propagate_cdg_uncertainty(
        cdg,
        profiles={"bimodal_op": bimodal_prof},
        source_regimes={"x": src_ood},
        risk_budget=RiskBudget(global_alpha=0.05),
    )
    assert "n1" in rep_ood.regime_violated_nodes
    assert not rep_ood.meets_budget


def test_infeasible_pullback_credit_assignment_and_optuna_early_prune() -> None:
    """Intrinsic noise exceeding sink precision triggers pullback infeasibility & early prune."""
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Clean Preprocessor",
                description="Low-noise scaling",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="h", type_desc="np.ndarray")],
            ),
            AlgorithmicNode(
                node_id="n2",
                name="Noisy Estimator",
                description="Injects large intrinsic approximation noise",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                inputs=[IOSpec(name="h", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            ),
        ],
        edges=[
            DependencyEdge(
                source_id="n1",
                target_id="n2",
                output_name="h",
                input_name="h",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )

    prof_clean = AtomUncertaintyProfile(
        atom_name="clean_prep",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "h"): LowRankPlusTailOperator.from_dense(0.5 * np.eye(2))},
                intrinsic_output_cov={"h": 1e-4 * np.eye(2)},
            ),
        ),
    )
    prof_noisy = AtomUncertaintyProfile(
        atom_name="noisy_est",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("h", "y"): LowRankPlusTailOperator.from_dense(0.8 * np.eye(2))},
                # Intrinsic variance 0.40 exceeds sink max_output_variance=0.10!
                intrinsic_output_cov={"y": 0.40 * np.eye(2)},
            ),
        ),
    )

    backend = ProfileCatalogBackend({"clean_prep": prof_clean, "noisy_est": prof_noisy})
    matches = [_make_match("n1", "clean_prep"), _make_match("n2", "noisy_est")]
    budget = RiskBudget(max_output_variance=0.10, default_source_variance=0.05)

    sim_report = run_ghost_simulation(
        cdg,
        matches,
        uncertainty_backend=backend,
        risk_budget=budget,
    )
    assert sim_report.infeasible_pullback_nodes == ["n2"]

    # CreditAssigner identifies n2 as top precision bottleneck with pullback infeasibility reason
    grads = CreditAssigner().compute_gradients(
        cdg,
        BenchmarkResult(global_loss=1.0),
        sim_report,
        OptimizationMetric.PRECISION,
    )
    assert len(grads) >= 1
    assert grads[0].node_id == "n2"
    assert "pullback infeasible" in grads[0].bottleneck_reason

    # OptunaManager prunes trial immediately due to mathematically unreachable precision budget
    with pytest.raises(TrialPrunedEarly, match="Infeasible precision pullback"):
        OptunaManager.check_early_prune(sim_report)


def test_candidate_selection_ledger_variant_and_heuristic_ranker() -> None:
    """Uncertainty priors steer LedgerVariantFamily and HeuristicCandidateRanker to low-risk atoms."""
    prof_noisy = AtomUncertaintyProfile(
        atom_name="filter_unstable",
        mode="empirical",
        confidence=0.9,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(5.0 * np.eye(2))},
                intrinsic_output_cov={"y": 0.25 * np.eye(2)},
            ),
        ),
    )
    prof_stable = AtomUncertaintyProfile(
        atom_name="filter_stabilized",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(0.4 * np.eye(2))},
                intrinsic_output_cov={"y": 0.005 * np.eye(2)},
            ),
        ),
    )
    backend = ProfileCatalogBackend(
        {"filter_unstable": prof_noisy, "filter_stabilized": prof_stable}
    )

    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n_filt",
                name="Filter Stage",
                description="Denoise signal",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="filter_unstable",
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            )
        ],
        edges=[],
    )
    u_rep = propagate_cdg_uncertainty(
        cdg,
        backend=backend,
        risk_budget=RiskBudget(max_output_variance=0.08, default_source_variance=0.2),
    )
    ranked = rank_candidates_by_uncertainty(
        ["filter_unstable", "filter_stabilized"],
        node_eval=u_rep.nodes["n_filt"],
        backend=backend,
    )
    assert ranked[0].atom_name == "filter_stabilized"
    assert ranked[0].meets_budget
    assert not ranked[1].meets_budget

    # LedgerVariantFamily cold-start swap driven by uncertainty_report + backend
    catalog = PrimitiveCatalog()
    for pname in ("filter_unstable", "filter_stabilized"):
        catalog.add(
            AlgorithmicPrimitive(
                name=pname,
                source="synthetic",
                category=ConceptType.SIGNAL_FILTER,
                description="Synthetic filter",
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            )
        )

    mut_res = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Filter Stage",
        atom_ledger=AtomLedger(),
        catalog=catalog,
        uncertainty_backend=backend,
        uncertainty_report=u_rep,
    )
    assert mut_res.applied
    assert mut_res.variant_name == "filter_stabilized"

    # HeuristicCandidateRanker breaks an exact lexical tie in favor of filter_stabilized
    class _NeverCalledFallback:
        async def complete(self, _sys: str, _usr: str) -> str:
            raise AssertionError("Fallback should not be called when uncertainty breaks tie")

    ranker = HeuristicCandidateRanker(
        _NeverCalledFallback(),
        min_margin=0.05,
        uncertainty_backend=backend,
        node_eval=u_rep.nodes["n_filt"],
    )
    prompt = (
        "Statement: (x: ndarray) -> ndarray\n"
        "Description: apply filter to signal\n"
        "[0] filter_unstable : (x: ndarray) -> ndarray\n"
        "[1] filter_stabilized : (x: ndarray) -> ndarray\n"
    )
    choice = json.loads(asyncio.run(ranker.complete("system", prompt)))
    assert choice[0] == 1


def test_constructive_gap_mining_and_bounty_spec_generation() -> None:
    """Gap miner distinguishes catalog-rescuable slots from novel bounty specifications."""
    prof_high_gain = AtomUncertaintyProfile(
        atom_name="coarse_solver",
        mode="empirical",
        confidence=0.9,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(6.0 * np.eye(2))},
                intrinsic_output_cov={"y": 0.02 * np.eye(2)},
            ),
        ),
    )
    prof_rescuing = AtomUncertaintyProfile(
        atom_name="fine_solver",
        mode="empirical",
        confidence=0.95,
        components=(
            ComponentChannel(
                component_index=0,
                operators={("x", "y"): LowRankPlusTailOperator.from_dense(0.2 * np.eye(2))},
                intrinsic_output_cov={"y": 0.001 * np.eye(2)},
            ),
        ),
    )
    profiles = {"coarse_solver": prof_high_gain, "fine_solver": prof_rescuing}

    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n_solve",
                name="Linear Solver",
                description="Solve linear system",
                concept_type=ConceptType.ALGEBRA,
                status=NodeStatus.ATOMIC,
                matched_primitive="coarse_solver",
                inputs=[IOSpec(name="x", type_desc="np.ndarray")],
                outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            )
        ],
        edges=[],
    )
    u_rep = propagate_cdg_uncertainty(
        cdg,
        profiles=profiles,
        risk_budget=RiskBudget(max_output_variance=0.05, default_source_variance=0.25, global_alpha=0.01),
    )
    assert not u_rep.meets_budget

    # 1) When fine_solver is available in candidate pool, slot is rescued without a gap
    rescued_rep = mine_cdg_uncertainty_gaps(
        cdg,
        u_rep,
        candidates_by_node={"n_solve": ["coarse_solver", "fine_solver"]},
        profiles=profiles,
    )
    assert len(rescued_rep.gaps) == 0
    assert rescued_rep.rescued_by_existing_candidates == {"n_solve": "fine_solver"}

    # 2) When no candidate meets the budget, emits constructive UncertaintyGapSpec & BountyCreateRequest
    gap_rep = mine_cdg_uncertainty_gaps(
        cdg,
        u_rep,
        candidates_by_node={"n_solve": ["coarse_solver"]},
        profiles=profiles,
    )
    assert len(gap_rep.gaps) == 1
    gap = gap_rep.gaps[0]
    assert gap.node_id == "n_solve"
    assert gap.failure_mode == "stiff_jacobian_amplification"
    assert gap.max_stiff_singular_value == pytest.approx(np.sqrt(0.5 / (20.0 * 0.25)), rel=1e-4)
    assert gap.max_intrinsic_noise_eigenvalues["y"] == pytest.approx(0.025, rel=1e-4)
    assert gap.min_calibration_trials == 99

    bounty_dict = gap.to_bounty_request_dict(escrow_amount=750.0, domain_tags=["linalg"])
    validated_bounty = BountyCreateRequest(**bounty_dict)
    assert validated_bounty.escrow_amount == 750.0
    assert validated_bounty.config_yml["verification_kind"] == "uncertainty_profile_v2"

    cluster = gap.to_expansion_gap_cluster()
    assert cluster.cluster_id == "uncertainty_gap:n_solve"
    assert cluster.recommended_action == "develop_certified_atom"
