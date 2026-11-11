"""Mixture-of-Regimes and Spectral Channel Uncertainty Engine (Phase 1 Core).

Provides:
- ``ExponentialFamilyMixture`` base class and canonical regime subclasses
  (``GaussianMixtureRegime``, ``CountMixtureRegime``, ``GammaMixtureRegime``,
  ``SimplexMixtureRegime``).
- ``SufficientStatProjector`` embeddings mapping arbitrary runtime arrays/signals
  to compact sufficient-statistic vectors.
- ``LowRankPlusTailOperator`` for stiff/sloppy singular structure, covariance
  pushforward, Stein distributional Jacobians, and Fisher precision pullbacks.
- ``ConformalEnvelope`` for distribution-free finite-sample Tier 1 OOD bounds.
- ``AtomUncertaintyProfile`` unifying v2 ``uncertainty.json`` serialization with
  v1 ``AtomUncertaintyEstimate`` compatibility.
"""

from __future__ import annotations

from sciona.uncertainty.calibration import (
    CalibrationConfig,
    CalibrationError,
    PortCalibrationConfig,
    calibrate_atom,
    introspect_atom_signature,
    sample_port_regime_value,
    try_calibrate_atom,
)
from sciona.uncertainty.conformal import (
    ConfidenceEnvelopeViolation,
    ConformalEnvelope,
    EnvelopeCheckResult,
    Tier1ConfidenceEnvelopeViolation,
    TrustTierFilter,
    allocate_dag_alpha_budget,
    matches_trust_tier,
)
from sciona.uncertainty.distributions import (
    CountMixtureRegime,
    ExponentialFamilyMixture,
    GammaMixtureRegime,
    GaussianMixtureRegime,
    SimplexMixtureRegime,
    mixture_from_dict,
)
from sciona.uncertainty.gap_spec import (
    CandidateUncertaintyScore,
    UncertaintyGapReport,
    UncertaintyGapSpec,
    mine_cdg_uncertainty_gaps,
    rank_candidates_by_uncertainty,
    score_candidate_for_node,
    uncertainty_priors_for_candidates,
)
from sciona.uncertainty.models import (
    AtomUncertaintyProfile,
    ComponentChannel,
    InputEvaluationReport,
    PortEvaluation,
    PortRegimeSpec,
    dump_uncertainty_profile_json,
    load_uncertainty_profile_json,
)
from sciona.uncertainty.operators import (
    LowRankPlusTailOperator,
    PullbackResult,
    SpectralSummary,
)
from sciona.uncertainty.projectors import (
    CountEventProjector,
    GenericNumericProjector,
    IdentityProjector,
    MatrixConditionProjector,
    PositiveScaleProjector,
    ProjectionResult,
    ProjectorSpec,
    SimplexProjector,
    SufficientStatProjector,
    WaveformProjector,
    infer_projector,
    projector_from_spec,
)
from sciona.uncertainty.propagation import (
    CDGUncertaintyReport,
    NodeUncertaintyEvaluation,
    RiskBudget,
    compute_risk_ratio,
    propagate_cdg_uncertainty,
    resolve_atom_profile,
)
from sciona.uncertainty.runtime_guard import (
    ConfidenceGuardConfig,
    NodeGuardSpec,
    clear_runtime_guard_cache,
    evaluate_runtime_guard,
    plan_cdg_confidence_guards,
    resolve_unit_profile,
    slim_profile_for_guard,
)

__all__ = [
    "AtomUncertaintyProfile",
    "CDGUncertaintyReport",
    "CalibrationConfig",
    "CalibrationError",
    "CandidateUncertaintyScore",
    "ComponentChannel",
    "ConfidenceEnvelopeViolation",
    "ConfidenceGuardConfig",
    "ConformalEnvelope",
    "CountEventProjector",
    "CountMixtureRegime",
    "EnvelopeCheckResult",
    "ExponentialFamilyMixture",
    "GammaMixtureRegime",
    "GaussianMixtureRegime",
    "GenericNumericProjector",
    "IdentityProjector",
    "InputEvaluationReport",
    "LowRankPlusTailOperator",
    "MatrixConditionProjector",
    "NodeGuardSpec",
    "NodeUncertaintyEvaluation",
    "PortCalibrationConfig",
    "PortEvaluation",
    "PortRegimeSpec",
    "PositiveScaleProjector",
    "ProjectionResult",
    "ProjectorSpec",
    "PullbackResult",
    "RiskBudget",
    "SimplexMixtureRegime",
    "SimplexProjector",
    "SpectralSummary",
    "SufficientStatProjector",
    "Tier1ConfidenceEnvelopeViolation",
    "TrustTierFilter",
    "UncertaintyGapReport",
    "UncertaintyGapSpec",
    "WaveformProjector",
    "allocate_dag_alpha_budget",
    "calibrate_atom",
    "clear_runtime_guard_cache",
    "compute_risk_ratio",
    "dump_uncertainty_profile_json",
    "evaluate_runtime_guard",
    "infer_projector",
    "introspect_atom_signature",
    "load_uncertainty_profile_json",
    "matches_trust_tier",
    "mine_cdg_uncertainty_gaps",
    "mixture_from_dict",
    "plan_cdg_confidence_guards",
    "projector_from_spec",
    "propagate_cdg_uncertainty",
    "rank_candidates_by_uncertainty",
    "resolve_atom_profile",
    "resolve_unit_profile",
    "sample_port_regime_value",
    "score_candidate_for_node",
    "slim_profile_for_guard",
    "try_calibrate_atom",
    "uncertainty_priors_for_candidates",
]
