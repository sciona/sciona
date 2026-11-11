"""Cross-domain 100% synthetic executable prototyping benchmarks.

Provides 4 self-contained, deterministic multi-rung prototyping benchmarks
used by both the Visualizer Live Prototyping API (``/api/cdg/proto/*``) and
the automated Prototyping Speed & Quality Gate (``benchmark_gate.py``):

1. ``waveform_event_cadence``: 1D composite pulse waveform with oscillatory
   interference & artifact interval -> Rung 1 filter HPO + Rung 2 robust sibling swap.
2. ``stateful_tabular_classifier``: Stateful ``fit -> model_state -> predict``
   binary classification with extreme sensor spikes -> Rung 3 DPO outlier-clipper
   expansion + Rung 1 regularization HPO.
3. ``calibrated_linear_solver``: Ill-conditioned SPD linear system with Phase 3
   calibrated ``AtomUncertaintyProfile``s & ``RiskBudget`` -> Tier 0/Rung 2
   uncertainty-guided stabilized solver swap + Rung 1 damping HPO.
4. ``novel_nonlinear_micro_atom``: Piecewise deadzone + cubic sensor calibration
   with a true catalog gap -> Rung 4b ``@icontract`` micro-atom synthesis +
   Tier 1 metamorphic check + standalone ``.py`` export.

Strictly synthetic: zero external datasets, files, or real-world identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.graph_rewriter import Morphism, RewriteRule
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
    StatePortSpec,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
)
from sciona.prototyping.funnel import FunnelConfig
from sciona.prototyping.harness import (
    EvalCase,
    MetamorphicRelation,
    ResidentSliceHarness,
)
from sciona.prototyping.ladder import MicroAtomSynthesisSpec
from sciona.prototyping.live_runner import LiveCDGRunner
from sciona.prototyping.session import PrototypingSession
from sciona.uncertainty.distributions import GaussianMixtureRegime
from sciona.uncertainty.models import (
    AtomUncertaintyProfile,
    ComponentChannel,
    PortRegimeSpec,
)
from sciona.uncertainty.operators import LowRankPlusTailOperator
from sciona.uncertainty.projectors import IdentityProjector
from sciona.uncertainty.propagation import RiskBudget

_MOD = __name__


# ---------------------------------------------------------------------------
# Port & Profile Builders
# ---------------------------------------------------------------------------


def _arr_port(
    name: str,
    *,
    ndim: int = 1,
    required: bool = True,
    default_repr: str = "",
) -> IOSpec:
    return IOSpec(
        name=name,
        type_desc="np.ndarray",
        required=required,
        default_value_repr=default_repr,
        port_type=PortTypeSpec(
            raw="np.ndarray",
            kind="array",
            dtype="float",
            ndim=ndim,
        ),
    )


def _scalar_port(
    name: str,
    type_desc: str = "float",
    *,
    required: bool = False,
    default_repr: str = "1.0",
) -> IOSpec:
    return IOSpec(
        name=name,
        type_desc=type_desc,
        required=required,
        default_value_repr=default_repr,
        port_type=PortTypeSpec(raw=type_desc, kind="scalar", dtype=type_desc),
    )


def _calibrated_profile(
    atom_name: str,
    *,
    in_port: str,
    out_port: str,
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
        input_regimes={
            in_port: PortRegimeSpec(
                port_name=in_port, projector_spec=proj_spec, mixture=mix
            )
        },
        output_projectors={out_port: proj_spec},
        components=(
            ComponentChannel(
                component_index=0,
                operators={(in_port, out_port): op},
                intrinsic_output_cov={
                    out_port: np.array([[intrinsic_var]], dtype=np.float64)
                },
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Domain 1: 1D Waveform Conditioning & Event Cadence Estimation
# ---------------------------------------------------------------------------


def waveform_ema_conditioner(
    signal: np.ndarray[Any, Any],
    alpha: float = 0.88,
) -> np.ndarray[Any, Any]:
    """First-order zero-lag bidirectional exponential smoother (alpha in (0, 1])."""
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    a = float(np.clip(alpha, 0.02, 1.0))
    fwd = np.empty_like(arr)
    fwd[0] = arr[0]
    for i in range(1, arr.size):
        fwd[i] = a * arr[i] + (1.0 - a) * fwd[i - 1]
    bwd = np.empty_like(arr)
    bwd[-1] = fwd[-1]
    for i in range(arr.size - 2, -1, -1):
        bwd[i] = a * fwd[i] + (1.0 - a) * bwd[i + 1]
    return bwd


def waveform_peak_detector(
    filtered: np.ndarray[Any, Any],
    threshold: float = 0.42,
) -> np.ndarray[Any, Any]:
    """Detect local peak sample indices above threshold with 15-sample refractory window."""
    arr = np.asarray(filtered, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return np.array([0.0, 100.0], dtype=np.float64)
    peaks: list[int] = []
    last_idx = -100
    for idx in range(1, arr.size - 1):
        if arr[idx] > threshold and arr[idx] >= arr[idx - 1] and arr[idx] > arr[idx + 1]:
            if idx - last_idx >= 15:
                peaks.append(idx)
                last_idx = idx
    if len(peaks) < 2:
        return np.array([0.0, max(1.0, float(arr.size - 1))], dtype=np.float64)
    return np.asarray(peaks, dtype=np.float64)


def waveform_mean_cadence(
    peaks: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Naive mean instantaneous cadence at 200 Hz (events/min) — vulnerable to outlier intervals."""
    p = np.asarray(peaks, dtype=np.float64).reshape(-1)
    if p.size < 2:
        return np.array([0.0], dtype=np.float64)
    diffs = np.maximum(np.diff(p), 1.0)
    inst_rates = 60.0 * 200.0 / diffs
    return np.array([float(np.mean(inst_rates))], dtype=np.float64)


def waveform_median_cadence(
    peaks: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Robust median instantaneous cadence at 200 Hz (events/min) — rejects isolated ectopic spikes."""
    p = np.asarray(peaks, dtype=np.float64).reshape(-1)
    if p.size < 2:
        return np.array([0.0], dtype=np.float64)
    diffs = np.maximum(np.diff(p), 1.0)
    inst_rates = 60.0 * 200.0 / diffs
    return np.array([float(np.median(inst_rates))], dtype=np.float64)


def _synthesize_pulse_waveform(
    *,
    cadence_epm: float,
    fs: float = 200.0,
    duration_s: float = 10.0,
    hf_amp: float = 0.52,
    artifact_at_s: float | None = 4.35,
    seed: int = 0,
) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n, dtype=np.float64) / fs
    period_s = 60.0 / cadence_epm
    sig = np.zeros(n, dtype=np.float64)
    center = 0.4
    while center < duration_s - 0.2:
        jittered = center + float(rng.normal(0.0, 0.008))
        sig += 1.15 * np.exp(-0.5 * ((t - jittered) / 0.032) ** 2)
        center += period_s
    # High-frequency narrow-band interference (causes false local maxima unless smoothed)
    sig += hf_amp * np.sin(2.0 * np.pi * 28.0 * t)
    sig += 0.03 * rng.normal(size=n)
    # Single genuine ectopic pulse creating one spurious short interval (~300+ epm!)
    if artifact_at_s is not None:
        sig += 1.25 * np.exp(-0.5 * ((t - artifact_at_s) / 0.032) ** 2)
    return sig


# ---------------------------------------------------------------------------
# Domain 2: Stateful Tabular ML (fit -> model_state -> predict)
# ---------------------------------------------------------------------------


def tabular_pass_features(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Raw unconditioned numeric feature matrix passthrough."""
    arr = np.asarray(features, dtype=np.float64)
    return arr if arr.ndim == 2 else arr.reshape(-1, 1)


def tabular_iqr_clipper(
    features: np.ndarray[Any, Any],
    clip_bound: float = 3.5,
) -> np.ndarray[Any, Any]:
    """Deterministic feature spike clipper protecting linear/logistic estimators."""
    arr = np.asarray(features, dtype=np.float64)
    mat = arr if arr.ndim == 2 else arr.reshape(-1, 1)
    b = max(0.5, float(clip_bound))
    return np.clip(mat, -b, b)


def tabular_fit_ridge_logit(
    train_x: np.ndarray[Any, Any],
    train_y: np.ndarray[Any, Any],
    l2_reg: float = 1e-4,
) -> np.ndarray[Any, Any]:
    """Fit L2-regularized linear-logit weight vector (explicit stateful output)."""
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    n = min(x.shape[0], y.size)
    if n == 0:
        return np.zeros(x.shape[1] + 1, dtype=np.float64)
    x = x[:n]
    y_centered = 2.0 * y[:n] - 1.0
    design = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    reg = max(1e-6, float(l2_reg))
    gram = design.T @ design + reg * np.eye(design.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, design.T @ y_centered)
    return np.asarray(weights, dtype=np.float64)


def tabular_predict_probs(
    model_state: np.ndarray[Any, Any],
    eval_x: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Predict calibrated binary probabilities from fitted model_state weights."""
    w = np.asarray(model_state, dtype=np.float64).reshape(-1)
    x = np.asarray(eval_x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
    if w.size != design.shape[1]:
        # Graceful dimension adaptation for synthetic Tier 1 smoke probes
        w_adj = np.zeros(design.shape[1], dtype=np.float64)
        k = min(w.size, design.shape[1])
        w_adj[:k] = w[:k]
        w = w_adj
    with np.errstate(all="ignore"):
        logits = np.clip(design @ w, -30.0, 30.0)
        logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    return 1.0 / (1.0 + np.exp(-2.2 * logits))


class _TabularOutlierClipRuleSet:
    """Deterministic DPO rule splicing ``tabular_iqr_clipper`` on extreme feature spikes."""

    @property
    def name(self) -> str:
        return "tabular_outlier_clip_rules"

    @property
    def domain(self) -> str:
        return "tabular_ml"

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        if any("tabular_iqr_clipper" in (n.matched_primitive or "") for n in cdg.nodes):
            return []
        raw_x = context.intermediates.get("prep_train")
        max_abs = 0.0
        if isinstance(raw_x, np.ndarray) and raw_x.size > 0:
            max_abs = float(np.max(np.abs(raw_x)))
        elif isinstance(raw_x, dict):
            max_abs = abs(float(raw_x.get("max", 0.0)))
        if max_abs > 12.0:
            return [
                ExpansionDiagnostic(
                    rule_name="insert_tabular_outlier_clipper",
                    severity=0.95,
                    evidence=f"Training feature spike max_abs={max_abs:.1f} > 12.0",
                    metric_name="max_abs_feature",
                    metric_value=max_abs,
                    threshold=12.0,
                    source_domain=self.domain,
                )
            ]
        return []

    def rules(self) -> list[RewriteRule]:
        p_prep = AlgorithmicNode(
            node_id="p_prep",
            name="prep",
            description="feature prep",
            concept_type=ConceptType.DATA_ASSEMBLY,
            matched_primitive=f"{_MOD}.tabular_pass_features",
            status=NodeStatus.ATOMIC,
            inputs=[_arr_port("features", ndim=2)],
            outputs=[_arr_port("train_x", ndim=2)],
        )
        p_fit = AlgorithmicNode(
            node_id="p_fit",
            name="fit",
            description="fit estimator",
            concept_type=ConceptType.CUSTOM,
            matched_primitive=f"{_MOD}.tabular_fit_ridge_logit",
            status=NodeStatus.ATOMIC,
            inputs=[
                _arr_port("train_x", ndim=2),
                _arr_port("train_y", ndim=1),
                _scalar_port("l2_reg", default_repr="0.0001"),
            ],
            outputs=[_arr_port("model_state", ndim=1)],
        )
        p_clip = AlgorithmicNode(
            node_id="clip_train_outliers",
            name="Clip Feature Spikes",
            description="Bound extreme tabular sensor spikes prior to fitting",
            concept_type=ConceptType.SIGNAL_FILTER,
            matched_primitive=f"{_MOD}.tabular_iqr_clipper",
            status=NodeStatus.ATOMIC,
            inputs=[
                _arr_port("features", ndim=2),
                _scalar_port("clip_bound", default_repr="3.5"),
            ],
            outputs=[_arr_port("train_x", ndim=2)],
        )
        lhs = CDGExport(
            nodes=[p_prep, p_fit],
            edges=[
                DependencyEdge(
                    source_id="p_prep",
                    target_id="p_fit",
                    output_name="train_x",
                    input_name="train_x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                )
            ],
        )
        interface = CDGExport(nodes=[p_prep, p_fit], edges=[])
        rhs = CDGExport(
            nodes=[p_prep, p_clip, p_fit],
            edges=[
                DependencyEdge(
                    source_id="p_prep",
                    target_id="clip_train_outliers",
                    output_name="train_x",
                    input_name="features",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
                DependencyEdge(
                    source_id="clip_train_outliers",
                    target_id="p_fit",
                    output_name="train_x",
                    input_name="train_x",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
            ],
        )
        m = Morphism(node_map={"p_prep": "p_prep", "p_fit": "p_fit"}, edge_map={})
        return [
            RewriteRule(
                name="insert_tabular_outlier_clipper",
                lhs=lhs,
                rhs=rhs,
                interface=interface,
                l_morphism=m,
                r_morphism=m,
                priority=10,
            )
        ]


# ---------------------------------------------------------------------------
# Domain 3: Ill-Conditioned Linear System + Calibrated Uncertainty Rescue
# ---------------------------------------------------------------------------


def lin_precondition_rhs(rhs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Scale-preserving RHS conditioning stage."""
    return np.asarray(rhs, dtype=np.float64).reshape(-1)


def _hilbert_like_operator(n: int) -> np.ndarray[Any, Any]:
    idx = np.arange(1, n + 1, dtype=np.float64)
    u = np.cos(np.outer(idx - 0.5, idx - 1.0) * (np.pi / max(1, n))) * np.sqrt(2.0 / max(1, n))
    u[:, 0] /= np.sqrt(2.0)
    s = np.geomspace(1.0, 2e-4, num=n)
    with np.errstate(all="ignore"):
        return np.asarray((u * s) @ u.T, dtype=np.float64)


def lin_solve_unregularized_normal_eq(
    rhs_cond: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Unregularized normal-equation solver on fixed ill-conditioned Hilbert-like operator."""
    b = np.asarray(rhs_cond, dtype=np.float64).reshape(-1)
    n = b.size
    if n == 0:
        return b.copy()
    a = _hilbert_like_operator(n)
    with np.errstate(all="ignore"):
        ata = a.T @ a + 1e-12 * np.eye(n, dtype=np.float64)
        rhs_vec = np.nan_to_num(a.T @ b, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.linalg.solve(ata, rhs_vec)


def lin_solve_tikhonov_damped(
    rhs_cond: np.ndarray[Any, Any],
    damping: float = 0.25,
) -> np.ndarray[Any, Any]:
    """Calibrated Tikhonov-regularized solver stabilizing small singular modes."""
    b = np.asarray(rhs_cond, dtype=np.float64).reshape(-1)
    n = b.size
    if n == 0:
        return b.copy()
    a = _hilbert_like_operator(n)
    lam = max(1e-5, float(damping))
    with np.errstate(all="ignore"):
        ata = a.T @ a + (lam ** 2) * np.eye(n, dtype=np.float64)
        rhs_vec = np.nan_to_num(a.T @ b, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.linalg.solve(ata, rhs_vec)


def lin_post_project(sol_raw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Readout projection stage."""
    return np.asarray(sol_raw, dtype=np.float64).reshape(-1)


def _hilbert_like_forward(x_true: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    a = _hilbert_like_operator(x_true.size)
    with np.errstate(all="ignore"):
        return np.asarray(a @ x_true, dtype=np.float64)


# ---------------------------------------------------------------------------
# Domain 4: Nonlinear Sensor Deadzone Calibration (Rung 4b Micro-Atom Gap)
# ---------------------------------------------------------------------------


def sensor_gain_stage(raw_volts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Normalize raw sensor frame voltage."""
    return np.asarray(raw_volts, dtype=np.float64) * 0.5


def sensor_linear_placeholder(scaled_volts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Naive linear placeholder missing soft-cubic saturation compensation."""
    return np.asarray(scaled_volts, dtype=np.float64)


def sensor_frame_readout(compensated: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Final calibrated sensor frame readout."""
    return np.asarray(compensated, dtype=np.float64)


async def _synthesize_cubic_calibrator(spec: MicroAtomSynthesisSpec) -> str:
    """Deterministic Rung 4b micro-atom synthesizer closing the nonlinear sensor gap."""
    fn_name = spec.suggested_function_name or f"micro_calibrator_{spec.slot_node_id}"
    return (
        "import icontract\n"
        "import numpy as np\n\n"
        "@icontract.require(lambda scaled_volts: scaled_volts is not None)\n"
        "@icontract.ensure(lambda result: np.all(np.isfinite(result)))\n"
        f"def {fn_name}(scaled_volts: np.ndarray) -> np.ndarray:\n"
        "    v = np.asarray(scaled_volts, dtype=np.float64)\n"
        "    return 1.6 * v + 0.35 * (v ** 3) / (1.0 + v ** 2)\n"
    )


# ---------------------------------------------------------------------------
# Benchmark Preset Registry & Builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrototypingBenchmarkPreset:
    """Metadata descriptor for a cross-domain synthetic prototyping benchmark."""

    preset_id: str
    title: str
    domain: str
    repo_alias: str
    goal: str
    objective: str
    expected_rungs: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "title": self.title,
            "domain": self.domain,
            "repo_alias": self.repo_alias,
            "goal": self.goal,
            "objective": self.objective,
            "expected_rungs": list(self.expected_rungs),
            "description": self.description,
        }


BENCHMARK_PRESETS: dict[str, PrototypingBenchmarkPreset] = {
    "waveform_event_cadence": PrototypingBenchmarkPreset(
        preset_id="waveform_event_cadence",
        title="1D Waveform Conditioning & Cadence Estimation",
        domain="signal_processing",
        repo_alias="showcase/public-waveform-rate",
        goal="Estimate recurring event cadence from an oscillatory, artifact-corrupted synthetic waveform",
        objective="rmse",
        expected_rungs=("rung1_node_hpo", "rung2_sibling_swap"),
        description=(
            "Tunes conditioner smoothing alpha via Rung 1 Optuna HPO with cached downstream reuse, "
            "then swaps mean cadence for median cadence via Rung 2 sibling search (0 LLM calls)."
        ),
    ),
    "stateful_tabular_classifier": PrototypingBenchmarkPreset(
        preset_id="stateful_tabular_classifier",
        title="Stateful Tabular Fit/Predict Binary Classifier",
        domain="tabular_ml",
        repo_alias="showcase/public-tabular-classification",
        goal="Fit and evaluate a stateful binary classifier resilient to extreme tabular sensor spikes",
        objective="rmse",
        expected_rungs=("rung3_dpo_expansion", "rung1_node_hpo"),
        description=(
            "Executes explicit fit -> model_state -> predict state edges, diagnoses extreme feature "
            "spikes on live intermediates to splice an IQR clipper via Rung 3 DPO, and tunes regularization (0 LLM calls)."
        ),
    ),
    "calibrated_linear_solver": PrototypingBenchmarkPreset(
        preset_id="calibrated_linear_solver",
        title="Ill-Conditioned Linear System & Uncertainty Rescue",
        domain="numerical_linear_algebra",
        repo_alias="showcase/public-calibrated-solver",
        goal="Solve an ill-conditioned linear system within an analytical output precision budget",
        objective="rmse",
        expected_rungs=("rung2_sibling_swap", "rung1_node_hpo"),
        description=(
            "Propagates Phase 3 analytical uncertainty across a poorly conditioned normal-equation solver, "
            "rescues the RiskBudget via Rung 2 Tikhonov solver swap, and tunes damping via Rung 1 HPO (0 LLM calls)."
        ),
    ),
    "novel_nonlinear_micro_atom": PrototypingBenchmarkPreset(
        preset_id="novel_nonlinear_micro_atom",
        title="Nonlinear Sensor Calibration (Rung 4b Micro-Atom)",
        domain="sensor_calibration",
        repo_alias="showcase/public-nonlinear-calibration",
        goal="Calibrate nonlinear saturating sensor frames with a true catalog gap",
        objective="rmse",
        expected_rungs=("rung4b_micro_atom_synthesis",),
        description=(
            "Exhausts deterministic Rungs 1-3 on a genuine catalog gap, invokes Rung 4b to synthesize "
            "a single @icontract-guarded leaf micro-atom, verifies metamorphic invariants, and exports standalone Python."
        ),
    ),
}

REPO_ALIAS_TO_PRESET: dict[str, str] = {
    "showcase/public-waveform-rate": "waveform_event_cadence",
    "biosppy/ecg_mismatch": "waveform_event_cadence",
    "showcase/public-tabular-classification": "stateful_tabular_classifier",
    "sklearn/tabular_ml": "stateful_tabular_classifier",
    "showcase/public-calibrated-solver": "calibrated_linear_solver",
    "umap/scientific_computing": "calibrated_linear_solver",
    "showcase/public-nonlinear-calibration": "novel_nonlinear_micro_atom",
}


def list_benchmark_presets() -> list[dict[str, Any]]:
    """Return serializable metadata for all registered cross-domain presets."""
    return [preset.to_dict() for preset in BENCHMARK_PRESETS.values()]


def resolve_preset_id(identifier: str) -> str:
    """Resolve either a canonical ``preset_id`` or a Visualizer ``repo`` alias."""
    key = identifier.strip()
    if key in BENCHMARK_PRESETS:
        return key
    if key in REPO_ALIAS_TO_PRESET:
        return REPO_ALIAS_TO_PRESET[key]
    raise KeyError(f"Unknown prototyping benchmark preset: {identifier!r}")


def build_benchmark_session(
    preset_id: str,
    *,
    session_id: str | None = None,
) -> PrototypingSession:
    """Construct a fresh, un-stepped ``PrototypingSession`` for ``preset_id``."""
    canonical = resolve_preset_id(preset_id)
    sid = session_id or f"proto-{canonical}"
    if canonical == "waveform_event_cadence":
        return _build_waveform_session(sid)
    if canonical == "stateful_tabular_classifier":
        return _build_tabular_session(sid)
    if canonical == "calibrated_linear_solver":
        return _build_linear_solver_session(sid)
    if canonical == "novel_nonlinear_micro_atom":
        return _build_nonlinear_micro_atom_session(sid)
    raise KeyError(f"Unhandled preset: {canonical!r}")


def _build_waveform_session(session_id: str) -> PrototypingSession:
    preset = BENCHMARK_PRESETS["waveform_event_cadence"]
    n_cond = AlgorithmicNode(
        node_id="condition",
        name="Condition Sampled Waveform",
        description="Suppress high-frequency interference before peak detection",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.waveform_ema_conditioner",
        inputs=[_arr_port("signal"), _scalar_port("alpha", default_repr="0.88")],
        outputs=[_arr_port("filtered")],
        depth=1,
    )
    n_det = AlgorithmicNode(
        node_id="detect",
        name="Detect Recurring Events",
        description="Locate local event peaks on conditioned waveform",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.waveform_peak_detector",
        inputs=[_arr_port("filtered"), _scalar_port("threshold", default_repr="0.42")],
        outputs=[_arr_port("peaks")],
        depth=1,
    )
    n_meas = AlgorithmicNode(
        node_id="measure",
        name="Measure Event Rate",
        description="Estimate instantaneous event cadence in events per minute",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.waveform_mean_cadence",
        inputs=[_arr_port("peaks")],
        outputs=[_arr_port("cadence")],
        depth=1,
    )
    cdg = CDGExport(
        nodes=[n_cond, n_det, n_meas],
        edges=[
            DependencyEdge(
                source_id="condition",
                target_id="detect",
                output_name="filtered",
                input_name="filtered",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="detect",
                target_id="measure",
                output_name="peaks",
                input_name="peaks",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ],
        metadata={
            "goal": preset.goal,
            "paradigm": "signal_detect_measure",
            "repo": preset.repo_alias,
            "dataset_id": "synthetic-waveform-cadence-v1",
        },
    )

    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name=f"{_MOD}.waveform_mean_cadence",
            source="sciona-synthetic",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Mean instantaneous event cadence estimator",
            inputs=[_arr_port("peaks")],
            outputs=[_arr_port("cadence")],
        )
    )
    cat.add(
        AlgorithmicPrimitive(
            name=f"{_MOD}.waveform_median_cadence",
            source="sciona-synthetic",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Robust median instantaneous event cadence estimator",
            inputs=[_arr_port("peaks")],
            outputs=[_arr_port("cadence")],
        )
    )

    val_cases = [
        EvalCase(
            case_id="val_63epm",
            inputs={"signal": _synthesize_pulse_waveform(cadence_epm=63.5, seed=11)},
            expected=np.array([63.5], dtype=np.float64),
        ),
        EvalCase(
            case_id="val_81epm",
            inputs={
                "signal": _synthesize_pulse_waveform(
                    cadence_epm=81.5, artifact_at_s=5.1, seed=12
                )
            },
            expected=np.array([81.5], dtype=np.float64),
        ),
    ]
    holdout_cases = [
        EvalCase(
            case_id="holdout_54epm",
            inputs={
                "signal": _synthesize_pulse_waveform(
                    cadence_epm=54.0, artifact_at_s=3.8, seed=21
                )
            },
            expected=np.array([54.0], dtype=np.float64),
        ),
        EvalCase(
            case_id="holdout_74epm",
            inputs={
                "signal": _synthesize_pulse_waveform(
                    cadence_epm=74.0, artifact_at_s=6.2, seed=22
                )
            },
            expected=np.array([74.0], dtype=np.float64),
        ),
    ]
    harness = ResidentSliceHarness(
        val_cases,
        holdout_cases=holdout_cases,
        root_inputs=[_arr_port("signal")],
        root_outputs=[_arr_port("cadence")],
        metric="rmse",
    )
    return PrototypingSession(
        cdg,
        goal=preset.goal,
        session_id=session_id,
        harness=harness,
        catalog=cat,
        hpo_trials_per_rung=10,
    )


def _make_spiked_tabular_case(
    case_id: str,
    *,
    seed: int,
    n_train: int = 80,
    n_eval: int = 40,
) -> EvalCase:
    rng = np.random.default_rng(seed)
    w_true = np.array([1.5, -1.2, 0.8, 0.0], dtype=np.float64)
    x_tr = rng.normal(0.0, 1.0, size=(n_train, 4))
    probs_tr = 1.0 / (1.0 + np.exp(-2.0 * (x_tr @ w_true)))
    y_tr = (probs_tr > 0.5).astype(np.float64)
    # Inject extreme sensor glitches into 4 rows on informative column 0 (inverts unclipped fit!)
    glitch_rows = [3, 17, 35, 59]
    for r in glitch_rows:
        x_tr[r, 0] = 65.0 if y_tr[r] < 0.5 else -65.0

    x_ev = rng.normal(0.0, 1.0, size=(n_eval, 4))
    probs_ev = 1.0 / (1.0 + np.exp(-2.0 * (x_ev @ w_true)))
    y_ev = (probs_ev > 0.5).astype(np.float64)
    return EvalCase(
        case_id=case_id,
        inputs={"features": x_tr, "train_y": y_tr, "eval_x": x_ev},
        expected=y_ev,
    )


def _build_tabular_session(session_id: str) -> PrototypingSession:
    preset = BENCHMARK_PRESETS["stateful_tabular_classifier"]
    n_prep = AlgorithmicNode(
        node_id="prep_train",
        name="Assemble Training Matrix",
        description="Prepare numeric training feature matrix",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.tabular_pass_features",
        inputs=[_arr_port("features", ndim=2)],
        outputs=[_arr_port("train_x", ndim=2)],
        depth=1,
    )
    n_fit = AlgorithmicNode(
        node_id="fit",
        name="Fit Regularized Classifier",
        description="Fit binary linear-logit state vector on training partition",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.tabular_fit_ridge_logit",
        inputs=[
            _arr_port("train_x", ndim=2),
            _arr_port("train_y", ndim=1),
            _scalar_port("l2_reg", default_repr="0.0001"),
        ],
        outputs=[_arr_port("model_state", ndim=1)],
        state_ports=[
            StatePortSpec(
                port_name="model_state",
                direction="produce",
                type_desc="np.ndarray",
            )
        ],
        depth=1,
    )
    n_pred = AlgorithmicNode(
        node_id="predict",
        name="Predict Held-Out Probabilities",
        description="Score held-out evaluation rows using fitted model_state",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.tabular_predict_probs",
        inputs=[
            _arr_port("model_state", ndim=1),
            _arr_port("eval_x", ndim=2),
        ],
        outputs=[_arr_port("probabilities", ndim=1)],
        state_ports=[
            StatePortSpec(
                port_name="model_state",
                direction="consume",
                type_desc="np.ndarray",
            )
        ],
        depth=1,
    )
    cdg = CDGExport(
        nodes=[n_prep, n_fit, n_pred],
        edges=[
            DependencyEdge(
                source_id="prep_train",
                target_id="fit",
                output_name="train_x",
                input_name="train_x",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="fit",
                target_id="predict",
                output_name="model_state",
                input_name="model_state",
                source_type="np.ndarray",
                target_type="np.ndarray",
                edge_kind=EdgeKind.DATA_FLOW,
                data_kind="state",
            ),
        ],
        metadata={
            "goal": preset.goal,
            "paradigm": "stateful_fit_predict",
            "repo": preset.repo_alias,
            "dataset_id": "synthetic-tabular-classification-v1",
        },
    )

    val_cases = [
        _make_spiked_tabular_case("tab_val_1", seed=101),
        _make_spiked_tabular_case("tab_val_2", seed=102),
    ]
    holdout_cases = [
        _make_spiked_tabular_case("tab_holdout_1", seed=201),
        _make_spiked_tabular_case("tab_holdout_2", seed=202),
    ]
    harness = ResidentSliceHarness(
        val_cases,
        holdout_cases=holdout_cases,
        root_inputs=[
            _arr_port("features", ndim=2),
            _arr_port("train_y", ndim=1),
            _arr_port("eval_x", ndim=2),
        ],
        root_outputs=[_arr_port("probabilities", ndim=1)],
        metric="rmse",
    )
    engine = ExpansionEngine([_TabularOutlierClipRuleSet()])
    return PrototypingSession(
        cdg,
        goal=preset.goal,
        session_id=session_id,
        harness=harness,
        expansion_engine=engine,
        hpo_trials_per_rung=8,
    )


def _build_linear_solver_session(session_id: str) -> PrototypingSession:
    preset = BENCHMARK_PRESETS["calibrated_linear_solver"]
    n_pre = AlgorithmicNode(
        node_id="precondition",
        name="Condition System RHS",
        description="Normalize right-hand-side observation vector",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.lin_precondition_rhs",
        inputs=[_arr_port("rhs")],
        outputs=[_arr_port("rhs_cond")],
        depth=1,
    )
    n_solve = AlgorithmicNode(
        node_id="solve",
        name="Solve Linear System",
        description="Invert linear forward operator under observation noise",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.lin_solve_unregularized_normal_eq",
        inputs=[_arr_port("rhs_cond")],
        outputs=[_arr_port("sol_raw")],
        depth=1,
    )
    n_post = AlgorithmicNode(
        node_id="project",
        name="Project Solution Vector",
        description="Emit reconstructed state vector",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.lin_post_project",
        inputs=[_arr_port("sol_raw")],
        outputs=[_arr_port("solution")],
        depth=1,
    )
    cdg = CDGExport(
        nodes=[n_pre, n_solve, n_post],
        edges=[
            DependencyEdge(
                source_id="precondition",
                target_id="solve",
                output_name="rhs_cond",
                input_name="rhs_cond",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="solve",
                target_id="project",
                output_name="sol_raw",
                input_name="sol_raw",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ],
        metadata={
            "goal": preset.goal,
            "paradigm": "calibrated_numerical_solver",
            "repo": preset.repo_alias,
            "dataset_id": "synthetic-ill-conditioned-spd-v1",
        },
    )

    cat = PrimitiveCatalog()
    cat.add(
        AlgorithmicPrimitive(
            name=f"{_MOD}.lin_solve_unregularized_normal_eq",
            source="sciona-synthetic",
            category=ConceptType.ALGEBRA,
            description="Unregularized normal-equation solver",
            inputs=[_arr_port("rhs_cond")],
            outputs=[_arr_port("sol_raw")],
        )
    )
    cat.add(
        AlgorithmicPrimitive(
            name=f"{_MOD}.lin_solve_tikhonov_damped",
            source="sciona-synthetic",
            category=ConceptType.ALGEBRA,
            description="Calibrated Tikhonov-damped linear solver",
            inputs=[_arr_port("rhs_cond"), _scalar_port("damping", default_repr="0.25")],
            outputs=[_arr_port("sol_raw")],
        )
    )

    profiles = {
        f"{_MOD}.lin_precondition_rhs": _calibrated_profile(
            f"{_MOD}.lin_precondition_rhs",
            in_port="rhs",
            out_port="rhs_cond",
            singular_val=1.0,
            intrinsic_var=1e-4,
        ),
        f"{_MOD}.lin_solve_unregularized_normal_eq": _calibrated_profile(
            f"{_MOD}.lin_solve_unregularized_normal_eq",
            in_port="rhs_cond",
            out_port="sol_raw",
            singular_val=12.0,
            intrinsic_var=25.0,
        ),
        f"{_MOD}.lin_solve_tikhonov_damped": _calibrated_profile(
            f"{_MOD}.lin_solve_tikhonov_damped",
            in_port="rhs_cond",
            out_port="sol_raw",
            singular_val=0.35,
            intrinsic_var=0.002,
        ),
        f"{_MOD}.lin_post_project": _calibrated_profile(
            f"{_MOD}.lin_post_project",
            in_port="sol_raw",
            out_port="solution",
            singular_val=1.0,
            intrinsic_var=1e-4,
        ),
    }
    risk_budget = RiskBudget(sink_precisions={"solution": 2.0})

    def _make_lin_case(cid: str, seed: int) -> EvalCase:
        rng = np.random.default_rng(seed)
        grid = np.linspace(0.0, np.pi, 16, dtype=np.float64)
        x_true = np.sin(grid) + 0.5 * np.cos(2.0 * grid)
        b_noisy = _hilbert_like_forward(x_true) + 0.002 * rng.normal(size=16)
        return EvalCase(
            case_id=cid,
            inputs={"rhs": b_noisy},
            expected=x_true,
        )

    harness = ResidentSliceHarness(
        [_make_lin_case("lin_val_1", 301), _make_lin_case("lin_val_2", 302)],
        holdout_cases=[_make_lin_case("lin_holdout_1", 401), _make_lin_case("lin_holdout_2", 402)],
        root_inputs=[_arr_port("rhs")],
        root_outputs=[_arr_port("solution")],
        risk_budget=risk_budget,
        metric="rmse",
    )
    runner = LiveCDGRunner(uncertainty_profiles=profiles)
    return PrototypingSession(
        cdg,
        goal=preset.goal,
        session_id=session_id,
        harness=harness,
        runner=runner,
        funnel_config=FunnelConfig(),
        catalog=cat,
        hpo_trials_per_rung=10,
    )


def _build_nonlinear_micro_atom_session(session_id: str) -> PrototypingSession:
    preset = BENCHMARK_PRESETS["novel_nonlinear_micro_atom"]
    n_gain = AlgorithmicNode(
        node_id="gain",
        name="Normalize Sensor Voltage",
        description="Scale raw frame voltage into normalized range",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.sensor_gain_stage",
        inputs=[_arr_port("raw_volts")],
        outputs=[_arr_port("scaled_volts")],
        depth=1,
    )
    n_comp = AlgorithmicNode(
        node_id="compensate",
        name="Nonlinear Saturation Compensator",
        description="Compensate cubic rational sensor saturation curve",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.sensor_linear_placeholder",
        inputs=[_arr_port("scaled_volts")],
        outputs=[_arr_port("compensated")],
        depth=1,
    )
    n_out = AlgorithmicNode(
        node_id="readout",
        name="Emit Calibrated Frame",
        description="Final calibrated frame readout",
        concept_type=ConceptType.SIGNAL_TRANSFORM,
        status=NodeStatus.ATOMIC,
        matched_primitive=f"{_MOD}.sensor_frame_readout",
        inputs=[_arr_port("compensated")],
        outputs=[_arr_port("calibrated")],
        depth=1,
    )
    cdg = CDGExport(
        nodes=[n_gain, n_comp, n_out],
        edges=[
            DependencyEdge(
                source_id="gain",
                target_id="compensate",
                output_name="scaled_volts",
                input_name="scaled_volts",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="compensate",
                target_id="readout",
                output_name="compensated",
                input_name="compensated",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ],
        metadata={
            "goal": preset.goal,
            "paradigm": "nonlinear_sensor_calibration",
            "repo": preset.repo_alias,
            "dataset_id": "synthetic-nonlinear-sensor-v1",
        },
    )

    def _true_response(raw_v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        sv = 0.5 * raw_v
        return 1.6 * sv + 0.35 * (sv ** 3) / (1.0 + sv ** 2)

    def _make_sensor_case(cid: str, seed: int) -> EvalCase:
        rng = np.random.default_rng(seed)
        raw_v = rng.uniform(-3.5, 3.5, size=48)
        return EvalCase(
            case_id=cid,
            inputs={"raw_volts": raw_v},
            expected=_true_response(raw_v),
        )

    # Odd-symmetry metamorphic relation: f(-x) == -f(x)!
    def _check_odd_polarity(
        out_base: Mapping[str, Any], out_neg: Mapping[str, Any]
    ) -> tuple[bool, str]:
        ok = bool(
            np.allclose(
                np.asarray(out_neg["calibrated"], dtype=np.float64),
                -np.asarray(out_base["calibrated"], dtype=np.float64),
                atol=1e-6,
            )
        )
        return (ok, "" if ok else "Violated odd-polarity sensor symmetry f(-x) == -f(x)")

    odd_symmetry_rel = MetamorphicRelation(
        name="odd_polarity_inversion",
        transform_inputs=lambda inp: {"raw_volts": -np.asarray(inp["raw_volts"], dtype=np.float64)},
        verify_outputs=_check_odd_polarity,
    )

    harness = ResidentSliceHarness(
        [_make_sensor_case("sens_val_1", 501), _make_sensor_case("sens_val_2", 502)],
        holdout_cases=[_make_sensor_case("sens_holdout_1", 601), _make_sensor_case("sens_holdout_2", 602)],
        root_inputs=[_arr_port("raw_volts")],
        root_outputs=[_arr_port("calibrated")],
        metamorphic_relations=[odd_symmetry_rel],
        metric="rmse",
    )
    return PrototypingSession(
        cdg,
        goal=preset.goal,
        session_id=session_id,
        harness=harness,
        runner=LiveCDGRunner(),
        micro_atom_synthesizer=_synthesize_cubic_calibrator,
        hpo_trials_per_rung=6,
    )
