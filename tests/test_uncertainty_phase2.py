"""Phase 2 tests: Empirical multi-regime calibration, CLI harness, & Supabase v2 backfill.

All fixtures and test functions use strictly synthetic mathematical functions and
generated random numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.backfill_uncertainty import (
    backfill_uncertainty,
    build_uncertainty_rows,
    build_uncertainty_rows_from_payload,
)
from scripts.measure_uncertainty import (
    discover_atoms_in_roots,
    run_measurement,
    write_v2_uncertainty_json,
)
from sciona.synthesizer.uncertainty import load_uncertainty_json
from sciona.uncertainty import (
    AtomUncertaintyProfile,
    CalibrationConfig,
    CountMixtureRegime,
    GammaMixtureRegime,
    GaussianMixtureRegime,
    IdentityProjector,
    PortCalibrationConfig,
    SimplexMixtureRegime,
    Tier1ConfidenceEnvelopeViolation,
    calibrate_atom,
    load_uncertainty_profile_json,
)


def test_calibrate_linear_stiff_sloppy_operator_recovery() -> None:
    """Calibrating y = A @ x recovers stiff & sloppy singular values and ~0 noise."""
    rng = np.random.default_rng(123)
    q_u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    q_v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    true_sv = np.array([12.0, 3.5, 0.25, 0.02], dtype=np.float64)
    a_mat = (q_u * true_sv[None, :]) @ q_v.T

    def linear_transform(x: np.ndarray) -> np.ndarray:
        return a_mat @ np.asarray(x, dtype=np.float64)

    def sampler(gen: np.random.Generator, regime_idx: int) -> dict[str, Any]:
        shift = [-2.0, 2.0][regime_idx % 2]
        return {"x": shift + 0.6 * gen.standard_normal(4)}

    profile = calibrate_atom(
        "linear_transform",
        linear_transform,
        config=CalibrationConfig(
            n_trials=120,
            fit_fraction=0.5,
            n_components=1,
            operator_max_rank=4,
            n_anchor_probes=32,
            seed=42,
        ),
        port_configs={
            "x": PortCalibrationConfig(port_name="x", projector=IdentityProjector(dim=4))
        },
        regime_sampler=sampler,
    )

    assert profile.schema_version == "2.0"
    assert profile.mode == "empirical"
    assert len(profile.components) == 1
    comp = profile.components[0]
    op = comp.operators[("x", "out")]
    np.testing.assert_allclose(op.singular_values, true_sv, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(comp.intrinsic_output_cov["out"], np.zeros((4, 4)), atol=1e-6)
    assert profile.effective_scalar_factor() == pytest.approx(12.0, rel=1e-3)


def test_calibrate_bimodal_regime_switching_atom_bic_and_channels() -> None:
    """BIC selects multi-component mixture and separates low-gain vs high-gain Jacobians."""

    def piecewise_gain(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if float(np.mean(arr)) < 0.0:
            return 0.4 * arr
        return 6.5 * arr

    def bimodal_sampler(gen: np.random.Generator, regime_idx: int) -> dict[str, Any]:
        if regime_idx % 2 == 0:
            return {"x": -4.0 + 0.35 * gen.standard_normal(3)}
        return {"x": 4.0 + 0.35 * gen.standard_normal(3)}

    profile = calibrate_atom(
        "piecewise_gain",
        piecewise_gain,
        config=CalibrationConfig(
            n_trials=200,
            fit_fraction=0.6,
            n_components=None,
            candidate_components=(1, 2, 3),
            n_anchor_probes=24,
            seed=7,
        ),
        port_configs={
            "x": PortCalibrationConfig(port_name="x", projector=IdentityProjector(dim=3))
        },
        regime_sampler=bimodal_sampler,
    )

    assert profile.n_components >= 2
    assert isinstance(profile.input_regimes["x"].mixture, GaussianMixtureRegime)

    eval_low = profile.evaluate_inputs({"x": np.array([-4.0, -4.0, -4.0])}, alpha=0.05)
    eval_high = profile.evaluate_inputs({"x": np.array([4.0, 4.0, 4.0])}, alpha=0.05)
    assert eval_low.in_confidence_bounds
    assert eval_high.in_confidence_bounds
    assert eval_low.active_component != eval_high.active_component

    low_op = profile.components[eval_low.active_component].operators[("x", "out")]
    high_op = profile.components[eval_high.active_component].operators[("x", "out")]
    assert low_op.spectral_norm == pytest.approx(0.4, rel=0.08)
    assert high_op.spectral_norm == pytest.approx(6.5, rel=0.08)


def test_calibrate_domain_families_and_preflight_constraint_probes() -> None:
    """Automatic domain probing handles >0 Gamma, Count, Simplex, SPD Matrix, & Stochastic atoms."""

    # 1. Generic parameter name with strict positivity contract (e.g. icontract x > 0)
    def log_energy(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if np.any(arr <= 0.0):
            raise ValueError("strictly positive input required")
        return np.sqrt(arr)

    pos_prof = calibrate_atom(
        "log_energy",
        log_energy,
        config=CalibrationConfig(n_trials=90, n_anchor_probes=18, seed=11),
    )
    assert isinstance(pos_prof.input_regimes["x"].mixture, GammaMixtureRegime)

    # 2. Count port -> CountMixtureRegime
    def thin_counts(event_counts: np.ndarray) -> np.ndarray:
        arr = np.asarray(event_counts, dtype=np.float64)
        if np.any(arr < 0.0):
            raise ValueError("counts must be >= 0")
        return np.floor(0.5 * arr)

    cnt_prof = calibrate_atom(
        "thin_counts",
        thin_counts,
        config=CalibrationConfig(n_trials=90, n_anchor_probes=18, seed=13),
    )
    assert isinstance(cnt_prof.input_regimes["event_counts"].mixture, CountMixtureRegime)

    # 3. Simplex port -> SimplexMixtureRegime
    def sharpen_simplex(probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        sq = p * p
        return sq / np.sum(sq)

    simp_prof = calibrate_atom(
        "sharpen_simplex",
        sharpen_simplex,
        config=CalibrationConfig(n_trials=90, n_anchor_probes=18, seed=17),
    )
    assert isinstance(simp_prof.input_regimes["probabilities"].mixture, SimplexMixtureRegime)

    # 4. SPD matrix contract -> MatrixConditionProjector + SPD-preserving perturbations
    def cholesky_logdet(covariance: np.ndarray) -> np.ndarray:
        mat = np.asarray(covariance, dtype=np.float64)
        if not np.allclose(mat, mat.T, atol=1e-7):
            raise ValueError("symmetric matrix required")
        chol = np.linalg.cholesky(mat)
        return np.array([2.0 * float(np.sum(np.log(np.diag(chol))))], dtype=np.float64)

    spd_prof = calibrate_atom(
        "cholesky_logdet",
        cholesky_logdet,
        config=CalibrationConfig(n_trials=80, n_anchor_probes=16, seed=19),
    )
    assert spd_prof.input_regimes["covariance"].projector_spec.kind == "matrix_condition"

    # 5. Stochastic + thresholded waveform atom -> non-zero intrinsic noise covariance
    noise_rng = np.random.default_rng(99)

    def noisy_peak_counter(signal: np.ndarray) -> np.ndarray:
        s = np.asarray(signal, dtype=np.float64)
        above = float(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > 0.5)))
        jitter = float(noise_rng.normal(0.0, 0.4))
        return np.array([above + jitter, float(np.std(s))], dtype=np.float64)

    stoch_prof = calibrate_atom(
        "noisy_peak_counter",
        noisy_peak_counter,
        config=CalibrationConfig(n_trials=100, n_components=1, n_anchor_probes=24, seed=23),
    )
    cov_eps = stoch_prof.components[0].intrinsic_output_cov["out"]
    assert cov_eps.shape == (2, 2)
    assert float(cov_eps[0, 0]) > 0.02


def test_tier1_split_conformal_empirical_coverage_and_ood_abort() -> None:
    """Disjoint split-conformal calibration bounds false-abort rate on fresh Tier 1 draws."""

    def smooth_filter(signal: np.ndarray) -> np.ndarray:
        s = np.asarray(signal, dtype=np.float64)
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        return np.convolve(s, kernel, mode="same")

    def exchange_sampler(gen: np.random.Generator, regime_idx: int) -> dict[str, Any]:
        t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        freq = 2.0 if regime_idx % 2 == 0 else 6.0
        amp = 1.0 if regime_idx % 2 == 0 else 2.0
        sig = amp * np.sin(freq * t) + 0.15 * gen.standard_normal(128)
        return {"signal": sig}

    profile = calibrate_atom(
        "smooth_filter",
        smooth_filter,
        config=CalibrationConfig(
            n_trials=240,
            fit_fraction=0.5,
            n_components=2,
            trust_tier=1,
            n_anchor_probes=18,
            seed=101,
        ),
        regime_sampler=exchange_sampler,
    )

    env = profile.input_regimes["signal"].conformal_envelope
    assert env is not None
    assert env.n_calibration == 120
    assert env.min_certifiable_alpha < 0.01

    test_rng = np.random.default_rng(2026)
    false_aborts = 0
    n_test = 300
    for idx in range(n_test):
        sample = exchange_sampler(test_rng, idx)
        report = profile.evaluate_inputs(
            sample,
            alpha=0.02,
            require_conformal_for_tier1=True,
        )
        if not report.in_confidence_bounds:
            false_aborts += 1

    # Target alpha = 0.02 -> allow finite-sample binomial slack up to 0.055
    assert (false_aborts / n_test) <= 0.055

    # Severe OOD waveform (1000x amplitude DC + impulses) must trigger Tier 1 abort
    ood_signal = np.full(128, 500.0, dtype=np.float64)
    ood_signal[::8] = -1500.0
    ood_report = profile.evaluate_inputs(
        {"signal": ood_signal},
        alpha=0.01,
        require_conformal_for_tier1=True,
    )
    assert not ood_report.in_confidence_bounds
    with pytest.raises(Tier1ConfidenceEnvelopeViolation):
        ood_report.raise_if_violated(node_id="n_filter", atom_name="smooth_filter")


def test_cli_measure_uncertainty_discovery_and_multi_atom_artifacts(tmp_path: Path) -> None:
    """CLI runner writes single-atom and multi-atom _artifacts v2 JSONs cleanly."""
    pkg_single = tmp_path / "sciona" / "atoms" / "synth_domain" / "single_pkg"
    pkg_multi = tmp_path / "sciona" / "atoms" / "synth_domain" / "multi_pkg"
    pkg_single.mkdir(parents=True)
    pkg_multi.mkdir(parents=True)

    (pkg_single / "atoms.py").write_text(
        "import numpy as np\n\n"
        "def scale_twice(x: np.ndarray) -> np.ndarray:\n"
        "    return 2.0 * np.asarray(x, dtype=float)\n"
    )
    # Seed an analytical estimate to verify non-empirical v1 entries are preserved
    (pkg_single / "uncertainty.json").write_text(
        json.dumps(
            {
                "atom": "scale_twice",
                "estimates": [
                    {
                        "mode": "analytical",
                        "scalar_factor": 2.0,
                        "confidence": 1.0,
                        "n_trials": 0,
                        "epsilon": 0.0,
                        "input_regime": "exact",
                        "notes": "exact norm",
                    }
                ],
            }
        )
    )

    (pkg_multi / "atoms.py").write_text(
        "import numpy as np\n\n"
        "def shift_up(x: np.ndarray) -> np.ndarray:\n"
        "    return np.asarray(x, dtype=float) + 1.0\n\n"
        "def triple_vec(x: np.ndarray) -> np.ndarray:\n"
        "    return 3.0 * np.asarray(x, dtype=float)\n"
    )

    discovered = discover_atoms_in_roots([tmp_path], domain="synth_domain")
    assert {d.name for d in discovered} == {"scale_twice", "shift_up", "triple_vec"}

    stats = run_measurement(
        [tmp_path],
        domain="synth_domain",
        config=CalibrationConfig(n_trials=40, n_components=1, n_anchor_probes=10, seed=5),
    )
    assert stats == {"discovered": 3, "calibrated": 3, "skipped": 0}

    single_uj = pkg_single / "uncertainty.json"
    multi_uj_a = pkg_multi / "_artifacts" / "shift_up" / "uncertainty.json"
    multi_uj_b = pkg_multi / "_artifacts" / "triple_vec" / "uncertainty.json"
    assert single_uj.is_file()
    assert multi_uj_a.is_file()
    assert multi_uj_b.is_file()

    # Check preserved analytical entry + v2 empirical entry
    single_raw = json.loads(single_uj.read_text())
    assert single_raw["schema_version"] == "2.0"
    modes = [e["mode"] for e in single_raw["estimates"]]
    assert modes == ["analytical", "empirical"]

    # Both v2 and v1 loaders read the generated artifacts
    prof_v2 = load_uncertainty_profile_json(multi_uj_b)
    assert prof_v2.atom_name == "triple_vec"
    assert prof_v2.is_calibrated
    name_v1, est_v1 = load_uncertainty_json(multi_uj_b)
    assert name_v1 == "triple_vec"
    assert est_v1.scalar_factor is not None


class _MockTable:
    def __init__(self, parent: _MockSupabase, name: str) -> None:
        self._parent = parent
        self._name = name
        self._eq_val = ""

    def select(self, _cols: str) -> _MockTable:
        return self

    def eq(self, _col: str, val: str) -> _MockTable:
        self._eq_val = val
        return self

    def like(self, _col: str, val: str) -> _MockTable:
        self._eq_val = val
        return self

    def limit(self, _n: int) -> _MockTable:
        return self

    def delete(self) -> _MockTable:
        return self

    def is_(self, _col: str, _val: str) -> _MockTable:
        return self

    def insert(self, rows: list[dict[str, Any]]) -> _MockTable:
        # Enforce PostgREST PGRST102 rule: all dicts in bulk insert must share identical keys
        if len(rows) > 1:
            ref_keys = set(rows[0].keys())
            for r in rows[1:]:
                assert set(r.keys()) == ref_keys
        self._parent.inserted.extend(rows)
        return self

    def execute(self) -> Any:
        class _Resp:
            def __init__(self, data: Any) -> None:
                self.data = data

        if self._name == "atoms":
            atom_id = self._parent.atom_map.get(self._eq_val)
            return _Resp([{"atom_id": atom_id}] if atom_id else [])
        return _Resp([])


class _MockSupabase:
    def __init__(self, atom_map: dict[str, str]) -> None:
        self.atom_map = atom_map
        self.inserted: list[dict[str, Any]] = []

    def table(self, name: str) -> _MockTable:
        return _MockTable(self, name)


def test_supabase_migration_and_v2_backfill_roundtrip(tmp_path: Path) -> None:
    """Migration DDL exists and v2 payloads backfill & hydrate cleanly without breaking v1."""
    mig_path = (
        Path(__file__).resolve().parent.parent
        / "supabase"
        / "migrations"
        / "20261110000000_uncertainty_v2_mixture_channels.sql"
    )
    assert mig_path.is_file()
    ddl = mig_path.read_text()
    for col in ("schema_version", "input_regimes", "output_projectors", "components"):
        assert col in ddl

    # Legacy v1 build_uncertainty_rows preserves exact 9-key dict
    v1_rows = build_uncertainty_rows(
        "atom-legacy",
        [{"scalar_factor": 1.2, "confidence": 0.8}],
    )
    assert set(v1_rows[0].keys()) == {
        "atom_id",
        "version_id",
        "mode",
        "scalar_factor",
        "confidence",
        "n_trials",
        "epsilon",
        "input_regime",
        "notes",
    }

    # Calibrate a synthetic atom, write v2 JSON alongside an analytical estimate, and backfill
    def affine_vec(x: np.ndarray) -> np.ndarray:
        return 2.5 * np.asarray(x, dtype=np.float64) - 0.5

    prof = calibrate_atom(
        "affine_vec",
        affine_vec,
        config=CalibrationConfig(n_trials=50, n_components=1, n_anchor_probes=12, seed=31),
        port_configs={
            "x": PortCalibrationConfig(port_name="x", projector=IdentityProjector(dim=3))
        },
        regime_sampler=lambda gen, _r: {"x": gen.standard_normal(3)},
    )

    atom_dir = tmp_path / "sciona" / "atoms" / "demo" / "affine"
    atom_dir.mkdir(parents=True)
    uj_path = atom_dir / "uncertainty.json"
    uj_path.write_text(
        json.dumps(
            {
                "atom": "affine_vec",
                "estimates": [
                    {
                        "mode": "analytical",
                        "scalar_factor": 2.5,
                        "confidence": 1.0,
                        "input_regime": "R^3",
                        "notes": "exact Jacobian norm",
                    }
                ],
            }
        )
    )
    write_v2_uncertainty_json(prof, uj_path)

    payload = json.loads(uj_path.read_text())
    rows = build_uncertainty_rows_from_payload("uuid-affine", payload)
    assert len(rows) == 2
    v2_row = next(r for r in rows if r["schema_version"] == "2.0")
    v1_aux = next(r for r in rows if r["schema_version"] == "1.0")
    assert v2_row["mode"] == "empirical"
    assert v2_row["scalar_factor"] == pytest.approx(2.5, rel=1e-2)
    assert "x" in v2_row["input_regimes"]
    assert len(v2_row["components"]) == 1
    assert v1_aux["mode"] == "analytical"

    # Hydrate AtomUncertaintyProfile directly from the Supabase row
    hydrated = AtomUncertaintyProfile.from_supabase_row("affine_vec", v2_row, trust_tier=1)
    assert hydrated.trust_tier == 1
    assert hydrated.effective_scalar_factor() == pytest.approx(2.5, rel=1e-2)
    eval_rep = hydrated.evaluate_inputs({"x": np.zeros(3)}, alpha=0.05)
    assert eval_rep.in_confidence_bounds

    # Full backfill_uncertainty against mock Supabase client (enforcing PostgREST key uniformity)
    mock_sb = _MockSupabase({"sciona.atoms.demo.affine.affine_vec": "uuid-affine"})
    stats = backfill_uncertainty(mock_sb, atoms_root=tmp_path, dry_run=False)
    assert stats == {"found": 1, "inserted": 2, "skipped_no_atom": 0, "errors": 0}
    assert len(mock_sb.inserted) == 2
