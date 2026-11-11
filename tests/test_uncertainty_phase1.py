"""Comprehensive synthetic test suite for Phase 1 of ``sciona.uncertainty``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sciona.synthesizer.uncertainty import (
    AtomUncertaintyEstimate,
    ProfileCatalogBackend,
    load_uncertainty_json,
)
from sciona.uncertainty import (
    AtomUncertaintyProfile,
    ComponentChannel,
    ConformalEnvelope,
    CountEventProjector,
    CountMixtureRegime,
    GammaMixtureRegime,
    GaussianMixtureRegime,
    GenericNumericProjector,
    IdentityProjector,
    LowRankPlusTailOperator,
    MatrixConditionProjector,
    PortRegimeSpec,
    PositiveScaleProjector,
    SimplexMixtureRegime,
    SimplexProjector,
    Tier1ConfidenceEnvelopeViolation,
    WaveformProjector,
    allocate_dag_alpha_budget,
    dump_uncertainty_profile_json,
    infer_projector,
    load_uncertainty_profile_json,
    mixture_from_dict,
    projector_from_spec,
)
from sciona.uncertainty._math import (
    digamma,
    loggamma,
    logsumexp,
    trigamma,
    woodbury_inverse_and_logdet,
)


# ---------------------------------------------------------------------------
# 1. Special functions & Woodbury linear algebra (_math.py)
# ---------------------------------------------------------------------------


class TestMathPrimitives:
    def test_digamma_known_values_and_derivative(self) -> None:
        euler_gamma = 0.5772156649015329
        assert float(digamma(1.0)) == pytest.approx(-euler_gamma, abs=1e-14)
        assert float(digamma(0.5)) == pytest.approx(
            -euler_gamma - 2.0 * math.log(2.0), abs=1e-14
        )

        pts = np.array([0.2, 1.3, 4.7, 12.5], dtype=np.float64)
        eps = 1e-6
        fd = (loggamma(pts + eps) - loggamma(pts - eps)) / (2.0 * eps)
        np.testing.assert_allclose(digamma(pts), fd, rtol=1e-8, atol=1e-8)

    def test_trigamma_known_values_and_derivative(self) -> None:
        assert float(trigamma(1.0)) == pytest.approx((math.pi ** 2) / 6.0, abs=1e-14)
        assert float(trigamma(0.5)) == pytest.approx((math.pi ** 2) / 2.0, abs=1e-14)

        pts = np.array([0.35, 1.8, 5.2, 15.0], dtype=np.float64)
        eps = 1e-6
        fd = (digamma(pts + eps) - digamma(pts - eps)) / (2.0 * eps)
        np.testing.assert_allclose(trigamma(pts), fd, rtol=1e-8, atol=1e-8)

    def test_woodbury_matches_dense_inverse_and_logdet(self) -> None:
        rng = np.random.default_rng(7)
        d = rng.uniform(0.2, 2.5, size=10)
        w = rng.standard_normal((10, 3))
        dense = np.diag(d) + w @ w.T

        prec, ldet = woodbury_inverse_and_logdet(d, w)
        dense_inv = np.linalg.inv(dense)
        _, dense_ldet = np.linalg.slogdet(dense)

        np.testing.assert_allclose(prec, dense_inv, rtol=1e-12, atol=1e-12)
        assert ldet == pytest.approx(float(dense_ldet), abs=1e-12)

    def test_logsumexp_handles_neg_inf(self) -> None:
        arr = np.array([[-np.inf, -np.inf], [0.0, np.log(3.0)]], dtype=np.float64)
        res = logsumexp(arr, axis=1)
        assert res[0] == -np.inf
        assert res[1] == pytest.approx(math.log(4.0), abs=1e-14)


# ---------------------------------------------------------------------------
# 2. Sufficient-statistic projectors (projectors.py)
# ---------------------------------------------------------------------------


class TestProjectors:
    def test_waveform_projector_and_nan_rejection(self) -> None:
        t = np.linspace(0.0, 1.0, 512, endpoint=False)
        sig = np.sin(2.0 * np.pi * 12.0 * t) + 0.1 * np.cos(2.0 * np.pi * 60.0 * t)
        proj = WaveformProjector()
        res = proj.project(sig)
        assert res.in_domain is True
        assert res.features.shape == (proj.dim,)
        assert np.all(np.isfinite(res.features))

        bad = sig.copy()
        bad[10] = np.nan
        bad_res = proj.project(bad)
        assert bad_res.in_domain is False
        assert "NaN" in bad_res.violation_reason

    def test_count_event_projector_counts_and_events(self) -> None:
        proj = CountEventProjector(n_bins=8)
        events = np.array([10.0, 25.0, 42.0, 59.0, 78.0, 95.0])
        res = proj.project(events)
        assert res.in_domain is True
        assert res.features[0] == pytest.approx(6.0)
        assert np.all(res.features >= 0.0)

        neg_res = proj.project(np.array([1.0, -2.0, 3.0]))
        assert neg_res.in_domain is False

    def test_positive_scale_projector_domain(self) -> None:
        proj = PositiveScaleProjector()
        ok = proj.project(np.array([0.5, 1.2, 3.4]))
        assert ok.in_domain is True

        zero_bad = proj.project(np.array([0.0, 1.0]))
        assert zero_bad.in_domain is False

    def test_simplex_projector_and_spec_roundtrip(self) -> None:
        proj = SimplexProjector(n_categories=3)
        res = proj.project(np.array([0.2, 0.5, 0.3]))
        assert res.in_domain is True
        assert float(np.sum(res.features)) == pytest.approx(1.0)

        off_simplex = proj.project(np.array([0.2, 0.2, 0.2]))
        assert off_simplex.in_domain is False

        spec_dict = proj.to_spec().to_dict()
        rebuilt = projector_from_spec(proj.to_spec().from_dict(spec_dict))
        assert rebuilt.kind == "simplex"
        assert rebuilt.dim == 3

    def test_matrix_and_infer_projector(self) -> None:
        mat = np.diag([10.0, 2.0, 1.0, 0.5])
        proj = infer_projector("covariance_matrix", "ndarray")
        assert isinstance(proj, MatrixConditionProjector)
        res = proj.project(mat)
        assert res.in_domain is True
        assert res.features.shape == (6,)

        assert isinstance(infer_projector("rpeaks", "ndarray"), CountEventProjector)
        assert isinstance(infer_projector("ecg_signal", "ndarray"), WaveformProjector)
        assert isinstance(infer_projector("alpha", "float"), IdentityProjector)
        assert isinstance(infer_projector("misc_buf", "ndarray"), GenericNumericProjector)


# ---------------------------------------------------------------------------
# 3. ExponentialFamilyMixture hierarchy (distributions.py)
# ---------------------------------------------------------------------------


class TestExponentialFamilyMixtures:
    def test_gaussian_exponential_family_identities_and_kl_reduce(self) -> None:
        gmm = GaussianMixtureRegime(
            weights=[0.3, 0.3, 0.4],
            means=[[-2.0, 0.0], [-1.9, 0.1], [3.0, -1.0]],
            diag_vars=[[0.5, 0.4], [0.5, 0.4], [0.8, 0.3]],
            factors=[
                [[0.2], [-0.1]],
                [[0.2], [-0.1]],
                [[-0.3], [0.4]],
            ],
        )
        z = np.array([[-2.0, 0.0], [0.5, -0.2], [3.1, -1.1]], dtype=np.float64)

        # Verify fast component_log_prob matches canonical <T(z), eta_k> + log h(z) - A(eta_k)
        fast_lp = gmm.component_log_prob(z)
        t_z = gmm.sufficient_statistics(z)
        for k in range(gmm.n_components):
            canonical_lp = (
                gmm.log_base_measure(z, k)
                + t_z @ gmm.natural_params(k)
                - gmm.log_partition(k)
            )
            np.testing.assert_allclose(fast_lp[:, k], canonical_lp, atol=1e-12)
            # Bregman divergence vanishes at component mean
            mu_k = gmm.component_mean(k).reshape(1, -1)
            assert float(gmm.component_bregman_divergence(mu_k, k)[0]) == pytest.approx(
                0.0, abs=1e-13
            )

        # Verify Runnalls KL reduction preserves exact mixture mean and covariance
        reduced = gmm.kl_reduce(max_components=2)
        assert reduced.n_components == 2
        np.testing.assert_allclose(reduced.mixture_mean(), gmm.mixture_mean(), atol=1e-12)
        np.testing.assert_allclose(
            reduced.mixture_covariance(), gmm.mixture_covariance(), atol=1e-8
        )

        # Round-trip serialization
        rebuilt = mixture_from_dict(gmm.to_dict())
        np.testing.assert_allclose(rebuilt.log_prob(z), gmm.log_prob(z), atol=1e-12)

    def test_count_mixture_poisson_and_negbin(self) -> None:
        pmm = CountMixtureRegime(
            weights=[0.6, 0.4],
            rates=[[2.0, 5.0], [15.0, 20.0]],
        )
        # Support check: negative values yield -inf log_prob
        assert pmm.log_prob(np.array([-1.0, 2.0])) == -np.inf

        # Poisson PMF exact check at z = (2, 5)
        z0 = np.array([[2.0, 5.0]])
        lp_k0 = pmm.component_log_prob(z0)[0, 0]
        expected_k0 = (2.0 * math.log(2.0) - 2.0 - math.lgamma(3.0)) + (
            5.0 * math.log(5.0) - 5.0 - math.lgamma(6.0)
        )
        assert lp_k0 == pytest.approx(expected_k0, abs=1e-13)

        # Negative-Binomial overdispersed fit & serialization
        rng = np.random.default_rng(11)
        nb_true = CountMixtureRegime(
            weights=[0.5, 0.5],
            rates=[[3.0, 4.0], [25.0, 30.0]],
            dispersions=[[1.5, 2.0], [2.0, 3.0]],
        )
        samples = nb_true.sample(600, rng=rng)
        fitted = CountMixtureRegime.fit_em(samples, n_components=2, seed=11)
        np.testing.assert_allclose(
            fitted.mixture_mean(), nb_true.mixture_mean(), rtol=0.15
        )
        rebuilt = mixture_from_dict(fitted.to_dict())
        np.testing.assert_allclose(
            rebuilt.log_prob(samples[:5]), fitted.log_prob(samples[:5]), atol=1e-12
        )

    def test_gamma_mixture_moments_fim_and_em(self) -> None:
        gam = GammaMixtureRegime(
            weights=[0.4, 0.6],
            shapes=[[3.0, 5.0], [10.0, 8.0]],
            rates=[[1.5, 2.5], [1.0, 2.0]],
        )
        assert gam.log_prob(np.array([0.0, 1.0])) == -np.inf

        rng = np.random.default_rng(23)
        draws = gam.sample(4000, rng=rng)
        np.testing.assert_allclose(
            np.mean(draws, axis=0), gam.mixture_mean(), rtol=0.05
        )
        np.testing.assert_allclose(
            np.cov(draws, rowvar=False), gam.mixture_covariance(), rtol=0.1
        )

        # Fisher information matrix must be strictly SPD
        eigvals = np.linalg.eigvalsh(gam.natural_fisher_information(0))
        assert np.all(eigvals > 0.0)

        fitted = GammaMixtureRegime.fit_em(draws[:400], n_components=2, seed=23)
        rebuilt = mixture_from_dict(fitted.to_dict())
        np.testing.assert_allclose(
            rebuilt.log_prob(draws[:5]), fitted.log_prob(draws[:5]), atol=1e-12
        )

    def test_simplex_mixture_dirichlet_moments_and_em(self) -> None:
        dir_mix = SimplexMixtureRegime(
            weights=[0.5, 0.5],
            concentrations=[[12.0, 2.0, 2.0], [2.0, 10.0, 4.0]],
        )
        # Off-simplex vector rejected
        assert dir_mix.log_prob(np.array([0.5, 0.5, 0.5])) == -np.inf

        rng = np.random.default_rng(31)
        draws = dir_mix.sample(4000, rng=rng)
        np.testing.assert_allclose(
            np.mean(draws, axis=0), dir_mix.mixture_mean(), atol=0.02
        )
        np.testing.assert_allclose(
            np.cov(draws, rowvar=False), dir_mix.mixture_covariance(), atol=0.01
        )

        fitted = SimplexMixtureRegime.fit_em(draws[:300], n_components=2, seed=31)
        rebuilt = mixture_from_dict(fitted.to_dict())
        np.testing.assert_allclose(
            rebuilt.log_prob(draws[:5]), fitted.log_prob(draws[:5]), atol=1e-12
        )


# ---------------------------------------------------------------------------
# 4. Stiff/Sloppy Operators, Pullbacks, & Stein Jacobians (operators.py)
# ---------------------------------------------------------------------------


class TestOperators:
    def test_stiff_vs_sloppy_cancellation_and_pullback(self) -> None:
        # J has stiff singular value 10.0 along e_0 and sloppy 0.01 along e_1, e_2
        jac = np.diag([10.0, 0.01, 0.005])
        op = LowRankPlusTailOperator.from_dense(jac, max_rank=1)

        assert op.spectral_norm == pytest.approx(10.0)
        assert op.tail_singular_value == pytest.approx(0.01)

        # Upstream noise strictly along sloppy direction e_1:
        sloppy_cov_in = np.diag([0.0, 1.0, 0.0])
        out_cov = op.pushforward_covariance(sloppy_cov_in)
        # Output total variance is <= 0.01^2 = 1e-4, NOT 10^2 = 100!
        assert float(np.trace(out_cov)) == pytest.approx(1e-4, rel=1e-6)

        # Upstream noise along stiff direction e_0:
        stiff_cov_in = np.diag([1.0, 0.0, 0.0])
        out_cov_stiff = op.pushforward_covariance(stiff_cov_in)
        assert out_cov_stiff[0, 0] == pytest.approx(100.0)

        # Pullback of unit output precision: requires 100.0 along e_0, 1e-4 along e_1
        pb = op.pullback_precision(np.eye(3))
        assert pb.feasible is True
        assert pb.required_input_precision[0, 0] == pytest.approx(100.0)
        assert pb.required_input_precision[1, 1] == pytest.approx(1e-4)

        spec = op.stiff_sloppy_spectrum(stiff_threshold=1.0, sloppy_threshold=0.05)
        assert spec.stiff_rank == 1
        assert spec.sloppy_rank == 2
        assert spec.condition_number == pytest.approx(1000.0)

    def test_pullback_detects_intrinsic_noise_infeasibility(self) -> None:
        op = LowRankPlusTailOperator.from_dense(np.eye(2), max_rank=2)
        # Downstream requires output variance <= 0.1 (precision = 10.0 * I),
        # but intrinsic approximation variance is 0.25 > 0.1!
        pb = op.pullback_precision(
            required_output_precision=10.0 * np.eye(2),
            intrinsic_output_cov=0.25 * np.eye(2),
        )
        assert pb.feasible is False
        assert pb.slack_min_eigenvalue == pytest.approx(-0.15)
        assert "Intrinsic output covariance exceeds" in pb.violation_reason

    def test_stein_distributional_jacobian_on_step_function(self) -> None:
        # Discontinuous step threshold along x[0]; pointwise derivative at x0=0.02 is 0!
        def step_detector(x: np.ndarray) -> np.ndarray:
            return np.array([1.0 if x[0] > 0.0 else 0.0, 0.5 * x[1]], dtype=np.float64)

        op = LowRankPlusTailOperator.from_stein_perturbation(
            step_detector,
            x0=np.array([0.02, 1.0]),
            sigma=0.1,
            n_probes=400,
            max_rank=2,
            seed=42,
        )
        fim = op.fisher_information()
        # Stein smoothing detects strong sensitivity along coordinate 0 and ~0.25 along coordinate 1
        assert fim[0, 0] > 5.0
        assert fim[1, 1] == pytest.approx(0.25, rel=0.25)


# ---------------------------------------------------------------------------
# 5. Split-Conformal Envelopes & Alpha-Spending (conformal.py)
# ---------------------------------------------------------------------------


class TestConformalEnvelope:
    def test_finite_sample_exchangeable_coverage_guarantee(self) -> None:
        rng = np.random.default_rng(99)
        alpha = 0.05
        n_cal = 199  # (199 + 1) * 0.95 = 190 -> exact integer quantile!
        false_aborts = 0
        n_mc = 2000

        for _ in range(n_mc):
            # Heavy-tailed Student-t nonconformity scores
            all_scores = np.abs(rng.standard_t(df=3.0, size=n_cal + 1))
            env = ConformalEnvelope(all_scores[:n_cal])
            chk = env.check(float(all_scores[n_cal]), alpha=alpha)
            if not chk.passed:
                false_aborts += 1

        emp_rate = false_aborts / float(n_mc)
        assert emp_rate == pytest.approx(alpha, abs=0.015)

    def test_insufficient_calibration_size_and_alpha_budget(self) -> None:
        env = ConformalEnvelope([1.0, 2.0, 3.0, 4.0])  # N = 4 => min alpha = 0.2
        chk_loose = env.check(10.0, alpha=0.01, require_sufficient_calibration=False)
        assert chk_loose.threshold == float("inf")
        assert chk_loose.passed is True

        chk_strict = env.check(2.0, alpha=0.01, require_sufficient_calibration=True)
        assert chk_strict.passed is False
        assert "insufficient" in chk_strict.reason

        alloc = allocate_dag_alpha_budget(
            ["n1", "n2", "n3"],
            global_alpha=0.03,
            risk_weights={"n1": 1.0, "n2": 2.0, "n3": 3.0},
        )
        assert sum(alloc.values()) == pytest.approx(0.03)
        assert alloc["n3"] == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# 6. AtomUncertaintyProfile, Tier 1 Guard, & v1/v2 Bridge (models.py)
# ---------------------------------------------------------------------------


class TestAtomUncertaintyProfileAndBridge:
    def _build_tier1_profile(self) -> AtomUncertaintyProfile:
        rng = np.random.default_rng(123)
        proj = IdentityProjector(dim=2)
        gmm = GaussianMixtureRegime(
            weights=[0.5, 0.5],
            means=[[-3.0, 0.0], [3.0, 0.0]],
            diag_vars=[[0.25, 0.25], [0.25, 0.25]],
        )
        cal_draws = gmm.sample(299, rng=rng)
        cal_scores = gmm.nonconformity_score(cal_draws)
        env = ConformalEnvelope(cal_scores)

        # Component 0 is gentle (sigma=0.5), Component 1 is stiff along x_0 (sigma=8.0)
        op_k0 = LowRankPlusTailOperator.from_dense(0.5 * np.eye(2), max_rank=2)
        op_k1 = LowRankPlusTailOperator.from_dense(np.diag([8.0, 0.1]), max_rank=2)

        return AtomUncertaintyProfile(
            atom_name="adaptive_filter",
            mode="empirical",
            confidence=0.95,
            trust_tier=1,
            input_regimes={
                "signal": PortRegimeSpec(
                    port_name="signal",
                    projector_spec=proj.to_spec(),
                    mixture=gmm,
                    conformal_envelope=env,
                )
            },
            output_projectors={"filtered": proj.to_spec()},
            components=(
                ComponentChannel(
                    component_index=0,
                    operators={("signal", "filtered"): op_k0},
                    intrinsic_output_cov={"filtered": 1e-4 * np.eye(2)},
                ),
                ComponentChannel(
                    component_index=1,
                    operators={("signal", "filtered"): op_k1},
                    intrinsic_output_cov={"filtered": 1e-3 * np.eye(2)},
                ),
            ),
            n_trials=300,
            epsilon=1e-6,
            notes="synthetic tier-1 bimodal test profile",
        )

    def test_regime_dispatch_and_tier1_abort(self) -> None:
        profile = self._build_tier1_profile()

        # Input near mode 0 (-3, 0): passes Tier 1 bounds and activates component 0
        rep0 = profile.evaluate_inputs(
            {"signal": np.array([-3.0, 0.05])},
            alpha=0.02,
            require_conformal_for_tier1=True,
        )
        assert rep0.in_confidence_bounds is True
        assert rep0.active_component == 0
        rep0.raise_if_violated(node_id="node_filt", atom_name="adaptive_filter")

        # Pushforward under component 0 vs component 1 reflects regime-specific conditioning!
        cov0 = profile.pushforward_covariances(
            {"signal": np.eye(2)}, responsibilities=rep0.combined_responsibilities
        )["filtered"]
        rep1 = profile.evaluate_inputs(
            {"signal": np.array([3.0, -0.05])},
            alpha=0.02,
            require_conformal_for_tier1=True,
        )
        assert rep1.active_component == 1
        cov1 = profile.pushforward_covariances(
            {"signal": np.eye(2)}, responsibilities=rep1.combined_responsibilities
        )["filtered"]
        assert cov0[0, 0] == pytest.approx(0.2501, rel=1e-2)
        assert cov1[0, 0] == pytest.approx(64.001, rel=1e-2)

        # Far OOD input (50, 50) triggers rigorous Tier 1 abort!
        rep_ood = profile.evaluate_inputs(
            {"signal": np.array([50.0, 50.0])},
            alpha=0.02,
            require_conformal_for_tier1=True,
        )
        assert rep_ood.in_confidence_bounds is False
        with pytest.raises(Tier1ConfidenceEnvelopeViolation) as exc_info:
            rep_ood.raise_if_violated(node_id="node_filt", atom_name="adaptive_filter")
        assert exc_info.value.node_id == "node_filt"
        assert exc_info.value.port_name == "signal"

    def test_v2_json_persistence_and_v1_loader_bridge(self, tmp_path) -> None:
        profile = self._build_tier1_profile()
        uj_path = dump_uncertainty_profile_json(profile, tmp_path / "uncertainty.json")

        # Full v2 reload preserves regimes, operators, and conformal scores
        reloaded = load_uncertainty_profile_json(uj_path)
        assert reloaded.atom_name == "adaptive_filter"
        assert reloaded.trust_tier == 1
        assert reloaded.n_components == 2
        assert reloaded.effective_scalar_factor() == pytest.approx(8.0)

        # Legacy v1 load_uncertainty_json reads the same v2 file seamlessly!
        name, v1_est = load_uncertainty_json(uj_path)
        assert name == "adaptive_filter"
        assert v1_est.mode == "empirical"
        assert v1_est.scalar_factor == pytest.approx(8.0)
        assert v1_est.confidence == pytest.approx(0.95)
        assert "signal:gaussian(K=2,d=2)" in v1_est.input_regime

        # ProfileCatalogBackend serves both v1 UncertaintyBackend and v2 estimate_profile
        backend = ProfileCatalogBackend({"adaptive_filter": reloaded})
        assert backend.estimate("pkg.adaptive_filter").scalar_factor == pytest.approx(8.0)
        assert backend.estimate_profile("pkg.adaptive_filter").trust_tier == 1
        assert backend.estimate("unknown_atom").mode == "unknown"

    def test_legacy_v1_file_lifts_to_v2_profile(self, tmp_path) -> None:
        legacy_path = tmp_path / "uncertainty.json"
        legacy_path.write_text(
            '{"atom": "fft", "estimates": [{"mode": "empirical", "scalar_factor": 1.42, "confidence": 0.9, "n_trials": 500}]}'
        )
        prof = load_uncertainty_profile_json(legacy_path)
        assert prof.atom_name == "fft"
        assert prof.effective_scalar_factor() == pytest.approx(1.42)
        v1_back = AtomUncertaintyProfile.from_v1_estimate(
            "fft", AtomUncertaintyEstimate(mode="heuristic", scalar_factor=1.5, confidence=0.2)
        ).to_v1_estimate()
        assert v1_back.scalar_factor == pytest.approx(1.5)
