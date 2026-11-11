"""Exponential-family mixture regimes ``p_cal(z) = sum_k pi_k p(z | eta_k)``.

Provides a unified mathematical base class grounded in canonical exponential
families ``p(z | eta) = h(z) exp(<T(z), eta> - A(eta))`` along with four
concrete regime subclasses covering the primary data geometries in ``sciona``:

- ``GaussianMixtureRegime`` — Factor-Analyzer (diagonal + low-rank) covariance
  on ``R^d`` using Woodbury inversions and Runnalls KL mixture reduction.
- ``CountMixtureRegime`` — Poisson and overdispersed Negative-Binomial mixtures
  on ``R_{>=0}^d`` for discrete counts and event streams.
- ``GammaMixtureRegime`` — Gamma mixtures on ``R_{>0}^d`` for strictly positive
  physical scales, variances, and inter-event intervals.
- ``SimplexMixtureRegime`` — Dirichlet mixtures on the open simplex ``Delta^{d-1}``
  for probability vectors and normalized weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sciona.uncertainty._math import (
    digamma,
    loggamma,
    logsumexp,
    symmetrize,
    trigamma,
    woodbury_inverse_and_logdet,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _validate_weights(weights: Any, n_components: int) -> FloatArray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n_components or n_components < 1:
        raise ValueError(
            f"Expected {n_components} mixture weights (>= 1), got shape {w.shape}."
        )
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("Mixture weights must be finite and strictly positive.")
    total = float(np.sum(w))
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Mixture weights must sum to 1.0, got {total:.6f}.")
    return np.asarray(w / total, dtype=np.float64)


def _as_batch(z: Any, dim: int) -> tuple[FloatArray, bool]:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 0:
        if dim != 1:
            raise ValueError(f"Scalar input given to {dim}-D mixture.")
        return arr.reshape(1, 1), True
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"Expected vector of length {dim}, got {arr.shape[0]}.")
        return arr.reshape(1, dim), True
    if arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(f"Expected batch width {dim}, got {arr.shape[1]}.")
        return arr, False
    raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}.")


class ExponentialFamilyMixture(ABC):
    """Base class for finite mixtures of canonical exponential-family regimes."""

    family: str = "abstract"

    def __init__(self, weights: Any, dim: int) -> None:
        if dim < 1:
            raise ValueError("Dimension must be >= 1.")
        w_raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        self._weights = _validate_weights(w_raw, w_raw.size)
        self._dim = int(dim)

    @property
    def weights(self) -> FloatArray:
        return self._weights.copy()

    @property
    def n_components(self) -> int:
        return int(self._weights.size)

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Canonical exponential-family interface
    # ------------------------------------------------------------------

    @abstractmethod
    def in_support(self, z: FloatArray) -> BoolArray:
        """Boolean mask of shape ``(N,)`` indicating points inside support."""

    @abstractmethod
    def sufficient_statistics(self, z: FloatArray) -> FloatArray:
        """Evaluate canonical sufficient statistics ``T(z)`` of shape ``(N, s)``."""

    @abstractmethod
    def natural_params(self, k: int) -> FloatArray:
        """Natural parameter vector ``eta_k`` of shape ``(s,)`` for component ``k``."""

    @abstractmethod
    def expectation_params(self, k: int) -> FloatArray:
        """Expectation parameter ``E_k[T(Z)] = grad A(eta_k)`` of shape ``(s,)``."""

    @abstractmethod
    def log_partition(self, k: int) -> float:
        """Log-partition scalar ``A_k(eta_k)`` for component ``k``."""

    @abstractmethod
    def log_base_measure(self, z: FloatArray, k: int) -> FloatArray:
        """Log base measure ``log h_k(z)`` of shape ``(N,)`` on valid support."""

    @abstractmethod
    def natural_fisher_information(self, k: int) -> FloatArray:
        """Natural-parameter Fisher matrix ``I_k(eta) = grad^2 A(eta_k)``."""

    @abstractmethod
    def component_mean(self, k: int) -> FloatArray:
        """Mean ``E_k[Z]`` in ``R^d`` for component ``k``."""

    @abstractmethod
    def component_covariance(self, k: int) -> FloatArray:
        """Covariance ``Cov_k(Z)`` of shape ``(d, d)`` for component ``k``."""

    @abstractmethod
    def component_bregman_divergence(self, z: FloatArray, k: int) -> FloatArray:
        """Non-negative deviance / Bregman divergence ``D(z || component_k)``."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        """Draw ``n`` i.i.d. samples of shape ``(n, d)`` from the mixture."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize regime parameters to JSON-compatible dict."""

    # ------------------------------------------------------------------
    # Unified mixture evaluation, responsibilities, & moments
    # ------------------------------------------------------------------

    def component_log_prob(self, z: Any) -> FloatArray:
        """Evaluate ``log p(z | eta_k)`` returning shape ``(N, K)`` (or ``(K,)``)."""
        batch, squeeze = _as_batch(z, self._dim)
        n = batch.shape[0]
        k_count = self.n_components
        out = np.full((n, k_count), -np.inf, dtype=np.float64)

        valid = self.in_support(batch)
        if np.any(valid):
            z_valid = batch[valid]
            t_z = self.sufficient_statistics(z_valid)
            for k in range(k_count):
                eta = self.natural_params(k)
                a_eta = self.log_partition(k)
                log_h = self.log_base_measure(z_valid, k)
                out[valid, k] = log_h + (t_z @ eta) - a_eta

        return out[0] if squeeze else out

    def log_prob(self, z: Any) -> FloatArray | float:
        """Evaluate mixture log-density ``log sum_k pi_k p(z | eta_k)``."""
        batch, squeeze = _as_batch(z, self._dim)
        comp_lp = self.component_log_prob(batch)  # (N, K)
        weighted = comp_lp + np.log(self._weights)[None, :]
        lp = logsumexp(weighted, axis=1)
        return float(lp[0]) if squeeze else lp

    def nonconformity_score(self, z: Any) -> FloatArray | float:
        """Conformal OOD nonconformity score ``s(z) = -log p_cal(z)``."""
        lp = self.log_prob(z)
        if isinstance(lp, float):
            return -lp
        return np.asarray(-lp, dtype=np.float64)

    def responsibilities(self, z: Any) -> FloatArray:
        """Posterior component weights ``gamma_{n,k} = P(K=k | z_n)``."""
        batch, squeeze = _as_batch(z, self._dim)
        comp_lp = self.component_log_prob(batch)
        log_joint = comp_lp + np.log(self._weights)[None, :]
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)

        resp = np.empty_like(log_joint)
        finite_rows = np.isfinite(log_norm[:, 0])
        if np.any(finite_rows):
            resp[finite_rows] = np.exp(
                log_joint[finite_rows] - log_norm[finite_rows]
            )
        if np.any(~finite_rows):
            # Outside support: fall back to prior mixture weights
            resp[~finite_rows] = self._weights[None, :]
        return np.asarray(resp[0] if squeeze else resp, dtype=np.float64)

    def active_component(self, z: Any) -> NDArray[np.int64] | int:
        """Return MAP component index ``argmax_k P(K=k | z)``."""
        resp = self.responsibilities(z)
        if resp.ndim == 1:
            return int(np.argmax(resp))
        return np.asarray(np.argmax(resp, axis=1), dtype=np.int64)

    def min_bregman_divergence(self, z: Any) -> FloatArray | float:
        """Minimum weight-penalized Bregman divergence ``min_k [D_k(z) - log pi_k]``."""
        batch, squeeze = _as_batch(z, self._dim)
        n = batch.shape[0]
        valid = self.in_support(batch)
        scores = np.full((n, self.n_components), np.inf, dtype=np.float64)
        if np.any(valid):
            z_valid = batch[valid]
            for k in range(self.n_components):
                scores[valid, k] = self.component_bregman_divergence(
                    z_valid, k
                ) - float(np.log(self._weights[k]))
        best = np.min(scores, axis=1)
        return float(best[0]) if squeeze else best

    def mixture_mean(self) -> FloatArray:
        """Exact marginal mean ``E[Z] = sum_k pi_k E_k[Z]``."""
        mean = np.zeros(self._dim, dtype=np.float64)
        for k in range(self.n_components):
            mean += self._weights[k] * self.component_mean(k)
        return mean

    def mixture_covariance(self) -> FloatArray:
        """Exact marginal covariance via the law of total covariance."""
        overall_mean = self.mixture_mean()
        cov = np.zeros((self._dim, self._dim), dtype=np.float64)
        for k in range(self.n_components):
            mu_k = self.component_mean(k)
            diff = mu_k - overall_mean
            cov += self._weights[k] * (
                self.component_covariance(k) + np.outer(diff, diff)
            )
        return symmetrize(cov)

    def kl_collapse_moments(self) -> tuple[FloatArray, FloatArray]:
        """Collapse mixture to a single moment-matched ``(mean, covariance)``."""
        return self.mixture_mean(), self.mixture_covariance()


# ---------------------------------------------------------------------------
# 1. GaussianMixtureRegime (Factor-Analyzer: Sigma_k = diag(d_k) + W_k W_k^T)
# ---------------------------------------------------------------------------


class GaussianMixtureRegime(ExponentialFamilyMixture):
    """Gaussian mixture with low-rank + diagonal covariance per component.

    Each component ``k`` models ``Sigma_k = diag(diag_vars[k]) + W_k @ W_k.T``
    where ``W_k`` has shape ``(d, r)``.  Setting ``r = 0`` yields a diagonal GMM;
    setting ``r = d`` captures arbitrary full-rank correlations while remaining
    invertible in ``O(d r^2)`` via the Woodbury identity.
    """

    family = "gaussian"

    def __init__(
        self,
        weights: Any,
        means: Any,
        diag_vars: Any,
        factors: Any | None = None,
    ) -> None:
        means_arr = np.asarray(means, dtype=np.float64)
        if means_arr.ndim == 1:
            means_arr = means_arr.reshape(1, -1)
        if means_arr.ndim != 2:
            raise ValueError(f"means must have shape (K, d), got {means_arr.shape}.")
        k_count, dim = means_arr.shape
        super().__init__(weights, dim)
        if k_count != self.n_components:
            raise ValueError(
                f"means has {k_count} components, weights has {self.n_components}."
            )

        diag_arr = np.asarray(diag_vars, dtype=np.float64)
        if diag_arr.ndim == 1:
            diag_arr = diag_arr.reshape(1, -1)
        if diag_arr.shape != (k_count, dim) or np.any(diag_arr <= 0.0):
            raise ValueError(
                f"diag_vars must be strictly positive of shape {(k_count, dim)}."
            )

        if factors is None:
            factors_arr = np.zeros((k_count, dim, 0), dtype=np.float64)
        else:
            factors_arr = np.asarray(factors, dtype=np.float64)
            if factors_arr.ndim != 3 or factors_arr.shape[:2] != (k_count, dim):
                raise ValueError(
                    f"factors must have shape (K, d, r), got {factors_arr.shape}."
                )

        if not (np.all(np.isfinite(means_arr)) and np.all(np.isfinite(factors_arr))):
            raise ValueError("GaussianMixtureRegime parameters must be finite.")

        self._means = means_arr.copy()
        self._diag_vars = diag_arr.copy()
        self._factors = factors_arr.copy()

        # Precompute Woodbury precision and log-determinants per component
        self._precisions: list[FloatArray] = []
        self._logdets: list[float] = []
        for k in range(k_count):
            prec, ldet = woodbury_inverse_and_logdet(
                self._diag_vars[k], self._factors[k]
            )
            self._precisions.append(prec)
            self._logdets.append(ldet)

    @property
    def means(self) -> FloatArray:
        return self._means.copy()

    @property
    def diag_vars(self) -> FloatArray:
        return self._diag_vars.copy()

    @property
    def factors(self) -> FloatArray:
        return self._factors.copy()

    @property
    def factor_rank(self) -> int:
        return int(self._factors.shape[2])

    def in_support(self, z: FloatArray) -> BoolArray:
        return np.asarray(np.all(np.isfinite(z), axis=1), dtype=np.bool_)

    def sufficient_statistics(self, z: FloatArray) -> FloatArray:
        # T(z) = [z, vec(z z^T)] of shape (N, d + d^2)
        outer = (z[:, :, None] * z[:, None, :]).reshape(z.shape[0], -1)
        return np.concatenate([z, outer], axis=1)

    def natural_params(self, k: int) -> FloatArray:
        prec = self._precisions[k]
        eta_1 = prec @ self._means[k]
        eta_2 = -0.5 * prec.reshape(-1)
        return np.concatenate([eta_1, eta_2])

    def expectation_params(self, k: int) -> FloatArray:
        mu = self._means[k]
        second = self.component_covariance(k) + np.outer(mu, mu)
        return np.concatenate([mu, second.reshape(-1)])

    def log_partition(self, k: int) -> float:
        mu = self._means[k]
        quad = float(mu @ self._precisions[k] @ mu)
        return 0.5 * (quad + self._logdets[k])

    def log_base_measure(self, z: FloatArray, k: int) -> FloatArray:
        del k
        const = -0.5 * self._dim * np.log(2.0 * np.pi)
        return np.full(z.shape[0], const, dtype=np.float64)

    def component_log_prob(self, z: Any) -> FloatArray:
        """Fast quadratic evaluation avoiding explicit ``d^2`` outer products."""
        batch, squeeze = _as_batch(z, self._dim)
        n = batch.shape[0]
        out = np.full((n, self.n_components), -np.inf, dtype=np.float64)
        valid = self.in_support(batch)
        if np.any(valid):
            zv = batch[valid]
            log2pi = self._dim * np.log(2.0 * np.pi)
            for k in range(self.n_components):
                diff = zv - self._means[k]
                mahal2 = np.einsum("ni,ij,nj->n", diff, self._precisions[k], diff)
                out[valid, k] = -0.5 * (log2pi + self._logdets[k] + mahal2)
        return out[0] if squeeze else out

    def natural_fisher_information(self, k: int) -> FloatArray:
        """Location Fisher information matrix ``I_k(mu) = Sigma_k^{-1}``."""
        return self._precisions[k].copy()

    def component_precision(self, k: int) -> FloatArray:
        return self._precisions[k].copy()

    def component_mean(self, k: int) -> FloatArray:
        return np.asarray(self._means[k].copy(), dtype=np.float64)

    def component_covariance(self, k: int) -> FloatArray:
        w = self._factors[k]
        cov = np.diag(self._diag_vars[k])
        if w.shape[1] > 0:
            cov = cov + w @ w.T
        return symmetrize(cov)

    def component_bregman_divergence(self, z: FloatArray, k: int) -> FloatArray:
        """Half squared Mahalanobis distance ``0.5 * (z - mu_k)^T Sigma_k^{-1} (z - mu_k)``."""
        diff = z - self._means[k]
        mahal2 = np.einsum("ni,ij,nj->n", diff, self._precisions[k], diff)
        return np.asarray(np.maximum(0.5 * mahal2, 0.0), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        if n < 0:
            raise ValueError("n must be >= 0")
        gen = rng or np.random.default_rng()
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float64)
        comp_idx = gen.choice(self.n_components, size=n, p=self._weights)
        samples = np.empty((n, self._dim), dtype=np.float64)
        r = self.factor_rank
        for k in range(self.n_components):
            mask = comp_idx == k
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            eps_diag = gen.standard_normal((cnt, self._dim)) * np.sqrt(
                self._diag_vars[k]
            )
            if r > 0:
                latent = gen.standard_normal((cnt, r))
                eps_lowrank = latent @ self._factors[k].T
            else:
                eps_lowrank = 0.0
            samples[mask] = self._means[k] + eps_diag + eps_lowrank
        return samples

    def kl_reduce(
        self,
        max_components: int,
        *,
        factor_rank: int | None = None,
        min_diag: float = 1e-9,
    ) -> GaussianMixtureRegime:
        """Greedy Runnalls (2007) KL-bound mixture reduction to ``<= max_components``.

        Prevents ``K^L`` component explosion when pushing Gaussian mixtures
        forward across multi-hop CDGs while preserving first and second moments.
        """
        if max_components < 1:
            raise ValueError("max_components must be >= 1.")
        rank = self.factor_rank if factor_rank is None else max(0, int(factor_rank))
        if self.n_components <= max_components and rank == self.factor_rank:
            return self

        weights = list(float(w) for w in self._weights)
        means = [self.component_mean(k) for k in range(self.n_components)]
        covs = [self.component_covariance(k) for k in range(self.n_components)]
        logdets = list(self._logdets)

        def _merged_triple(i: int, j: int) -> tuple[float, FloatArray, FloatArray, float]:
            w_ij = weights[i] + weights[j]
            alpha_i = weights[i] / w_ij
            alpha_j = weights[j] / w_ij
            mu_ij = alpha_i * means[i] + alpha_j * means[j]
            diff = means[i] - means[j]
            cov_ij = symmetrize(
                alpha_i * covs[i]
                + alpha_j * covs[j]
                + alpha_i * alpha_j * np.outer(diff, diff)
            )
            # Ensure strict SPD
            eigvals = np.linalg.eigvalsh(cov_ij)
            if eigvals[0] <= min_diag:
                cov_ij = cov_ij + (min_diag - eigvals[0] + min_diag) * np.eye(self._dim)
            sign, ldet = np.linalg.slogdet(cov_ij)
            if sign <= 0:
                ldet = float(np.sum(np.log(np.maximum(eigvals, min_diag))))
            return w_ij, mu_ij, cov_ij, float(ldet)

        while len(weights) > max_components:
            best_pair = (0, 1)
            best_bound = float("inf")
            best_merged: tuple[float, FloatArray, FloatArray, float] | None = None
            m_cur = len(weights)
            for i in range(m_cur):
                for j in range(i + 1, m_cur):
                    w_ij, mu_ij, cov_ij, ldet_ij = _merged_triple(i, j)
                    # Runnalls upper bound on KL divergence increase:
                    bound = 0.5 * (
                        w_ij * ldet_ij
                        - weights[i] * logdets[i]
                        - weights[j] * logdets[j]
                    )
                    if bound < best_bound:
                        best_bound = bound
                        best_pair = (i, j)
                        best_merged = (w_ij, mu_ij, cov_ij, ldet_ij)

            assert best_merged is not None
            i, j = best_pair
            w_ij, mu_ij, cov_ij, ldet_ij = best_merged
            # Replace i with merged, remove j
            weights[i] = w_ij
            means[i] = mu_ij
            covs[i] = cov_ij
            logdets[i] = ldet_ij
            del weights[j]
            del means[j]
            del covs[j]
            del logdets[j]

        return GaussianMixtureRegime.from_full_covariances(
            weights=np.asarray(weights, dtype=np.float64),
            means=np.stack(means, axis=0),
            covariances=np.stack(covs, axis=0),
            factor_rank=rank,
            min_diag=min_diag,
        )

    @classmethod
    def from_full_covariances(
        cls,
        weights: Any,
        means: Any,
        covariances: Any,
        *,
        factor_rank: int = 4,
        min_diag: float = 1e-6,
    ) -> GaussianMixtureRegime:
        """Factorize full SPD covariances into ``diag(d_k) + W_k W_k^T``."""
        cov_arr = np.asarray(covariances, dtype=np.float64)
        means_arr = np.asarray(means, dtype=np.float64)
        if means_arr.ndim == 1:
            means_arr = means_arr.reshape(1, -1)
        if cov_arr.ndim == 2:
            cov_arr = cov_arr.reshape(1, cov_arr.shape[0], cov_arr.shape[1])

        k_count, dim = means_arr.shape
        rank = max(0, min(int(factor_rank), dim))
        diag_out = np.empty((k_count, dim), dtype=np.float64)
        factors_out = np.zeros((k_count, dim, rank), dtype=np.float64)

        for k in range(k_count):
            sym = symmetrize(cov_arr[k])
            if rank == 0:
                diag_out[k] = np.maximum(np.diag(sym), min_diag)
                continue
            eigvals, eigvecs = np.linalg.eigh(sym)
            # Descending order
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            if rank < dim:
                tail_floor = float(
                    max(np.mean(np.maximum(eigvals[rank:], min_diag)), min_diag)
                )
            else:
                tail_floor = float(min_diag)
            excess = np.maximum(eigvals[:rank] - tail_floor, 0.0)
            w_k = eigvecs[:, :rank] * np.sqrt(excess)[None, :]
            residual_diag = np.maximum(
                np.diag(sym - w_k @ w_k.T),
                min_diag,
            )
            diag_out[k] = residual_diag
            factors_out[k] = w_k

        return cls(
            weights=weights,
            means=means_arr,
            diag_vars=diag_out,
            factors=factors_out,
        )

    @classmethod
    def fit_em(
        cls,
        data: Any,
        *,
        n_components: int = 2,
        factor_rank: int = 2,
        max_iter: int = 60,
        tol: float = 1e-4,
        reg_covar: float = 1e-5,
        seed: int = 0,
    ) -> GaussianMixtureRegime:
        """Fit a Factor-Analyzer ``GaussianMixtureRegime`` via EM."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, dim = x.shape
        if n_samples < 1:
            raise ValueError("Cannot fit GaussianMixtureRegime on empty data.")
        k_count = max(1, min(int(n_components), n_samples))
        rng = np.random.default_rng(seed)

        # K-means++ style seeding
        centers = np.empty((k_count, dim), dtype=np.float64)
        first_idx = int(rng.integers(0, n_samples))
        centers[0] = x[first_idx]
        for k in range(1, k_count):
            dists = np.min(
                np.sum((x[:, None, :] - centers[None, :k, :]) ** 2, axis=2),
                axis=1,
            )
            total_d = float(np.sum(dists))
            if total_d > 1e-18:
                probs = dists / total_d
                idx = int(rng.choice(n_samples, p=probs))
            else:
                idx = int(rng.integers(0, n_samples))
            centers[k] = x[idx]

        global_var = np.maximum(np.var(x, axis=0), reg_covar)
        weights = np.full(k_count, 1.0 / k_count, dtype=np.float64)
        covs = np.repeat(
            (np.diag(global_var) + reg_covar * np.eye(dim))[None, :, :],
            k_count,
            axis=0,
        )

        model = cls.from_full_covariances(
            weights=weights,
            means=centers,
            covariances=covs,
            factor_rank=factor_rank,
            min_diag=reg_covar,
        )
        prev_ll = -float("inf")

        for _ in range(max_iter):
            lp = np.asarray(model.log_prob(x), dtype=np.float64)
            ll = float(np.mean(lp))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            resp = model.responsibilities(x)  # (N, K)
            nk = np.maximum(np.sum(resp, axis=0), 1e-12)
            new_weights = np.maximum(nk / n_samples, 1e-8)
            new_weights /= float(np.sum(new_weights))
            new_means = (resp.T @ x) / nk[:, None]

            new_covs = np.empty((k_count, dim, dim), dtype=np.float64)
            for k in range(k_count):
                diff = x - new_means[k]
                weighted_diff = resp[:, k : k + 1] * diff
                cov_k = (weighted_diff.T @ diff) / nk[k]
                new_covs[k] = symmetrize(cov_k) + reg_covar * np.eye(dim)

            model = cls.from_full_covariances(
                weights=new_weights,
                means=new_means,
                covariances=new_covs,
                factor_rank=factor_rank,
                min_diag=reg_covar,
            )

        return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dim": self._dim,
            "weights": self._weights.tolist(),
            "means": self._means.tolist(),
            "diag_vars": self._diag_vars.tolist(),
            "factors": self._factors.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GaussianMixtureRegime:
        factors_raw = payload.get("factors")
        return cls(
            weights=payload["weights"],
            means=payload["means"],
            diag_vars=payload["diag_vars"],
            factors=factors_raw,
        )


# ---------------------------------------------------------------------------
# 2. CountMixtureRegime (Poisson & Negative-Binomial on R_{>=0}^d)
# ---------------------------------------------------------------------------


class CountMixtureRegime(ExponentialFamilyMixture):
    """Poisson or overdispersed Negative-Binomial mixture on ``R_{>=0}^d``.

    When ``dispersions`` is ``None``, models independent Poisson rates
    ``rates[k, j] > 0``.  When ``dispersions[k, j] = r_{k,j} > 0`` is given,
    models Negative-Binomial components with mean ``lambda_{k,j}`` and variance
    ``lambda_{k,j} + lambda_{k,j}^2 / r_{k,j}`` (reducing smoothly to Poisson
    as ``r -> inf``).
    """

    family = "count"

    def __init__(
        self,
        weights: Any,
        rates: Any,
        dispersions: Any | None = None,
    ) -> None:
        rates_arr = np.asarray(rates, dtype=np.float64)
        if rates_arr.ndim == 1:
            rates_arr = rates_arr.reshape(1, -1)
        if rates_arr.ndim != 2 or np.any(rates_arr <= 0.0) or np.any(~np.isfinite(rates_arr)):
            raise ValueError("CountMixtureRegime rates must be finite, > 0, shape (K, d).")
        k_count, dim = rates_arr.shape
        super().__init__(weights, dim)
        if k_count != self.n_components:
            raise ValueError("CountMixtureRegime rates/weights component count mismatch.")

        self._rates = rates_arr.copy()
        if dispersions is None:
            self._dispersions: FloatArray | None = None
        else:
            disp_arr = np.asarray(dispersions, dtype=np.float64)
            if disp_arr.ndim == 1:
                disp_arr = disp_arr.reshape(1, -1)
            if disp_arr.shape != (k_count, dim) or np.any(disp_arr <= 0.0) or np.any(~np.isfinite(disp_arr)):
                raise ValueError(
                    f"dispersions must be finite, > 0, shape {(k_count, dim)}."
                )
            self._dispersions = disp_arr.copy()

    @property
    def rates(self) -> FloatArray:
        return self._rates.copy()

    @property
    def dispersions(self) -> FloatArray | None:
        return None if self._dispersions is None else self._dispersions.copy()

    def in_support(self, z: FloatArray) -> BoolArray:
        return np.asarray(
            np.all(np.isfinite(z) & (z >= 0.0), axis=1),
            dtype=np.bool_,
        )

    def sufficient_statistics(self, z: FloatArray) -> FloatArray:
        return z.copy()

    def natural_params(self, k: int) -> FloatArray:
        lam = self._rates[k]
        if self._dispersions is None:
            return np.asarray(np.log(lam), dtype=np.float64)
        r = self._dispersions[k]
        return np.asarray(np.log(lam / (lam + r)), dtype=np.float64)

    def expectation_params(self, k: int) -> FloatArray:
        return np.asarray(self._rates[k].copy(), dtype=np.float64)

    def log_partition(self, k: int) -> float:
        lam = self._rates[k]
        if self._dispersions is None:
            return float(np.sum(lam))
        r = self._dispersions[k]
        return float(np.sum(r * np.log((lam + r) / r)))

    def log_base_measure(self, z: FloatArray, k: int) -> FloatArray:
        log_fact = loggamma(z + 1.0)
        if self._dispersions is None:
            return np.asarray(-np.sum(log_fact, axis=1), dtype=np.float64)
        r = self._dispersions[k][None, :]
        return np.asarray(
            np.sum(loggamma(z + r) - loggamma(r) - log_fact, axis=1),
            dtype=np.float64,
        )

    def natural_fisher_information(self, k: int) -> FloatArray:
        return self.component_covariance(k)

    def component_mean(self, k: int) -> FloatArray:
        return np.asarray(self._rates[k].copy(), dtype=np.float64)

    def component_covariance(self, k: int) -> FloatArray:
        lam = self._rates[k]
        if self._dispersions is None:
            return np.diag(lam)
        r = self._dispersions[k]
        return np.diag(lam + (lam * lam) / r)

    def component_bregman_divergence(self, z: FloatArray, k: int) -> FloatArray:
        lam = self._rates[k][None, :]
        # Safe z * log(z / lam) with 0 * log(0) = 0
        pos = z > 0.0
        ratio_term = np.zeros_like(z)
        ratio_term[pos] = z[pos] * np.log(z[pos] / np.broadcast_to(lam, z.shape)[pos])
        if self._dispersions is None:
            dev = ratio_term - (z - lam)
        else:
            r = self._dispersions[k][None, :]
            dev = ratio_term - (z + r) * np.log((z + r) / (lam + r))
        return np.asarray(np.maximum(np.sum(dev, axis=1), 0.0), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        if n < 0:
            raise ValueError("n must be >= 0")
        gen = rng or np.random.default_rng()
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float64)
        comp_idx = gen.choice(self.n_components, size=n, p=self._weights)
        out = np.empty((n, self._dim), dtype=np.float64)
        for k in range(self.n_components):
            mask = comp_idx == k
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            lam = self._rates[k]
            if self._dispersions is None:
                out[mask] = gen.poisson(lam=lam, size=(cnt, self._dim)).astype(np.float64)
            else:
                # Gamma-Poisson mixture representation of Negative Binomial
                r = self._dispersions[k]
                gamma_rates = gen.gamma(shape=r, scale=lam / r, size=(cnt, self._dim))
                out[mask] = gen.poisson(lam=gamma_rates).astype(np.float64)
        return out

    @classmethod
    def fit_em(
        cls,
        data: Any,
        *,
        n_components: int = 2,
        estimate_dispersion: bool = True,
        max_iter: int = 50,
        tol: float = 1e-4,
        min_rate: float = 1e-4,
        seed: int = 0,
    ) -> CountMixtureRegime:
        """Fit a Poisson / Negative-Binomial mixture via EM + moment dispersion."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] < 1 or np.any(x < 0.0):
            raise ValueError("CountMixtureRegime.fit_em requires non-empty non-negative data.")
        n_samples, dim = x.shape
        k_count = max(1, min(int(n_components), n_samples))
        rng = np.random.default_rng(seed)

        idx = rng.choice(n_samples, size=k_count, replace=False)
        rates = np.maximum(x[idx], min_rate)
        weights = np.full(k_count, 1.0 / k_count, dtype=np.float64)
        dispersions: FloatArray | None = None

        model = cls(weights=weights, rates=rates, dispersions=dispersions)
        prev_ll = -float("inf")
        for _ in range(max_iter):
            lp = np.asarray(model.log_prob(x), dtype=np.float64)
            ll = float(np.mean(lp))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            resp = model.responsibilities(x)
            nk = np.maximum(np.sum(resp, axis=0), 1e-12)
            weights = np.maximum(nk / n_samples, 1e-8)
            weights /= float(np.sum(weights))
            rates = np.maximum((resp.T @ x) / nk[:, None], min_rate)

            if estimate_dispersion:
                disp = np.full((k_count, dim), 1e4, dtype=np.float64)
                has_overdisp = False
                for k in range(k_count):
                    diff2 = (x - rates[k]) ** 2
                    var_k = (resp[:, k] @ diff2) / nk[k]
                    excess = var_k - rates[k]
                    over = excess > 1e-3 * rates[k]
                    if np.any(over):
                        has_overdisp = True
                        r_est = np.where(
                            over,
                            (rates[k] ** 2) / np.maximum(excess, 1e-9),
                            1e4,
                        )
                        disp[k] = np.clip(r_est, 1e-2, 1e6)
                dispersions = disp if has_overdisp else None

            model = cls(weights=weights, rates=rates, dispersions=dispersions)
        return model

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family": self.family,
            "dim": self._dim,
            "weights": self._weights.tolist(),
            "rates": self._rates.tolist(),
        }
        if self._dispersions is not None:
            payload["dispersions"] = self._dispersions.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CountMixtureRegime:
        return cls(
            weights=payload["weights"],
            rates=payload["rates"],
            dispersions=payload.get("dispersions"),
        )


# ---------------------------------------------------------------------------
# 3. GammaMixtureRegime (Strictly Positive Scales on R_{>0}^d)
# ---------------------------------------------------------------------------


class GammaMixtureRegime(ExponentialFamilyMixture):
    """Independent-coordinate Gamma mixture on ``R_{>0}^d`` with (shape, rate)."""

    family = "gamma"

    def __init__(self, weights: Any, shapes: Any, rates: Any) -> None:
        shapes_arr = np.asarray(shapes, dtype=np.float64)
        rates_arr = np.asarray(rates, dtype=np.float64)
        if shapes_arr.ndim == 1:
            shapes_arr = shapes_arr.reshape(1, -1)
        if rates_arr.ndim == 1:
            rates_arr = rates_arr.reshape(1, -1)
        if (
            shapes_arr.ndim != 2
            or shapes_arr.shape != rates_arr.shape
            or np.any(shapes_arr <= 0.0)
            or np.any(rates_arr <= 0.0)
            or np.any(~np.isfinite(shapes_arr))
            or np.any(~np.isfinite(rates_arr))
        ):
            raise ValueError(
                "GammaMixtureRegime shapes and rates must be finite, > 0, matching (K, d)."
            )
        k_count, dim = shapes_arr.shape
        super().__init__(weights, dim)
        if k_count != self.n_components:
            raise ValueError("GammaMixtureRegime shapes/weights component count mismatch.")
        self._shapes = shapes_arr.copy()
        self._rates = rates_arr.copy()

    @property
    def shapes(self) -> FloatArray:
        return self._shapes.copy()

    @property
    def rates(self) -> FloatArray:
        return self._rates.copy()

    def in_support(self, z: FloatArray) -> BoolArray:
        return np.asarray(
            np.all(np.isfinite(z) & (z > 0.0), axis=1),
            dtype=np.bool_,
        )

    def sufficient_statistics(self, z: FloatArray) -> FloatArray:
        # T(z) = [log z, -z] of shape (N, 2d)
        return np.concatenate([np.log(z), -z], axis=1)

    def natural_params(self, k: int) -> FloatArray:
        return np.concatenate([self._shapes[k] - 1.0, self._rates[k]])

    def expectation_params(self, k: int) -> FloatArray:
        alpha = self._shapes[k]
        beta = self._rates[k]
        return np.concatenate([digamma(alpha) - np.log(beta), -alpha / beta])

    def log_partition(self, k: int) -> float:
        alpha = self._shapes[k]
        beta = self._rates[k]
        return float(np.sum(loggamma(alpha) - alpha * np.log(beta)))

    def log_base_measure(self, z: FloatArray, k: int) -> FloatArray:
        del k
        return np.zeros(z.shape[0], dtype=np.float64)

    def natural_fisher_information(self, k: int) -> FloatArray:
        """2d x 2d Hessian ``grad^2 A(eta_k)`` wrt ``(alpha - 1, beta)``."""
        alpha = self._shapes[k]
        beta = self._rates[k]
        d = self._dim
        fim = np.zeros((2 * d, 2 * d), dtype=np.float64)
        psi1 = trigamma(alpha)
        for j in range(d):
            fim[j, j] = float(psi1[j])
            fim[j, d + j] = float(-1.0 / beta[j])
            fim[d + j, j] = float(-1.0 / beta[j])
            fim[d + j, d + j] = float(alpha[j] / (beta[j] * beta[j]))
        return fim

    def component_mean(self, k: int) -> FloatArray:
        return np.asarray(self._shapes[k] / self._rates[k], dtype=np.float64)

    def component_covariance(self, k: int) -> FloatArray:
        var = self._shapes[k] / (self._rates[k] ** 2)
        return np.diag(var)

    def component_bregman_divergence(self, z: FloatArray, k: int) -> FloatArray:
        """Shape-weighted Itakura-Saito / Gamma deviance ``sum_j alpha_j (z_j/m_j - log(z_j/m_j) - 1)``."""
        alpha = self._shapes[k][None, :]
        mean = (self._shapes[k] / self._rates[k])[None, :]
        ratio = z / mean
        dev = alpha * (ratio - np.log(ratio) - 1.0)
        return np.asarray(np.maximum(np.sum(dev, axis=1), 0.0), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        if n < 0:
            raise ValueError("n must be >= 0")
        gen = rng or np.random.default_rng()
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float64)
        comp_idx = gen.choice(self.n_components, size=n, p=self._weights)
        out = np.empty((n, self._dim), dtype=np.float64)
        for k in range(self.n_components):
            mask = comp_idx == k
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            out[mask] = gen.gamma(
                shape=self._shapes[k],
                scale=1.0 / self._rates[k],
                size=(cnt, self._dim),
            )
        return out

    @classmethod
    def fit_em(
        cls,
        data: Any,
        *,
        n_components: int = 2,
        max_iter: int = 50,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> GammaMixtureRegime:
        """Fit a ``GammaMixtureRegime`` via EM with Minka-Newton shape updates."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] < 1 or np.any(x <= 0.0):
            raise ValueError("GammaMixtureRegime.fit_em requires strictly positive data.")
        n_samples, dim = x.shape
        k_count = max(1, min(int(n_components), n_samples))
        rng = np.random.default_rng(seed)

        log_x = np.log(x)
        idx = rng.choice(n_samples, size=k_count, replace=False)
        means = np.maximum(x[idx], 1e-6)
        global_var = np.maximum(np.var(x, axis=0), 1e-6)
        shapes = np.clip((means ** 2) / global_var[None, :], 0.25, 100.0)
        rates = np.maximum(shapes / means, 1e-6)
        weights = np.full(k_count, 1.0 / k_count, dtype=np.float64)

        model = cls(weights=weights, shapes=shapes, rates=rates)
        prev_ll = -float("inf")
        for _ in range(max_iter):
            lp = np.asarray(model.log_prob(x), dtype=np.float64)
            ll = float(np.mean(lp))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            resp = model.responsibilities(x)
            nk = np.maximum(np.sum(resp, axis=0), 1e-12)
            weights = np.maximum(nk / n_samples, 1e-8)
            weights /= float(np.sum(weights))

            bar_x = np.maximum((resp.T @ x) / nk[:, None], 1e-9)
            bar_log_x = (resp.T @ log_x) / nk[:, None]
            # Generalized log-mean gap s = log(bar_x) - bar_log_x > 0
            s = np.maximum(np.log(bar_x) - bar_log_x, 1e-6)
            # Choi & Wette initial approximation + 3 Minka-Newton steps
            alpha = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
            alpha = np.clip(alpha, 1e-2, 1e4)
            for _newton in range(3):
                inv_a = 1.0 / alpha + (bar_log_x - np.log(bar_x) + np.log(alpha) - digamma(alpha)) / np.maximum(
                    alpha * alpha * (1.0 / alpha - trigamma(alpha)),
                    1e-12,
                )
                alpha = np.clip(1.0 / np.maximum(inv_a, 1e-6), 1e-2, 1e4)
            rates = np.maximum(alpha / bar_x, 1e-9)
            model = cls(weights=weights, shapes=alpha, rates=rates)
        return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dim": self._dim,
            "weights": self._weights.tolist(),
            "shapes": self._shapes.tolist(),
            "rates": self._rates.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GammaMixtureRegime:
        return cls(
            weights=payload["weights"],
            shapes=payload["shapes"],
            rates=payload["rates"],
        )


# ---------------------------------------------------------------------------
# 4. SimplexMixtureRegime (Dirichlet Mixture on Open Simplex Delta^{d-1})
# ---------------------------------------------------------------------------


class SimplexMixtureRegime(ExponentialFamilyMixture):
    """Dirichlet mixture on the open probability simplex ``Delta^{d-1}``."""

    family = "simplex"

    def __init__(self, weights: Any, concentrations: Any) -> None:
        conc = np.asarray(concentrations, dtype=np.float64)
        if conc.ndim == 1:
            conc = conc.reshape(1, -1)
        if conc.ndim != 2 or conc.shape[1] < 2 or np.any(conc <= 0.0) or np.any(~np.isfinite(conc)):
            raise ValueError(
                "SimplexMixtureRegime concentrations must be finite, > 0, shape (K, d>=2)."
            )
        k_count, dim = conc.shape
        super().__init__(weights, dim)
        if k_count != self.n_components:
            raise ValueError("SimplexMixtureRegime concentrations/weights mismatch.")
        self._concentrations = conc.copy()

    @property
    def concentrations(self) -> FloatArray:
        return self._concentrations.copy()

    def in_support(self, z: FloatArray) -> BoolArray:
        finite_pos = np.all(np.isfinite(z) & (z > 0.0), axis=1)
        sum_one = np.abs(np.sum(z, axis=1) - 1.0) <= 1e-4
        return np.asarray(finite_pos & sum_one, dtype=np.bool_)

    def sufficient_statistics(self, z: FloatArray) -> FloatArray:
        return np.asarray(np.log(z), dtype=np.float64)

    def natural_params(self, k: int) -> FloatArray:
        return np.asarray(self._concentrations[k] - 1.0, dtype=np.float64)

    def expectation_params(self, k: int) -> FloatArray:
        alpha = self._concentrations[k]
        alpha0 = float(np.sum(alpha))
        return np.asarray(digamma(alpha) - digamma(alpha0), dtype=np.float64)

    def log_partition(self, k: int) -> float:
        alpha = self._concentrations[k]
        alpha0 = float(np.sum(alpha))
        return float(np.sum(loggamma(alpha)) - float(loggamma(alpha0)))

    def log_base_measure(self, z: FloatArray, k: int) -> FloatArray:
        del k
        return np.zeros(z.shape[0], dtype=np.float64)

    def natural_fisher_information(self, k: int) -> FloatArray:
        """Exact Dirichlet FIM ``diag(psi_1(alpha)) - psi_1(sum alpha) 1 1^T``."""
        alpha = self._concentrations[k]
        alpha0 = float(np.sum(alpha))
        psi1 = trigamma(alpha)
        psi1_0 = float(trigamma(alpha0))
        return symmetrize(np.diag(psi1) - psi1_0 * np.ones((self._dim, self._dim)))

    def component_mean(self, k: int) -> FloatArray:
        alpha = self._concentrations[k]
        return np.asarray(alpha / float(np.sum(alpha)), dtype=np.float64)

    def component_covariance(self, k: int) -> FloatArray:
        mean = self.component_mean(k)
        alpha0 = float(np.sum(self._concentrations[k]))
        return symmetrize((np.diag(mean) - np.outer(mean, mean)) / (alpha0 + 1.0))

    def component_bregman_divergence(self, z: FloatArray, k: int) -> FloatArray:
        """KL divergence from empirical coordinate vector ``z`` to Dirichlet mean ``m_k``
        scaled by concentration ``alpha_{k,0}``.
        """
        alpha0 = float(np.sum(self._concentrations[k]))
        mean = self.component_mean(k)[None, :]
        kl = np.sum(z * np.log(z / mean), axis=1)
        return np.asarray(np.maximum(alpha0 * kl, 0.0), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        if n < 0:
            raise ValueError("n must be >= 0")
        gen = rng or np.random.default_rng()
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float64)
        comp_idx = gen.choice(self.n_components, size=n, p=self._weights)
        out = np.empty((n, self._dim), dtype=np.float64)
        for k in range(self.n_components):
            mask = comp_idx == k
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            out[mask] = gen.dirichlet(self._concentrations[k], size=cnt)
        return out

    @classmethod
    def fit_em(
        cls,
        data: Any,
        *,
        n_components: int = 2,
        max_iter: int = 50,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> SimplexMixtureRegime:
        """Fit a Dirichlet mixture via EM with fixed-point Minka updates."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
            raise ValueError("SimplexMixtureRegime.fit_em requires (N, d>=2) simplex data.")
        x_clipped = np.maximum(x, 1e-9)
        x_norm = x_clipped / np.sum(x_clipped, axis=1, keepdims=True)
        log_x = np.log(x_norm)

        n_samples, dim = x_norm.shape
        k_count = max(1, min(int(n_components), n_samples))
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, size=k_count, replace=False)
        conc = np.maximum(x_norm[idx] * float(dim * 4), 0.5)
        weights = np.full(k_count, 1.0 / k_count, dtype=np.float64)

        model = cls(weights=weights, concentrations=conc)
        prev_ll = -float("inf")
        for _ in range(max_iter):
            lp = np.asarray(model.log_prob(x_norm), dtype=np.float64)
            ll = float(np.mean(lp))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            resp = model.responsibilities(x_norm)
            nk = np.maximum(np.sum(resp, axis=0), 1e-12)
            weights = np.maximum(nk / n_samples, 1e-8)
            weights /= float(np.sum(weights))

            bar_log_x = (resp.T @ log_x) / nk[:, None]
            new_conc = model.concentrations
            for k in range(k_count):
                alpha_k = new_conc[k].copy()
                for _fp in range(8):
                    alpha0 = float(np.sum(alpha_k))
                    target_psi = bar_log_x[k] + float(digamma(alpha0))
                    # Newton step to invert digamma(alpha_j) = target_psi_j
                    for _nw in range(3):
                        alpha_k = np.clip(
                            alpha_k - (digamma(alpha_k) - target_psi) / trigamma(alpha_k),
                            1e-2,
                            1e4,
                        )
                new_conc[k] = alpha_k
            model = cls(weights=weights, concentrations=new_conc)
        return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dim": self._dim,
            "weights": self._weights.tolist(),
            "concentrations": self._concentrations.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimplexMixtureRegime:
        return cls(
            weights=payload["weights"],
            concentrations=payload["concentrations"],
        )


def mixture_from_dict(payload: dict[str, Any]) -> ExponentialFamilyMixture:
    """Polymorphic deserializer for ``ExponentialFamilyMixture`` subclasses."""
    family = str(payload.get("family", "")).strip().lower()
    if family == "gaussian":
        return GaussianMixtureRegime.from_dict(payload)
    if family in {"count", "poisson", "negative_binomial"}:
        return CountMixtureRegime.from_dict(payload)
    if family == "gamma":
        return GammaMixtureRegime.from_dict(payload)
    if family in {"simplex", "dirichlet"}:
        return SimplexMixtureRegime.from_dict(payload)
    raise ValueError(f"Unsupported ExponentialFamilyMixture family: {family!r}")
