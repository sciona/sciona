"""Low-rank + isotropic-tail port operators, Fisher pullbacks, and Stein Jacobians.

Represents local linearized sensitivity ``J_k: R^{d_in} -> R^{d_out}`` inside a
calibrated mixture component ``k`` without dense ``O(d_out * d_in)`` storage:

    J_k ≈ U_r diag(sigma_1, ..., sigma_r) V_r^T  +  sigma_tail * Pi_perp

where:
- ``(U_r, sigma_1..sigma_r, V_r)`` capture **stiff** singular directions, and
- ``sigma_tail >= 0`` bounds the **sloppy** residual spectrum on ``span(V_r)^perp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from sciona.uncertainty._math import spd_eigendecomp, symmetrize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpectralSummary:
    """Stiff/sloppy spectral characterization of a port operator or FIM."""

    spectral_norm: float
    """Largest singular value ``sigma_max`` (worst-case amplification)."""
    sloppy_floor: float
    """Smallest tracked singular value / isotropic residual ``sigma_tail``."""
    condition_number: float
    """Ratio ``spectral_norm / max(sloppy_floor, 1e-15)``."""
    effective_dimension: float
    """Participation ratio / ridge effective dimension ``sum_i lambda_i / (lambda_i + reg)``."""
    stiff_rank: int
    """Number of singular values strictly above ``stiff_threshold``."""
    sloppy_rank: int
    """Number of input dimensions at or below ``sloppy_threshold``."""


@dataclass(frozen=True)
class PullbackResult:
    """Result of pulling a downstream precision budget backward across a port."""

    required_input_precision: FloatArray
    """Symmetric positive semi-definite matrix ``Lambda_req`` on ``R^{d_in}``."""
    feasible: bool
    """False if intrinsic noise ``Sigma_eps`` alone violates ``Lambda_out^{-1}``."""
    slack_min_eigenvalue: float
    """Minimum eigenvalue of ``Lambda_out^{-1} - Sigma_eps`` (<= 0 means infeasible)."""
    violation_reason: str = ""


class LowRankPlusTailOperator:
    """Factored Jacobian ``U diag(s) V^T + s_tail * Pi_perp`` between projected ports."""

    def __init__(
        self,
        dim_out: int,
        dim_in: int,
        singular_values: Any = (),
        left_vectors: Any | None = None,
        right_vectors: Any | None = None,
        tail_singular_value: float = 0.0,
    ) -> None:
        if dim_out < 1 or dim_in < 1:
            raise ValueError("Operator dimensions dim_out and dim_in must be >= 1.")
        if tail_singular_value < 0.0 or not np.isfinite(tail_singular_value):
            raise ValueError("tail_singular_value must be finite and >= 0.")

        s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
        r = int(s.size)
        if r > min(dim_out, dim_in):
            raise ValueError(
                f"Rank {r} cannot exceed min(dim_out={dim_out}, dim_in={dim_in})."
            )
        if np.any(~np.isfinite(s)) or np.any(s < 0.0):
            raise ValueError("Singular values must be finite and >= 0.")

        if r == 0:
            u = np.zeros((dim_out, 0), dtype=np.float64)
            v = np.zeros((dim_in, 0), dtype=np.float64)
        else:
            if left_vectors is None or right_vectors is None:
                raise ValueError(
                    "left_vectors and right_vectors are required when rank > 0."
                )
            u = np.asarray(left_vectors, dtype=np.float64)
            v = np.asarray(right_vectors, dtype=np.float64)
            if u.shape != (dim_out, r) or v.shape != (dim_in, r):
                raise ValueError(
                    f"Expected U={(dim_out, r)} and V={(dim_in, r)}, got U={u.shape}, V={v.shape}."
                )
            # Sort singular triplets descending
            order = np.argsort(-s)
            s = s[order]
            u = u[:, order]
            v = v[:, order]

        self._dim_out = int(dim_out)
        self._dim_in = int(dim_in)
        self._singular_values = s
        self._left_vectors = u
        self._right_vectors = v
        self._tail = float(tail_singular_value)

    @property
    def dim_out(self) -> int:
        return self._dim_out

    @property
    def dim_in(self) -> int:
        return self._dim_in

    @property
    def rank(self) -> int:
        return int(self._singular_values.size)

    @property
    def singular_values(self) -> FloatArray:
        return self._singular_values.copy()

    @property
    def left_vectors(self) -> FloatArray:
        return self._left_vectors.copy()

    @property
    def right_vectors(self) -> FloatArray:
        return self._right_vectors.copy()

    @property
    def tail_singular_value(self) -> float:
        return self._tail

    @property
    def spectral_norm(self) -> float:
        if self._singular_values.size > 0:
            return float(max(self._singular_values[0], self._tail))
        return self._tail

    @classmethod
    def from_scalar_factor(
        cls,
        factor: float,
        *,
        dim_out: int = 1,
        dim_in: int = 1,
    ) -> LowRankPlusTailOperator:
        """Construct an isotropic rank-0 tail operator (v1 backward compatibility)."""
        return cls(
            dim_out=dim_out,
            dim_in=dim_in,
            singular_values=(),
            tail_singular_value=max(0.0, float(factor)),
        )

    @classmethod
    def from_dense(
        cls,
        jacobian: Any,
        *,
        max_rank: int = 8,
        tail_mode: str = "spectral_upper_bound",
    ) -> LowRankPlusTailOperator:
        """Factorize a dense ``(m, n)`` Jacobian into top-``r`` SVD + sloppy tail.

        Args:
            jacobian: 2-D array of shape ``(dim_out, dim_in)``.
            max_rank: Maximum number of explicit stiff singular triplets to retain.
            tail_mode: ``"spectral_upper_bound"`` sets ``sigma_tail = sigma_{r+1}``
                (conservative Loewner upper bound on residual directions);
                ``"rms"`` sets ``sigma_tail`` to RMS of truncated singular values.
        """
        jac = np.asarray(jacobian, dtype=np.float64)
        if jac.ndim != 2:
            raise ValueError(f"Expected 2-D Jacobian, got shape {jac.shape}.")
        m, n = jac.shape
        u, s, vt = np.linalg.svd(jac, full_matrices=False)
        r = max(0, min(int(max_rank), s.size))
        if r < s.size:
            rem = s[r:]
            if tail_mode == "spectral_upper_bound":
                tail = float(rem[0])
            elif tail_mode == "rms":
                tail = float(np.sqrt(np.mean(rem * rem)))
            else:
                raise ValueError(f"Unknown tail_mode: {tail_mode!r}")
        else:
            tail = 0.0
        return cls(
            dim_out=m,
            dim_in=n,
            singular_values=s[:r],
            left_vectors=u[:, :r],
            right_vectors=vt[:r, :].T,
            tail_singular_value=tail,
        )

    @classmethod
    def from_stein_perturbation(
        cls,
        fn: Callable[[FloatArray], FloatArray],
        x0: Any,
        *,
        sigma: float = 1e-3,
        n_probes: int = 128,
        max_rank: int = 8,
        seed: int = 0,
    ) -> LowRankPlusTailOperator:
        """Estimate smoothed distributional Jacobian via Stein's identity.

        Works even when ``fn`` is piecewise-constant, thresholded, or discrete
        (where pointwise finite differences are zero or undefined) by computing:
            ``J_sigma(x0) = E_{delta ~ N(0, sigma^2 I)}[(f(x0 + delta) - E[f]) delta^T / sigma^2]``
        using antithetic Gaussian perturbations for variance reduction.
        """
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0.")
        x_base = np.asarray(x0, dtype=np.float64).reshape(-1)
        dim_in = x_base.size
        half = max(2, int(n_probes) // 2)
        rng = np.random.default_rng(seed)

        z = rng.standard_normal((half, dim_in))
        y_plus_list: list[FloatArray] = []
        y_minus_list: list[FloatArray] = []
        valid_z: list[FloatArray] = []

        for i in range(half):
            delta = sigma * z[i]
            yp = np.asarray(fn(x_base + delta), dtype=np.float64).reshape(-1)
            ym = np.asarray(fn(x_base - delta), dtype=np.float64).reshape(-1)
            if np.all(np.isfinite(yp)) and np.all(np.isfinite(ym)):
                y_plus_list.append(yp)
                y_minus_list.append(ym)
                valid_z.append(z[i])

        if not valid_z:
            raise RuntimeError("All Stein perturbation evaluations returned non-finite outputs.")

        yp_mat = np.stack(y_plus_list, axis=0)  # (M, dim_out)
        ym_mat = np.stack(y_minus_list, axis=0)  # (M, dim_out)
        z_mat = np.stack(valid_z, axis=0)        # (M, dim_in)

        # Antithetic Stein estimator: (1 / (2 M sigma)) * (Y+ - Y-)^T @ Z
        jac_est = ((yp_mat - ym_mat).T @ z_mat) / (2.0 * float(len(valid_z)) * sigma)
        return cls.from_dense(jac_est, max_rank=max_rank)

    def pushforward_covariance(
        self,
        sigma_in: Any,
        *,
        intrinsic_output_cov: Any | None = None,
    ) -> FloatArray:
        """Push input covariance ``Sigma_x`` forward to ``J Sigma_x J^T + Sigma_eps``.

        Decomposes ``Sigma_x`` along the stiff subspace ``span(V_r)`` and its
        orthogonal complement ``Pi_perp = I - V_r V_r^T``.  Noise strictly inside
        a sloppy direction (``V_r^T xi = 0``) scales only by ``sigma_tail^2``
        rather than ``sigma_max^2``!
        """
        cov_x = symmetrize(np.asarray(sigma_in, dtype=np.float64))
        if cov_x.shape != (self._dim_in, self._dim_in):
            raise ValueError(
                f"sigma_in shape {cov_x.shape} does not match dim_in={self._dim_in}."
            )

        r = self.rank
        if r > 0:
            v = self._right_vectors
            u = self._left_vectors
            s = self._singular_values
            # Low-rank stiff pushforward: U diag(s) (V^T Sigma_x V) diag(s) U^T
            v_cov_v = v.T @ cov_x @ v
            scaled = (s[:, None] * v_cov_v) * s[None, :]
            out_cov = u @ scaled @ u.T
            trace_stiff_in = float(np.trace(v_cov_v))
        else:
            out_cov = np.zeros((self._dim_out, self._dim_out), dtype=np.float64)
            trace_stiff_in = 0.0

        if self._tail > 0.0:
            # Residual input variance in V_r^perp: Tr((I - V V^T) Sigma_x)
            trace_perp = max(0.0, float(np.trace(cov_x)) - trace_stiff_in)
            if r > 0 and r < self._dim_out:
                u = self._left_vectors
                perp_out = np.eye(self._dim_out) - u @ u.T
                denom = float(self._dim_out - r)
                out_cov = out_cov + (self._tail ** 2) * (trace_perp / denom) * perp_out
            else:
                denom = float(self._dim_out)
                out_cov = out_cov + (self._tail ** 2) * (
                    trace_perp / denom
                ) * np.eye(self._dim_out)

        if intrinsic_output_cov is not None:
            eps_cov = np.asarray(intrinsic_output_cov, dtype=np.float64)
            if eps_cov.ndim == 1:
                eps_cov = np.diag(eps_cov)
            if eps_cov.shape != (self._dim_out, self._dim_out):
                raise ValueError(
                    f"intrinsic_output_cov shape {eps_cov.shape} != {(self._dim_out, self._dim_out)}."
                )
            out_cov = out_cov + eps_cov

        return symmetrize(out_cov)

    def fisher_information(
        self,
        output_precision: Any | None = None,
    ) -> FloatArray:
        """Compute input-space pullback Fisher matrix ``I_X = J^T Lambda_Y J``."""
        if output_precision is None:
            lam_y = np.eye(self._dim_out, dtype=np.float64)
        else:
            lam_y = symmetrize(np.asarray(output_precision, dtype=np.float64))
            if lam_y.shape != (self._dim_out, self._dim_out):
                raise ValueError(
                    f"output_precision shape {lam_y.shape} != {(self._dim_out, self._dim_out)}."
                )

        r = self.rank
        if r > 0:
            u = self._left_vectors
            v = self._right_vectors
            s = self._singular_values
            u_lam_u = u.T @ lam_y @ u
            scaled = (s[:, None] * u_lam_u) * s[None, :]
            fim = v @ scaled @ v.T
        else:
            fim = np.zeros((self._dim_in, self._dim_in), dtype=np.float64)

        if self._tail > 0.0 and r < self._dim_in:
            # Conservative tail precision scaled by spectral norm of Lambda_Y on U^perp
            if r > 0 and r < self._dim_out:
                u = self._left_vectors
                p_u_perp = np.eye(self._dim_out) - u @ u.T
                tail_lam_scale = float(
                    np.max(np.linalg.eigvalsh(symmetrize(p_u_perp @ lam_y @ p_u_perp)))
                )
            else:
                tail_lam_scale = float(np.max(np.linalg.eigvalsh(lam_y)))
            tail_lam_scale = max(0.0, tail_lam_scale)

            if r > 0:
                v = self._right_vectors
                p_v_perp = np.eye(self._dim_in) - v @ v.T
            else:
                p_v_perp = np.eye(self._dim_in)
            fim = fim + (self._tail ** 2) * tail_lam_scale * p_v_perp

        return symmetrize(fim)

    def pullback_precision(
        self,
        required_output_precision: Any,
        *,
        intrinsic_output_cov: Any | None = None,
        spd_tol: float = 1e-12,
    ) -> PullbackResult:
        """Pull a downstream required precision ``Lambda_out`` back to input space.

        Enforces ``J Sigma_x J^T + Sigma_eps <= Lambda_out^{-1}``:
        1. Checks whether residual covariance budget ``S_avail = Lambda_out^{-1} - Sigma_eps``
           is strictly positive-definite.  If not, the atom's intrinsic error alone
           exceeds the budget (``feasible=False``).
        2. Otherwise returns ``Lambda_req = J^T S_avail^{-1} J``.
        """
        lam_out = symmetrize(np.asarray(required_output_precision, dtype=np.float64))
        if lam_out.shape != (self._dim_out, self._dim_out):
            raise ValueError(
                f"required_output_precision shape {lam_out.shape} != {(self._dim_out, self._dim_out)}."
            )

        eig_lam, vec_lam, lam_spd = spd_eigendecomp(lam_out, tol=spd_tol)
        if not lam_spd:
            # Semi-definite output precision (e.g., unconstrained directions):
            # regularize zero directions slightly to invert onto constrained subspace
            safe_eigs = np.maximum(eig_lam, spd_tol)
            cov_budget = (vec_lam * (1.0 / safe_eigs)[None, :]) @ vec_lam.T
        else:
            cov_budget = (vec_lam * (1.0 / eig_lam)[None, :]) @ vec_lam.T
        cov_budget = symmetrize(cov_budget)

        if intrinsic_output_cov is not None:
            eps_cov = np.asarray(intrinsic_output_cov, dtype=np.float64)
            if eps_cov.ndim == 1:
                eps_cov = np.diag(eps_cov)
            slack_cov = symmetrize(cov_budget - eps_cov)
        else:
            slack_cov = cov_budget

        eig_slack, vec_slack, feasible = spd_eigendecomp(slack_cov, tol=spd_tol)
        min_eig = float(np.min(eig_slack))
        if not feasible:
            # Still compute a high-penalty directional precision along violated axes
            clamped = np.maximum(eig_slack, spd_tol)
            eff_out_prec = symmetrize((vec_slack * (1.0 / clamped)[None, :]) @ vec_slack.T)
            req = self.fisher_information(eff_out_prec)
            return PullbackResult(
                required_input_precision=req,
                feasible=False,
                slack_min_eigenvalue=min_eig,
                violation_reason=(
                    f"Intrinsic output covariance exceeds downstream tolerance "
                    f"(min slack eigenvalue={min_eig:.3e} <= 0)."
                ),
            )

        eff_out_prec = symmetrize(
            (vec_slack * (1.0 / eig_slack)[None, :]) @ vec_slack.T
        )
        req = self.fisher_information(eff_out_prec)
        return PullbackResult(
            required_input_precision=req,
            feasible=True,
            slack_min_eigenvalue=min_eig,
        )

    def stiff_sloppy_spectrum(
        self,
        *,
        output_precision: Any | None = None,
        stiff_threshold: float = 1.0,
        sloppy_threshold: float = 1e-2,
        ridge_gamma: float = 1e-2,
    ) -> SpectralSummary:
        """Summarize stiff vs. sloppy eigenstructure of ``I_X = J^T Lambda_Y J``."""
        fim = self.fisher_information(output_precision)
        eigvals = np.maximum(np.linalg.eigvalsh(fim), 0.0)[::-1]
        svals = np.sqrt(eigvals)
        s_max = float(svals[0]) if svals.size > 0 else 0.0
        s_min = float(svals[-1]) if svals.size > 0 else 0.0
        cond = s_max / max(s_min, 1e-15) if s_max > 0.0 else 1.0
        eff_dim = float(np.sum(eigvals / np.maximum(eigvals + ridge_gamma, 1e-30)))
        stiff_rank = int(np.sum(svals > stiff_threshold))
        sloppy_rank = int(np.sum(svals <= sloppy_threshold))
        return SpectralSummary(
            spectral_norm=s_max,
            sloppy_floor=s_min,
            condition_number=cond,
            effective_dimension=eff_dim,
            stiff_rank=stiff_rank,
            sloppy_rank=sloppy_rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_out": self._dim_out,
            "dim_in": self._dim_in,
            "singular_values": self._singular_values.tolist(),
            "left_vectors": self._left_vectors.tolist(),
            "right_vectors": self._right_vectors.tolist(),
            "tail_singular_value": self._tail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LowRankPlusTailOperator:
        svals = payload.get("singular_values", [])
        return cls(
            dim_out=int(payload["dim_out"]),
            dim_in=int(payload["dim_in"]),
            singular_values=svals,
            left_vectors=payload.get("left_vectors") if len(svals) > 0 else None,
            right_vectors=payload.get("right_vectors") if len(svals) > 0 else None,
            tail_singular_value=float(payload.get("tail_singular_value", 0.0)),
        )
