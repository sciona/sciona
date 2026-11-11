"""Sufficient-statistic / domain-feature projectors ``phi: X -> R^d``.

Maps variable-length signals, event streams, positive physical quantities,
simplex vectors, and matrices into compact, fixed-dimensional summary vectors
so exponential-family regime mixtures and low-rank Jacobians stay tractable
without losing stiff/sloppy conditioning structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionResult:
    """Result of projecting a raw port value into sufficient-statistic space."""

    features: FloatArray
    """1-D float64 vector of length ``projector.dim``."""
    in_domain: bool
    """False when raw value violates hard mathematical support (e.g. NaN, <=0)."""
    violation_reason: str = ""
    """Human-readable explanation when ``in_domain`` is False."""


@dataclass(frozen=True)
class ProjectorSpec:
    """Serializable specification for reconstructing a ``SufficientStatProjector``."""

    kind: str
    dim: int
    feature_names: tuple[str, ...]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "feature_names": list(self.feature_names),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectorSpec:
        return cls(
            kind=str(payload["kind"]),
            dim=int(payload["dim"]),
            feature_names=tuple(str(x) for x in payload.get("feature_names", ())),
            params=dict(payload.get("params", {})),
        )


@runtime_checkable
class SufficientStatProjector(Protocol):
    """Protocol for deterministic ``O(n)`` input/output summary projections."""

    @property
    def kind(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def preferred_family(self) -> str: ...

    def project(self, value: Any) -> ProjectionResult: ...

    def to_spec(self) -> ProjectorSpec: ...


class IdentityProjector:
    """Direct pass-through for small fixed-size vectors/scalars (``d <= max_dim``)."""

    def __init__(self, dim: int = 1) -> None:
        if dim < 1:
            raise ValueError("IdentityProjector dim must be >= 1.")
        self._dim = int(dim)

    @property
    def kind(self) -> str:
        return "identity"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"x_{i}" for i in range(self._dim))

    @property
    def preferred_family(self) -> str:
        return "gaussian"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.zeros(self._dim, dtype=np.float64),
                in_domain=False,
                violation_reason="non-finite entries in vector",
            )
        if arr.size != self._dim:
            return ProjectionResult(
                features=np.resize(arr, self._dim).astype(np.float64)
                if arr.size > 0
                else np.zeros(self._dim, dtype=np.float64),
                in_domain=False,
                violation_reason=f"expected length {self._dim}, got {arr.size}",
            )
        return ProjectionResult(features=arr.copy(), in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={"dim": self._dim},
        )


class WaveformProjector:
    """8-D spectral and amplitude envelope summary for continuous waveforms."""

    _FEATURES = (
        "mean",
        "log_std",
        "p05",
        "p50",
        "p95",
        "crest_factor",
        "spectral_centroid",
        "spectral_flatness",
    )

    @property
    def kind(self) -> str:
        return "waveform"

    @property
    def dim(self) -> int:
        return len(self._FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._FEATURES

    @property
    def preferred_family(self) -> str:
        return "gaussian"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="empty waveform",
            )
        if not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="waveform contains NaN or Inf",
            )

        mean = float(np.mean(arr))
        centered = arr - mean
        std = float(np.std(centered))
        log_std = float(np.log(max(std, 1e-12)))
        p05, p50, p95 = (float(q) for q in np.quantile(arr, [0.05, 0.50, 0.95]))
        rms = float(np.sqrt(np.mean(arr * arr)))
        max_abs = float(np.max(np.abs(arr)))
        crest = max_abs / max(rms, 1e-12)

        # Normalized one-sided power spectrum features in [0, 1]
        if arr.size >= 4 and std > 1e-15:
            spec = np.abs(np.fft.rfft(centered)) ** 2
            spec = spec[1:] if spec.size > 1 else spec
            power_sum = float(np.sum(spec))
            if power_sum > 1e-24 and spec.size > 0:
                freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float64)
                centroid = float(np.sum(freqs * spec) / power_sum)
                geom_mean = float(np.exp(np.mean(np.log(np.maximum(spec, 1e-30)))))
                arith_mean = float(np.mean(spec))
                flatness = float(np.clip(geom_mean / max(arith_mean, 1e-30), 0.0, 1.0))
            else:
                centroid = 0.0
                flatness = 1.0
        else:
            centroid = 0.0
            flatness = 1.0

        feats = np.array(
            [mean, log_std, p05, p50, p95, crest, centroid, flatness],
            dtype=np.float64,
        )
        return ProjectionResult(features=feats, in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={},
        )


class CountEventProjector:
    """Non-negative count summary for event sequences and count arrays.

    Emits 4 non-negative count/binned-count statistics compatible with
    ``CountMixtureRegime`` (Poisson / Negative-Binomial mixtures):
    total count, non-zero bin count, max bin count, and 90th-percentile count.
    """

    _FEATURES = ("total_count", "nonzero_count", "max_count", "p90_count")

    def __init__(self, *, mode: str = "auto", n_bins: int = 16) -> None:
        if mode not in {"auto", "counts", "event_indices"}:
            raise ValueError(f"Unsupported CountEventProjector mode: {mode!r}")
        self._mode = mode
        self._n_bins = max(2, int(n_bins))

    @property
    def kind(self) -> str:
        return "count_event"

    @property
    def dim(self) -> int:
        return len(self._FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._FEATURES

    @property
    def preferred_family(self) -> str:
        return "count"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="count/event array contains NaN or Inf",
            )
        if arr.size == 0:
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=True,
            )
        if np.any(arr < 0.0):
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="count/event values must be non-negative",
            )

        is_monotone_events = (
            self._mode == "event_indices"
            or (
                self._mode == "auto"
                and arr.size >= 3
                and bool(np.all(np.diff(arr) > 0))
                and float(arr[-1]) > float(arr.size)
            )
        )
        if is_monotone_events:
            # Histogram event timestamps/indices into n_bins equal windows
            span_lo = float(arr[0])
            span_hi = max(float(arr[-1]), span_lo + 1.0)
            counts, _ = np.histogram(arr, bins=self._n_bins, range=(span_lo, span_hi))
            c = counts.astype(np.float64)
        else:
            c = arr

        total_count = float(np.sum(c))
        nonzero_count = float(np.sum(c > 0))
        max_count = float(np.max(c))
        p90_count = float(np.quantile(c, 0.90))
        feats = np.array(
            [total_count, nonzero_count, max_count, p90_count],
            dtype=np.float64,
        )
        return ProjectionResult(features=feats, in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={"mode": self._mode, "n_bins": self._n_bins},
        )


class PositiveScaleProjector:
    """Strictly positive summary statistics compatible with ``GammaMixtureRegime``."""

    _FEATURES = ("mean_pos", "rms_pos", "p10_pos", "p50_pos", "p90_pos")

    @property
    def kind(self) -> str:
        return "positive_scale"

    @property
    def dim(self) -> int:
        return len(self._FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._FEATURES

    @property
    def preferred_family(self) -> str:
        return "gamma"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.ones(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="empty or non-finite positive quantity",
            )
        if np.any(arr <= 0.0):
            return ProjectionResult(
                features=np.ones(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="positive_scale requires strictly positive values (> 0)",
            )
        mean_pos = float(np.mean(arr))
        rms_pos = float(np.sqrt(np.mean(arr * arr)))
        p10, p50, p90 = (float(q) for q in np.quantile(arr, [0.10, 0.50, 0.90]))
        feats = np.array([mean_pos, rms_pos, p10, p50, p90], dtype=np.float64)
        return ProjectionResult(features=feats, in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={},
        )


class SimplexProjector:
    """Projects probability vectors / normalized weights onto the open simplex."""

    def __init__(self, n_categories: int = 4, *, smoothing: float = 1e-8) -> None:
        if n_categories < 2:
            raise ValueError("SimplexProjector requires n_categories >= 2.")
        self._n_categories = int(n_categories)
        self._smoothing = float(smoothing)

    @property
    def kind(self) -> str:
        return "simplex"

    @property
    def dim(self) -> int:
        return self._n_categories

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"prob_{i}" for i in range(self._n_categories))

    @property
    def preferred_family(self) -> str:
        return "simplex"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(arr)) or arr.size == 0:
            uniform = np.full(self._n_categories, 1.0 / self._n_categories)
            return ProjectionResult(
                features=uniform,
                in_domain=False,
                violation_reason="empty or non-finite simplex input",
            )
        # Support batch of probability vectors (N, K) by averaging across batch
        if arr.ndim >= 2 and arr.shape[-1] == self._n_categories:
            vec = np.mean(arr.reshape(-1, self._n_categories), axis=0)
        else:
            vec = arr.reshape(-1)

        if vec.size != self._n_categories:
            uniform = np.full(self._n_categories, 1.0 / self._n_categories)
            return ProjectionResult(
                features=uniform,
                in_domain=False,
                violation_reason=(
                    f"expected {self._n_categories} simplex coordinates, got {vec.size}"
                ),
            )

        if np.any(vec < -1e-6):
            uniform = np.full(self._n_categories, 1.0 / self._n_categories)
            return ProjectionResult(
                features=uniform,
                in_domain=False,
                violation_reason="simplex coordinates cannot be negative",
            )
        total = float(np.sum(vec))
        if abs(total - 1.0) > 1e-2:
            uniform = np.full(self._n_categories, 1.0 / self._n_categories)
            return ProjectionResult(
                features=uniform,
                in_domain=False,
                violation_reason=f"simplex vector sums to {total:.6f}, expected ~1.0",
            )
        clipped = np.maximum(vec, 0.0) + self._smoothing
        normalized = clipped / float(np.sum(clipped))
        return ProjectionResult(features=normalized.astype(np.float64), in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={"n_categories": self._n_categories, "smoothing": self._smoothing},
        )


class MatrixConditionProjector:
    """6-D matrix conditioning and structural summary for linear-algebra ports."""

    _FEATURES = (
        "log_frobenius",
        "trace_per_dim",
        "diag_dominance",
        "symmetry_defect",
        "log_cond_est",
        "effective_rank_ratio",
    )

    def __init__(self, *, max_svd_dim: int = 64) -> None:
        self._max_svd_dim = max(4, int(max_svd_dim))

    @property
    def kind(self) -> str:
        return "matrix_condition"

    @property
    def dim(self) -> int:
        return len(self._FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._FEATURES

    @property
    def preferred_family(self) -> str:
        return "gaussian"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim < 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="expected finite 2-D+ matrix",
            )
        mat = arr.reshape(arr.shape[0], -1)
        m, n = mat.shape
        frob = float(np.linalg.norm(mat, ord="fro"))
        log_frob = float(np.log(max(frob, 1e-12)))

        min_dim = min(m, n)
        diag = np.diag(mat[:min_dim, :min_dim])
        trace_per_dim = float(np.mean(diag)) if min_dim > 0 else 0.0

        abs_mat = np.abs(mat[:min_dim, :min_dim])
        abs_diag = np.abs(diag)
        off_diag_sum = np.sum(abs_mat, axis=1) - abs_diag
        diag_dom = (
            float(np.mean(abs_diag / np.maximum(abs_diag + off_diag_sum, 1e-12)))
            if min_dim > 0
            else 0.0
        )

        if m == n and f_denom_ok(frob):
            sym_defect = float(np.linalg.norm(mat - mat.T, ord="fro") / (2.0 * frob))
        else:
            sym_defect = 1.0

        # Subsample symmetrically if matrix exceeds max_svd_dim to stay fast
        if max(m, n) > self._max_svd_dim:
            idx_r = np.linspace(0, m - 1, min(m, self._max_svd_dim), dtype=int)
            idx_c = np.linspace(0, n - 1, min(n, self._max_svd_dim), dtype=int)
            sub = mat[np.ix_(idx_r, idx_c)]
        else:
            sub = mat

        svals = np.linalg.svd(sub, compute_uv=False)
        if svals.size > 0 and svals[0] > 1e-15:
            s_max = float(svals[0])
            s_min = float(max(svals[-1], 1e-15 * s_max))
            log_cond = float(np.log(s_max / s_min))
            p = svals / float(np.sum(svals))
            p_pos = p[p > 1e-30]
            entropy = -float(np.sum(p_pos * np.log(p_pos)))
            eff_rank_ratio = float(np.exp(entropy) / svals.size)
        else:
            log_cond = 0.0
            eff_rank_ratio = 0.0

        feats = np.array(
            [log_frob, trace_per_dim, diag_dom, sym_defect, log_cond, eff_rank_ratio],
            dtype=np.float64,
        )
        return ProjectionResult(features=feats, in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={"max_svd_dim": self._max_svd_dim},
        )


def f_denom_ok(val: float) -> bool:
    return val > 1e-15


class GenericNumericProjector:
    """6-D summary fallback for arbitrary numeric arrays or scalars."""

    _FEATURES = ("mean", "log_std", "p05", "p50", "p95", "log_size")

    @property
    def kind(self) -> str:
        return "generic_numeric"

    @property
    def dim(self) -> int:
        return len(self._FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._FEATURES

    @property
    def preferred_family(self) -> str:
        return "gaussian"

    def project(self, value: Any) -> ProjectionResult:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return ProjectionResult(
                features=np.zeros(self.dim, dtype=np.float64),
                in_domain=False,
                violation_reason="empty or non-finite numeric value",
            )
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        log_std = float(np.log(max(std, 1e-12)))
        p05, p50, p95 = (float(q) for q in np.quantile(arr, [0.05, 0.50, 0.95]))
        log_size = float(np.log1p(arr.size))
        feats = np.array([mean, log_std, p05, p50, p95, log_size], dtype=np.float64)
        return ProjectionResult(features=feats, in_domain=True)

    def to_spec(self) -> ProjectorSpec:
        return ProjectorSpec(
            kind=self.kind,
            dim=self.dim,
            feature_names=self.feature_names,
            params={},
        )


def projector_from_spec(spec: ProjectorSpec) -> SufficientStatProjector:
    """Reconstruct a ``SufficientStatProjector`` from its serialized ``ProjectorSpec``."""
    kind = spec.kind
    params = spec.params
    if kind == "identity":
        return IdentityProjector(dim=int(params.get("dim", spec.dim)))
    if kind == "waveform":
        return WaveformProjector()
    if kind == "count_event":
        return CountEventProjector(
            mode=str(params.get("mode", "auto")),
            n_bins=int(params.get("n_bins", 16)),
        )
    if kind == "positive_scale":
        return PositiveScaleProjector()
    if kind == "simplex":
        return SimplexProjector(
            n_categories=int(params.get("n_categories", spec.dim)),
            smoothing=float(params.get("smoothing", 1e-8)),
        )
    if kind == "matrix_condition":
        return MatrixConditionProjector(
            max_svd_dim=int(params.get("max_svd_dim", 64)),
        )
    if kind == "generic_numeric":
        return GenericNumericProjector()
    raise ValueError(f"Unknown projector kind: {kind!r}")


def infer_projector(
    port_name: str,
    type_desc: str = "",
    constraints: str = "",
    *,
    sample_value: Any = None,
) -> SufficientStatProjector:
    """Infer the appropriate ``SufficientStatProjector`` from IOSpec metadata or sample."""
    name_l = port_name.strip().lower()
    type_l = type_desc.strip().lower()
    cons_l = constraints.strip().lower()
    combined = f"{name_l} {type_l} {cons_l}"

    if any(tok in combined for tok in ("simplex", "dirichlet", "probabilities", "prob_")):
        n_cat = 4
        if sample_value is not None:
            arr = np.asarray(sample_value)
            if arr.ndim >= 1 and arr.shape[-1] >= 2:
                n_cat = int(arr.shape[-1])
        return SimplexProjector(n_categories=n_cat)

    if any(
        tok in combined
        for tok in ("event", "rpeak", "peaks", "beats", "onsets", "counts", "poisson")
    ):
        return CountEventProjector()

    if any(
        tok in combined
        for tok in ("matrix", "covariance", "laplacian", "jacobian", "hessian", "2d")
    ):
        return MatrixConditionProjector()

    if any(
        tok in combined
        for tok in ("positive", "> 0", ">0", "variance", "scale", "interval", "sampling_rate")
    ) and "semi-definite" not in combined:
        return PositiveScaleProjector()

    if sample_value is not None:
        arr = np.asarray(sample_value)
        if arr.ndim == 2 and min(arr.shape) >= 2:
            return MatrixConditionProjector()
        if arr.size <= 4 and arr.dtype.kind in {"f", "i", "u"}:
            return IdentityProjector(dim=max(1, int(arr.size)))

    if any(
        tok in combined
        for tok in ("signal", "waveform", "ecg", "ppg", "eeg", "audio", "filtered")
    ):
        return WaveformProjector()

    if type_l in {"float", "double", "int", "integer"}:
        return IdentityProjector(dim=1)

    return GenericNumericProjector()
