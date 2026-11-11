"""Empirical multi-regime atom calibration & spectral channel estimation (Phase 2).

Transforms black-box callable atoms ``y = fn(**kwargs)`` into calibrated v2
``AtomUncertaintyProfile`` specifications containing:
1. Per-input-port ``PortRegimeSpec`` (``SufficientStatProjector`` + EM-fitted
   ``ExponentialFamilyMixture`` with joint BIC component selection).
2. Exact finite-sample ``ConformalEnvelope`` computed strictly on a disjoint
   held-out calibration split ``D_cal``.
3. Per-regime ``ComponentChannel`` operators (``LowRankPlusTailOperator`` stiff/sloppy
   Jacobians estimated via domain-respecting multi-scale antithetic probes +
   intrinsic approximation/projection residual covariances ``Sigma_{eps, k}``).
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sciona.uncertainty._math import symmetrize
from sciona.uncertainty.conformal import ConformalEnvelope
from sciona.uncertainty.distributions import (
    CountMixtureRegime,
    ExponentialFamilyMixture,
    GammaMixtureRegime,
    GaussianMixtureRegime,
    SimplexMixtureRegime,
)
from sciona.uncertainty.models import (
    AtomUncertaintyProfile,
    ComponentChannel,
    PortRegimeSpec,
)
from sciona.uncertainty.operators import LowRankPlusTailOperator
from sciona.uncertainty.projectors import (
    CountEventProjector,
    GenericNumericProjector,
    IdentityProjector,
    MatrixConditionProjector,
    PositiveScaleProjector,
    ProjectorSpec,
    SimplexProjector,
    SufficientStatProjector,
    WaveformProjector,
    infer_projector,
)

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class CalibrationError(RuntimeError):
    """Raised when an atom cannot be calibrated empirically."""


@dataclass(frozen=True)
class PortCalibrationConfig:
    """Optional override metadata for a single atom input port."""

    port_name: str
    projector: SufficientStatProjector | None = None
    type_desc: str = ""
    constraints: str = ""
    perturbed: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    """Hyperparameters governing empirical v2 atom calibration."""

    n_trials: int = 300
    """Total valid input-output trials targeted across synthetic regimes."""
    fit_fraction: float = 0.5
    """Fraction of valid trials used for EM + Jacobian fitting; remainder forms D_cal."""
    n_components: int | None = None
    """Fixed mixture component count K, or None for joint BIC selection."""
    candidate_components: tuple[int, ...] = (1, 2, 3)
    """Candidate component counts evaluated under joint BIC when n_components is None."""
    factor_rank: int = 3
    """Low-rank Factor-Analyzer dimension r for GaussianMixtureRegime."""
    operator_max_rank: int = 6
    """Maximum explicit stiff singular triplets retained per LowRankPlusTailOperator."""
    n_anchor_probes: int = 36
    """Antithetic perturbation pairs per input port per mixture component."""
    probe_scales: tuple[float, ...] = (1e-3, 1e-2, 5e-2)
    """Multi-scale perturbation magnitudes capturing both smooth & threshold sensitivities."""
    failure_threshold: float = 0.5
    """Minimum trial success ratio required to accept calibration."""
    trust_tier: int = 3
    """Assurance tier tag recorded on the resulting AtomUncertaintyProfile."""
    seed: int | None = None
    """Deterministic PRNG seed (derived from atom_name when None)."""


# ---------------------------------------------------------------------------
# Default configuration scalar heuristics (for unannotated / multi-arg atoms)
# ---------------------------------------------------------------------------

_SCALAR_DEFAULTS: dict[str, float | int | str | bool] = {
    "sampling_rate": 250.0,
    "fs": 250.0,
    "sample_rate": 250.0,
    "sr": 250.0,
    "order": 4,
    "n": 256,
    "cutoff": 0.1,
    "cutoff_frequency": 0.1,
    "wn": 0.1,
    "low": 0.05,
    "high": 0.4,
    "lowcut": 0.05,
    "highcut": 0.4,
    "peakwindow": 0.5,
    "beatwindow": 1.0,
    "beatoffset": 0.0,
    "mindelay": 0.4,
    "window_size": 32,
    "size": 32,
    "alarm_size": 16,
    "threshold": 0.5,
    "atol": 1e-6,
    "tol": 1e-6,
    "transition_threshold": 0.5,
    "rtol": 1e-5,
    "smoothing": 0.1,
    "epsilon": 1.0,
    "alpha": 0.01,
    "beta": 1.0,
    "gamma": 1.0,
    "lambda_": 0.01,
    "maxiter": 100,
    "popsize": 15,
    "mutation": 0.5,
    "recombination": 0.7,
    "step_size": 0.01,
    "dt": 0.01,
    "time_step": 0.01,
    "degree": 3,
    "k": 3,
    "neighbors": 8,
    "num_components": 2,
    "n_components": 2,
    "n_iter": 50,
    "iterations": 50,
    "max_iterations": 50,
    "seed": 42,
    "mode": "time",
    "norm": "ortho",
    "bc_type": "not-a-knot",
    "kernel": "linear",
    "strategy": "best1bin",
    "updating": "immediate",
    "side": "left",
    "btype": "low",
    "ftype": "butter",
    "disp": False,
    "polish": False,
    "verbose": False,
    "normalize": False,
    "correct": True,
    "detrend": True,
}

_TYPE_DEFAULTS: dict[str, float | int | str | bool] = {
    "float": 1.0,
    "int": 4,
    "str": "default",
    "bool": False,
}

_MATRIX_PARAM_NAMES = {"F", "H", "Q", "R", "P", "B", "A", "matrix", "covariance", "cov", "gram"}


def _stable_seed(atom_name: str) -> int:
    digest = hashlib.sha256(atom_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _generate_default_for_param(
    name: str,
    ann_str: str,
    rng: np.random.Generator,
) -> Any:
    if name in _SCALAR_DEFAULTS:
        return _SCALAR_DEFAULTS[name]

    low_ann = ann_str.lower()
    low_name = name.lower()

    if any(tok in low_ann for tok in ("ndarray", "ndarr", "array")):
        if (
            name in _MATRIX_PARAM_NAMES
            or "matrix" in low_name
            or "covariance" in low_name
            or "laplacian" in low_name
        ):
            m = rng.standard_normal((4, 4))
            return m @ m.T + np.eye(4)
        if any(tok in low_name for tok in ("simplex", "prob", "weights_simplex")):
            return rng.dirichlet(np.ones(4))
        if any(tok in low_name for tok in ("count", "poisson", "events", "rpeak", "peaks", "beats")):
            return rng.poisson(lam=4.0, size=64).astype(np.float64)
        if any(tok in low_name for tok in ("positive", "scale", "variance")):
            return rng.gamma(shape=3.0, scale=1.0, size=64)
        t = np.linspace(0.0, 2.0 * math.pi, 256)
        return np.sin(3.0 * t) + 0.1 * rng.standard_normal(256)

    if "callable" in low_ann:
        return lambda x: x
    if "dict" in low_ann:
        return {}
    if "list" in low_ann and "tuple" in low_ann:
        return [(-5.0, 5.0)] * 4
    if "tuple" in low_ann:
        return (0.0, 1.0)
    if not low_ann or low_ann == "none":
        return None

    for tname, default in _TYPE_DEFAULTS.items():
        if tname in low_ann:
            return default

    if any(kw in low_name for kw in ("rate", "freq", "hz")):
        return 250.0
    if any(kw in low_name for kw in ("order", "degree", "size", "count", "num")):
        return 4
    if any(kw in low_name for kw in ("threshold", "tol", "eps")):
        return 0.5
    return None


def introspect_atom_signature(
    fn: Callable[..., Any],
    *,
    atom_name: str = "",
    rng: np.random.Generator | None = None,
    port_configs: Mapping[str, PortCalibrationConfig] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Introspect ``fn`` and return ``(base_kwargs, calibrated_ports, annotations_map)``."""
    del atom_name
    gen = rng or np.random.default_rng(0)
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as exc:
        raise CalibrationError(f"Could not introspect callable signature: {exc}") from exc

    base_kwargs: dict[str, Any] = {}
    calibrated_ports: list[str] = []
    annotations_map: dict[str, str] = {}
    cfg_map = dict(port_configs or {})

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        ann = param.annotation
        ann_str = "" if ann is inspect.Parameter.empty else str(ann)
        annotations_map[pname] = ann_str
        p_cfg = cfg_map.get(pname)

        if param.default is not inspect.Parameter.empty:
            val = param.default
            if val is None and any(tok in ann_str.lower() for tok in ("ndarray", "ndarr", "array")):
                val = _generate_default_for_param(pname, ann_str, gen)
            base_kwargs[pname] = val
        else:
            val = _generate_default_for_param(pname, ann_str, gen)
            if val is None and (p_cfg is not None or not base_kwargs):
                t = np.linspace(0.0, 2.0 * math.pi, 256)
                val = np.sin(3.0 * t) + 0.1 * gen.standard_normal(256)
            base_kwargs[pname] = val

        if p_cfg is not None:
            if p_cfg.perturbed:
                calibrated_ports.append(pname)
            continue

        if pname in _SCALAR_DEFAULTS:
            continue

        if isinstance(base_kwargs[pname], np.ndarray) or any(
            tok in ann_str.lower() for tok in ("ndarray", "ndarr", "array")
        ):
            if not isinstance(base_kwargs[pname], np.ndarray):
                t = np.linspace(0.0, 2.0 * math.pi, 256)
                base_kwargs[pname] = np.sin(3.0 * t) + 0.1 * gen.standard_normal(256)
            calibrated_ports.append(pname)

    # Fallback: if no array ports detected, treat first numeric/untyped required arg as signal/vector
    if not calibrated_ports:
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls") or pname in _SCALAR_DEFAULTS:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            val = base_kwargs.get(pname)
            if isinstance(val, bool) or callable(val) or isinstance(val, (str, dict)):
                continue
            if val is None:
                t = np.linspace(0.0, 2.0 * math.pi, 256)
                base_kwargs[pname] = np.sin(3.0 * t) + 0.1 * gen.standard_normal(256)
            calibrated_ports.append(pname)
            break

    return base_kwargs, calibrated_ports, annotations_map


@dataclass(frozen=True)
class _DomainProbeInfo:
    requires_positive: bool = False
    requires_nonneg: bool = False
    requires_spd: bool = False
    monotone_events: bool = False


def _probe_port_constraints(
    fn: Callable[..., Any],
    base_kwargs: Mapping[str, Any],
    port_name: str,
    rng: np.random.Generator,
) -> _DomainProbeInfo:
    """Lightweight preflight probe detecting domain invariants (e.g. icontract >0 or SPD)."""
    val = base_kwargs[port_name]
    arr = np.asarray(val)
    if arr.dtype.kind not in {"f", "i", "u"} or arr.size == 0:
        return _DomainProbeInfo()

    low_name = port_name.lower()
    is_event_name = any(
        tok in low_name for tok in ("rpeak", "peaks", "beats", "onsets", "event_indices")
    )
    if is_event_name:
        return _DomainProbeInfo(requires_nonneg=True, monotone_events=True)

    # Check 2-D matrix symmetry/SPD requirement
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
        d = arr.shape[0]
        asym = rng.standard_normal((d, d)) + 2.0 * np.eye(d)
        test_kw = dict(base_kwargs)
        test_kw[port_name] = asym
        try:
            out = fn(**test_kw)
            _ = out
            requires_spd = False
        except Exception:
            requires_spd = True
        return _DomainProbeInfo(requires_spd=requires_spd)

    # Test whether negative values are rejected
    neg_probe = -np.abs(np.asarray(arr, dtype=np.float64)) - 0.5
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        neg_val: Any = float(neg_probe.reshape(-1)[0])
    else:
        neg_val = neg_probe
    test_neg = dict(base_kwargs)
    test_neg[port_name] = neg_val
    neg_ok = True
    try:
        res_neg = fn(**test_neg)
        flat = np.asarray(res_neg, dtype=np.float64).reshape(-1)
        if flat.size > 0 and not np.all(np.isfinite(flat)):
            neg_ok = False
    except Exception:
        neg_ok = False

    if neg_ok:
        return _DomainProbeInfo()

    # Negative failed; check whether zero is allowed (non-negative vs strictly positive)
    zero_probe = np.zeros_like(np.asarray(arr, dtype=np.float64))
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        zero_val: Any = 0.0
    else:
        zero_val = zero_probe
    test_zero = dict(base_kwargs)
    test_zero[port_name] = zero_val
    zero_ok = True
    try:
        res_zero = fn(**test_zero)
        flat_z = np.asarray(res_zero, dtype=np.float64).reshape(-1)
        if flat_z.size > 0 and not np.all(np.isfinite(flat_z)):
            zero_ok = False
    except Exception:
        zero_ok = False

    if not zero_ok:
        return _DomainProbeInfo(requires_positive=True, requires_nonneg=True)
    return _DomainProbeInfo(requires_positive=False, requires_nonneg=True)


def sample_port_regime_value(
    port_name: str,
    projector: SufficientStatProjector,
    base_value: Any,
    regime_idx: int,
    rng: np.random.Generator,
    *,
    domain_info: _DomainProbeInfo | None = None,
) -> Any:
    """Draw a synthetic input value from operating regime ``regime_idx``."""
    del port_name
    info = domain_info or _DomainProbeInfo()
    is_scalar = isinstance(base_value, (int, float)) and not isinstance(base_value, bool)
    base_arr = np.asarray(base_value, dtype=np.float64)
    shape = base_arr.shape if base_arr.ndim > 0 else (1,)
    size = int(np.prod(shape))
    r_mod = int(regime_idx) % 3
    kind = projector.kind

    if kind == "simplex":
        d = max(2, shape[-1] if len(shape) >= 1 else projector.dim)
        if r_mod == 0:
            alpha = np.full(d, 1.0, dtype=np.float64)
            alpha[0] = 7.0
        elif r_mod == 1:
            alpha = np.full(d, 4.0, dtype=np.float64)
        else:
            alpha = np.full(d, 0.8, dtype=np.float64)
            alpha[-1] = 6.0
        if len(shape) <= 1:
            return rng.dirichlet(alpha)
        batch = int(np.prod(shape[:-1]))
        return rng.dirichlet(alpha, size=batch).reshape(shape)

    if kind == "positive_scale" or info.requires_positive:
        params = [(4.5, 0.08), (6.0, 0.30), (2.5, 2.8)][r_mod]
        sh, sc = params
        draw = rng.gamma(shape=sh, scale=sc, size=shape)
        draw = np.maximum(draw, 1e-4)
        return float(draw.reshape(-1)[0]) if is_scalar else draw

    if kind == "count_event" or info.monotone_events:
        if info.monotone_events:
            n_ev = [10, 24, 48][r_mod]
            gaps = rng.gamma(shape=[4.0, 2.0, 0.9][r_mod], scale=[25.0, 12.0, 6.0][r_mod], size=n_ev)
            events = np.cumsum(np.maximum(gaps, 1.0))
            return np.round(events).astype(np.float64)
        if r_mod == 0:
            cnt = rng.poisson(lam=1.5, size=shape).astype(np.float64)
        elif r_mod == 1:
            cnt = rng.poisson(lam=8.0, size=shape).astype(np.float64)
        else:
            rates = rng.gamma(shape=2.0, scale=11.0, size=shape)
            cnt = rng.poisson(lam=rates).astype(np.float64)
        return float(cnt.reshape(-1)[0]) if is_scalar else cnt

    if kind == "matrix_condition" and len(shape) == 2:
        m, n = shape
        p = min(m, n)
        q1, _ = np.linalg.qr(rng.standard_normal((m, m)))
        q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        cond_bounds = [(1.2, 3.5), (10.0, 35.0), (90.0, 350.0)][r_mod]
        kappa = float(rng.uniform(cond_bounds[0], cond_bounds[1]))
        scale_top = [1.0, 2.5, 6.0][r_mod] * float(rng.lognormal(0.0, 0.15))
        s_vals = scale_top * np.geomspace(1.0, 1.0 / kappa, num=p)
        if info.requires_spd and m == n:
            return symmetrize((q1 * s_vals[None, :]) @ q1.T)
        s_mat = np.zeros((m, n), dtype=np.float64)
        for idx in range(p):
            s_mat[idx, idx] = s_vals[idx]
        return q1 @ s_mat @ q2.T

    if kind == "identity":
        centers = [-2.2, 0.0, 2.6][r_mod]
        spread = [0.25, 0.75, 1.5][r_mod]
        draw = centers + spread * rng.standard_normal(shape)
        if info.requires_nonneg:
            draw = np.abs(draw) + [0.1, 1.0, 3.5][r_mod]
        return float(draw.reshape(-1)[0]) if is_scalar else draw

    # Waveform or GenericNumeric array regimes
    t = np.linspace(0.0, 2.0 * math.pi, max(1, size), endpoint=False)
    if r_mod == 0:
        freq = float(rng.uniform(1.0, 3.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        amp = float(rng.lognormal(0.0, 0.2))
        flat = amp * np.sin(freq * t + phase) + 0.06 * amp * rng.standard_normal(size)
    elif r_mod == 1:
        f1 = float(rng.uniform(4.0, 9.0))
        f2 = float(rng.uniform(12.0, 22.0))
        amp = 2.2 * float(rng.lognormal(0.0, 0.25))
        flat = (
            0.5
            + amp * (0.65 * np.sin(f1 * t) + 0.35 * np.cos(f2 * t))
            + 0.35 * amp * rng.standard_normal(size)
        )
    else:
        amp = 3.8 * float(rng.lognormal(0.0, 0.3))
        flat = -0.8 + 0.7 * amp * rng.standard_normal(size)
        n_spikes = max(1, size // 32)
        spike_idx = rng.choice(size, size=n_spikes, replace=False)
        flat[spike_idx] += amp * rng.uniform(2.5, 4.5, size=n_spikes)

    if info.requires_nonneg:
        flat = np.abs(flat) + 0.05
    reshaped = flat.reshape(shape)
    return float(reshaped.reshape(-1)[0]) if is_scalar else reshaped


def _perturb_port_pair(
    anchor_value: Any,
    projector: SufficientStatProjector,
    scale: float,
    rng: np.random.Generator,
    *,
    domain_info: _DomainProbeInfo | None = None,
) -> tuple[Any, Any]:
    """Generate an antithetic domain-valid pair ``(v_plus, v_minus)`` around ``anchor_value``."""
    info = domain_info or _DomainProbeInfo()
    is_scalar = isinstance(anchor_value, (int, float)) and not isinstance(anchor_value, bool)
    arr = np.asarray(anchor_value, dtype=np.float64)
    shape = arr.shape if arr.ndim > 0 else (1,)
    flat_arr = arr.reshape(shape)
    kind = projector.kind

    if kind == "simplex":
        z = rng.standard_normal(shape)
        vp = np.maximum(flat_arr * np.exp(scale * z), 1e-12)
        vm = np.maximum(flat_arr * np.exp(-scale * z), 1e-12)
        vp = vp / np.sum(vp, axis=-1, keepdims=True)
        vm = vm / np.sum(vm, axis=-1, keepdims=True)
        return vp, vm

    if kind == "positive_scale" or info.requires_positive:
        z = rng.standard_normal(shape)
        vp = np.maximum(flat_arr * np.exp(scale * z), 1e-12)
        vm = np.maximum(flat_arr * np.exp(-scale * z), 1e-12)
        if is_scalar:
            return float(vp.reshape(-1)[0]), float(vm.reshape(-1)[0])
        return vp, vm

    if kind == "count_event" or info.monotone_events:
        eff_scale = max(scale, 0.12)
        if info.monotone_events:
            # Perturb inter-arrival spacings multiplicatively so event count/density shifts smoothly
            base_1d = flat_arr.reshape(-1)
            if base_1d.size <= 1:
                step = max(1.0, eff_scale * 5.0)
                return np.maximum(base_1d + step, 0.0), np.maximum(base_1d - step, 0.0)
            diffs = np.diff(np.concatenate([[0.0], base_1d]))
            z_d = rng.standard_normal(diffs.shape)
            dp = np.maximum(1.0, np.round(diffs * np.exp(eff_scale * z_d)))
            dm = np.maximum(1.0, np.round(diffs * np.exp(-eff_scale * z_d)))
            return np.cumsum(dp), np.cumsum(dm)

        z = rng.standard_normal(shape)
        base_pos = np.maximum(flat_arr, 0.5)
        vp = np.maximum(0.0, np.round(base_pos * np.exp(eff_scale * z)))
        vm = np.maximum(0.0, np.round(base_pos * np.exp(-eff_scale * z)))
        if np.allclose(vp, vm):
            delta = np.ones(shape, dtype=np.float64)
            vp = flat_arr + delta
            vm = np.maximum(0.0, flat_arr - delta)
        if is_scalar:
            return float(vp.reshape(-1)[0]), float(vm.reshape(-1)[0])
        return vp, vm

    if kind == "matrix_condition" and len(shape) == 2 and info.requires_spd and shape[0] == shape[1]:
        d = shape[0]
        eigvals, eigvecs = np.linalg.eigh(symmetrize(flat_arr))
        eigvals = np.maximum(eigvals, 1e-8)
        z_e = rng.standard_normal(d)
        # Skew-symmetric generator for small orthogonal basis rotation
        raw_skew = rng.standard_normal((d, d))
        skew = 0.5 * (raw_skew - raw_skew.T)
        rot_p, _ = np.linalg.qr(np.eye(d) + scale * skew)
        rot_m, _ = np.linalg.qr(np.eye(d) - scale * skew)
        qp = eigvecs @ rot_p
        qm = eigvecs @ rot_m
        vp = symmetrize((qp * (eigvals * np.exp(scale * z_e))[None, :]) @ qp.T)
        vm = symmetrize((qm * (eigvals * np.exp(-scale * z_e))[None, :]) @ qm.T)
        return vp, vm

    rms = float(np.sqrt(np.mean(flat_arr * flat_arr)))
    step = scale * max(rms, 1.0)
    z = rng.standard_normal(shape)
    vp = flat_arr + step * z
    vm = flat_arr - step * z
    if info.requires_nonneg:
        vp = np.maximum(vp, 0.0)
        vm = np.maximum(vm, 0.0)
    if is_scalar:
        return float(vp.reshape(-1)[0]), float(vm.reshape(-1)[0])
    return vp, vm


# ---------------------------------------------------------------------------
# Output port unpacking & projector inference
# ---------------------------------------------------------------------------


def _is_numeric_arraylike(val: Any) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    if isinstance(val, (int, float, np.number)):
        return True
    try:
        arr = np.asarray(val)
    except Exception:
        return False
    return arr.dtype.kind in {"f", "i", "u"} and arr.size > 0


def _unpack_outputs(
    raw_out: Any,
    output_port_names: Sequence[str] | None = None,
) -> dict[str, FloatArray] | None:
    """Normalize an atom's return value into ``{output_port: float64_ndarray}``."""
    if isinstance(raw_out, Mapping):
        out_dict: dict[str, FloatArray] = {}
        for k, v in raw_out.items():
            if output_port_names and str(k) not in output_port_names:
                continue
            if _is_numeric_arraylike(v):
                arr = np.asarray(v, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    return None
                out_dict[str(k)] = arr
        return out_dict if out_dict else None

    if isinstance(raw_out, tuple):
        out_tuple: dict[str, FloatArray] = {}
        for idx, item in enumerate(raw_out):
            if not _is_numeric_arraylike(item):
                continue
            pname = (
                str(output_port_names[idx])
                if output_port_names and idx < len(output_port_names)
                else f"out_{idx}"
            )
            arr = np.asarray(item, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                return None
            out_tuple[pname] = arr
        return out_tuple if out_tuple else None

    if _is_numeric_arraylike(raw_out):
        arr = np.asarray(raw_out, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return None
        pname = str(output_port_names[0]) if output_port_names else "out"
        return {pname: arr}

    return None


def _infer_output_projector(
    port_name: str,
    samples: Sequence[FloatArray],
) -> SufficientStatProjector:
    """Infer an output ``SufficientStatProjector`` guaranteed to cover observed outputs."""
    first = samples[0]
    low = port_name.lower()

    if any(tok in low for tok in ("rpeak", "peaks", "beats", "onsets", "counts")):
        if all(bool(np.all(s >= 0.0)) for s in samples):
            return CountEventProjector()

    if first.ndim == 2 and min(first.shape) >= 2:
        return MatrixConditionProjector()

    # Fixed small output vectors/scalars keep exact coordinates via IdentityProjector
    sizes = {int(s.size) for s in samples}
    if len(sizes) == 1:
        only_size = next(iter(sizes))
        if 1 <= only_size <= 4:
            return IdentityProjector(dim=only_size)

    if any(tok in low for tok in ("signal", "waveform", "filtered", "audio", "ecg", "eeg", "ppg")):
        return WaveformProjector()

    if first.ndim == 1 and first.size >= 32:
        return WaveformProjector()

    return GenericNumericProjector()


# ---------------------------------------------------------------------------
# Mixture fitting, BIC selection, and cross-port label alignment
# ---------------------------------------------------------------------------


def _fit_family_mixture(
    family: str,
    z_fit: FloatArray,
    *,
    n_components: int,
    factor_rank: int,
    seed: int,
) -> ExponentialFamilyMixture:
    k_eff = max(1, min(int(n_components), z_fit.shape[0]))
    if family == "count":
        return CountMixtureRegime.fit_em(
            np.maximum(z_fit, 0.0),
            n_components=k_eff,
            estimate_dispersion=True,
            seed=seed,
        )
    if family == "gamma":
        return GammaMixtureRegime.fit_em(
            np.maximum(z_fit, 1e-8),
            n_components=k_eff,
            seed=seed,
        )
    if family == "simplex" and z_fit.shape[1] >= 2:
        return SimplexMixtureRegime.fit_em(
            z_fit,
            n_components=k_eff,
            seed=seed,
        )
    eff_rank = min(int(factor_rank), z_fit.shape[1])
    return GaussianMixtureRegime.fit_em(
        z_fit,
        n_components=k_eff,
        factor_rank=eff_rank,
        seed=seed,
    )


def _mixture_free_params(mixture: ExponentialFamilyMixture) -> int:
    k = mixture.n_components
    d = mixture.dim
    if isinstance(mixture, GaussianMixtureRegime):
        r = mixture.factors.shape[2]
        # Effective parameters accounting for rotational gauge r(r-1)/2 in W_k
        cov_params = d + max(0, d * r - (r * (r - 1)) // 2)
        return (k - 1) + k * d + k * cov_params
    if isinstance(mixture, CountMixtureRegime):
        disp_params = k * d if mixture.dispersions is not None else 0
        return (k - 1) + k * d + disp_params
    if isinstance(mixture, GammaMixtureRegime):
        return (k - 1) + 2 * k * d
    if isinstance(mixture, SimplexMixtureRegime):
        return (k - 1) + k * d
    return (k - 1) + 2 * k * d


def _permute_mixture(
    mixture: ExponentialFamilyMixture,
    order: Sequence[int],
) -> ExponentialFamilyMixture:
    idx = np.asarray(list(order), dtype=np.int64)
    w = mixture.weights[idx]
    if isinstance(mixture, GaussianMixtureRegime):
        return GaussianMixtureRegime(
            weights=w,
            means=mixture.means[idx],
            diag_vars=mixture.diag_vars[idx],
            factors=mixture.factors[idx],
        )
    if isinstance(mixture, CountMixtureRegime):
        disp = mixture.dispersions
        return CountMixtureRegime(
            weights=w,
            rates=mixture.rates[idx],
            dispersions=None if disp is None else disp[idx],
        )
    if isinstance(mixture, GammaMixtureRegime):
        return GammaMixtureRegime(
            weights=w,
            shapes=mixture.shapes[idx],
            rates=mixture.rates[idx],
        )
    if isinstance(mixture, SimplexMixtureRegime):
        return SimplexMixtureRegime(
            weights=w,
            concentrations=mixture.concentrations[idx],
        )
    return mixture


def _align_mixture_to_reference(
    reference_resp: FloatArray,
    candidate: ExponentialFamilyMixture,
    z_fit: FloatArray,
) -> ExponentialFamilyMixture:
    """Permute ``candidate`` components to maximize overlap with ``reference_resp``."""
    k = candidate.n_components
    if k <= 1 or k > 6 or reference_resp.shape[1] != k:
        return candidate
    cand_resp = candidate.responsibilities(z_fit)  # (N, K)
    agreement = reference_resp.T @ cand_resp       # (K, K)
    best_perm: tuple[int, ...] = tuple(range(k))
    best_score = -float("inf")
    for perm in itertools.permutations(range(k)):
        score = sum(float(agreement[c_idx, perm[c_idx]]) for c_idx in range(k))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm == tuple(range(k)):
        return candidate
    return _permute_mixture(candidate, best_perm)


# ---------------------------------------------------------------------------
# Damped SVD Jacobian Regression
# ---------------------------------------------------------------------------


def _solve_damped_jacobian(
    dz: FloatArray,
    dy: FloatArray,
    *,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Solve ``dy ≈ dz @ J^T`` via damped SVD, returning ``J`` of shape ``(d_out, d_in)``."""
    if dz.shape[0] == 0:
        return np.zeros((dy.shape[1], dz.shape[1]), dtype=np.float64)

    if weights is not None:
        sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
        dz_w = sqrt_w * dz
        dy_w = sqrt_w * dy
    else:
        dz_w = dz
        dy_w = dy

    u, s, vt = np.linalg.svd(dz_w, full_matrices=False)
    if s.size == 0 or float(s[0]) <= 1e-15:
        return np.zeros((dy.shape[1], dz.shape[1]), dtype=np.float64)

    s_max = float(s[0])
    lam = max(1e-10 * s_max * s_max, 1e-15)
    active = s > (1e-11 * s_max)
    inv_damped = np.where(active, s / (s * s + lam), 0.0)
    # J = dy_w^T @ U @ diag(inv_damped) @ Vt
    jac = (dy_w.T @ (u * inv_damped[None, :])) @ vt
    return np.asarray(jac, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public Calibration Entrypoints
# ---------------------------------------------------------------------------


def calibrate_atom(
    atom_name: str,
    fn: Callable[..., Any],
    *,
    config: CalibrationConfig | None = None,
    port_configs: Mapping[str, PortCalibrationConfig] | None = None,
    output_port_names: Sequence[str] | None = None,
    regime_sampler: Callable[[np.random.Generator, int], Mapping[str, Any]] | None = None,
    calibration_inputs: Sequence[Mapping[str, Any]] | None = None,
) -> AtomUncertaintyProfile:
    """Empirically calibrate an atom into a v2 ``AtomUncertaintyProfile``.

    Splits successful trials into a disjoint fit split ``D_fit`` (for EM regime
    estimation and local spectral ``ComponentChannel`` extraction) and a held-out
    calibration split ``D_cal`` (for finite-sample ``ConformalEnvelope`` bounds).
    """
    cfg = config or CalibrationConfig()
    seed = cfg.seed if cfg.seed is not None else _stable_seed(atom_name)
    rng = np.random.default_rng(seed)

    base_kwargs, calibrated_ports, annotations_map = introspect_atom_signature(
        fn,
        atom_name=atom_name,
        rng=rng,
        port_configs=port_configs,
    )
    if not calibrated_ports and calibration_inputs:
        calibrated_ports = list(calibration_inputs[0].keys())
        for key_name, val_item in calibration_inputs[0].items():
            base_kwargs.setdefault(key_name, val_item)

    if not calibrated_ports:
        raise CalibrationError(f"Atom '{atom_name}' has no calibratable dynamic input ports.")

    cfg_map = dict(port_configs or {})
    domain_infos: dict[str, _DomainProbeInfo] = {}
    in_projectors: dict[str, SufficientStatProjector] = {}

    for pname in calibrated_ports:
        p_cfg = cfg_map.get(pname)
        d_info = _probe_port_constraints(fn, base_kwargs, pname, rng)
        domain_infos[pname] = d_info
        if p_cfg is not None and p_cfg.projector is not None:
            in_projectors[pname] = p_cfg.projector
            continue

        type_desc = p_cfg.type_desc if p_cfg and p_cfg.type_desc else annotations_map.get(pname, "")
        constraints = p_cfg.constraints if p_cfg else ""
        if d_info.requires_positive and "positive" not in constraints.lower():
            constraints = f"{constraints} > 0".strip()
        elif d_info.monotone_events and "event" not in constraints.lower():
            constraints = f"{constraints} event_indices".strip()

        # Refresh base_kwargs[pname] if domain probe detected positivity/SPD constraint
        if d_info.requires_positive or d_info.requires_nonneg or d_info.requires_spd or d_info.monotone_events:
            probe_proj = infer_projector(
                pname,
                type_desc,
                constraints,
                sample_value=base_kwargs.get(pname),
            )
            base_kwargs[pname] = sample_port_regime_value(
                pname,
                probe_proj,
                base_kwargs[pname],
                1,
                rng,
                domain_info=d_info,
            )

        in_projectors[pname] = infer_projector(
            pname,
            type_desc,
            constraints,
            sample_value=base_kwargs.get(pname),
        )

    # Collect multi-regime input-output trials
    raw_trials_in: list[dict[str, Any]] = []
    raw_trials_out: list[dict[str, FloatArray]] = []
    proj_trials_in: dict[str, list[FloatArray]] = {p: [] for p in calibrated_ports}

    n_target = len(calibration_inputs) if calibration_inputs is not None else max(16, cfg.n_trials)
    max_attempts = n_target if calibration_inputs is not None else int(math.ceil(n_target / max(cfg.failure_threshold, 0.1)))
    attempts = 0

    while len(raw_trials_in) < n_target and attempts < max_attempts:
        regime_idx = attempts % 3
        if calibration_inputs is not None:
            call_kw = dict(base_kwargs)
            call_kw.update(calibration_inputs[attempts])
        elif regime_sampler is not None:
            call_kw = dict(base_kwargs)
            call_kw.update(regime_sampler(rng, regime_idx))
        else:
            call_kw = dict(base_kwargs)
            for pname in calibrated_ports:
                call_kw[pname] = sample_port_regime_value(
                    pname,
                    in_projectors[pname],
                    base_kwargs[pname],
                    regime_idx,
                    rng,
                    domain_info=domain_infos[pname],
                )
        attempts += 1

        # Project inputs and ensure support validity
        step_proj_in: dict[str, FloatArray] = {}
        in_ok = True
        for pname in calibrated_ports:
            pres = in_projectors[pname].project(call_kw[pname])
            if not pres.in_domain or not np.all(np.isfinite(pres.features)):
                in_ok = False
                break
            step_proj_in[pname] = pres.features
        if not in_ok:
            continue

        try:
            raw_y = fn(**call_kw)
        except Exception:
            continue

        unpacked = _unpack_outputs(raw_y, output_port_names=output_port_names)
        if not unpacked:
            continue

        raw_trials_in.append(call_kw)
        raw_trials_out.append(unpacked)
        for pname in calibrated_ports:
            proj_trials_in[pname].append(step_proj_in[pname])

    n_valid = len(raw_trials_in)
    success_ratio = float(n_valid) / float(max(1, attempts))
    if n_valid < 8 or success_ratio < cfg.failure_threshold:
        raise CalibrationError(
            f"Atom '{atom_name}' calibration failed: {n_valid}/{attempts} valid trials "
            f"(ratio={success_ratio:.2f} < {cfg.failure_threshold:.2f})."
        )

    # Infer output projectors across all valid trials and project outputs
    common_out_ports = sorted(
        set.intersection(*(set(d.keys()) for d in raw_trials_out))
    )
    if not common_out_ports:
        raise CalibrationError(f"Atom '{atom_name}' produced no consistent numeric output ports.")

    out_projectors: dict[str, SufficientStatProjector] = {}
    proj_trials_out: dict[str, FloatArray] = {}
    valid_mask = np.ones(n_valid, dtype=bool)

    for qname in common_out_ports:
        q_samples = [d[qname] for d in raw_trials_out]
        q_proj = _infer_output_projector(qname, q_samples)
        out_projectors[qname] = q_proj
        q_feats: list[FloatArray] = []
        for idx, sample in enumerate(q_samples):
            q_res = q_proj.project(sample)
            if not q_res.in_domain or not np.all(np.isfinite(q_res.features)):
                valid_mask[idx] = False
                q_feats.append(np.zeros(q_proj.dim, dtype=np.float64))
            else:
                q_feats.append(q_res.features)
        proj_trials_out[qname] = np.stack(q_feats, axis=0)

    keep_indices = np.flatnonzero(valid_mask)
    if keep_indices.size < 8:
        raise CalibrationError(f"Atom '{atom_name}' had fewer than 8 valid projected output trials.")

    raw_trials_in = [raw_trials_in[int(i)] for i in keep_indices]
    raw_trials_out = [raw_trials_out[int(i)] for i in keep_indices]
    z_in_all: dict[str, FloatArray] = {
        pname: np.stack([proj_trials_in[pname][int(i)] for i in keep_indices], axis=0)
        for pname in calibrated_ports
    }
    z_out_all: dict[str, FloatArray] = {
        qname: proj_trials_out[qname][keep_indices] for qname in common_out_ports
    }
    n_valid = int(keep_indices.size)

    # Disjoint random partition into D_fit and D_cal (for exact split-conformal coverage)
    perm_idx = rng.permutation(n_valid)
    n_fit = max(4, min(n_valid - 4, int(round(cfg.fit_fraction * n_valid))))
    fit_idx = perm_idx[:n_fit]
    cal_idx = perm_idx[n_fit:]

    z_in_fit = {p: z_in_all[p][fit_idx] for p in calibrated_ports}
    z_in_cal = {p: z_in_all[p][cal_idx] for p in calibrated_ports}
    z_out_fit = {q: z_out_all[q][fit_idx] for q in common_out_ports}
    fit_raw_in = [raw_trials_in[int(i)] for i in fit_idx]

    # Select K (fixed or via joint BIC across calibrated input ports)
    cand_ks: tuple[int, ...]
    if cfg.n_components is not None:
        cand_ks = (max(1, min(int(cfg.n_components), n_fit // 2)),)
    else:
        cand_ks = tuple(
            sorted({max(1, min(int(k_val), max(1, n_fit // 4))) for k_val in cfg.candidate_components})
        )

    best_k = 1
    best_bic = float("inf")
    best_mixtures: dict[str, ExponentialFamilyMixture] = {}

    for k_try in cand_ks:
        trial_mixtures: dict[str, ExponentialFamilyMixture] = {}
        total_ll = 0.0
        total_params = 0
        ref_resp: FloatArray | None = None
        for p_ord, pname in enumerate(calibrated_ports):
            fam = in_projectors[pname].preferred_family
            mix = _fit_family_mixture(
                fam,
                z_in_fit[pname],
                n_components=k_try,
                factor_rank=cfg.factor_rank,
                seed=seed + 17 * k_try + p_ord,
            )
            if ref_resp is None:
                ref_resp = mix.responsibilities(z_in_fit[pname])
            else:
                mix = _align_mixture_to_reference(ref_resp, mix, z_in_fit[pname])
            lp = np.asarray(mix.log_prob(z_in_fit[pname]), dtype=np.float64)
            finite_lp = lp[np.isfinite(lp)]
            ll_sum = float(np.sum(finite_lp)) if finite_lp.size == n_fit else -1e12
            total_ll += ll_sum
            total_params += _mixture_free_params(mix)
            trial_mixtures[pname] = mix

        bic = -2.0 * total_ll + float(total_params) * math.log(max(2, n_fit))
        if bic < best_bic or not best_mixtures:
            best_bic = bic
            best_k = k_try
            best_mixtures = trial_mixtures

    # Build PortRegimeSpecs with disjoint split-conformal envelopes on D_cal
    input_regimes: dict[str, PortRegimeSpec] = {}
    for pname in calibrated_ports:
        mix = best_mixtures[pname]
        cal_scores = np.asarray(mix.nonconformity_score(z_in_cal[pname]), dtype=np.float64).reshape(-1)
        # Clip extreme +inf scores on rare numeric boundary draws to finite max for storage safety
        finite_mask = np.isfinite(cal_scores)
        if np.any(finite_mask):
            hi_cap = float(np.max(cal_scores[finite_mask])) + 1e6
            cal_scores = np.where(finite_mask, cal_scores, hi_cap)
        else:
            cal_scores = np.zeros(max(1, cal_scores.size), dtype=np.float64)
        envelope = ConformalEnvelope(cal_scores)
        input_regimes[pname] = PortRegimeSpec(
            port_name=pname,
            projector_spec=in_projectors[pname].to_spec(),
            mixture=mix,
            conformal_envelope=envelope,
        )

    # Joint component responsibilities across D_fit
    log_joint = np.zeros((n_fit, best_k), dtype=np.float64)
    prior_w = best_mixtures[calibrated_ports[0]].weights
    log_joint += np.log(np.maximum(prior_w, 1e-12))[None, :]
    for pname in calibrated_ports:
        comp_lp = best_mixtures[pname].component_log_prob(z_in_fit[pname])
        comp_lp = np.where(np.isfinite(comp_lp), comp_lp, -1e6)
        log_joint += comp_lp
    row_max = np.max(log_joint, axis=1, keepdims=True)
    unnorm = np.exp(log_joint - row_max)
    joint_resp = unnorm / np.maximum(np.sum(unnorm, axis=1, keepdims=True), 1e-15)

    # Estimate per-component LowRankPlusTailOperator channels & intrinsic output covariances
    components_list: list[ComponentChannel] = []
    raw_amplifications: list[float] = []

    for k in range(best_k):
        gamma_k = joint_resp[:, k]
        nk = float(np.sum(gamma_k))
        top_order = np.argsort(-gamma_k)
        n_anchors = min(3, n_fit)
        anchor_indices = [int(i) for i in top_order[:n_anchors]]

        mu_z_k = {
            p: (gamma_k @ z_in_fit[p]) / max(nk, 1e-12) for p in calibrated_ports
        }
        mu_y_k = {
            q: (gamma_k @ z_out_fit[q]) / max(nk, 1e-12) for q in common_out_ports
        }

        dense_jacs: dict[tuple[str, str], FloatArray] = {}
        comp_ops: dict[tuple[str, str], LowRankPlusTailOperator] = {}

        probes_per_anchor = max(2, int(math.ceil(cfg.n_anchor_probes / max(1, n_anchors))))

        for pname in calibrated_ports:
            dz_probes: list[FloatArray] = []
            dy_probes: dict[str, list[FloatArray]] = {q: [] for q in common_out_ports}

            if cfg.n_anchor_probes > 0:
                for a_idx in anchor_indices:
                    anchor_kw = fit_raw_in[a_idx]
                    for p_i in range(probes_per_anchor):
                        scale = cfg.probe_scales[p_i % len(cfg.probe_scales)]
                        vp, vm = _perturb_port_pair(
                            anchor_kw[pname],
                            in_projectors[pname],
                            scale,
                            rng,
                            domain_info=domain_infos[pname],
                        )
                        proj_p = in_projectors[pname].project(vp)
                        proj_m = in_projectors[pname].project(vm)
                        if not (proj_p.in_domain and proj_m.in_domain):
                            continue

                        kw_p = dict(anchor_kw)
                        kw_m = dict(anchor_kw)
                        kw_p[pname] = vp
                        kw_m[pname] = vm
                        try:
                            out_p = _unpack_outputs(fn(**kw_p), output_port_names=output_port_names)
                            out_m = _unpack_outputs(fn(**kw_m), output_port_names=output_port_names)
                        except Exception:
                            continue
                        if not out_p or not out_m:
                            continue

                        # Track raw perturbation amplification ratio for v1 legacy_scalar_factor
                        try:
                            dx_raw = np.asarray(vp, dtype=np.float64).ravel() - np.asarray(vm, dtype=np.float64).ravel()
                            first_q = common_out_ports[0]
                            dy_raw = out_p[first_q].ravel() - out_m[first_q].ravel()
                            nx = float(np.linalg.norm(dx_raw)) / math.sqrt(max(1, dx_raw.size))
                            ny = float(np.linalg.norm(dy_raw)) / math.sqrt(max(1, dy_raw.size))
                            if nx > 1e-14 and math.isfinite(ny / nx):
                                raw_amplifications.append(ny / nx)
                        except Exception:
                            pass

                        q_deltas: dict[str, FloatArray] = {}
                        all_q_ok = True
                        for qname in common_out_ports:
                            if qname not in out_p or qname not in out_m:
                                all_q_ok = False
                                break
                            qp = out_projectors[qname].project(out_p[qname])
                            qm = out_projectors[qname].project(out_m[qname])
                            if not (qp.in_domain and qm.in_domain):
                                all_q_ok = False
                                break
                            q_deltas[qname] = qp.features - qm.features
                        if not all_q_ok:
                            continue

                        dz_probes.append(proj_p.features - proj_m.features)
                        for qname in common_out_ports:
                            dy_probes[qname].append(q_deltas[qname])

            # Combine causal local anchor probes with lightly-weighted within-component fit variations
            dz_fit_centered = z_in_fit[pname] - mu_z_k[pname][None, :]
            for qname in common_out_ports:
                dy_fit_centered = z_out_fit[qname] - mu_y_k[qname][None, :]
                if dz_probes:
                    dz_mat = np.stack(dz_probes, axis=0)
                    dy_mat = np.stack(dy_probes[qname], axis=0)
                    if len(calibrated_ports) == 1:
                        # Blend high-weight causal probes with low-weight regime-wide variations
                        probe_w = np.ones(dz_mat.shape[0], dtype=np.float64)
                        fit_w = 0.05 * gamma_k
                        jac = _solve_damped_jacobian(
                            np.concatenate([dz_mat, dz_fit_centered], axis=0),
                            np.concatenate([dy_mat, dy_fit_centered], axis=0),
                            weights=np.concatenate([probe_w, fit_w], axis=0),
                        )
                    else:
                        jac = _solve_damped_jacobian(dz_mat, dy_mat)
                else:
                    jac = _solve_damped_jacobian(
                        dz_fit_centered,
                        dy_fit_centered,
                        weights=gamma_k,
                    )

                dense_jacs[(pname, qname)] = jac
                comp_ops[(pname, qname)] = LowRankPlusTailOperator.from_dense(
                    jac,
                    max_rank=cfg.operator_max_rank,
                    tail_mode="spectral_upper_bound",
                )

        # Estimate intrinsic output noise covariance Sigma_{eps, k, q}
        # 1) Responsibility-weighted linearization + projection residual on D_fit
        # 2) Repeatability check at top anchor for purely stochastic atoms
        anchor0_kw = fit_raw_in[anchor_indices[0]]
        repeat_feats: dict[str, list[FloatArray]] = {q: [] for q in common_out_ports}
        for _rep in range(4):
            try:
                rep_out = _unpack_outputs(fn(**anchor0_kw), output_port_names=output_port_names)
            except Exception:
                rep_out = None
            if rep_out:
                for qname in common_out_ports:
                    if qname in rep_out:
                        r_proj = out_projectors[qname].project(rep_out[qname])
                        if r_proj.in_domain and np.all(np.isfinite(r_proj.features)):
                            repeat_feats[qname].append(r_proj.features)

        intrinsic_covs: dict[str, FloatArray] = {}
        for qname in common_out_ports:
            pred_dev = np.zeros_like(z_out_fit[qname])
            for pname in calibrated_ports:
                dz_p = z_in_fit[pname] - mu_z_k[pname][None, :]
                pred_dev += dz_p @ dense_jacs[(pname, qname)].T
            residuals = (z_out_fit[qname] - mu_y_k[qname][None, :]) - pred_dev
            weighted_res = (gamma_k[:, None] * residuals).T @ residuals / max(nk, 1e-12)
            cov_eps = symmetrize(weighted_res)

            if len(repeat_feats[qname]) >= 2:
                rep_mat = np.stack(repeat_feats[qname], axis=0)
                rep_cov = symmetrize(np.cov(rep_mat, rowvar=False, bias=False))
                if rep_cov.ndim == 0:
                    rep_cov = np.array([[float(rep_cov)]], dtype=np.float64)
                # Ensure cov_eps dominates pure stochastic repeatability variance
                diff_eig, diff_vec = np.linalg.eigh(rep_cov - cov_eps)
                pos_excess = np.maximum(diff_eig, 0.0)
                if np.any(pos_excess > 0.0):
                    cov_eps = symmetrize(
                        cov_eps + (diff_vec * pos_excess[None, :]) @ diff_vec.T
                    )

            intrinsic_covs[qname] = cov_eps

        components_list.append(
            ComponentChannel(
                component_index=k,
                operators=comp_ops,
                intrinsic_output_cov=intrinsic_covs,
            )
        )

    n_cal = int(cal_idx.size)
    cal_adequacy = min(1.0, float(n_cal) / 100.0)
    confidence = round(min(0.98, 0.55 * success_ratio + 0.43 * cal_adequacy), 4)

    legacy_sf: float | None = None
    if raw_amplifications:
        legacy_sf = round(float(np.median(np.asarray(raw_amplifications, dtype=np.float64))), 6)

    port_summaries = [
        f"{p}:{input_regimes[p].projector_spec.kind}->{input_regimes[p].mixture.family}(K={best_k},d={input_regimes[p].mixture.dim})"
        for p in calibrated_ports
    ]
    summary_str = "; ".join(port_summaries)
    out_ProjectorSpecs: dict[str, ProjectorSpec] = {
        q: out_projectors[q].to_spec() for q in common_out_ports
    }

    return AtomUncertaintyProfile(
        atom_name=atom_name,
        mode="empirical",
        confidence=confidence,
        trust_tier=cfg.trust_tier,
        input_regimes=input_regimes,
        output_projectors=out_ProjectorSpecs,
        components=tuple(components_list),
        legacy_scalar_factor=legacy_sf,
        n_trials=n_valid,
        epsilon=float(cfg.probe_scales[0]) if cfg.probe_scales else 0.0,
        input_regime_summary=summary_str,
        notes=(
            f"v2 mixture+spectral calibration: K={best_k}, "
            f"n_fit={n_fit}, n_cal={n_cal}, probes={cfg.n_anchor_probes}"
        ),
        schema_version="2.0",
    )


def try_calibrate_atom(
    atom_name: str,
    fn: Callable[..., Any],
    *,
    config: CalibrationConfig | None = None,
    port_configs: Mapping[str, PortCalibrationConfig] | None = None,
    output_port_names: Sequence[str] | None = None,
    regime_sampler: Callable[[np.random.Generator, int], Mapping[str, Any]] | None = None,
    calibration_inputs: Sequence[Mapping[str, Any]] | None = None,
) -> AtomUncertaintyProfile | None:
    """Wrapper around ``calibrate_atom`` returning ``None`` if calibration fails."""
    try:
        return calibrate_atom(
            atom_name,
            fn,
            config=config,
            port_configs=port_configs,
            output_port_names=output_port_names,
            regime_sampler=regime_sampler,
            calibration_inputs=calibration_inputs,
        )
    except CalibrationError as exc:
        logger.info("Skipping atom %s: %s", atom_name, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error calibrating atom %s: %s", atom_name, exc)
        return None
