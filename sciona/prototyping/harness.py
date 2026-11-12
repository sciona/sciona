"""Problem-agnostic evaluation harnesses, synthetic fixture generators, and metamorphic invariants.

Every optimization problem — signal processing, tabular ML, graph algorithms,
numerical PDE/linear algebra, or streaming state estimation — plugs into the
same ``ProblemHarness`` protocol so ``EvaluationFunnel`` and ``Principal`` run
identically across domains.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import BoundaryKind, EdgeKind, IOSpec, PortTypeSpec
from sciona.uncertainty.propagation import RiskBudget

DEFAULT_SYNTHETIC_REGIMES: tuple[str, ...] = (
    "nominal",
    "noisy_outliers",
    "ill_conditioned",
    "boundary_small",
    "scale_probe",
)


@dataclass
class EvalCase:
    """Single evaluation input case (stateless or stateful ``fit_inputs -> inputs``)."""

    case_id: str
    inputs: dict[str, Any]
    expected: Any = None
    fit_inputs: dict[str, Any] | None = None
    regime: str = "nominal"
    tier: int = 1
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseScore:
    """Outcome of scoring one ``EvalCase``."""

    case_id: str
    loss: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    invariant_violations: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: str = ""
    failed_node_id: str = ""


@dataclass(frozen=True)
class MetamorphicRelation:
    """Metamorphic property relating ``f(x)`` and ``f(T(x))`` without ground-truth labels."""

    name: str
    transform_inputs: Callable[[Mapping[str, Any]], dict[str, Any]]
    verify_outputs: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[bool, str]]

    @classmethod
    def shift_invariance(
        cls,
        input_port: str,
        *,
        shift: float = 5.0,
        output_port: str | None = None,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> "MetamorphicRelation":
        """Assert ``f(x + c) == f(x)`` (e.g. demeaned variance, peak spacing, translation-invariant stats)."""

        def _tx(inp: Mapping[str, Any]) -> dict[str, Any]:
            out = dict(inp)
            if input_port in out:
                val = out[input_port]
                out[input_port] = val + shift if isinstance(val, (int, float, np.ndarray, np.number)) else val
            return out

        def _check(base_out: Mapping[str, Any], trans_out: Mapping[str, Any]) -> tuple[bool, str]:
            a = _pick_output_value(base_out, output_port)
            b = _pick_output_value(trans_out, output_port)
            if _values_close(a, b, atol=atol, rtol=rtol):
                return True, ""
            return False, f"shift_invariance({input_port}, +{shift}) violated"

        return cls(name=f"shift_invariance[{input_port}]", transform_inputs=_tx, verify_outputs=_check)

    @classmethod
    def scale_equivariance(
        cls,
        input_port: str,
        *,
        scale: float = 2.0,
        power: float = 1.0,
        output_port: str | None = None,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> "MetamorphicRelation":
        """Assert ``f(c * x) == (c ** power) * f(x)`` (``power=0`` tests scale invariance)."""

        def _tx(inp: Mapping[str, Any]) -> dict[str, Any]:
            out = dict(inp)
            if input_port in out:
                val = out[input_port]
                out[input_port] = val * scale if isinstance(val, (int, float, np.ndarray, np.number)) else val
            return out

        def _check(base_out: Mapping[str, Any], trans_out: Mapping[str, Any]) -> tuple[bool, str]:
            a = _pick_output_value(base_out, output_port)
            b = _pick_output_value(trans_out, output_port)
            factor = float(scale) ** float(power)
            expected_b = a * factor if isinstance(a, (int, float, np.ndarray, np.number)) else a
            if _values_close(expected_b, b, atol=atol, rtol=rtol):
                return True, ""
            return False, f"scale_equivariance({input_port}, *{scale}^{power}) violated"

        return cls(name=f"scale_equivariance[{input_port}]", transform_inputs=_tx, verify_outputs=_check)

    @classmethod
    def permutation_invariance(
        cls,
        input_port: str,
        *,
        axis: int = 0,
        output_port: str | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        seed: int = 17,
    ) -> "MetamorphicRelation":
        """Assert ``f(P x) == f(x)`` under deterministic row/element permutation."""

        def _tx(inp: Mapping[str, Any]) -> dict[str, Any]:
            out = dict(inp)
            if input_port in out:
                val = out[input_port]
                rng = np.random.default_rng(seed)
                if isinstance(val, np.ndarray) and val.ndim > axis and val.shape[axis] > 1:
                    perm = rng.permutation(val.shape[axis])
                    out[input_port] = np.take(val, perm, axis=axis)
                elif isinstance(val, list) and len(val) > 1:
                    perm = rng.permutation(len(val))
                    out[input_port] = [val[int(i)] for i in perm]
            return out

        def _check(base_out: Mapping[str, Any], trans_out: Mapping[str, Any]) -> tuple[bool, str]:
            a = _pick_output_value(base_out, output_port)
            b = _pick_output_value(trans_out, output_port)
            if _values_close(a, b, atol=atol, rtol=rtol):
                return True, ""
            return False, f"permutation_invariance({input_port}, axis={axis}) violated"

        return cls(name=f"permutation_invariance[{input_port}]", transform_inputs=_tx, verify_outputs=_check)


@runtime_checkable
class ProblemHarness(Protocol):
    """Protocol implemented by every prototyping evaluation harness."""

    @property
    def root_inputs(self) -> Sequence[IOSpec]: ...

    @property
    def root_outputs(self) -> Sequence[IOSpec]: ...

    @property
    def risk_budget(self) -> RiskBudget | None: ...

    def synthetic_cases(self) -> Sequence[EvalCase]: ...

    def validation_cases(self) -> Sequence[EvalCase]: ...

    def holdout_cases(self) -> Sequence[EvalCase]: ...

    def score(self, case: EvalCase, outputs: Mapping[str, Any]) -> CaseScore: ...


def infer_cdg_root_ports(cdg: CDGExport) -> tuple[list[IOSpec], list[IOSpec]]:
    """Extract unwired root ``IOSpec`` inputs and sink ``IOSpec`` outputs from *cdg*."""
    leaves = cdg.leaf_nodes()
    if not leaves:
        leaves = [n for n in cdg.nodes if not n.children]
    leaf_ids = {n.node_id for n in leaves}
    data_edges = [
        e
        for e in cdg.edges
        if e.source_id in leaf_ids
        and e.target_id in leaf_ids
        and e.edge_kind != EdgeKind.CALLABLE_INJECTION
    ]
    wired_targets = {(e.target_id, e.input_name) for e in data_edges}
    consumed_sources = {(e.source_id, e.output_name) for e in data_edges}
    outgoing_nodes = {e.source_id for e in data_edges}

    seen_in: set[str] = set()
    root_in: list[IOSpec] = []
    for node in leaves:
        for inp in node.inputs:
            if (node.node_id, inp.name) not in wired_targets and inp.name not in seen_in:
                seen_in.add(inp.name)
                root_in.append(inp)

    seen_out: set[str] = set()
    root_out: list[IOSpec] = []
    for node in leaves:
        is_sink = node.node_id not in outgoing_nodes or node.boundary_kind == BoundaryKind.ROOT_OUTPUT
        for outp in node.outputs:
            if ((node.node_id, outp.name) not in consumed_sources or is_sink) and outp.name not in seen_out:
                seen_out.add(outp.name)
                root_out.append(outp)

    return root_in, root_out


def _pick_output_value(outputs: Mapping[str, Any], preferred_port: str | None = None) -> Any:
    if preferred_port and preferred_port in outputs:
        return outputs[preferred_port]
    # Prefer unqualified sink port names over 'node_id.port'
    unqualified = [v for k, v in outputs.items() if "." not in str(k)]
    if len(unqualified) == 1:
        return unqualified[0]
    if unqualified:
        return tuple(unqualified)
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return dict(outputs)


def _values_close(a: Any, b: Any, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if arr_a.shape != arr_b.shape:
            return False
        if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
            return bool(np.allclose(arr_a, arr_b, atol=atol, rtol=rtol))
        return bool(np.array_equal(arr_a, arr_b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_values_close(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
    return bool(a == b)


def check_output_invariants(
    outputs: Mapping[str, Any],
    expected_specs: Sequence[IOSpec] | None,
    *,
    allow_nonfinite: bool = False,
) -> list[str]:
    """Validate finiteness, non-emptiness, and ``PortTypeSpec`` compatibility on outputs."""
    violations: list[str] = []
    specs = list(expected_specs) if expected_specs else []
    if not outputs and specs:
        violations.append("CDG produced empty outputs dict")
        return violations

    for spec in specs:
        if spec.name not in outputs:
            if len(specs) == 1 and outputs:
                val = _pick_output_value(outputs)
            elif spec.required:
                violations.append(f"Missing required output port '{spec.name}'")
                continue
            else:
                continue
        else:
            val = outputs[spec.name]

        ptype = spec.resolved_port_type()
        violations.extend(
            _validate_single_value_against_port(
                spec.name,
                val,
                ptype,
                allow_nonfinite=allow_nonfinite,
            )
        )
    return violations


def _validate_single_value_against_port(
    port_name: str,
    val: Any,
    ptype: PortTypeSpec,
    *,
    allow_nonfinite: bool,
) -> list[str]:
    issues: list[str] = []
    if val is None:
        if not ptype.optional and ptype.kind != "none":
            issues.append(f"Output '{port_name}' is None for non-optional port ({ptype.raw})")
        return issues

    if isinstance(val, (float, np.floating)):
        if not allow_nonfinite and not math.isfinite(float(val)):
            issues.append(f"Output '{port_name}' is non-finite scalar ({val})")
        if ptype.kind == "array":
            issues.append(f"Output '{port_name}' expected array ({ptype.raw}), got scalar")
        return issues

    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        if ptype.kind == "array":
            issues.append(f"Output '{port_name}' expected array ({ptype.raw}), got int scalar")
        return issues

    if isinstance(val, np.ndarray):
        if val.size == 0:
            issues.append(f"Output '{port_name}' is an empty ndarray (shape={val.shape})")
        elif not allow_nonfinite and np.issubdtype(val.dtype, np.floating):
            if not bool(np.isfinite(val).all()):
                n_bad = int((~np.isfinite(val)).sum())
                issues.append(f"Output '{port_name}' contains {n_bad} NaN/Inf element(s)")
        if ptype.kind == "scalar" and val.ndim > 0 and val.size != 1:
            issues.append(f"Output '{port_name}' expected scalar ({ptype.raw}), got shape {val.shape}")
        if ptype.ndim is not None and val.ndim != ptype.ndim:
            issues.append(
                f"Output '{port_name}' ndim mismatch: expected {ptype.ndim}D, got {val.ndim}D (shape={val.shape})"
            )
        return issues

    cls_mod = type(val).__module__
    cls_name = type(val).__name__
    if cls_mod.startswith("pandas."):
        if ptype.kind == "scalar":
            issues.append(f"Output '{port_name}' expected scalar, got {cls_name}")
        if hasattr(val, "isna") and not allow_nonfinite:
            try:
                na_total = int(val.isna().sum().sum() if cls_name == "DataFrame" else val.isna().sum())
                if na_total > 0:
                    issues.append(f"Output '{port_name}' ({cls_name}) contains {na_total} NA value(s)")
            except Exception:
                pass
        return issues

    if isinstance(val, tuple) and ptype.kind == "tuple" and ptype.tuple_elements:
        if len(val) != len(ptype.tuple_elements):
            issues.append(
                f"Output '{port_name}' tuple arity mismatch: expected {len(ptype.tuple_elements)}, got {len(val)}"
            )
        else:
            for idx, (elem_val, elem_spec) in enumerate(zip(val, ptype.tuple_elements)):
                issues.extend(
                    _validate_single_value_against_port(
                        f"{port_name}[{idx}]",
                        elem_val,
                        elem_spec,
                        allow_nonfinite=allow_nonfinite,
                    )
                )
    return issues


def generate_synthetic_value(
    port: IOSpec,
    *,
    regime: str = "nominal",
    rng: np.random.Generator,
    base_length: int = 256,
    n_features: int = 6,
) -> Any:
    """Deterministically synthesize a typed test input from ``IOSpec`` + *regime*."""
    ptype = port.resolved_port_type()
    pname = port.name.strip().lower()
    tags = {t.lower() for t in ptype.domain_tags}

    # Semantic parameter heuristics for common domain scalar ports
    if ptype.kind in {"scalar", "any"} and any(
        tok == pname or pname.endswith(f"_{tok}")
        for tok in ("sampling_rate", "sample_rate", "fs", "sr", "rate_hz")
    ):
        return 250.0 if ptype.dtype != "int" else 250

    if ptype.kind == "scalar":
        if ptype.dtype == "bool":
            return True
        if ptype.dtype in {"int", "int32", "int64"}:
            if any(tok in pname for tok in ("k", "clusters", "window", "order", "degree", "lags", "bins")):
                return 3
            if "n_" in pname or pname in {"n", "size", "length"}:
                return 16 if regime == "boundary_small" else 64
            return int(rng.integers(2, 12))
        # Float scalar
        if any(tok in pname for tok in ("alpha", "tol", "lr", "eps", "cutoff", "prob", "quantile")):
            return 0.05
        if any(tok in pname for tok in ("lowcut", "f_low")):
            return 0.5
        if any(tok in pname for tok in ("highcut", "f_high")):
            return 40.0
        val = float(rng.normal(1.0, 0.25))
        if "positive" in port.constraints.lower() or "std" in pname or "scale" in pname or "sigma" in pname:
            val = abs(val) + 0.25
        return val

    # Sample length scaling by regime
    if regime == "boundary_small":
        length = 16
        rows = 12
    elif regime == "scale_probe":
        length = max(base_length * 4, 1024)
        rows = max(128, base_length)
    else:
        length = base_length
        rows = 64

    if ptype.kind == "tuple":
        if ptype.tuple_elements:
            return tuple(
                generate_synthetic_value(
                    IOSpec(name=f"{port.name}_{i}", type_desc=sub.raw, port_type=sub),
                    regime=regime,
                    rng=rng,
                    base_length=base_length,
                    n_features=n_features,
                )
                for i, sub in enumerate(ptype.tuple_elements)
            )
        return (
            rng.normal(0.0, 1.0, size=rows).astype(np.float64),
            rng.normal(0.0, 1.0, size=rows).astype(np.float64),
        )

    if ptype.kind == "dataframe":
        import pandas as pd  # type: ignore[import-untyped]

        mat = _generate_matrix(rows, n_features, regime=regime, rng=rng)
        cols = [f"f_{j}" for j in range(n_features)]
        return pd.DataFrame(mat, columns=cols)

    if ptype.kind == "series":
        import pandas as pd

        vec = _generate_1d_signal_or_vector(length, pname=pname, tags=tags, regime=regime, rng=rng)
        return pd.Series(vec, name=port.name)

    if ptype.kind == "sequence":
        if ptype.dtype in {"int", "int32", "int64"}:
            return [int(x) for x in rng.integers(0, max(4, length // 4), size=min(length, 32))]
        return [float(x) for x in rng.normal(0.0, 1.0, size=min(length, 32))]

    # Default / array / matrix / tensor generation
    ndim = ptype.ndim
    if ndim is None:
        if any(t in tags for t in ("matrix", "tabular", "image", "graph")) or pname in {"x_train", "features", "matrix", "cov", "adjacency"}:
            ndim = 2
        else:
            ndim = 1

    if ndim == 1:
        if ptype.dtype in {"int", "int32", "int64"} or any(
            tok in pname for tok in ("peaks", "rpeaks", "indices", "events", "labels", "y_class")
        ):
            if any(tok in pname for tok in ("peaks", "rpeaks", "indices", "events")):
                step = max(4, length // 6)
                return np.arange(step // 2, length, step, dtype=np.int64)
            return rng.integers(0, 2, size=rows, dtype=np.int64)
        return _generate_1d_signal_or_vector(length, pname=pname, tags=tags, regime=regime, rng=rng)

    if ndim == 2:
        if "graph" in tags or "adj" in pname or "laplacian" in pname:
            n_v = min(rows, 24)
            raw_g = rng.uniform(0.0, 1.0, size=(n_v, n_v))
            sym = ((raw_g + raw_g.T) > 1.35).astype(np.float64)
            np.fill_diagonal(sym, 0.0)
            return sym
        if "cov" in pname or "spd" in port.constraints.lower() or "gram" in pname:
            a = rng.normal(0.0, 1.0, size=(n_features, n_features))
            eps = 1e-5 if regime == "ill_conditioned" else 0.25
            return (a @ a.T + eps * np.eye(n_features)).astype(np.float64)
        return _generate_matrix(rows, n_features, regime=regime, rng=rng)

    shape = tuple([min(rows, 16)] + [4] * (ndim - 1))
    arr = rng.normal(0.0, 1.0, size=shape).astype(np.float64)
    if regime == "noisy_outliers" and arr.size > 0:
        arr.ravel()[0] = 25.0
    return arr


def _generate_1d_signal_or_vector(
    length: int,
    *,
    pname: str,
    tags: set[str],
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    t = np.linspace(0.0, 4.0, num=length, endpoint=False, dtype=np.float64)
    if "signal" in tags or any(tok in pname for tok in ("signal", "waveform", "ecg", "ppg", "eeg", "audio")):
        # Realistic smooth composite oscillation + periodic sharp R-like pulses
        clean = 0.6 * np.sin(2.0 * np.pi * 1.2 * t) + 0.15 * np.cos(2.0 * np.pi * 3.0 * t)
        for center in (0.6, 1.4, 2.2, 3.0, 3.8):
            clean += 2.0 * np.exp(-0.5 * ((t - center) / 0.025) ** 2)
    else:
        clean = np.sin(2.0 * np.pi * 0.5 * t) + 0.5 * rng.normal(0.0, 1.0, size=length)

    if regime == "nominal":
        return (clean + 0.03 * rng.normal(0.0, 1.0, size=length)).astype(np.float64)
    if regime == "noisy_outliers":
        noisy = clean + 0.2 * rng.normal(0.0, 1.0, size=length)
        spike_idx = rng.choice(length, size=max(1, length // 32), replace=False)
        noisy[spike_idx] += rng.choice([-12.0, 12.0], size=spike_idx.size)
        return noisy.astype(np.float64)
    if regime == "ill_conditioned":
        # Strong low-frequency baseline drift + tiny dynamic range component
        drift = 15.0 * np.linspace(-1.0, 1.0, num=length) + 1e-4 * clean
        return drift.astype(np.float64)
    return clean.astype(np.float64)


def _generate_matrix(
    rows: int,
    cols: int,
    *,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    mat = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float64)
    if regime == "noisy_outliers" and rows > 2:
        mat[0, :] += 15.0
        mat[-1, 0] = -20.0
    elif regime == "ill_conditioned" and cols >= 2:
        # Near-collinear columns with condition number ~1e5
        mat[:, 1] = mat[:, 0] + 1e-5 * rng.normal(0.0, 1.0, size=rows)
    return mat


def compute_objective_metric(
    metric: str,
    predicted: Any,
    expected: Any,
) -> float:
    """Compute a standard scalar loss to minimize (`rmse`, `mse`, `mae`, `logloss`, `error_rate`)."""
    m = metric.strip().lower()
    pred_arr = np.asarray(predicted, dtype=np.float64)
    exp_arr = np.asarray(expected, dtype=np.float64)

    if pred_arr.shape != exp_arr.shape and pred_arr.size == exp_arr.size:
        pred_arr = pred_arr.reshape(exp_arr.shape)

    if m in {"mse", "l2"}:
        return float(np.mean((pred_arr - exp_arr) ** 2))
    if m in {"rmse", "precision", "default"}:
        return float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))
    if m in {"mae", "l1"}:
        return float(np.mean(np.abs(pred_arr - exp_arr)))
    if m in {"logloss", "bce", "cross_entropy"}:
        eps = 1e-12
        clipped = np.clip(pred_arr, eps, 1.0 - eps)
        return float(-np.mean(exp_arr * np.log(clipped) + (1.0 - exp_arr) * np.log(1.0 - clipped)))
    if m in {"error_rate", "zero_one"}:
        pred_labels = (pred_arr >= 0.5).astype(np.int64) if np.issubdtype(pred_arr.dtype, np.floating) and np.all((exp_arr == 0) | (exp_arr == 1)) else pred_arr
        return float(np.mean(pred_labels != exp_arr))
    return float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))


class SyntheticSuiteHarness:
    """Deterministic multi-regime synthetic harness built directly from root ``IOSpec``s."""

    def __init__(
        self,
        root_inputs: Sequence[IOSpec],
        root_outputs: Sequence[IOSpec],
        *,
        regimes: Sequence[str] = DEFAULT_SYNTHETIC_REGIMES,
        seed: int = 42,
        base_length: int = 256,
        n_features: int = 6,
        reference_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        score_fn: Callable[[EvalCase, Mapping[str, Any]], float | CaseScore] | None = None,
        metamorphic_relations: Sequence[MetamorphicRelation] = (),
        risk_budget: RiskBudget | None = None,
        allow_nonfinite_outputs: bool = False,
        metric: str = "rmse",
    ) -> None:
        self._root_inputs = list(root_inputs)
        self._root_outputs = list(root_outputs)
        self.regimes = tuple(regimes)
        self.seed = int(seed)
        self.base_length = int(base_length)
        self.n_features = int(n_features)
        self.reference_fn = reference_fn
        self.score_fn = score_fn
        self.metamorphic_relations = list(metamorphic_relations)
        self._risk_budget = risk_budget
        self.allow_nonfinite_outputs = allow_nonfinite_outputs
        self.metric = metric
        self._cached_synthetic: list[EvalCase] | None = None

    @classmethod
    def from_cdg(
        cls,
        cdg: CDGExport,
        *,
        regimes: Sequence[str] = DEFAULT_SYNTHETIC_REGIMES,
        seed: int = 42,
        reference_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        score_fn: Callable[[EvalCase, Mapping[str, Any]], float | CaseScore] | None = None,
        metamorphic_relations: Sequence[MetamorphicRelation] = (),
        risk_budget: RiskBudget | None = None,
    ) -> "SyntheticSuiteHarness":
        """Infer unwired root inputs and sink outputs from *cdg* and build a harness."""
        r_in, r_out = infer_cdg_root_ports(cdg)
        return cls(
            r_in,
            r_out,
            regimes=regimes,
            seed=seed,
            reference_fn=reference_fn,
            score_fn=score_fn,
            metamorphic_relations=metamorphic_relations,
            risk_budget=risk_budget,
        )

    @property
    def root_inputs(self) -> Sequence[IOSpec]:
        return self._root_inputs

    @property
    def root_outputs(self) -> Sequence[IOSpec]:
        return self._root_outputs

    @property
    def risk_budget(self) -> RiskBudget | None:
        return self._risk_budget

    def synthetic_cases(self) -> Sequence[EvalCase]:
        if self._cached_synthetic is not None:
            return self._cached_synthetic
        cases: list[EvalCase] = []
        for idx, reg in enumerate(self.regimes):
            rng = np.random.default_rng(self.seed + idx * 1009)
            inp_map: dict[str, Any] = {}
            for p_spec in self._root_inputs:
                inp_map[p_spec.name] = generate_synthetic_value(
                    p_spec,
                    regime=reg,
                    rng=rng,
                    base_length=self.base_length,
                    n_features=self.n_features,
                )
            expected = self.reference_fn(inp_map) if self.reference_fn is not None else None
            cases.append(
                EvalCase(
                    case_id=f"synth_{idx}_{reg}",
                    inputs=inp_map,
                    expected=expected,
                    regime=reg,
                    tier=1,
                )
            )
        self._cached_synthetic = cases
        return cases

    def validation_cases(self) -> Sequence[EvalCase]:
        # When used standalone without a separate dataset, Tier 2 evaluates the synthetic cases.
        return self.synthetic_cases()

    def holdout_cases(self) -> Sequence[EvalCase]:
        rng_offset = self.seed + 99_991
        holdout: list[EvalCase] = []
        for idx, reg in enumerate(("nominal", "noisy_outliers")):
            rng = np.random.default_rng(rng_offset + idx * 1009)
            inp_map = {
                p_spec.name: generate_synthetic_value(
                    p_spec,
                    regime=reg,
                    rng=rng,
                    base_length=self.base_length,
                    n_features=self.n_features,
                )
                for p_spec in self._root_inputs
            }
            expected = self.reference_fn(inp_map) if self.reference_fn is not None else None
            holdout.append(
                EvalCase(
                    case_id=f"holdout_{idx}_{reg}",
                    inputs=inp_map,
                    expected=expected,
                    regime=reg,
                    tier=3,
                )
            )
        return holdout

    def score(self, case: EvalCase, outputs: Mapping[str, Any]) -> CaseScore:
        violations = check_output_invariants(
            outputs,
            self._root_outputs,
            allow_nonfinite=self.allow_nonfinite_outputs,
        )
        if violations:
            return CaseScore(
                case_id=case.case_id,
                loss=float("inf"),
                passed=False,
                invariant_violations=violations,
                error="; ".join(violations),
            )

        if self.score_fn is not None:
            custom = self.score_fn(case, outputs)
            if isinstance(custom, CaseScore):
                return custom
            f_loss = float(custom)
            ok = math.isfinite(f_loss)
            return CaseScore(
                case_id=case.case_id,
                loss=f_loss if ok else float("inf"),
                passed=ok,
                metrics={"custom_loss": f_loss} if ok else {},
            )

        if case.expected is not None:
            pred = (
                outputs[self._root_outputs[0].name]
                if len(self._root_outputs) == 1 and self._root_outputs[0].name in outputs
                else _pick_output_value(outputs)
            )
            try:
                loss_val = compute_objective_metric(self.metric, pred, case.expected)
            except Exception as exc:
                return CaseScore(
                    case_id=case.case_id,
                    loss=float("inf"),
                    passed=False,
                    error=f"Scoring error: {exc}",
                )
            ok = math.isfinite(loss_val)
            return CaseScore(
                case_id=case.case_id,
                loss=loss_val if ok else float("inf"),
                passed=ok,
                metrics={self.metric: loss_val} if ok else {},
            )

        return CaseScore(case_id=case.case_id, loss=0.0, passed=True, metrics={"contract_ok": 1.0})


class ResidentSliceHarness:
    """In-RAM evaluation harness with Tier 1 synthetic checks, Tier 2 validation, and Tier 3 holdout."""

    def __init__(
        self,
        cases: Sequence[EvalCase],
        *,
        root_inputs: Sequence[IOSpec] = (),
        root_outputs: Sequence[IOSpec] = (),
        holdout_cases: Sequence[EvalCase] | None = None,
        holdout_fraction: float = 0.25,
        metric: str = "rmse",
        output_port: str | None = None,
        score_fn: Callable[[EvalCase, Mapping[str, Any]], float | CaseScore] | None = None,
        synthetic_harness: SyntheticSuiteHarness | None = None,
        metamorphic_relations: Sequence[MetamorphicRelation] = (),
        risk_budget: RiskBudget | None = None,
        allow_nonfinite_outputs: bool = False,
        seed: int = 42,
    ) -> None:
        self._root_inputs = list(root_inputs)
        self._root_outputs = list(root_outputs)
        self.metric = metric
        self.output_port = output_port
        self.score_fn = score_fn
        self.metamorphic_relations = list(metamorphic_relations)
        self._risk_budget = risk_budget
        self.allow_nonfinite_outputs = allow_nonfinite_outputs

        all_cases = list(cases)
        if holdout_cases is not None:
            self._validation_cases = all_cases
            self._holdout_cases = list(holdout_cases)
        elif len(all_cases) >= 4 and 0.0 < holdout_fraction < 1.0:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(all_cases))
            n_hold = max(1, int(round(len(all_cases) * holdout_fraction)))
            hold_set = {int(i) for i in indices[:n_hold]}
            self._validation_cases = [c for idx, c in enumerate(all_cases) if idx not in hold_set]
            self._holdout_cases = [c for idx, c in enumerate(all_cases) if idx in hold_set]
        else:
            self._validation_cases = all_cases
            self._holdout_cases = list(all_cases)

        if synthetic_harness is not None:
            self._synthetic_harness: SyntheticSuiteHarness | None = synthetic_harness
        elif self._root_inputs:
            self._synthetic_harness = SyntheticSuiteHarness(
                self._root_inputs,
                self._root_outputs,
                seed=seed,
                metamorphic_relations=metamorphic_relations,
                risk_budget=risk_budget,
                allow_nonfinite_outputs=allow_nonfinite_outputs,
            )
        else:
            self._synthetic_harness = None

    @classmethod
    def from_dataset_file(
        cls,
        dataset_path: str | Path,
        *,
        root_inputs: Sequence[IOSpec] = (),
        root_outputs: Sequence[IOSpec] = (),
        target_key: str = "expected",
        metric: str = "rmse",
        output_port: str | None = None,
        score_fn: Callable[[EvalCase, Mapping[str, Any]], float | CaseScore] | None = None,
        holdout_fraction: float = 0.25,
        risk_budget: RiskBudget | None = None,
        seed: int = 42,
    ) -> "ResidentSliceHarness":
        """Load a local dataset slice (``.json``, ``.npz``, ``.npy``, ``.csv``, ``.parquet``) ONCE into RAM."""
        path = Path(dataset_path)
        suffix = path.suffix.lower()
        loaded_cases: list[EvalCase] = []

        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("cases", [payload])
            for idx, item in enumerate(records):
                if not isinstance(item, Mapping):
                    continue
                if "inputs" in item and isinstance(item["inputs"], Mapping):
                    inp = {
                        str(k): (np.asarray(v) if isinstance(v, list) else v)
                        for k, v in item["inputs"].items()
                    }
                    exp = item.get(target_key, item.get("expected"))
                else:
                    inp = {
                        str(k): (np.asarray(v) if isinstance(v, list) else v)
                        for k, v in item.items()
                        if k != target_key
                    }
                    exp = item.get(target_key)
                if isinstance(exp, list):
                    exp = np.asarray(exp)
                loaded_cases.append(EvalCase(case_id=f"slice_{idx}", inputs=inp, expected=exp, tier=2))
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.files)
                exp_val = archive[target_key] if target_key in archive else (archive["y"] if "y" in archive else None)
                inp_dict = {
                    k: archive[k]
                    for k in keys
                    if k not in {target_key, "y"}
                }
                loaded_cases.append(EvalCase(case_id="npz_0", inputs=inp_dict, expected=exp_val, tier=2))
        elif suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            port_name = root_inputs[0].name if root_inputs else "x"
            loaded_cases.append(EvalCase(case_id="npy_0", inputs={port_name: arr}, tier=2))
        elif suffix in {".csv", ".parquet"}:
            import pandas as pd

            df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
            exp_series = df[target_key].to_numpy() if target_key in df.columns else None
            feat_df = df.drop(columns=[target_key]) if target_key in df.columns else df
            port_name = root_inputs[0].name if root_inputs else "features"
            loaded_cases.append(
                EvalCase(case_id="table_0", inputs={port_name: feat_df}, expected=exp_series, tier=2)
            )
        else:
            raise ValueError(f"Unsupported resident dataset format: {path.suffix}")

        return cls(
            loaded_cases,
            root_inputs=root_inputs,
            root_outputs=root_outputs,
            metric=metric,
            output_port=output_port,
            score_fn=score_fn,
            holdout_fraction=holdout_fraction,
            risk_budget=risk_budget,
            seed=seed,
        )

    @property
    def root_inputs(self) -> Sequence[IOSpec]:
        return self._root_inputs

    @property
    def root_outputs(self) -> Sequence[IOSpec]:
        return self._root_outputs

    @property
    def risk_budget(self) -> RiskBudget | None:
        return self._risk_budget

    def synthetic_cases(self) -> Sequence[EvalCase]:
        if self._synthetic_harness is not None:
            return self._synthetic_harness.synthetic_cases()
        return ()

    def validation_cases(self) -> Sequence[EvalCase]:
        return self._validation_cases

    def holdout_cases(self) -> Sequence[EvalCase]:
        return self._holdout_cases

    def score(self, case: EvalCase, outputs: Mapping[str, Any]) -> CaseScore:
        violations = check_output_invariants(
            outputs,
            self._root_outputs,
            allow_nonfinite=self.allow_nonfinite_outputs,
        )
        if violations:
            return CaseScore(
                case_id=case.case_id,
                loss=float("inf"),
                passed=False,
                invariant_violations=violations,
                error="; ".join(violations),
            )

        if self.score_fn is not None:
            custom = self.score_fn(case, outputs)
            if isinstance(custom, CaseScore):
                return custom
            f_loss = float(custom)
            ok = math.isfinite(f_loss)
            return CaseScore(
                case_id=case.case_id,
                loss=f_loss if ok else float("inf"),
                passed=ok,
                metrics={self.metric: f_loss} if ok else {},
            )

        if case.expected is not None:
            pred_port = self.output_port or (
                self._root_outputs[0].name if len(self._root_outputs) == 1 else None
            )
            pred = _pick_output_value(outputs, pred_port)
            try:
                loss_val = compute_objective_metric(self.metric, pred, case.expected)
            except Exception as exc:
                return CaseScore(
                    case_id=case.case_id,
                    loss=float("inf"),
                    passed=False,
                    error=f"Scoring error: {exc}",
                )
            ok = math.isfinite(loss_val)
            return CaseScore(
                case_id=case.case_id,
                loss=loss_val if ok else float("inf"),
                passed=ok,
                metrics={self.metric: loss_val} if ok else {},
            )

        return CaseScore(case_id=case.case_id, loss=0.0, passed=True)
