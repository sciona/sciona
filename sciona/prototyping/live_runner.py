"""Zero-subprocess in-process CDG execution engine with content-addressed caching.

Replaces the legacy ``Assembler -> Python pkg write -> OS subprocess -> JSON parse``
inner loop during prototyping, HPO sweeps, structural DPO mutations, and
counterfactual credit assignment.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    BoundaryKind,
    ConceptType,
    DependencyEdge,
    EdgeKind,
    NodeStatus,
)
from sciona.architect.stage_resolution import is_non_atom_resolved_node
from sciona.atom_identity import known_atom_package_prefixes
from sciona.ghost.registry import REGISTRY
from sciona.principal.models import BenchmarkResult, NodeTelemetry
from sciona.synthesizer.models import AssemblyUnit
from sciona.synthesizer.toposort import detect_cycle_partition, toposort_nodes
from sciona.types import MatchResult
from sciona.uncertainty.models import AtomUncertaintyProfile
import dataclasses
import typing
from sciona.uncertainty.runtime_guard import (
    ConfidenceGuardConfig,
    NodeGuardSpec,
    evaluate_runtime_guard,
    plan_cdg_confidence_guards,
)

from sciona.prototyping.cache import (
    IntermediateCache,
    compute_node_cache_key,
    estimate_value_bytes,
    hash_runtime_value,
)

logger = logging.getLogger(__name__)

_ATOMS_LAZY_IMPORTED = False
_TYPE_HINTS_CACHE: dict[int, tuple[Any, dict[str, Any]]] = {}

_COMMON_PORT_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("signal", "raw_signal", "waveform", "ecg", "ppg", "eeg", "x", "data", "samples"),
    ("sampling_rate", "fs", "sample_rate", "rate_hz", "sr"),
    ("filtered_signal", "conditioned_signal", "filtered", "clean_signal"),
    ("events", "peaks", "rpeaks", "beats", "onsets", "indices"),
    ("rate", "heart_rate", "bpm", "cadence"),
    ("features", "X", "inputs", "matrix"),
    ("targets", "y", "labels", "target"),
)


class AtomResolutionError(LookupError):
    """Raised when an atomic CDG leaf cannot be resolved to a Python callable."""

    def __init__(self, node_id: str, primitive_name: str, detail: str = "") -> None:
        self.node_id = node_id
        self.primitive_name = primitive_name
        msg = f"Cannot resolve callable for node '{node_id}' (primitive='{primitive_name}')"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


class LiveExecutionError(RuntimeError):
    """Raised when a node fails during strict live execution."""

    def __init__(
        self,
        node_id: str,
        atom_fqn: str,
        cause: BaseException,
    ) -> None:
        self.node_id = node_id
        self.atom_fqn = atom_fqn
        self.cause = cause
        super().__init__(
            f"Node '{node_id}' ({atom_fqn}) failed with {type(cause).__name__}: {cause}"
        )


def _lazy_import_installed_atoms_once() -> None:
    """Import installed atom modules once as a fallback for short-name REGISTRY lookups."""
    global _ATOMS_LAZY_IMPORTED
    if _ATOMS_LAZY_IMPORTED:
        return
    _ATOMS_LAZY_IMPORTED = True
    try:
        from sciona.synthesizer.ghost_sim import _ensure_atoms_imported

        _ensure_atoms_imported()
    except Exception:
        logger.debug("Lazy atom import fallback encountered an error", exc_info=True)


def _get_cached_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    fid = id(fn)
    cached = _TYPE_HINTS_CACHE.get(fid)
    if cached is not None and cached[0] is fn:
        return cached[1]
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    if len(_TYPE_HINTS_CACHE) > 2048:
        _TYPE_HINTS_CACHE.clear()
    _TYPE_HINTS_CACHE[fid] = (fn, hints)
    return hints


def _coerce_arg_value(val: Any, param_type: Any) -> Any:
    """Coerce list/tuple -> np.ndarray or dict -> dataclass when annotated."""
    if param_type is inspect.Parameter.empty or param_type is None:
        return val
    if (
        param_type is np.ndarray or typing.get_origin(param_type) is np.ndarray
    ) and isinstance(val, (list, tuple)):
        return np.asarray(val)
    if dataclasses.is_dataclass(param_type) and isinstance(param_type, type) and isinstance(val, dict):
        try:
            return param_type(**val)
        except Exception:
            return val
    return val


def _try_import_dotted_callable(spec: str) -> tuple[Callable[..., Any], str] | None:
    """Attempt to import ``module.symbol`` or ``module:symbol`` into a Python callable."""
    cleaned = spec.strip().lstrip("@")
    if not cleaned:
        return None

    if ":" in cleaned:
        mod_name, sym_name = cleaned.split(":", 1)
        try:
            mod = importlib.import_module(mod_name.strip())
            attr = getattr(mod, sym_name.strip(), None)
            if callable(attr):
                return attr, f"{mod_name.strip()}.{sym_name.strip()}"
        except Exception:
            return None

    if "." in cleaned:
        mod_name, sym_name = cleaned.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
            attr = getattr(mod, sym_name, None)
            if callable(attr):
                return attr, cleaned
            if inspect.ismodule(attr):
                nested = getattr(attr, sym_name, None)
                if callable(nested):
                    return nested, f"{cleaned}.{sym_name}"
        except Exception:
            pass

        # Also check <pkg>.atoms.<sym> when declaration omits `.atoms`
        parts = cleaned.split(".")
        if len(parts) >= 2 and "atoms" not in parts:
            alt_mod = ".".join(parts[:-1] + ["atoms"])
            try:
                mod = importlib.import_module(alt_mod)
                attr = getattr(mod, sym_name, None)
                if callable(attr):
                    return attr, f"{alt_mod}.{sym_name}"
            except Exception:
                pass
    else:
        for prefix in known_atom_package_prefixes():
            for candidate_mod in (f"{prefix}.atoms", prefix):
                try:
                    mod = importlib.import_module(candidate_mod)
                    attr = getattr(mod, cleaned, None)
                    if callable(attr):
                        return attr, f"{candidate_mod}.{cleaned}"
                except Exception:
                    continue

    return None


def _lookup_in_registry(name: str) -> tuple[Callable[..., Any], str] | None:
    """Look up *name* in ``sciona.ghost.registry.REGISTRY`` by exact or unique suffix."""
    cleaned = name.strip().lstrip("@")
    if not cleaned:
        return None
    entry = REGISTRY.get(cleaned)
    if entry is not None and callable(entry.get("impl")):
        return entry["impl"], cleaned

    short = cleaned.rsplit(".", 1)[-1]
    entry_short = REGISTRY.get(short)
    if entry_short is not None and callable(entry_short.get("impl")):
        return entry_short["impl"], short

    suffix = f".{short}"
    matches = [
        (reg_key, reg_val["impl"])
        for reg_key, reg_val in REGISTRY.items()
        if (reg_key == short or reg_key.endswith(suffix)) and callable(reg_val.get("impl"))
    ]
    if len(matches) == 1:
        return matches[0][1], matches[0][0]
    return None


def resolve_atom_callable(
    primitive_name: str,
    *,
    node: AlgorithmicNode | None = None,
    match_map: Mapping[str, MatchResult] | None = None,
    extra_callables: Mapping[str, Callable[..., Any]] | None = None,
) -> tuple[Callable[..., Any], str]:
    """Resolve an atomic CDG leaf to ``(callable_fn, canonical_atom_fqn)``."""
    node_id = node.node_id if node is not None else ""
    candidates: list[str] = []

    def _push(raw: str | None) -> None:
        val = str(raw or "").strip()
        if val and val not in candidates:
            candidates.append(val)

    _push(primitive_name)
    if node is not None:
        _push(node.matched_primitive)

    if match_map and node_id and node_id in match_map:
        mr = match_map[node_id]
        vm = getattr(mr, "verified_match", None)
        cand_obj = getattr(vm, "candidate", None) if vm is not None else None
        decl = getattr(cand_obj, "declaration", None) if cand_obj is not None else None
        if decl is not None:
            _push(getattr(decl, "name", ""))
            proof_term = str(getattr(decl, "proof_term", "") or "").strip()
            if proof_term.startswith("@"):
                _push(proof_term[1:])
            source_lib = str(getattr(decl, "source_lib", "") or "").strip()
            decl_name = str(getattr(decl, "name", "") or "").strip()
            if source_lib and decl_name and not decl_name.startswith(source_lib):
                _push(f"{source_lib}.{decl_name}")

    # 1. Explicit in-memory callables (highest priority for fast prototyping / tests)
    if extra_callables:
        if node_id and node_id in extra_callables:
            fn = extra_callables[node_id]
            fqn = candidates[0] if candidates else f"extra.{node_id}"
            return fn, fqn
        for cand in candidates:
            if cand in extra_callables:
                return extra_callables[cand], cand
            short = cand.rsplit(".", 1)[-1]
            if short in extra_callables:
                return extra_callables[short], cand

    # 2. Already-registered ghost REGISTRY entries (O(1))
    for cand in candidates:
        found = _lookup_in_registry(cand)
        if found is not None:
            return found

    # 3. Direct importlib resolution (PEP 420 / Python module.symbol)
    for cand in candidates:
        imported = _try_import_dotted_callable(cand)
        if imported is not None:
            return imported

    # 4. Lazy import installed atom packages once and retry REGISTRY
    _lazy_import_installed_atoms_once()
    for cand in candidates:
        found = _lookup_in_registry(cand)
        if found is not None:
            return found
        imported = _try_import_dotted_callable(cand)
        if imported is not None:
            return imported

    # 5. Boundary passthrough fallback for explicit ROOT_INPUT / ROOT_OUTPUT nodes
    if node is not None and (
        node.boundary_kind in (BoundaryKind.ROOT_INPUT, BoundaryKind.ROOT_OUTPUT)
        or (not candidates and is_non_atom_resolved_node(node))
    ):
        def _passthrough(**kwargs: Any) -> Any:
            if not kwargs:
                return None
            if len(kwargs) == 1:
                return next(iter(kwargs.values()))
            return tuple(kwargs.values())

        return _passthrough, f"sciona.boundary.{node.boundary_kind.value}"

    raise AtomResolutionError(
        node_id or "<unknown>",
        primitive_name or (candidates[0] if candidates else ""),
    )


def summarize_runtime_value(value: Any) -> dict[str, Any]:
    """Produce compact telemetry statistics for ExpansionEngine & AdmissibilityEvaluator."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        return {
            "kind": "float",
            "value": fv if math.isfinite(fv) else None,
            "is_finite": math.isfinite(fv),
            "is_nan": math.isnan(fv),
        }
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "kind": "ndarray",
            "shape": list(value.shape),
            "ndim": int(value.ndim),
            "size": int(value.size),
            "dtype": str(value.dtype),
        }
        if value.size == 0:
            summary["empty"] = True
            return summary
        if np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
            arr_f = value.astype(np.float64, copy=False)
            nan_mask = np.isnan(arr_f)
            inf_mask = np.isinf(arr_f)
            nan_count = int(nan_mask.sum())
            inf_count = int(inf_mask.sum())
            summary["nan_count"] = nan_count
            summary["inf_count"] = inf_count
            summary["is_finite"] = nan_count == 0 and inf_count == 0
            finite_vals = arr_f[~(nan_mask | inf_mask)]
            if finite_vals.size > 0:
                summary["mean"] = float(np.mean(finite_vals))
                summary["std"] = float(np.std(finite_vals))
                summary["min"] = float(np.min(finite_vals))
                summary["max"] = float(np.max(finite_vals))
        return summary

    cls_mod = type(value).__module__
    cls_name = type(value).__name__
    if cls_mod.startswith("pandas.") and hasattr(value, "shape"):
        return {
            "kind": cls_name.lower(),
            "shape": list(value.shape),
            "columns": [str(c) for c in getattr(value, "columns", [])][:32],
        }
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "len": len(value)}
    if isinstance(value, Mapping):
        return {"kind": "dict", "len": len(value), "keys": [str(k) for k in list(value.keys())[:16]]}
    return {"kind": cls_name}


def _port_aliases(name: str) -> tuple[str, ...]:
    norm = name.strip().lower()
    for grp in _COMMON_PORT_ALIAS_GROUPS:
        if norm in grp:
            return grp
    return (norm,)


def _parse_default_repr(raw_repr: str) -> tuple[bool, Any]:
    """Parse ``IOSpec.default_value_repr`` when literal-evaluatable."""
    text = str(raw_repr or "").strip()
    if not text or text == "...":
        return False, None
    try:
        return True, ast.literal_eval(text)
    except Exception:
        lower = text.lower()
        if lower == "none":
            return True, None
        if lower == "true":
            return True, True
        if lower == "false":
            return True, False
        return False, None


@dataclass
class LiveRunResult:
    """Detailed outcome of an in-process ``LiveCDGRunner`` execution."""

    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_output_digests: dict[str, dict[str, str]] = field(default_factory=dict)
    state_store: dict[str, Any] = field(default_factory=dict)
    node_telemetry: dict[str, NodeTelemetry] = field(default_factory=dict)
    cache_hits: list[str] = field(default_factory=list)
    cache_misses: list[str] = field(default_factory=list)
    confidence_guards: dict[str, dict[str, Any]] = field(default_factory=dict)
    intermediate_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_time_ms: float = 0.0
    failed_node_id: str = ""
    error: str = ""
    exception_type: str = ""

    def to_benchmark_result(self, *, global_loss: float) -> BenchmarkResult:
        """Convert this live run into a ``Principal``-compatible ``BenchmarkResult``."""
        flat_intermediates: dict[str, Any] = {}
        for nid, port_map in self.node_outputs.items():
            for pname, val in port_map.items():
                flat_intermediates[f"{nid}.{pname}"] = val
                if len(port_map) == 1:
                    flat_intermediates.setdefault(nid, val)

        return BenchmarkResult(
            global_loss=float(global_loss) if self.ok else float("inf"),
            node_telemetry=dict(self.node_telemetry),
            runtime_artifacts={
                "ok": self.ok,
                "outputs": dict(self.outputs),
                "node_outputs": dict(self.node_outputs),
                "intermediates": flat_intermediates,
                "intermediate_summaries": dict(self.intermediate_summaries),
                "state_store": dict(self.state_store),
                "confidence_guards": dict(self.confidence_guards),
                "cache_hits": list(self.cache_hits),
                "cache_misses": list(self.cache_misses),
                "total_time_ms": self.total_time_ms,
                "failed_node_id": self.failed_node_id,
                "error": self.error,
                "exception_type": self.exception_type,
            },
        )


class LiveCDGRunner:
    """Zero-subprocess CDG runner with content-addressed prefix memoization."""

    def __init__(
        self,
        *,
        cache: IntermediateCache | None = None,
        extra_callables: Mapping[str, Callable[..., Any]] | None = None,
        confidence_guard: ConfidenceGuardConfig | None = None,
        uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
        max_fixed_point_iters: int = 64,
        fixed_point_tol: float = 1e-6,
    ) -> None:
        self.cache = cache if cache is not None else IntermediateCache()
        self.extra_callables: dict[str, Callable[..., Any]] = dict(extra_callables or {})
        self.confidence_guard = confidence_guard
        self.uncertainty_profiles = uncertainty_profiles
        self.max_fixed_point_iters = max(1, int(max_fixed_point_iters))
        self.fixed_point_tol = float(fixed_point_tol)

    def _build_guard_specs(
        self,
        leaf_nodes: Sequence[AlgorithmicNode],
        resolved_fqns: Mapping[str, str],
    ) -> dict[str, NodeGuardSpec]:
        cfg = self.confidence_guard
        if cfg is None or not cfg.enabled:
            return {}
        units = [
            AssemblyUnit(
                node_id=node.node_id,
                name=node.name,
                declaration_name=resolved_fqns.get(node.node_id, node.matched_primitive or node.name),
                type_signature=node.type_signature or "",
                matched_primitive=node.matched_primitive or "",
                inputs=list(node.inputs),
                outputs=list(node.outputs),
            )
            for node in leaf_nodes
        ]
        return plan_cdg_confidence_guards(
            units,
            profiles=self.uncertainty_profiles,
            trust_tier=cfg.trust_tier,
            global_alpha=cfg.global_alpha,
            require_conformal=cfg.require_conformal,
            abort_on_violation=cfg.abort_on_violation,
        )

    @staticmethod
    def _resolve_root_input(
        node: AlgorithmicNode,
        port_spec: Any,
        case_inputs: Mapping[str, Any],
        *,
        total_unwired_ports: int,
    ) -> tuple[bool, Any]:
        """Resolve an unwired root input port from *case_inputs*."""
        port_name = str(port_spec.name)
        qualified = f"{node.node_id}.{port_name}"
        if qualified in case_inputs:
            return True, case_inputs[qualified]
        if (node.node_id, port_name) in case_inputs:  # type: ignore[comparison-overlap]
            return True, case_inputs[(node.node_id, port_name)]  # type: ignore[index]
        if port_name in case_inputs:
            return True, case_inputs[port_name]

        lower_inputs = {str(k).lower(): v for k, v in case_inputs.items()}
        plow = port_name.lower()
        if plow in lower_inputs:
            return True, lower_inputs[plow]

        for alias in _port_aliases(port_name):
            if alias in case_inputs:
                return True, case_inputs[alias]
            if alias in lower_inputs:
                return True, lower_inputs[alias]

        if bool(getattr(port_spec, "required", True)) and total_unwired_ports == 1 and len(case_inputs) == 1:
            return True, next(iter(case_inputs.values()))

        return False, None

    @staticmethod
    def _bind_call_arguments(
        fn: Callable[..., Any],
        node: AlgorithmicNode,
        raw_port_values: Mapping[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Bind CDG port names + parameter overrides onto *fn*'s Python signature."""
        coerced_map = dict(raw_port_values)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return [], coerced_map

        type_hints = _get_cached_type_hints(fn)
        params = list(sig.parameters.values())
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        lower_pool = {str(k).lower(): (str(k), v) for k, v in coerced_map.items()}
        ordered_node_ports = [p.name for p in node.inputs if p.name in coerced_map]
        for k in coerced_map:
            if k not in ordered_node_ports:
                ordered_node_ports.append(str(k))

        used_keys: set[str] = set()
        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}

        def _claim(param_name: str) -> tuple[bool, Any]:
            if param_name in coerced_map and param_name not in used_keys:
                used_keys.add(param_name)
                return True, coerced_map[param_name]
            low = param_name.lower()
            if low in lower_pool and lower_pool[low][0] not in used_keys:
                orig_k, val = lower_pool[low]
                used_keys.add(orig_k)
                return True, val
            for alias in _port_aliases(param_name):
                if alias in coerced_map and alias not in used_keys:
                    used_keys.add(alias)
                    return True, coerced_map[alias]
                if alias in lower_pool and lower_pool[alias][0] not in used_keys:
                    orig_k, val = lower_pool[alias]
                    used_keys.add(orig_k)
                    return True, val
            return False, None

        for param in params:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            found, val = _claim(param.name)
            if not found and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                # Positional fallback along declared node.inputs order when CDG port
                # names differ from underlying library parameter names.
                for cand_key in ordered_node_ports:
                    if cand_key not in used_keys and cand_key not in sig.parameters:
                        used_keys.add(cand_key)
                        found = True
                        val = coerced_map[cand_key]
                        break
            if found:
                ptype = type_hints.get(param.name, param.annotation)
                val = _coerce_arg_value(val, ptype)
                if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    pos_args.append(val)
                else:
                    kw_args[param.name] = val

        if accepts_varkw:
            for k, v in coerced_map.items():
                if k not in used_keys and k not in kw_args:
                    kw_args[str(k)] = v

        return pos_args, kw_args

    @staticmethod
    def _unpack_node_outputs(
        node: AlgorithmicNode,
        raw_result: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Unpack *raw_result* into ``(outputs_by_port, state_updates_by_port)``."""
        out_specs = list(node.outputs)
        produced_state_ports = [
            sp.port_name
            for sp in node.state_ports
            if sp.port_name and sp.direction in ("produce", "update")
        ]

        outputs: dict[str, Any] = {}
        state_updates: dict[str, Any] = {}

        if not out_specs:
            if isinstance(raw_result, Mapping):
                for sp_name in produced_state_ports:
                    if sp_name in raw_result:
                        state_updates[sp_name] = raw_result[sp_name]
            elif len(produced_state_ports) == 1 and raw_result is not None:
                state_updates[produced_state_ports[0]] = raw_result
            return outputs, state_updates

        if len(out_specs) == 1:
            only_name = out_specs[0].name
            only_kind = out_specs[0].resolved_port_type().kind
            if (
                isinstance(raw_result, Mapping)
                and only_name in raw_result
                and only_kind not in {"dict", "object", "any"}
            ):
                outputs[only_name] = raw_result[only_name]
            elif (
                produced_state_ports
                and isinstance(raw_result, (tuple, list))
                and len(raw_result) == 1 + len(produced_state_ports)
                and only_kind != "tuple"
            ):
                outputs[only_name] = raw_result[0]
                for s_idx, sp_name in enumerate(produced_state_ports):
                    state_updates[sp_name] = raw_result[1 + s_idx]
            else:
                outputs[only_name] = raw_result
        else:
            if isinstance(raw_result, Mapping) and all(o.name in raw_result for o in out_specs):
                for o in out_specs:
                    outputs[o.name] = raw_result[o.name]
            elif isinstance(raw_result, (tuple, list)):
                for idx, o in enumerate(out_specs):
                    if idx < len(raw_result):
                        outputs[o.name] = raw_result[idx]
                # Extra tuple elements beyond declared data outputs map to produced state_ports
                extra_offset = len(out_specs)
                for s_idx, sp_name in enumerate(produced_state_ports):
                    if extra_offset + s_idx < len(raw_result):
                        state_updates[sp_name] = raw_result[extra_offset + s_idx]
            else:
                outputs[out_specs[0].name] = raw_result

        for sp_name in produced_state_ports:
            if sp_name in outputs:
                state_updates[sp_name] = outputs[sp_name]
            elif isinstance(raw_result, Mapping) and sp_name in raw_result:
                state_updates[sp_name] = raw_result[sp_name]

        return outputs, state_updates

    @staticmethod
    def _collect_sink_outputs(
        leaf_nodes: Sequence[AlgorithmicNode],
        data_edges: Sequence[DependencyEdge],
        node_outputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Collect CDG sink outputs (ports with no outgoing DATA_FLOW edges)."""
        consumed_ports: set[tuple[str, str]] = {
            (e.source_id, e.output_name) for e in data_edges
        }
        outgoing_nodes: set[str] = {e.source_id for e in data_edges}

        sink_outputs: dict[str, Any] = {}
        for node in leaf_nodes:
            nout = node_outputs.get(node.node_id, {})
            is_sink_node = node.node_id not in outgoing_nodes or node.boundary_kind == BoundaryKind.ROOT_OUTPUT
            for ospec in node.outputs:
                if (node.node_id, ospec.name) not in consumed_ports or is_sink_node:
                    if ospec.name in nout:
                        if ospec.name not in sink_outputs:
                            sink_outputs[ospec.name] = nout[ospec.name]
                        sink_outputs[f"{node.node_id}.{ospec.name}"] = nout[ospec.name]
            if not node.outputs and is_sink_node and nout:
                for k, v in nout.items():
                    sink_outputs.setdefault(k, v)
        return sink_outputs

    def execute(
        self,
        cdg: CDGExport,
        inputs: Mapping[str, Any],
        *,
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        initial_state: Mapping[str, Any] | None = None,
        raise_on_error: bool = False,
        use_cache: bool = True,
    ) -> LiveRunResult:
        """Execute *cdg* in-process on *inputs* with content-addressed memoization."""
        return self._execute_impl(
            cdg=cdg,
            inputs=inputs,
            node_params=node_params,
            match_results=match_results,
            initial_state=initial_state,
            baseline_node_outputs=None,
            override_node_outputs=None,
            raise_on_error=raise_on_error,
            use_cache=use_cache,
        )

    def execute_from_override(
        self,
        cdg: CDGExport,
        inputs: Mapping[str, Any],
        *,
        baseline_node_outputs: Mapping[str, Mapping[str, Any]],
        override_node_outputs: Mapping[str, Mapping[str, Any]],
        node_params: Mapping[str, Mapping[str, Any]] | None = None,
        match_results: Sequence[MatchResult] | None = None,
        initial_state: Mapping[str, Any] | None = None,
        raise_on_error: bool = False,
        use_cache: bool = True,
    ) -> LiveRunResult:
        """Counterfactual execution: seed *override_node_outputs* and run ONLY their downstream cone.

        Upstream and parallel branches outside the strict downstream reachable set of
        ``override_node_outputs.keys()`` are copied directly from *baseline_node_outputs*
        in ``O(1)`` without re-executing their Python callables.
        """
        return self._execute_impl(
            cdg=cdg,
            inputs=inputs,
            node_params=node_params,
            match_results=match_results,
            initial_state=initial_state,
            baseline_node_outputs=baseline_node_outputs,
            override_node_outputs=override_node_outputs,
            raise_on_error=raise_on_error,
            use_cache=use_cache,
        )

    def _execute_impl(
        self,
        *,
        cdg: CDGExport,
        inputs: Mapping[str, Any],
        node_params: Mapping[str, Mapping[str, Any]] | None,
        match_results: Sequence[MatchResult] | None,
        initial_state: Mapping[str, Any] | None,
        baseline_node_outputs: Mapping[str, Mapping[str, Any]] | None,
        override_node_outputs: Mapping[str, Mapping[str, Any]] | None,
        raise_on_error: bool,
        use_cache: bool,
    ) -> LiveRunResult:
        t_run_start = time.perf_counter()
        leaf_nodes = cdg.leaf_nodes()
        if not leaf_nodes:
            # Fallback if flat atomic nodes lack explicit leaf hierarchy
            leaf_nodes = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC and not n.children]

        leaf_ids = {n.node_id for n in leaf_nodes}
        node_map = {n.node_id: n for n in leaf_nodes}

        leaf_edges = [
            e for e in cdg.edges if e.source_id in leaf_ids and e.target_id in leaf_ids
        ]
        data_edges = [e for e in leaf_edges if e.edge_kind != EdgeKind.CALLABLE_INJECTION]
        callable_edges = [e for e in leaf_edges if e.edge_kind == EdgeKind.CALLABLE_INJECTION]

        incoming_data: dict[str, list[DependencyEdge]] = {nid: [] for nid in leaf_ids}
        incoming_callable: dict[str, list[DependencyEdge]] = {nid: [] for nid in leaf_ids}
        outgoing_adj: dict[str, list[str]] = {nid: [] for nid in leaf_ids}
        outgoing_data_sources: set[str] = set()
        outgoing_callable_sources: set[str] = set()
        wired_target_ports: set[tuple[str, str]] = set()
        for e in data_edges:
            incoming_data[e.target_id].append(e)
            outgoing_adj[e.source_id].append(e.target_id)
            outgoing_data_sources.add(e.source_id)
            wired_target_ports.add((e.target_id, e.input_name))
        for e in callable_edges:
            incoming_callable[e.target_id].append(e)
            outgoing_adj[e.source_id].append(e.target_id)
            outgoing_callable_sources.add(e.source_id)
            wired_target_ports.add((e.target_id, e.input_name))

        # Nodes whose ONLY outgoing edges are CALLABLE_INJECTION provide callables
        # and should not be eagerly invoked on root inputs as data-flow stages.
        callable_only_providers: set[str] = outgoing_callable_sources - outgoing_data_sources

        total_unwired_ports = sum(
            1
            for n in leaf_nodes
            if n.node_id not in callable_only_providers
            for p in n.inputs
            if p.required and (n.node_id, p.name) not in wired_target_ports
        )

        match_map: dict[str, MatchResult] = {}
        if match_results:
            for mr in match_results:
                pid = getattr(getattr(mr, "pdg_node", None), "predicate_id", "")
                if pid:
                    match_map[str(pid)] = mr

        # Resolve all leaf callables upfront so missing groundings fail fast before execution
        resolved_fns: dict[str, Callable[..., Any]] = {}
        resolved_fqns: dict[str, str] = {}
        override_keys = set(override_node_outputs.keys()) if override_node_outputs else set()
        downstream_cone: set[str] | None = None
        if override_node_outputs is not None:
            downstream_cone = set()
            q: deque[str] = deque(nid for nid in override_keys if nid in leaf_ids)
            while q:
                cur = q.popleft()
                for nxt in outgoing_adj.get(cur, ()):
                    if nxt not in override_keys and nxt not in downstream_cone:
                        downstream_cone.add(nxt)
                        q.append(nxt)

        for node in leaf_nodes:
            nid = node.node_id
            if downstream_cone is not None and nid not in downstream_cone:
                # Seeded from baseline_node_outputs or override_node_outputs; no callable needed!
                resolved_fqns[nid] = node.matched_primitive or node.name
                continue
            try:
                fn, fqn = resolve_atom_callable(
                    node.matched_primitive or "",
                    node=node,
                    match_map=match_map,
                    extra_callables=self.extra_callables,
                )
                resolved_fns[nid] = fn
                resolved_fqns[nid] = fqn
            except Exception as exc:
                if raise_on_error:
                    raise
                return LiveRunResult(
                    ok=False,
                    total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                    failed_node_id=nid,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )

        guard_specs = self._build_guard_specs(leaf_nodes, resolved_fqns)

        # Determine execution schedule via topological sort or Tarjan SCC condensation
        # (supporting valid FIXED_POINT / MESSAGE_PASSING cycles with downstream nodes).
        stages: list[tuple[bool, list[str]]] = []
        try:
            exec_order = toposort_nodes(leaf_nodes, leaf_edges)
            stages = [(False, [nid]) for nid in exec_order]
        except ValueError as exc:
            sccs = _tarjan_scc_topological(leaf_nodes, leaf_edges)
            valid_cycle_types = {ConceptType.FIXED_POINT, ConceptType.MESSAGE_PASSING}
            self_loops = {e.source_id for e in leaf_edges if e.source_id == e.target_id}
            invalid_cycle_nodes: list[str] = []
            for comp in sccs:
                is_cyc = len(comp) > 1 or (len(comp) == 1 and comp[0] in self_loops)
                if is_cyc:
                    for cid in comp:
                        if node_map[cid].concept_type not in valid_cycle_types:
                            invalid_cycle_nodes.append(cid)
                stages.append((is_cyc, comp))
            if invalid_cycle_nodes:
                if raise_on_error:
                    raise
                return LiveRunResult(
                    ok=False,
                    total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                    failed_node_id=invalid_cycle_nodes[0],
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )

        node_outputs: dict[str, dict[str, Any]] = {}
        node_output_digests: dict[str, dict[str, str]] = {}
        state_store: dict[str, Any] = dict(initial_state or {})
        state_digests: dict[str, str] = {
            k: hash_runtime_value(v) for k, v in state_store.items()
        }
        root_input_digest_memo: dict[int, tuple[Any, str]] = {}
        node_telemetry: dict[str, NodeTelemetry] = {}
        cache_hits: list[str] = []
        cache_misses: list[str] = []
        confidence_guards: dict[str, dict[str, Any]] = {}
        intermediate_summaries: dict[str, dict[str, Any]] = {}

        def _digest_of_root(val: Any) -> str:
            vid = id(val)
            memo = root_input_digest_memo.get(vid)
            if memo is not None and memo[0] is val:
                return memo[1]
            d = hash_runtime_value(val)
            root_input_digest_memo[vid] = (val, d)
            return d

        # Pre-seed baseline + override outputs when running counterfactual attribution
        if downstream_cone is not None:
            if baseline_node_outputs:
                for nid, pmap in baseline_node_outputs.items():
                    if nid in leaf_ids and nid not in downstream_cone and nid not in override_keys:
                        node_outputs[nid] = dict(pmap)
                        node_output_digests[nid] = {
                            pk: hash_runtime_value(pv) for pk, pv in pmap.items()
                        }
            if override_node_outputs:
                for nid, pmap in override_node_outputs.items():
                    if nid in leaf_ids:
                        node_outputs[nid] = dict(pmap)
                        node_output_digests[nid] = {
                            pk: hash_runtime_value(pv) for pk, pv in pmap.items()
                        }

        def _run_single_node(
            nid: str,
            *,
            allow_cache: bool,
            active_cycle_set: set[str] | None = None,
        ) -> LiveRunResult | None:
            if nid in callable_only_providers:
                return None
            node = node_map[nid]
            fn = resolved_fns[nid]
            fqn = resolved_fqns[nid]

            raw_port_values: dict[str, Any] = {}
            port_digests: dict[str, str] = {}

            # 1. Upstream DATA_FLOW edges (ordering non-cycle seed edges before back-edges
            # so fixed-point iteration 0 uses the seed value while iterations >= 1 use the
            # updated cycle state!)
            ordered_incoming = incoming_data[nid]
            if active_cycle_set:
                ordered_incoming = sorted(
                    ordered_incoming,
                    key=lambda ed: 1 if ed.source_id in active_cycle_set else 0,
                )
            for edge in ordered_incoming:
                src_map = node_outputs.get(edge.source_id, {})
                src_digests = node_output_digests.get(edge.source_id, {})
                if active_cycle_set and edge.source_id in active_cycle_set and not src_map:
                    # Uninitialized cycle back-edge on iteration 0: keep seed value if present
                    continue
                if edge.output_name in src_map:
                    val = src_map[edge.output_name]
                    dig = src_digests.get(edge.output_name) or hash_runtime_value(val)
                elif len(src_map) == 1:
                    only_k, val = next(iter(src_map.items()))
                    dig = src_digests.get(only_k) or hash_runtime_value(val)
                else:
                    err = KeyError(
                        f"Missing upstream output '{edge.output_name}' from node '{edge.source_id}' "
                        f"for target '{nid}.{edge.input_name}' (available: {list(src_map.keys())})"
                    )
                    if raise_on_error:
                        raise LiveExecutionError(nid, fqn, err) from err
                    return LiveRunResult(
                        ok=False,
                        node_outputs=node_outputs,
                        state_store=state_store,
                        node_telemetry=node_telemetry,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        confidence_guards=confidence_guards,
                        intermediate_summaries=intermediate_summaries,
                        total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                        failed_node_id=nid,
                        error=str(err),
                        exception_type="KeyError",
                    )
                raw_port_values[edge.input_name] = val
                port_digests[edge.input_name] = dig

            # 2. CALLABLE_INJECTION edges
            for edge in incoming_callable[nid]:
                injected_fn = resolved_fns.get(edge.source_id)
                if injected_fn is None:
                    injected_fn, _ = resolve_atom_callable(
                        node_map[edge.source_id].matched_primitive or "",
                        node=node_map[edge.source_id],
                        match_map=match_map,
                        extra_callables=self.extra_callables,
                    )
                raw_port_values[edge.input_name] = injected_fn
                port_digests[edge.input_name] = hash_runtime_value(injected_fn)

            # 3. Root boundary inputs + IOSpec literal defaults
            for p_spec in node.inputs:
                if p_spec.name in raw_port_values:
                    continue
                found, rval = self._resolve_root_input(
                    node,
                    p_spec,
                    inputs,
                    total_unwired_ports=total_unwired_ports,
                )
                if found:
                    raw_port_values[p_spec.name] = rval
                    port_digests[p_spec.name] = _digest_of_root(rval)
                elif p_spec.default_value_repr:
                    parsed_ok, def_val = _parse_default_repr(p_spec.default_value_repr)
                    if parsed_ok:
                        raw_port_values[p_spec.name] = def_val
                        port_digests[p_spec.name] = _digest_of_root(def_val)

            # 4. Consumed state_ports
            consumed_state_digests: dict[str, str] = {}
            for sp in node.state_ports:
                if sp.port_name and sp.direction in ("consume", "update"):
                    if sp.port_name in state_store:
                        sval = state_store[sp.port_name]
                        sdig = state_digests.get(sp.port_name) or hash_runtime_value(sval)
                        consumed_state_digests[sp.port_name] = sdig
                        raw_port_values.setdefault(sp.port_name, sval)
                    elif sp.required and sp.port_name not in raw_port_values:
                        err_state = KeyError(
                            f"Required state_port '{sp.port_name}' missing for node '{nid}'"
                        )
                        if raise_on_error:
                            raise LiveExecutionError(nid, fqn, err_state) from err_state
                        return LiveRunResult(
                            ok=False,
                            node_outputs=node_outputs,
                            state_store=state_store,
                            node_telemetry=node_telemetry,
                            cache_hits=cache_hits,
                            cache_misses=cache_misses,
                            confidence_guards=confidence_guards,
                            intermediate_summaries=intermediate_summaries,
                            total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                            failed_node_id=nid,
                            error=str(err_state),
                            exception_type="KeyError",
                        )

            # 5. Hyperparameter overrides (node_params[node_id] or node_params[fqn])
            override_params: dict[str, Any] = {}
            if node_params:
                if nid in node_params:
                    override_params.update(node_params[nid])
                elif fqn in node_params:
                    override_params.update(node_params[fqn])
                elif node.matched_primitive and node.matched_primitive in node_params:
                    override_params.update(node_params[node.matched_primitive])
            for pk, pv in override_params.items():
                raw_port_values[pk] = pv

            # 6. Runtime confidence guard check (Phase 4) BEFORE cache or execution
            if nid in guard_specs:
                try:
                    guard_summary = evaluate_runtime_guard(
                        guard_specs[nid].to_dict(),
                        raw_port_values,
                        callable_fn=fn,
                        ordered_names=[p.name for p in node.inputs],
                    )
                    confidence_guards[nid] = guard_summary
                except Exception as exc:
                    g_summary = getattr(exc, "guard_summary", None)
                    if isinstance(g_summary, dict):
                        confidence_guards[nid] = g_summary
                    if raise_on_error:
                        raise LiveExecutionError(nid, fqn, exc) from exc
                    return LiveRunResult(
                        ok=False,
                        node_outputs=node_outputs,
                        state_store=state_store,
                        node_telemetry=node_telemetry,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        confidence_guards=confidence_guards,
                        intermediate_summaries=intermediate_summaries,
                        total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                        failed_node_id=nid,
                        error=str(exc),
                        exception_type=type(exc).__name__,
                    )

            # 7. Content-addressed prefix cache lookup
            cache_key = ""
            if allow_cache and use_cache and self.cache.max_entries > 0:
                cache_key = compute_node_cache_key(
                    atom_fqn=fqn,
                    impl_callable=fn,
                    output_port_names=tuple(o.name for o in node.outputs),
                    node_params=override_params,
                    input_digests=port_digests,
                    state_digests=consumed_state_digests,
                )
                cached = self.cache.get(cache_key)
                if cached is not None:
                    cache_hits.append(nid)
                    outs = dict(cached.outputs)
                    node_outputs[nid] = outs
                    node_output_digests[nid] = dict(cached.output_digests)
                    if cached.state_updates:
                        state_store.update(cached.state_updates)
                        state_digests.update(cached.state_update_digests)
                    mem_bytes = estimate_value_bytes(outs)
                    node_telemetry[nid] = NodeTelemetry(
                        node_id=nid,
                        execution_time_ms=0.0,
                        peak_memory_bytes=mem_bytes,
                        error_expansion=0.0,
                    )
                    for pname, pval in outs.items():
                        intermediate_summaries[f"{nid}.{pname}"] = summarize_runtime_value(pval)
                    return None
                cache_misses.append(nid)
            else:
                cache_misses.append(nid)

            # 8. Execute Python callable directly in-process
            t0 = time.perf_counter()
            try:
                pos_args, kw_args = self._bind_call_arguments(fn, node, raw_port_values)
                raw_result = fn(*pos_args, **kw_args)
            except Exception as exc:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                node_telemetry[nid] = NodeTelemetry(
                    node_id=nid,
                    execution_time_ms=dt_ms,
                    peak_memory_bytes=0,
                    error_expansion=float("inf"),
                )
                if raise_on_error:
                    raise LiveExecutionError(nid, fqn, exc) from exc
                return LiveRunResult(
                    ok=False,
                    node_outputs=node_outputs,
                    state_store=state_store,
                    node_telemetry=node_telemetry,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    confidence_guards=confidence_guards,
                    intermediate_summaries=intermediate_summaries,
                    total_time_ms=(time.perf_counter() - t_run_start) * 1000.0,
                    failed_node_id=nid,
                    error=f"{type(exc).__name__}: {exc}",
                    exception_type=type(exc).__name__,
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            outs, st_updates = self._unpack_node_outputs(node, raw_result)
            out_digests = {k: hash_runtime_value(v) for k, v in outs.items()}
            st_digests = {k: hash_runtime_value(v) for k, v in st_updates.items()}

            node_outputs[nid] = outs
            node_output_digests[nid] = out_digests
            if st_updates:
                state_store.update(st_updates)
                state_digests.update(st_digests)

            if cache_key:
                self.cache.put(
                    cache_key,
                    atom_fqn=fqn,
                    outputs=outs,
                    output_digests=out_digests,
                    state_updates=st_updates,
                    state_update_digests=st_digests,
                    compute_time_ms=dt_ms,
                )

            mem_bytes = estimate_value_bytes(outs)
            node_telemetry[nid] = NodeTelemetry(
                node_id=nid,
                execution_time_ms=dt_ms,
                peak_memory_bytes=mem_bytes,
                error_expansion=0.0,
            )
            for pname, pval in outs.items():
                intermediate_summaries[f"{nid}.{pname}"] = summarize_runtime_value(pval)
            return None

        # Execute condensed stages in topological order
        for is_cycle_stage, comp_nodes in stages:
            if not is_cycle_stage:
                nid = comp_nodes[0]
                if downstream_cone is not None and nid not in downstream_cone:
                    continue
                err_res = _run_single_node(nid, allow_cache=True)
                if err_res is not None:
                    return err_res
            else:
                if downstream_cone is not None and not any(c in downstream_cone for c in comp_nodes):
                    continue
                cyc_set = set(comp_nodes)
                for _it in range(self.max_fixed_point_iters):
                    prev_snapshot = {
                        cid: dict(node_outputs.get(cid, {})) for cid in comp_nodes
                    }
                    for cid in comp_nodes:
                        err_res = _run_single_node(
                            cid,
                            allow_cache=False,
                            active_cycle_set=cyc_set,
                        )
                        if err_res is not None:
                            return err_res
                    if _it > 0 and _fixed_point_converged(
                        prev_snapshot,
                        {cid: node_outputs.get(cid, {}) for cid in comp_nodes},
                        tol=self.fixed_point_tol,
                    ):
                        break

        sink_outputs = self._collect_sink_outputs(leaf_nodes, data_edges, node_outputs)
        total_ms = (time.perf_counter() - t_run_start) * 1000.0
        return LiveRunResult(
            ok=True,
            outputs=sink_outputs,
            node_outputs=node_outputs,
            node_output_digests=node_output_digests,
            state_store=state_store,
            node_telemetry=node_telemetry,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            confidence_guards=confidence_guards,
            intermediate_summaries=intermediate_summaries,
            total_time_ms=total_ms,
        )


def _tarjan_scc_topological(
    nodes: Sequence[AlgorithmicNode],
    edges: Sequence[DependencyEdge],
) -> list[list[str]]:
    """Return strongly connected components in topological order (sources first, sinks last)."""
    ordered_ids = [n.node_id for n in nodes]
    id_set = set(ordered_ids)
    adj: dict[str, list[str]] = {nid: [] for nid in ordered_ids}
    for e in edges:
        if e.source_id in id_set and e.target_id in id_set:
            adj[e.source_id].append(e.target_id)

    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs_reverse_topo: list[list[str]] = []

    def _strongconnect(v: str) -> None:
        nonlocal index_counter
        indices[v] = index_counter
        lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj[v]:
            if w not in indices:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comp.reverse()
            sccs_reverse_topo.append(comp)

    for nid in ordered_ids:
        if nid not in indices:
            _strongconnect(nid)

    # Tarjan emits sink SCCs first; reversing yields source-to-sink topological order!
    sccs_reverse_topo.reverse()
    return sccs_reverse_topo


def _fixed_point_converged(
    prev: Mapping[str, Mapping[str, Any]],
    curr: Mapping[str, Mapping[str, Any]],
    *,
    tol: float,
) -> bool:
    """Check whether fixed-point cycle outputs have converged within *tol*."""
    for cid, curr_ports in curr.items():
        if "converged" in curr_ports and isinstance(curr_ports["converged"], (bool, np.bool_)):
            if bool(curr_ports["converged"]):
                return True
        prev_ports = prev.get(cid, {})
        if not prev_ports:
            return False
        for pk, cval in curr_ports.items():
            if pk not in prev_ports:
                return False
            pval = prev_ports[pk]
            if isinstance(cval, np.ndarray) and isinstance(pval, np.ndarray):
                if cval.shape != pval.shape:
                    return False
                if np.issubdtype(cval.dtype, np.number) and np.issubdtype(pval.dtype, np.number):
                    if not bool(np.allclose(cval, pval, atol=tol, rtol=tol, equal_nan=True)):
                        return False
                elif not np.array_equal(cval, pval):
                    return False
            elif isinstance(cval, (int, float, np.number)) and isinstance(pval, (int, float, np.number)):
                if abs(float(cval) - float(pval)) > tol:
                    return False
            elif cval != pval:
                return False
    return True
