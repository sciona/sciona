"""Runtime input confidence-envelope guards & DAG alpha-spending planner.

Bridges calibrated ``AtomUncertaintyProfile`` regimes with synthesized Python
skeletons and ``ContractGenerator`` wrappers.  Supports configurable
``trust_tier`` filtering:
- ``trust_tier=1`` (default): guards Tier 1 certified atoms.
- ``trust_tier=k`` or ``trust_tier=(1, 2)``: guards atoms matching the tier(s).
- ``trust_tier=None``: guards atoms of **all** trust tiers across the CDG.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sciona.uncertainty.conformal import (
    TrustTierFilter,
    allocate_dag_alpha_budget,
    matches_trust_tier,
)
from sciona.uncertainty.models import AtomUncertaintyProfile
from sciona.uncertainty.propagation import resolve_atom_profile

if TYPE_CHECKING:
    from sciona.synthesizer.models import AssemblyUnit
    from sciona.synthesizer.uncertainty import UncertaintyBackend

_PORT_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("signal", "conditioned_signal", "filtered", "filtered_signal", "waveform"),
    ("events", "rpeaks", "peaks", "beats", "onsets"),
    ("rate", "heart_rate", "bpm"),
)

# Runtime cache keyed by (node_id, id(profile_dict)) so loops don't re-parse dicts
_PROFILE_CACHE: dict[tuple[str, int], AtomUncertaintyProfile] = {}


def clear_runtime_guard_cache() -> None:
    """Clear cached ``AtomUncertaintyProfile`` instances used by runtime guards."""
    _PROFILE_CACHE.clear()


def _serialize_trust_tier_filter(trust_tier: TrustTierFilter) -> int | list[int] | None:
    if trust_tier is None:
        return None
    if isinstance(trust_tier, bool):
        raise TypeError(f"Unsupported trust_tier filter: {trust_tier!r}")
    if isinstance(trust_tier, int):
        return int(trust_tier)
    return sorted(int(t) for t in trust_tier)


@dataclass(frozen=True)
class ConfidenceGuardConfig:
    """Configuration for runtime confidence-bound abort guards."""

    enabled: bool = True
    trust_tier: TrustTierFilter = 1
    """Target atom trust tier(s); ``None`` applies to atoms of **all** tiers."""
    global_alpha: float = 0.01
    """CDG-wide family-wise false-abort budget ``sum_v alpha_v = global_alpha``."""
    require_conformal: bool = True
    """Fail closed when a guarded atom lacks a ``ConformalEnvelope`` or sufficient N."""
    abort_on_violation: bool = True
    """Raise ``ConfidenceEnvelopeViolation`` when inputs exceed confidence bounds."""

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trust_tier": _serialize_trust_tier_filter(self.trust_tier),
            "global_alpha": self.global_alpha,
            "require_conformal": self.require_conformal,
            "abort_on_violation": self.abort_on_violation,
        }


@dataclass(frozen=True)
class NodeGuardSpec:
    """Compiled runtime confidence guard specification for one atomic CDG node."""

    node_id: str
    atom_name: str
    trust_tier: int
    allocated_alpha: float
    require_conformal: bool
    abort_on_violation: bool
    profile_dict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "atom_name": self.atom_name,
            "trust_tier": self.trust_tier,
            "allocated_alpha": self.allocated_alpha,
            "require_conformal": self.require_conformal,
            "abort_on_violation": self.abort_on_violation,
            "profile": self.profile_dict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NodeGuardSpec:
        return cls(
            node_id=str(payload["node_id"]),
            atom_name=str(payload["atom_name"]),
            trust_tier=int(payload.get("trust_tier", 3)),
            allocated_alpha=float(payload.get("allocated_alpha", 0.01)),
            require_conformal=bool(payload.get("require_conformal", True)),
            abort_on_violation=bool(payload.get("abort_on_violation", True)),
            profile_dict=dict(payload["profile"]),
        )


def slim_profile_for_guard(profile: AtomUncertaintyProfile) -> dict[str, Any]:
    """Serialize only the regime & conformal fields needed for runtime input checks."""
    return {
        "schema_version": "2.0",
        "atom": profile.atom_name,
        "mode": profile.mode,
        "confidence": profile.confidence,
        "trust_tier": profile.trust_tier,
        "n_trials": profile.n_trials,
        "epsilon": profile.epsilon,
        "input_regime_summary": profile.input_regime_summary,
        "input_regimes": {
            pname: reg.to_dict()
            for pname, reg in sorted(profile.input_regimes.items())
        },
        "output_projectors": {},
        "components": [],
    }


def resolve_unit_profile(
    declaration_name: str,
    *,
    matched_primitive: str = "",
    unit_name: str = "",
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
    default_trust_tier: int | None = None,
) -> AtomUncertaintyProfile:
    """Resolve an ``AtomUncertaintyProfile`` across qualified/short/primitive names."""
    candidates: list[str] = []
    for raw in (declaration_name, matched_primitive, unit_name):
        clean = str(raw or "").strip()
        if clean and clean not in candidates:
            candidates.append(clean)
        if "." in clean:
            short = clean.rsplit(".", 1)[-1]
            if short and short not in candidates:
                candidates.append(short)

    for cand in candidates:
        prof = resolve_atom_profile(cand, backend=backend, profiles=profiles)
        if prof.mode != "unknown":
            return prof

    fallback_name = candidates[0] if candidates else "unknown_atom"
    tier = int(default_trust_tier) if default_trust_tier is not None else 3
    return AtomUncertaintyProfile(
        atom_name=fallback_name,
        mode="unknown",
        trust_tier=tier,
    )


def plan_cdg_confidence_guards(
    units: Sequence[AssemblyUnit],
    *,
    backend: UncertaintyBackend | None = None,
    profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
    trust_tier: TrustTierFilter = 1,
    global_alpha: float = 0.01,
    require_conformal: bool = True,
    abort_on_violation: bool = True,
    risk_weights: Mapping[str, float] | None = None,
    default_trust_tier: int | None = None,
) -> dict[str, NodeGuardSpec]:
    """Select guarded atomic units and allocate DAG family-wise ``alpha`` budget.

    Args:
        units: Topologically ordered ``AssemblyUnit``s for atomic leaves.
        backend: Optional ``UncertaintyBackend`` for profile lookup.
        profiles: Optional explicit ``{atom_name: AtomUncertaintyProfile}`` map.
        trust_tier: Target trust tier(s) to guard.  Pass ``None`` to guard atoms
            of **all** trust tiers.
        global_alpha: Total CDG false-abort budget ``in (0, 1)``.
        require_conformal: When True, every unit matching ``trust_tier`` is guarded
            (uncalibrated units fail closed at runtime).  When False, only
            tier-matching units with calibrated ``input_regimes`` are guarded.
        abort_on_violation: Raise ``ConfidenceEnvelopeViolation`` on violation.
        risk_weights: Optional positive alpha-share weights keyed by ``node_id``.
        default_trust_tier: Fallback trust tier for uncalibrated atoms.
    """
    selected: list[tuple[str, AtomUncertaintyProfile]] = []
    for unit in units:
        matched_prim = str(getattr(unit, "matched_primitive", "") or "")
        prof = resolve_unit_profile(
            unit.declaration_name,
            matched_primitive=matched_prim,
            unit_name=unit.name,
            backend=backend,
            profiles=profiles,
            default_trust_tier=default_trust_tier,
        )
        if not matches_trust_tier(prof.trust_tier, trust_tier):
            continue
        if not require_conformal and not prof.input_regimes:
            continue
        selected.append((unit.node_id, prof))

    if not selected:
        return {}

    weights_dict = (
        {str(k): float(v) for k, v in risk_weights.items()}
        if risk_weights is not None
        else None
    )
    alpha_map = allocate_dag_alpha_budget(
        [nid for nid, _ in selected],
        global_alpha=global_alpha,
        risk_weights=weights_dict,
    )

    specs: dict[str, NodeGuardSpec] = {}
    for nid, prof in selected:
        specs[nid] = NodeGuardSpec(
            node_id=nid,
            atom_name=prof.atom_name,
            trust_tier=prof.trust_tier,
            allocated_alpha=alpha_map[nid],
            require_conformal=require_conformal,
            abort_on_violation=abort_on_violation,
            profile_dict=slim_profile_for_guard(prof),
        )
    return specs


def _aliases_for(port_name: str) -> tuple[str, ...]:
    norm = port_name.strip().lower()
    for grp in _PORT_ALIAS_GROUPS:
        if norm in grp:
            return grp
    return (norm,)


def _bind_runtime_inputs_for_profile(
    profile: AtomUncertaintyProfile,
    available: Mapping[str, Any],
    *,
    callable_fn: Any = None,
    ordered_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind runtime scaffold variables onto ``profile.input_regimes`` port names."""
    if not profile.input_regimes:
        return dict(available)

    # Start with direct & callable-signature bindings (mirrors _sciona_call)
    enriched: dict[str, Any] = dict(available)
    lower_map: dict[str, Any] = {str(k).lower(): v for k, v in available.items()}

    def _resolve_in_available(name: str) -> tuple[str | None, Any]:
        if name in available:
            return name, available[name]
        low = name.lower()
        if low in lower_map:
            return name, lower_map[low]
        for alias in _aliases_for(name):
            if alias in available:
                return alias, available[alias]
            if alias in lower_map:
                return alias, lower_map[alias]
        return None, None

    if callable_fn is not None and callable(callable_fn):
        try:
            sig = inspect.signature(callable_fn)
            ordered_pool: list[tuple[str, Any]] = []
            for oname in ordered_names:
                rname, rval = _resolve_in_available(oname)
                if rname is not None:
                    ordered_pool.append((rname, rval))
            if not ordered_pool:
                ordered_pool = [(str(k), v) for k, v in available.items()]

            used: set[str] = set()
            for param in sig.parameters.values():
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                rname, val = _resolve_in_available(param.name)
                if rname is not None:
                    used.add(rname)
                    enriched.setdefault(param.name, val)
                elif param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    picked: tuple[str, Any] | None = None
                    for cand_name, cand_val in ordered_pool:
                        if cand_name not in used:
                            picked = (cand_name, cand_val)
                            break
                    if picked is not None:
                        used.add(picked[0])
                        enriched.setdefault(param.name, picked[1])
                    elif param.default is not inspect.Parameter.empty:
                        enriched.setdefault(param.name, param.default)
                elif param.default is not inspect.Parameter.empty:
                    enriched.setdefault(param.name, param.default)
        except (TypeError, ValueError):
            pass

    enriched_lower = {str(k).lower(): v for k, v in enriched.items()}
    bound: dict[str, Any] = {}
    for port_name in profile.input_regimes:
        if port_name in enriched:
            bound[port_name] = enriched[port_name]
            continue
        plow = port_name.lower()
        if plow in enriched_lower:
            bound[port_name] = enriched_lower[plow]
            continue
        for alias in _aliases_for(port_name):
            if alias in enriched:
                bound[port_name] = enriched[alias]
                break
            if alias in enriched_lower:
                bound[port_name] = enriched_lower[alias]
                break

    # Unambiguous single-port positional fallback (e.g. CDG port 'x' -> atom param 'a')
    if (
        len(profile.input_regimes) == 1
        and not bound
        and len(available) == 1
    ):
        only_port = next(iter(profile.input_regimes))
        bound[only_port] = next(iter(available.values()))

    return bound


def evaluate_runtime_guard(
    spec_payload: Mapping[str, Any],
    available_inputs: Mapping[str, Any],
    *,
    callable_fn: Any = None,
    ordered_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate runtime inputs against a serialized ``NodeGuardSpec`` dict.

    Raises ``Tier1ConfidenceEnvelopeViolation`` (a ``ConfidenceEnvelopeViolation``)
    when ``abort_on_violation`` is True and any calibrated input port violates hard
    domain support or its allocated conformal nonconformity threshold.
    """
    node_id = str(spec_payload["node_id"])
    atom_name = str(spec_payload["atom_name"])
    trust_tier = int(spec_payload.get("trust_tier", 3))
    allocated_alpha = float(spec_payload.get("allocated_alpha", 0.01))
    require_conformal = bool(spec_payload.get("require_conformal", True))
    abort_on_violation = bool(spec_payload.get("abort_on_violation", True))

    prof_raw = spec_payload["profile"]
    cache_key = (node_id, id(prof_raw))
    profile = _PROFILE_CACHE.get(cache_key)
    if profile is None:
        profile = AtomUncertaintyProfile.from_dict(dict(prof_raw))
        _PROFILE_CACHE[cache_key] = profile

    bound = _bind_runtime_inputs_for_profile(
        profile,
        available_inputs,
        callable_fn=callable_fn,
        ordered_names=ordered_names,
    )
    # Pass trust_tier=None because this node was already selected for guarding
    # under the user's configured trust_tier filter at assembly/wrapper time.
    report = profile.evaluate_inputs(
        bound,
        alpha=allocated_alpha,
        require_conformal=require_conformal,
        trust_tier=None,
        require_all_calibrated_ports=True,
    )

    ports_summary: dict[str, dict[str, Any]] = {}
    for pname, pe in report.port_evaluations.items():
        p_dict: dict[str, Any] = {
            "in_domain": pe.in_domain,
            "nonconformity_score": pe.nonconformity_score,
            "bregman_divergence": pe.bregman_divergence,
        }
        if pe.envelope_check is not None:
            p_dict["passed"] = pe.envelope_check.passed
            p_dict["threshold"] = pe.envelope_check.threshold
            p_dict["p_value"] = pe.envelope_check.p_value
            p_dict["alpha"] = pe.envelope_check.alpha
        if pe.violation_reason:
            p_dict["reason"] = pe.violation_reason
        ports_summary[pname] = p_dict

    summary: dict[str, Any] = {
        "node_id": node_id,
        "atom_name": atom_name,
        "trust_tier": trust_tier,
        "allocated_alpha": allocated_alpha,
        "in_confidence_bounds": report.in_confidence_bounds,
        "active_component": report.active_component,
        "ports": ports_summary,
    }
    if report.violation_reason:
        summary["reason"] = report.violation_reason

    if abort_on_violation and not report.in_confidence_bounds:
        try:
            report.raise_if_violated(
                node_id=node_id,
                atom_name=atom_name,
                trust_tier=trust_tier,
            )
        except Exception as exc:
            setattr(exc, "guard_summary", summary)
            raise

    return summary
