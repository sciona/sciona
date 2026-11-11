"""Safe atom wrapper generator with icontract decorators and confidence guards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sciona.architect.models import IOSpec
from sciona.synthesizer.models import AssemblyUnit
from sciona.types import Declaration
from sciona.uncertainty.conformal import TrustTierFilter, matches_trust_tier
from sciona.uncertainty.runtime_guard import (
    NodeGuardSpec,
    resolve_unit_profile,
    slim_profile_for_guard,
)

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import UncertaintyBackend
    from sciona.uncertainty.models import AtomUncertaintyProfile

_UNSET_TIER: Any = object()


@dataclass
class ContractSpec:
    """A single contract (precondition or postcondition)."""

    kind: str  # "require" or "ensure"
    lambda_expr: str
    description: str = ""


@dataclass
class SafeAtomWrapper:
    """A Python function that wraps a library call with icontract decorators."""

    function_name: str
    original_qualname: str
    imports: list[str] = field(default_factory=list)
    parameters: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    return_type: str = ""
    preconditions: list[ContractSpec] = field(default_factory=list)
    postconditions: list[ContractSpec] = field(default_factory=list)
    body: str = ""
    confidence_guard_spec: dict[str, Any] | None = None


class ContractGenerator:
    """Generates icontract-decorated wrapper functions and runtime confidence guards."""

    def __init__(
        self,
        *,
        uncertainty_backend: UncertaintyBackend | None = None,
        uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
        enable_confidence_guards: bool | None = None,
        trust_tier: TrustTierFilter = 1,
        confidence_alpha: float = 0.01,
        require_conformal: bool = True,
    ) -> None:
        self._uncertainty_backend = uncertainty_backend
        self._uncertainty_profiles = (
            dict(uncertainty_profiles) if uncertainty_profiles is not None else None
        )
        self._enable_confidence_guards = enable_confidence_guards
        self._trust_tier: TrustTierFilter = trust_tier
        self._confidence_alpha = confidence_alpha
        self._require_conformal = require_conformal

    def generate_wrapper(
        self,
        unit: AssemblyUnit,
        declaration: Declaration,
        *,
        profile: AtomUncertaintyProfile | None = None,
        enable_confidence_guards: bool | None = None,
        trust_tier: TrustTierFilter | object = _UNSET_TIER,
        alpha: float | None = None,
        require_conformal: bool | None = None,
    ) -> SafeAtomWrapper:
        """Generate a SafeAtomWrapper from an AssemblyUnit and Declaration."""
        # Parse parameters from the declaration's type signature
        parameters: list[tuple[str, str]] = []
        for inp in unit.inputs:
            parameters.append((inp.name, inp.type_desc))

        return_type = ""
        if unit.outputs:
            return_type = unit.outputs[0].type_desc

        # Build contracts from IOSpec constraints
        preconditions: list[ContractSpec] = []
        for inp in unit.inputs:
            contract = self._iospec_to_contract(inp, "require")
            if contract is not None:
                preconditions.append(contract)

        postconditions: list[ContractSpec] = []
        for out in unit.outputs:
            contract = self._iospec_to_contract(out, "ensure")
            if contract is not None:
                postconditions.append(contract)

        # Infer imports from the declaration
        imports = ["import icontract"]
        module = declaration.source_lib
        if module:
            imports.append(f"import {module}")

        function_name = (
            unit.name.replace(" ", "_").replace("-", "_").lower() + "_wrapper"
        )

        # Optional runtime confidence guard emission
        eff_enable = (
            enable_confidence_guards
            if enable_confidence_guards is not None
            else (
                self._enable_confidence_guards
                if self._enable_confidence_guards is not None
                else (
                    profile is not None
                    or self._uncertainty_backend is not None
                    or self._uncertainty_profiles is not None
                )
            )
        )
        eff_tier: TrustTierFilter = (
            self._trust_tier if trust_tier is _UNSET_TIER else trust_tier  # type: ignore[assignment]
        )
        eff_alpha = float(alpha if alpha is not None else self._confidence_alpha)
        eff_req_conformal = (
            require_conformal
            if require_conformal is not None
            else self._require_conformal
        )

        guard_spec_dict: dict[str, Any] | None = None
        body_lines: list[str] = []
        if eff_enable:
            resolved_prof = (
                profile
                if profile is not None
                else resolve_unit_profile(
                    declaration.name,
                    matched_primitive=unit.matched_primitive,
                    unit_name=unit.name,
                    backend=self._uncertainty_backend,
                    profiles=self._uncertainty_profiles,
                )
            )
            if matches_trust_tier(resolved_prof.trust_tier, eff_tier) and (
                eff_req_conformal or bool(resolved_prof.input_regimes)
            ):
                guard_spec = NodeGuardSpec(
                    node_id=unit.node_id,
                    atom_name=resolved_prof.atom_name,
                    trust_tier=resolved_prof.trust_tier,
                    allocated_alpha=eff_alpha,
                    require_conformal=eff_req_conformal,
                    abort_on_violation=True,
                    profile_dict=slim_profile_for_guard(resolved_prof),
                )
                guard_spec_dict = guard_spec.to_dict()
                guard_import = (
                    "from sciona.uncertainty.runtime_guard import "
                    "evaluate_runtime_guard as _sciona_eval_guard"
                )
                if guard_import not in imports:
                    imports.append(guard_import)
                guard_const = f"_SCIONA_GUARD_{function_name.upper()}"
                avail_expr = (
                    "{"
                    + ", ".join(f"{name!r}: {name}" for name, _ in parameters)
                    + "}"
                )
                ordered_expr = repr(tuple(name for name, _ in parameters))
                body_lines.append(
                    f"_sciona_eval_guard({guard_const}, {avail_expr}, "
                    f"callable_fn={declaration.name}, ordered_names={ordered_expr})"
                )

        # Build body
        body_lines.append(
            f"return {declaration.name}({', '.join(name for name, _ in parameters)})"
        )
        body = "\n".join(body_lines)

        return SafeAtomWrapper(
            function_name=function_name,
            original_qualname=declaration.name,
            imports=imports,
            parameters=parameters,
            return_type=return_type,
            preconditions=preconditions,
            postconditions=postconditions,
            body=body,
            confidence_guard_spec=guard_spec_dict,
        )

    def render_wrapper(self, wrapper: SafeAtomWrapper) -> str:
        """Render a SafeAtomWrapper to Python source code."""
        lines: list[str] = []

        if wrapper.confidence_guard_spec is not None:
            guard_const = f"_SCIONA_GUARD_{wrapper.function_name.upper()}"
            lines.append(f"{guard_const} = {wrapper.confidence_guard_spec!r}")
            lines.append("")

        # Decorators
        for pre in wrapper.preconditions:
            desc = f', "{pre.description}"' if pre.description else ""
            lines.append(f"@icontract.require({pre.lambda_expr}{desc})")

        for post in wrapper.postconditions:
            desc = f', "{post.description}"' if post.description else ""
            lines.append(f"@icontract.ensure({post.lambda_expr}{desc})")

        # Function signature
        params = ", ".join(
            f"{name}: {typ}" if typ else name for name, typ in wrapper.parameters
        )
        ret = f" -> {wrapper.return_type}" if wrapper.return_type else ""
        lines.append(f"def {wrapper.function_name}({params}){ret}:")

        # Body
        if wrapper.body:
            for body_line in wrapper.body.splitlines():
                lines.append(f"    {body_line}")
        else:
            lines.append(
                f'    raise NotImplementedError("TODO: compose {wrapper.original_qualname}")'
            )

        return "\n".join(lines)

    def _iospec_to_contract(self, spec: IOSpec, kind: str) -> ContractSpec | None:
        """Convert an IOSpec constraint to a ContractSpec.

        Recognizes DSP-specific constraint patterns:
        - "round_trip: INVERSE(FORWARD(x))" -> epsilon-metric postcondition
        - "poles inside unit circle" -> stability postcondition
        - "positive semi-definite" -> eigenvalue check
        - "TV(result) <= TV(input)" -> total variation reduction
        """
        if not spec.constraints:
            return None

        constraint = spec.constraints.strip()

        # DSP pattern: round-trip epsilon-metric
        if constraint.startswith("round_trip:"):
            expr = constraint[len("round_trip:") :].strip()
            lambda_expr = (
                f"lambda result, {spec.name}: " f"np.allclose({expr}, atol=1e-10)"
            )
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description=f"Round-trip: {expr}",
            )

        # DSP pattern: filter stability
        if constraint == "poles inside unit circle":
            lambda_expr = "lambda result: _poles_inside_unit_circle(result[1])"
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Filter must be stable (poles inside unit circle)",
            )

        # DSP pattern: positive semi-definite
        if constraint == "positive semi-definite":
            lambda_expr = "lambda result: _eigenvalues_nonneg(result, k=1)"
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Matrix must be positive semi-definite",
            )

        # DSP pattern: total variation reduction
        if constraint == "TV(result) <= TV(input)":
            lambda_expr = (
                f"lambda result, {spec.name}: "
                f"_total_variation(L, result) <= _total_variation(L, {spec.name}) + 1e-8"
            )
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Output total variation must not exceed input total variation",
            )

        # Default: generic constraint
        if kind == "require":
            lambda_expr = f"lambda {spec.name}: {constraint}"
        else:
            lambda_expr = f"lambda result: {constraint}"

        return ContractSpec(
            kind=kind,
            lambda_expr=lambda_expr,
            description=f"{spec.name}: {constraint}",
        )
