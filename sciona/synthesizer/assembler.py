"""Core assembly logic: CDG + MatchResults -> compilable skeleton file."""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from sciona.atom_identity import known_atom_package_prefixes
from sciona.ghost.registry import REGISTRY as _GHOST_REGISTRY
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.synthesizer.models import AssemblyUnit, GlueEdge, SkeletonFile
from sciona.synthesizer.toposort import (
    detect_cycle_partition,
    toposort_nodes,
    toposort_with_fixed_points,
)
from sciona.types import MatchResult, Prover
from sciona.uncertainty.conformal import TrustTierFilter
from sciona.uncertainty.runtime_guard import (
    ConfidenceGuardConfig,
    NodeGuardSpec,
    plan_cdg_confidence_guards,
)

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import UncertaintyBackend
    from sciona.uncertainty.models import AtomUncertaintyProfile

_UNSET_TIER: Any = object()

# ---------------------------------------------------------------------------
# Telemetry helper emitted into instrumented Python skeletons
# ---------------------------------------------------------------------------

_PYTHON_PARAMS_HARNESS: list[str] = [
    "# --- Parameter override harness ---",
    "import argparse as _sciona_argparse",
    "_SCIONA_PARAMS = {}",
    "_sciona_parser = _sciona_argparse.ArgumentParser(add_help=False)",
    "_sciona_parser.add_argument('--params', default=None)",
    "_sciona_known, _ = _sciona_parser.parse_known_args()",
    "if _sciona_known.params:",
    "    import json as _sciona_json",
    "    from pathlib import Path as _sciona_Path",
    "    _SCIONA_PARAMS = _sciona_json.loads(_sciona_Path(_sciona_known.params).read_text())",
]

_PYTHON_TELEMETRY_HELPER: list[str] = [
    "",
    "from sciona.principal.runtime_context import summarize_named_value as _sciona_summarize_named_value",
    "",
    "_SCIONA_TRACE_PATH = 'trace.jsonl'",
    "",
    "",
    "def _sciona_summarize_outputs(output_names, result):",
    '    """Build compact named summaries for one node result."""',
    "    names = [str(name) for name in output_names or () if name]",
    "    if not names:",
    "        return {}",
    "    if len(names) == 1:",
    "        values = [result]",
    "    elif isinstance(result, (tuple, list)):",
    "        values = list(result)",
    "    else:",
    "        values = [result]",
    "    summaries = {}",
    "    for index, name in enumerate(names):",
    "        if index >= len(values):",
    "            break",
    "        try:",
    "            summaries[name] = _sciona_summarize_named_value(name, values[index])",
    "        except Exception:",
    "            continue",
    "    return summaries",
    "",
    "",
    "def _sciona_probe(node_id: str, fn, output_names=(), guard_fn=None):",
    '    """Execute *fn* and append a JSON-lines telemetry record."""',
    "    result = None",
    "    guard_info = None",
    "    aborted_error = None",
    "    tracemalloc.start()",
    "    t0 = time.perf_counter()",
    "    try:",
    "        if guard_fn is not None:",
    "            guard_info = guard_fn()",
    "        result = fn()",
    "    except Exception as _exc:",
    "        guard_info = getattr(_exc, 'guard_summary', guard_info)",
    "        aborted_error = f'{type(_exc).__name__}: {_exc}'",
    "        raise",
    "    finally:",
    "        elapsed_ms = (time.perf_counter() - t0) * 1000.0",
    "        _, peak = tracemalloc.get_traced_memory()",
    "        tracemalloc.stop()",
    "        record = {",
    '            "node_id": node_id,',
    '            "execution_time_ms": elapsed_ms,',
    '            "peak_memory_bytes": peak,',
    "        }",
    "        if guard_info is not None:",
    '            record["confidence_guard"] = guard_info',
    "        if aborted_error is not None:",
    '            record["aborted_error"] = aborted_error',
    "        summaries = _sciona_summarize_outputs(output_names, result)",
    "        if summaries:",
    '            record["output_summaries"] = summaries',
    "        with open(_SCIONA_TRACE_PATH, 'a') as _f:",
    "            _f.write(json.dumps(record) + '\\n')",
    "    return result",
]


class AssemblyError(Exception):
    """Raised when assembly fails due to missing or invalid inputs."""


def sanitize_name(name: str) -> str:
    """Convert a human-readable name to a valid Lean/Coq identifier."""
    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Lowercase
    s = s.lower()
    # Ensure it starts with a letter
    if s and not s[0].isalpha():
        s = "n_" + s
    return s or "unnamed"


def _unique_unit_names(units: list[AssemblyUnit]) -> dict[str, str]:
    """Map ``unit.node_id`` -> unique sanitized identifier, preserving bare names when unique."""
    counts: dict[str, int] = {}
    for unit in units:
        base = sanitize_name(unit.name)
        counts[base] = counts.get(base, 0) + 1

    used: set[str] = set()
    mapping: dict[str, str] = {}
    for unit in units:
        base = sanitize_name(unit.name)
        if counts[base] == 1 and base not in used:
            candidate = base
        else:
            candidate = f"{base}_{sanitize_name(unit.node_id)}"
            suffix = 2
            while candidate in used:
                candidate = f"{base}_{sanitize_name(unit.node_id)}_{suffix}"
                suffix += 1
        used.add(candidate)
        mapping[unit.node_id] = candidate
    return mapping


def _collect_atomic_descendants(
    root: AlgorithmicNode,
    decomposed_map: dict[str, AlgorithmicNode],
    unit_ids: set[str],
) -> set[str]:
    """Expand ``root.children`` transitively through nested DECOMPOSED nodes."""
    if not root.children:
        return set(unit_ids)
    resolved: set[str] = set()
    visited: set[str] = set()
    stack = list(root.children)
    while stack:
        cid = stack.pop()
        if cid in visited:
            continue
        visited.add(cid)
        if cid in unit_ids:
            resolved.add(cid)
        elif cid in decomposed_map:
            stack.extend(decomposed_map[cid].children)
    return resolved


_PORT_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("signal", "conditioned_signal", "filtered", "filtered_signal", "waveform"),
    ("events", "rpeaks", "peaks", "beats", "onsets"),
    ("rate", "heart_rate", "bpm"),
)


def _port_names_compatible(left: str, right: str) -> bool:
    left_norm = str(left or "").strip().lower()
    right_norm = str(right or "").strip().lower()
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    for group in _PORT_ALIAS_GROUPS:
        if left_norm in group and right_norm in group:
            return True
    return False


def _python_annotation_expr(type_desc: str) -> str:
    """Return a syntactically valid Python annotation expression.

    Conceptual labels from the architect catalog often contain spaces
    (for example ``filter specification``). Those are useful to preserve,
    but they are not valid Python expressions unless quoted.
    """
    annotation = type_desc.strip()
    if not annotation:
        return "object"
    try:
        ast.parse(annotation, mode="eval")
    except SyntaxError:
        return repr(annotation)
    return annotation


def _split_top_level(text: str, delimiter: str) -> list[str]:
    """Split *text* on a delimiter, ignoring nested bracket scopes."""
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _find_top_level(text: str, needle: str) -> int:
    """Find a top-level delimiter position, ignoring nested bracket scopes."""
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == needle and depth == 0:
            return idx
    return -1


def _sanitize_python_param_annotation(param: str) -> str:
    """Quote conceptual Python annotations inside a single function param."""
    colon = _find_top_level(param, ":")
    if colon < 0:
        return param

    left = param[:colon].rstrip()
    right = param[colon + 1 :]
    default_idx = _find_top_level(right, "=")
    if default_idx >= 0:
        annotation = right[:default_idx].strip()
        default = right[default_idx:]
    else:
        annotation = right.strip()
        default = ""

    sanitized = _python_annotation_expr(annotation)
    rebuilt = f"{left}: {sanitized}"
    if default:
        rebuilt += default
    return rebuilt


def sanitize_python_source_annotations(source: str) -> str:
    """Rewrite invalid Python def annotations into syntactically valid forms."""
    lines = source.splitlines()
    sanitized_lines: list[str] = []

    for line in lines:
        if not re.match(r"^\s*def\s+\w+\s*\(", line):
            sanitized_lines.append(line)
            continue

        open_paren = line.find("(")
        if open_paren < 0:
            sanitized_lines.append(line)
            continue

        depth = 0
        close_paren = -1
        for idx in range(open_paren, len(line)):
            char = line[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_paren = idx
                    break
        if close_paren < 0:
            sanitized_lines.append(line)
            continue

        prefix = line[: open_paren + 1]
        params_text = line[open_paren + 1 : close_paren]
        suffix = line[close_paren + 1 :]

        sanitized_params = ",".join(
            _sanitize_python_param_annotation(param)
            for param in _split_top_level(params_text, ",")
        )

        sanitized_suffix = suffix
        arrow_idx = suffix.find("->")
        colon_idx = suffix.rfind(":")
        if arrow_idx >= 0 and colon_idx > arrow_idx:
            annotation = suffix[arrow_idx + 2 : colon_idx].strip()
            sanitized_return = _python_annotation_expr(annotation)
            sanitized_suffix = f" -> {sanitized_return}{suffix[colon_idx:]}"

        sanitized_lines.append(f"{prefix}{sanitized_params}){sanitized_suffix}")

    result = "\n".join(sanitized_lines)
    if source.endswith("\n"):
        result += "\n"
    return result


def _infer_cast(source_type: str, target_type: str) -> str:
    """Infer a cast expression for a GlueEdge between source and target types.

    Returns an empty string when no cast is needed, or a code snippet
    appropriate for the type transformation.
    """
    src = source_type.strip()
    tgt = target_type.strip()

    if not src or not tgt:
        return ""

    # Same type -> identity (no cast)
    if src == tgt:
        return ""

    # ndarray shape changes -> np.reshape()
    if "ndarray" in src and "ndarray" in tgt:
        return "np.reshape({src}, {tgt_shape})"

    # Tuple destructure: (b, a) -> individual components
    if src.startswith("(") and "," in src and not tgt.startswith("("):
        return "({src})[0]"

    # Lean coercions
    if "Equiv" in src or "Equiv" in tgt:
        return "Equiv.toFun"

    return ""


def _uses_known_atom_package(module_name: str) -> bool:
    text = str(module_name or "").strip()
    if not text:
        return False
    for prefix in known_atom_package_prefixes():
        if text == prefix or text.startswith(prefix + "."):
            return True
    return False


class Assembler:
    """Assembles a CDG and match results into a compilable skeleton."""

    def __init__(
        self,
        prover: Prover | str,
        *,
        with_telemetry: bool = False,
        aot_target: str = "numpy",
        uncertainty_backend: UncertaintyBackend | None = None,
        uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
        enable_confidence_guards: bool | None = None,
        trust_tier: TrustTierFilter = 1,
        confidence_alpha: float = 0.01,
        require_conformal: bool = True,
    ) -> None:
        if isinstance(prover, str):
            self._prover = Prover(prover)
        else:
            self._prover = prover
        self._with_telemetry = with_telemetry
        self._aot_target = aot_target
        self._uncertainty_backend = uncertainty_backend
        self._uncertainty_profiles = (
            dict(uncertainty_profiles) if uncertainty_profiles is not None else None
        )
        self._enable_confidence_guards = enable_confidence_guards
        self._trust_tier: TrustTierFilter = trust_tier
        self._confidence_alpha = confidence_alpha
        self._require_conformal = require_conformal

    def assemble(
        self,
        cdg: CDGExport,
        match_results: list[MatchResult],
        *,
        with_telemetry: bool | None = None,
        tunable_params_by_primitive: dict[str, list[str]] | None = None,
        uncertainty_backend: UncertaintyBackend | None = None,
        uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
        enable_confidence_guards: bool | None = None,
        trust_tier: TrustTierFilter | object = _UNSET_TIER,
        confidence_alpha: float | None = None,
        require_conformal: bool | None = None,
        risk_weights: Mapping[str, float] | None = None,
    ) -> SkeletonFile:
        """Build a SkeletonFile from a CDG and its match results."""
        # Index match results by predicate_id (= node_id)
        match_map: dict[str, MatchResult] = {}
        for mr in match_results:
            match_map[mr.pdg_node.predicate_id] = mr

        # Validate: every atomic leaf must have a successful match
        atomic_leaves = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        missing: list[str] = []
        for leaf in atomic_leaves:
            mr = match_map.get(leaf.node_id)
            if mr is None or not mr.success:
                missing.append(leaf.name)
        if missing:
            raise AssemblyError(
                f"Missing verified matches for atomic leaves: {missing}"
            )

        # Build AssemblyUnits for atomic leaves
        units: list[AssemblyUnit] = []
        glue_node_ids: set[str] = set()
        for edge in cdg.edges:
            if edge.requires_glue:
                glue_node_ids.add(edge.target_id)

        for leaf in atomic_leaves:
            mr = match_map[leaf.node_id]
            assert mr.verified_match is not None
            decl = mr.verified_match.candidate.declaration
            primitive_name = str(leaf.matched_primitive or "").strip()
            units.append(
                AssemblyUnit(
                    node_id=leaf.node_id,
                    name=leaf.name,
                    declaration_name=decl.name,
                    type_signature=decl.type_signature or leaf.type_signature,
                    matched_primitive=primitive_name,
                    raw_code=decl.raw_code,
                    inputs=leaf.inputs,
                    outputs=leaf.outputs,
                    requires_glue=leaf.node_id in glue_node_ids,
                    tunable_param_names=list(
                        (tunable_params_by_primitive or {}).get(primitive_name, [])
                    ),
                )
            )

        # Build GlueEdges with cast inference and dim propagation
        node_map_for_dim = {n.node_id: n for n in cdg.nodes}
        glue_edges: list[GlueEdge] = []
        for edge in cdg.edges:
            cast = _infer_cast(edge.source_type, edge.target_type)

            # Propagate dim_signature from source output / target input IOSpecs
            src_dim = ""
            tgt_dim = ""
            src_node = node_map_for_dim.get(edge.source_id)
            tgt_node = node_map_for_dim.get(edge.target_id)
            if src_node:
                for out in src_node.outputs:
                    if _port_names_compatible(out.name, edge.output_name) and out.dim_signature:
                        src_dim = out.dim_signature
                        break
            if tgt_node:
                for inp in tgt_node.inputs:
                    if _port_names_compatible(inp.name, edge.input_name) and inp.dim_signature:
                        tgt_dim = inp.dim_signature
                        break

            glue_edges.append(
                GlueEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    output_name=edge.output_name,
                    input_name=edge.input_name,
                    source_type=edge.source_type,
                    target_type=edge.target_type,
                    cast_expr=cast,
                    source_dim=src_dim,
                    target_dim=tgt_dim,
                )
            )

        # Topological sort for emission order
        cycle_node_ids: set[str] = set()
        try:
            sorted_ids = toposort_nodes(cdg.nodes, cdg.edges)
        except ValueError:
            # Cycle detected — check whether it is a valid
            # MESSAGE_PASSING / FIXED_POINT cycle.
            acyclic_sorted, cycle_node_ids, is_valid = detect_cycle_partition(
                cdg.nodes, cdg.edges
            )
            if not is_valid:
                raise AssemblyError(
                    f"Cycle detected among non-iterative nodes: "
                    f"{sorted(cycle_node_ids)}"
                ) from None
            # Valid cycle: emit acyclic nodes first, cyclic nodes in
            # round-robin order at the end.
            sorted_ids = list(acyclic_sorted) + sorted(cycle_node_ids)

        # Check for combinator subtrees (FIXED_POINT / MAP_OVER)
        combinator_node_ids: set[str] = set()
        combinator_bodies: dict[str, list[str]] = {}
        for n in cdg.nodes:
            if n.concept_type in {ConceptType.FIXED_POINT, ConceptType.MAP_OVER}:
                combinator_node_ids.add(n.node_id)
        if combinator_node_ids:
            try:
                top_order, combinator_bodies = toposort_with_fixed_points(
                    cdg.nodes, cdg.edges
                )
                sorted_ids = list(top_order)
                for combinator_id in top_order:
                    sorted_ids.extend(combinator_bodies.get(combinator_id, []))
            except ValueError:
                logger.debug("Combinator toposort failed; using fallback order")

        # Reorder units by topological order
        unit_map = {u.node_id: u for u in units}
        sorted_units = [unit_map[nid] for nid in sorted_ids if nid in unit_map]

        # Find root nodes (decomposed, not atomic)
        root_nodes = [n for n in cdg.nodes if n.status == NodeStatus.DECOMPOSED]

        # Generate source code
        metadata = dict(cdg.metadata) if cdg.metadata else {}
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        telemetry = (
            with_telemetry if with_telemetry is not None else self._with_telemetry
        )

        eff_backend = (
            uncertainty_backend
            if uncertainty_backend is not None
            else self._uncertainty_backend
        )
        eff_profiles = (
            uncertainty_profiles
            if uncertainty_profiles is not None
            else self._uncertainty_profiles
        )
        eff_guards_enabled = (
            enable_confidence_guards
            if enable_confidence_guards is not None
            else (
                self._enable_confidence_guards
                if self._enable_confidence_guards is not None
                else (eff_backend is not None or eff_profiles is not None)
            )
        )
        eff_trust_tier: TrustTierFilter = (
            self._trust_tier if trust_tier is _UNSET_TIER else trust_tier  # type: ignore[assignment]
        )
        eff_alpha = float(
            confidence_alpha if confidence_alpha is not None else self._confidence_alpha
        )
        eff_req_conformal = (
            require_conformal
            if require_conformal is not None
            else self._require_conformal
        )

        guard_specs: dict[str, NodeGuardSpec] = {}
        if eff_guards_enabled:
            raw_cdg_tier = metadata.get("trust_tier")
            default_tier = (
                int(raw_cdg_tier)
                if isinstance(raw_cdg_tier, int) and not isinstance(raw_cdg_tier, bool)
                else None
            )
            guard_specs = plan_cdg_confidence_guards(
                sorted_units,
                backend=eff_backend,
                profiles=eff_profiles,
                trust_tier=eff_trust_tier,
                global_alpha=eff_alpha,
                require_conformal=eff_req_conformal,
                abort_on_violation=True,
                risk_weights=risk_weights,
                default_trust_tier=default_tier,
            )
            guard_cfg = ConfidenceGuardConfig(
                enabled=True,
                trust_tier=eff_trust_tier,
                global_alpha=eff_alpha,
                require_conformal=eff_req_conformal,
                abort_on_violation=True,
            )
            metadata["confidence_guards"] = {
                **guard_cfg.to_metadata_dict(),
                "guarded_nodes": {
                    nid: {
                        "atom_name": spec.atom_name,
                        "trust_tier": spec.trust_tier,
                        "allocated_alpha": spec.allocated_alpha,
                    }
                    for nid, spec in guard_specs.items()
                },
            }

        if self._prover == Prover.LEAN4:
            source, sorry_count = self._emit_lean4(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
                cycle_node_ids=cycle_node_ids,
                fp_bodies=combinator_bodies,
            )
        elif self._prover == Prover.PYTHON:
            source, sorry_count = self._emit_python(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
                cycle_node_ids=cycle_node_ids,
                fp_bodies=combinator_bodies,
                guard_specs=guard_specs,
            )
        else:
            source, sorry_count = self._emit_coq(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
                cycle_node_ids=cycle_node_ids,
                fp_bodies=combinator_bodies,
            )

        return SkeletonFile(
            prover=self._prover.value,
            source_code=source,
            units=sorted_units,
            glue_edges=glue_edges,
            sorry_count=sorry_count,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def _compose_python(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
        *,
        decomposed_map: dict[str, AlgorithmicNode] | None = None,
        unit_names: dict[str, str] | None = None,
    ) -> list[str]:
        """Generate Python composition body for a root node.

        Walks glue edges in topological order and emits variable assignments
        connecting source outputs to target inputs.
        """
        unit_map = {u.node_id: u for u in units}
        name_map = unit_names if unit_names is not None else _unique_unit_names(units)

        # Edges relevant to this root's transitive atomic children
        child_ids = _collect_atomic_descendants(
            root,
            decomposed_map or {},
            set(unit_map.keys()),
        )
        relevant_edges = [
            e
            for e in glue_edges
            if e.source_id in child_ids or e.target_id in child_ids
        ]

        if not relevant_edges and not units:
            return [
                '    raise NotImplementedError("TODO: compose {}")'.format(root.name)
            ]

        lines: list[str] = []

        # First, call each atomic unit in topological order
        called: set[str] = set()
        for unit in units:
            if unit.node_id not in child_ids:
                continue
            sname = name_map[unit.node_id]
            # Determine input arguments from edges or function inputs
            args: list[str] = []
            for inp in unit.inputs:
                # Check if an edge provides this input
                edge_for_inp = next(
                    (
                        e
                        for e in relevant_edges
                        if e.target_id == unit.node_id
                        and _port_names_compatible(e.input_name, inp.name)
                    ),
                    None,
                )
                if edge_for_inp:
                    src_unit = unit_map.get(edge_for_inp.source_id)
                    if src_unit:
                        src_var = name_map[src_unit.node_id] + "_result"
                        src_expr = src_var
                        if len(src_unit.outputs) > 1 and edge_for_inp.output_name:
                            out_idx = next(
                                (
                                    idx
                                    for idx, out_spec in enumerate(src_unit.outputs)
                                    if _port_names_compatible(
                                        edge_for_inp.output_name, out_spec.name
                                    )
                                ),
                                None,
                            )
                            if out_idx is not None:
                                out_key = edge_for_inp.output_name
                                src_expr = (
                                    f"({src_var}[{out_idx}] if isinstance({src_var}, (tuple, list)) "
                                    f"else ({src_var}[{out_key!r}] if isinstance({src_var}, dict) else {src_var}))"
                                )
                        if edge_for_inp.cast_expr:
                            args.append(
                                f"# cast: {edge_for_inp.cast_expr}\n    {src_expr}"
                            )
                        else:
                            args.append(src_expr)
                    else:
                        args.append(inp.name)
                else:
                    args.append(inp.name)

            args_str = ", ".join(args)
            lines.append(f"    {sname}_result = {sname}({args_str})")
            called.add(unit.node_id)

        # Return sink unit output(s)
        if called:
            called_units = [u for u in units if u.node_id in called]
            internal_sources = {
                e.source_id
                for e in relevant_edges
                if e.source_id in called and e.target_id in called
            }
            sink_units = [u for u in called_units if u.node_id not in internal_sources]
            if not sink_units:
                sink_units = [called_units[-1]]
            if len(sink_units) == 1:
                last_name = name_map[sink_units[0].node_id] + "_result"
                lines.append(f"    return {last_name}")
            else:
                tuple_items = ", ".join(
                    f"{name_map[u.node_id]}_result" for u in sink_units
                )
                lines.append(f"    return ({tuple_items})")
        else:
            lines.append("    # TODO: compose -- no atomic children resolved")
            lines.append(
                '    raise NotImplementedError("compose {}")'.format(root.name)
            )

        return lines

    def _compose_lean4(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
    ) -> list[str]:
        """Generate Lean 4 composition proof body for a root node.

        Uses direct term composition (f . g) or calc chains instead of sorry.
        """
        child_ids = set(root.children) if root.children else {u.node_id for u in units}
        relevant_units = [u for u in units if u.node_id in child_ids]

        if not relevant_units:
            return [
                "  -- TODO: compose {} -- no atomic children resolved".format(
                    root.name
                ),
                "  sorry",
            ]

        lines: list[str] = []

        if len(relevant_units) == 1:
            # Single child: direct application
            u = relevant_units[0]
            sname = sanitize_name(u.name)
            lines.append(f"  exact {sname}")
        else:
            # Multiple children: compose via term application
            names = [sanitize_name(u.name) for u in relevant_units]

            # Check for glue edges needing casts
            has_casts = any(
                e.cast_expr
                for e in glue_edges
                if e.source_id in child_ids or e.target_id in child_ids
            )
            if has_casts:
                for e in glue_edges:
                    if e.cast_expr and (
                        e.source_id in child_ids or e.target_id in child_ids
                    ):
                        lines.append(f"  -- GLUE: {e.source_type} -> {e.target_type}")
                lines.append(f"  exact ({' ∘ '.join(reversed(names))})")
            else:
                lines.append(f"  exact ({' ∘ '.join(reversed(names))})")

        return lines

    def _compose_coq(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
    ) -> list[str]:
        """Generate Coq composition proof body for a root node.

        Uses exact/apply chains instead of Admitted.
        """
        child_ids = set(root.children) if root.children else {u.node_id for u in units}
        relevant_units = [u for u in units if u.node_id in child_ids]

        if not relevant_units:
            return [
                "  (* TODO: compose {} -- no atomic children resolved *)".format(
                    root.name
                ),
                "  Admitted.",
            ]

        if len(relevant_units) == 1:
            u = relevant_units[0]
            sname = sanitize_name(u.name)
            return [f"Proof. exact {sname}. Qed."]
        else:
            names = [sanitize_name(u.name) for u in relevant_units]
            apply_chain = " ".join(f"apply {n}." for n in names)
            return [f"Proof. {apply_chain} Qed."]

    # ------------------------------------------------------------------
    # Prover-specific emitters
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cycle & FIXED_POINT loop emission helpers
    # ------------------------------------------------------------------

    def _emit_cycle_python(
        self,
        cycle_units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
    ) -> list[str]:
        """Emit a Python while-loop around cyclic nodes in round-robin order."""
        lines: list[str] = []
        lines.append("    # --- Iterative message-passing loop ---")
        lines.append("    _MAX_ITERS = 100")
        lines.append("    _converged = False")
        lines.append("    for _iter in range(_MAX_ITERS):")
        for unit in cycle_units:
            sname = sanitize_name(unit.name)
            args: list[str] = []
            for inp in unit.inputs:
                edge_for_inp = next(
                    (
                        e
                        for e in glue_edges
                        if e.target_id == unit.node_id
                        and _port_names_compatible(e.input_name, inp.name)
                    ),
                    None,
                )
                if edge_for_inp:
                    src_unit = next(
                        (u for u in cycle_units if u.node_id == edge_for_inp.source_id),
                        None,
                    )
                    if src_unit:
                        args.append(sanitize_name(src_unit.name) + "_result")
                    else:
                        args.append(inp.name)
                else:
                    args.append(inp.name)
            args_str = ", ".join(args)
            lines.append(f"        {sname}_result = {sname}({args_str})")
        lines.append("        # Check convergence (convention: last node returns bool)")
        if cycle_units:
            last = sanitize_name(cycle_units[-1].name)
            lines.append(f"        if {last}_result is True:")
            lines.append("            _converged = True")
            lines.append("            break")
        return lines

    def _emit_cycle_lean4(self, cycle_units: list[AssemblyUnit]) -> list[str]:
        """Emit a Lean 4 sorry-guarded placeholder for cyclic nodes."""
        lines: list[str] = []
        lines.append("  -- Iterative message-passing cycle (placeholder)")
        names = [sanitize_name(u.name) for u in cycle_units]
        lines.append(f"  -- Cycle nodes: {', '.join(names)}")
        lines.append("  sorry")
        return lines

    def _emit_cycle_coq(self, cycle_units: list[AssemblyUnit]) -> list[str]:
        """Emit a Coq Admitted placeholder for cyclic nodes."""
        lines: list[str] = []
        names = [sanitize_name(u.name) for u in cycle_units]
        lines.append(f"  (* Iterative message-passing cycle: {', '.join(names)} *)")
        lines.append("  Admitted.")
        return lines

    def _emit_fixed_point_python(
        self,
        fp_node: AlgorithmicNode,
        body_units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
    ) -> list[str]:
        """Emit a Python while-loop for a FIXED_POINT node."""
        max_iters = getattr(fp_node, "fixed_point_max_iterations", 0) or 100
        conv_field = getattr(fp_node, "fixed_point_convergence_field", "") or "converged"

        lines: list[str] = []
        fpname = sanitize_name(fp_node.name)
        lines.append(f"    # --- Fixed-point iteration: {fp_node.name} ---")
        lines.append(f"    _fp_max_iters_{fpname} = {max_iters}")
        lines.append(f"    _fp_converged_{fpname} = False")
        lines.append(f"    for _fp_iter_{fpname} in range(_fp_max_iters_{fpname}):")
        for unit in body_units:
            sname = sanitize_name(unit.name)
            args: list[str] = []
            for inp in unit.inputs:
                edge_for_inp = next(
                    (
                        e
                        for e in glue_edges
                        if e.target_id == unit.node_id
                        and _port_names_compatible(e.input_name, inp.name)
                    ),
                    None,
                )
                if edge_for_inp:
                    src_unit = next(
                        (u for u in body_units if u.node_id == edge_for_inp.source_id),
                        None,
                    )
                    if src_unit:
                        args.append(sanitize_name(src_unit.name) + "_result")
                    else:
                        args.append(inp.name)
                else:
                    args.append(inp.name)
            args_str = ", ".join(args)
            lines.append(f"        {sname}_result = {sname}({args_str})")
        # Convergence check
        if body_units:
            last = sanitize_name(body_units[-1].name)
            lines.append(f"        if hasattr({last}_result, '{conv_field}') and {last}_result.{conv_field}:")
            lines.append(f"            _fp_converged_{fpname} = True")
            lines.append("            break")
            lines.append(f"        if isinstance({last}_result, bool) and {last}_result:")
            lines.append(f"            _fp_converged_{fpname} = True")
            lines.append("            break")
        return lines

    def _emit_fixed_point_lean4(self, fp_node: AlgorithmicNode) -> list[str]:
        """Emit Lean 4 Nat.rec-based fuel or sorry-guarded partial def."""
        max_iters = getattr(fp_node, "fixed_point_max_iterations", 0) or 100
        lines: list[str] = []
        lines.append(f"  -- Fixed-point iteration: {fp_node.name} (fuel={max_iters})")
        lines.append("  -- TODO: Nat.rec-based termination proof")
        lines.append("  sorry")
        return lines

    def _emit_fixed_point_coq(self, fp_node: AlgorithmicNode) -> list[str]:
        """Emit Coq Fixpoint with fuel nat parameter."""
        max_iters = getattr(fp_node, "fixed_point_max_iterations", 0) or 100
        fpname = sanitize_name(fp_node.name)
        lines: list[str] = []
        lines.append(f"  (* Fixed-point iteration: {fp_node.name} (fuel={max_iters}) *)")
        lines.append(f"  (* Fixpoint {fpname}_loop (fuel : nat) := ... *)")
        lines.append("  Admitted.")
        return lines

    def _emit_map_over_python(
        self,
        map_node: AlgorithmicNode,
        body_units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
    ) -> list[str]:
        """Emit a Python for-loop over sliding windows for a MAP_OVER node."""
        window_size = getattr(map_node, "map_window_size", 0) or 1024
        hop_size = getattr(map_node, "map_hop_size", 0) or window_size

        lines: list[str] = []
        mname = sanitize_name(map_node.name)
        lines.append(f"    # --- MAP over windows: {map_node.name} ---")
        lines.append(f"    _map_window_{mname} = {window_size}")
        lines.append(f"    _map_hop_{mname} = {hop_size}")
        lines.append(f"    _map_results_{mname} = []")
        lines.append(
            f"    for _win_start_{mname} in range("
            f"0, len(signal) - _map_window_{mname} + 1, _map_hop_{mname}):"
        )
        lines.append(
            f"        _window_{mname} = signal["
            f"_win_start_{mname}:_win_start_{mname} + _map_window_{mname}]"
        )

        for unit in body_units:
            sname = sanitize_name(unit.name)
            args: list[str] = []
            for inp in unit.inputs:
                if inp.name == "signal":
                    args.append("_window_" + mname)
                    continue
                if inp.name == "window":
                    args.append("_window_" + mname)
                    continue
                edge_for_inp = next(
                    (
                        e
                        for e in glue_edges
                        if e.target_id == unit.node_id
                        and _port_names_compatible(e.input_name, inp.name)
                    ),
                    None,
                )
                if edge_for_inp:
                    src_unit = next(
                        (u for u in body_units if u.node_id == edge_for_inp.source_id),
                        None,
                    )
                    if src_unit:
                        args.append(sanitize_name(src_unit.name) + "_result")
                    else:
                        args.append(inp.name)
                else:
                    args.append(inp.name)
            args_str = ", ".join(args)
            lines.append(f"        {sname}_result = {sname}({args_str})")

        if body_units:
            last = sanitize_name(body_units[-1].name)
            lines.append(f"        _map_results_{mname}.append({last}_result)")

        lines.append(f"    return _map_results_{mname}")
        return lines

    def _emit_map_over_lean4(self, map_node: AlgorithmicNode) -> list[str]:
        """Emit a Lean 4 placeholder for a MAP_OVER node."""
        return [
            f"  -- MAP over windows: {map_node.name}",
            "  -- TODO: List.map-based proof obligation",
            "  sorry",
        ]

    def _emit_map_over_coq(self, map_node: AlgorithmicNode) -> list[str]:
        """Emit a Coq placeholder for a MAP_OVER node."""
        mname = sanitize_name(map_node.name)
        return [
            f"  (* MAP over windows: {map_node.name} *)",
            f"  (* map {mname}_body (windows signal) *)",
            "  Admitted.",
        ]

    # ------------------------------------------------------------------
    # Prover-specific emitters
    # ------------------------------------------------------------------

    def _emit_lean4(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
        cycle_node_ids: set[str] | None = None,
        fp_bodies: dict[str, list[str]] | None = None,
    ) -> tuple[str, int]:
        """Generate Lean 4 source. Returns (source_code, sorry_count)."""
        if telemetry:
            raise NotImplementedError(
                "Lean 4 telemetry instrumentation is not yet implemented. "
                "Lean's `timeit` tactic and custom `IO.monoMsNow` wrappers "
                "will be integrated here."
            )

        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("/-!")
        lines.append("  AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"  Goal: {goal}")
        lines.append(f"  Generated: {metadata.get('timestamp', '')}")
        lines.append("-/")
        lines.append("")
        lines.append("import Mathlib")
        lines.append("")

        # Emit atomic leaf definitions
        for unit in units:
            sname = sanitize_name(unit.name)
            lines.append(f"-- Node: {unit.name} ({unit.node_id})")
            lines.append(f"#check @{unit.declaration_name}")
            if unit.type_signature:
                lines.append(f"noncomputable def {sname} : {unit.type_signature} :=")
                lines.append(f"  @{unit.declaration_name}")
            else:
                lines.append(f"noncomputable def {sname} := @{unit.declaration_name}")
            lines.append("")

        # Emit cycle placeholder (if any)
        _cycle_ids = cycle_node_ids or set()
        if _cycle_ids:
            cycle_units = [u for u in units if u.node_id in _cycle_ids]
            if cycle_units:
                cycle_lines = self._emit_cycle_lean4(cycle_units)
                lines.extend(cycle_lines)
                for line in cycle_lines:
                    if "sorry" in line and not line.strip().startswith("--"):
                        sorry_count += 1
                lines.append("")

        # Emit FIXED_POINT blocks
        node_map = {n.node_id: n for n in root_nodes}
        _fp_bodies = fp_bodies or {}
        for fp_id, body_ids in _fp_bodies.items():
            fp_node = node_map.get(fp_id)
            if fp_node is None:
                continue
            if fp_node.concept_type == ConceptType.FIXED_POINT:
                fp_lines = self._emit_fixed_point_lean4(fp_node)
                lines.extend(fp_lines)
                for line in fp_lines:
                    if "sorry" in line and not line.strip().startswith("--"):
                        sorry_count += 1
                lines.append("")
            elif fp_node.concept_type == ConceptType.MAP_OVER:
                map_lines = self._emit_map_over_lean4(fp_node)
                lines.extend(map_lines)
                for line in map_lines:
                    if "sorry" in line and not line.strip().startswith("--"):
                        sorry_count += 1
                lines.append("")

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"-- Composition: {root.name} ({root.node_id})")
                lines.append(
                    f"theorem {rname}_composition : {root.type_signature} := by"
                )
                composition = self._compose_lean4(units, glue_edges, root)
                lines.extend(composition)
                # Count sorrys in generated composition
                for line in composition:
                    if "sorry" in line and not line.strip().startswith("--"):
                        sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count

    def _emit_python(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
        cycle_node_ids: set[str] | None = None,
        fp_bodies: dict[str, list[str]] | None = None,
        guard_specs: Mapping[str, NodeGuardSpec] | None = None,
    ) -> tuple[str, int]:
        """Generate Python source. Returns (source_code, sorry_count)."""
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append('"""')
        lines.append("AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"Goal: {goal}")
        lines.append(f"Generated: {metadata.get('timestamp', '')}")
        lines.append('"""')
        lines.append("")

        # Imports
        lines.append("import icontract")
        lines.append("import inspect")

        # Infer imports from declaration names
        imported_modules: list[str] = []
        imports_seen: set[str] = {"icontract", "inspect"}
        for unit in units:
            if "." in unit.declaration_name:
                module = unit.declaration_name.rsplit(".", 1)[0]
                if module not in imports_seen:
                    imports_seen.add(module)
                    imported_modules.append(module)
        if any(_uses_known_atom_package(module) for module in imported_modules):
            lines.append("from sciona.julia_runtime import configure_juliacall_env")
            lines.append("configure_juliacall_env()")
        for module in imported_modules:
            lines.append(f"import {module}")

        # Common scientific imports if not already present. Generated Python
        # annotations use ``np.ndarray`` heavily, so the alias needs to exist in
        # the whole-file compile path as well, not only during export cleanup.
        if "numpy" not in imports_seen:
            imports_seen.add("numpy")
            lines.append("import numpy as np")
        if "scipy" not in imports_seen:
            imports_seen.add("scipy")
            lines.append("import scipy")

        # Telemetry imports
        if telemetry:
            lines.append("import json")
            lines.append("import time")
            lines.append("import tracemalloc")

        _guard_specs = dict(guard_specs or {})
        if _guard_specs:
            lines.append(
                "from sciona.uncertainty.runtime_guard import "
                "evaluate_runtime_guard as _sciona_eval_guard"
            )

        lines.append("")

        # Telemetry helper
        if telemetry:
            lines.extend(_PYTHON_TELEMETRY_HELPER)
            lines.append("")

        # Runtime confidence guard specs & dispatcher
        if _guard_specs:
            serialized_guards = {
                nid: spec.to_dict() for nid, spec in _guard_specs.items()
            }
            lines.append("# --- Calibrated runtime confidence guards ---")
            lines.append(f"_SCIONA_CONFIDENCE_GUARDS = {serialized_guards!r}")
            lines.append("")
            lines.append(
                "def _sciona_check_confidence(node_id, fn=None, ordered_names=(), **available):"
            )
            lines.append("    _spec = _SCIONA_CONFIDENCE_GUARDS.get(node_id)")
            lines.append("    if _spec is None:")
            lines.append("        return None")
            lines.append(
                "    return _sciona_eval_guard(_spec, available, callable_fn=fn, ordered_names=ordered_names)"
            )
            lines.append("")

        # Params harness (only if any unit has tunables)
        has_tunables = any(unit.tunable_param_names for unit in units)
        if has_tunables:
            lines.extend(_PYTHON_PARAMS_HARNESS)
            lines.append("")

        lines.append("_SCIONA_ALIASES = {")
        lines.append("    'signal': ('signal', 'conditioned_signal', 'filtered', 'filtered_signal', 'waveform'),")
        lines.append("    'conditioned_signal': ('conditioned_signal', 'filtered', 'filtered_signal', 'signal'),")
        lines.append("    'filtered': ('filtered', 'conditioned_signal', 'filtered_signal', 'signal'),")
        lines.append("    'filtered_signal': ('filtered_signal', 'filtered', 'conditioned_signal', 'signal'),")
        lines.append("    'events': ('events', 'rpeaks', 'peaks', 'beats', 'onsets'),")
        lines.append("    'rpeaks': ('rpeaks', 'events', 'peaks', 'beats', 'onsets'),")
        lines.append("    'peaks': ('peaks', 'rpeaks', 'events', 'beats', 'onsets'),")
        lines.append("    'beats': ('beats', 'rpeaks', 'events', 'peaks', 'onsets'),")
        lines.append("    'heart_rate': ('heart_rate', 'rate', 'bpm'),")
        lines.append("    'rate': ('rate', 'heart_rate', 'bpm'),")
        lines.append("}")
        lines.append("")
        lines.append("def _sciona_resolve_available(name, available):")
        lines.append("    if name in available:")
        lines.append("        return name, available[name]")
        lines.append("    for alias in _SCIONA_ALIASES.get(name, (name,)):")
        lines.append("        if alias in available:")
        lines.append("            return alias, available[alias]")
        lines.append("    return None, None")
        lines.append("")
        lines.append("def _sciona_call(fn, ordered_names=(), **available):")
        lines.append('    """Bind available scaffold values to the runtime callable signature."""')
        lines.append("    signature = inspect.signature(fn)")
        lines.append("    ordered_pool = []")
        lines.append("    for name in ordered_names:")
        lines.append("        resolved_name, resolved_value = _sciona_resolve_available(name, available)")
        lines.append("        if resolved_name is not None:")
        lines.append("            ordered_pool.append((resolved_name, resolved_value))")
        lines.append("    used_names = set()")
        lines.append("    positional_args = []")
        lines.append("    keyword_args = {}")
        lines.append("    for param in signature.parameters.values():")
        lines.append(
            "        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):"
        )
        lines.append("            continue")
        lines.append("        resolved_name, value = _sciona_resolve_available(param.name, available)")
        lines.append("        if resolved_name is not None:")
        lines.append("            used_names.add(resolved_name)")
        lines.append("        elif param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):")
        lines.append("            fallback = None")
        lines.append("            for name, candidate in ordered_pool:")
        lines.append("                if name in used_names:")
        lines.append("                    continue")
        lines.append("                fallback = (name, candidate)")
        lines.append("                break")
        lines.append("            if fallback is None:")
        lines.append("                continue")
        lines.append("            used_names.add(fallback[0])")
        lines.append("            value = fallback[1]")
        lines.append("        else:")
        lines.append("            continue")
        lines.append(
            "        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):"
        )
        lines.append("            positional_args.append(value)")
        lines.append("        else:")
        lines.append("            keyword_args[param.name] = value")
        lines.append("    return fn(*positional_args, **keyword_args)")
        lines.append("")
        lines.append("")

        # Emit atomic leaf definitions
        unit_names = _unique_unit_names(units)
        for unit in units:
            sname = unit_names[unit.node_id]
            lines.append(f"# Node: {unit.name} ({unit.node_id})")

            # --- AOT path: if unit has a symbolic expression, emit inline NumPy/Mojo ---
            _reg_entry = (
                _GHOST_REGISTRY.get(unit.declaration_name)
                or _GHOST_REGISTRY.get(unit.name)
            )
            _symbolic = (
                _reg_entry.get("symbolic")
                if (_reg_entry and unit.node_id not in _guard_specs)
                else None
            )
            if _symbolic is not None:
                try:
                    if self._aot_target == "mojo":
                        from sciona.synthesizer.mojo_codegen import (
                            sympy_to_mojo_source,
                        )
                        aot_src = sympy_to_mojo_source(
                            _symbolic,
                            sname,
                            input_vars=[inp.name for inp in unit.inputs],
                            docstring=f"AOT-compiled: {unit.name} ({unit.node_id})",
                        )
                        lines.append("# AOT-compiled from SymPy to Mojo (native speed)")
                    else:
                        from sciona.synthesizer.numpy_codegen import (
                            sympy_to_numpy_source,
                        )
                        aot_src = sympy_to_numpy_source(
                            _symbolic,
                            sname,
                            input_vars=[inp.name for inp in unit.inputs],
                            docstring=f"AOT-compiled: {unit.name} ({unit.node_id})",
                        )
                        lines.append("# AOT-compiled from SymPy (zero sympy runtime dependency)")
                    lines.append(aot_src)
                    lines.append("")
                    lines.append("")
                    continue
                except Exception as _aot_exc:
                    logger.debug(
                        "AOT codegen failed for %s, falling back to _sciona_call: %s",
                        unit.name, _aot_exc,
                    )

            # --- Standard path: delegate to the registered implementation ---
            # Build function signature from unit inputs/outputs
            params: list[str] = []
            for inp in unit.inputs:
                if inp.type_desc:
                    params.append(f"{inp.name}: {_python_annotation_expr(inp.type_desc)}")
                else:
                    params.append(inp.name)

            ret_type = ""
            if len(unit.outputs) == 1:
                ret_type = _python_annotation_expr(unit.outputs[0].type_desc)
            elif len(unit.outputs) > 1:
                out_types = ", ".join(
                    _python_annotation_expr(o.type_desc) for o in unit.outputs
                )
                ret_type = f"tuple[{out_types}]"

            param_str = ", ".join(params)
            ret_str = f" -> {ret_type}" if ret_type else ""
            lines.append(f"def {sname}({param_str}){ret_str}:")

            if unit.type_signature:
                lines.append(f'    """Type: {unit.type_signature}"""')

            ordered_names_expr = repr(tuple(inp.name for inp in unit.inputs))
            available_args = ", ".join(f"{inp.name}={inp.name}" for inp in unit.inputs)
            if unit.tunable_param_names:
                params_expr = f"_SCIONA_PARAMS.get({unit.node_id!r}, {{}})"
                if available_args:
                    call_expr = (
                        f"_sciona_call({unit.declaration_name}, {ordered_names_expr}, {available_args}, **{params_expr})"
                    )
                else:
                    call_expr = (
                        f"_sciona_call({unit.declaration_name}, {ordered_names_expr}, **{params_expr})"
                    )
            else:
                if available_args:
                    call_expr = (
                        f"_sciona_call({unit.declaration_name}, {ordered_names_expr}, {available_args})"
                    )
                else:
                    call_expr = f"_sciona_call({unit.declaration_name}, {ordered_names_expr})"

            if telemetry:
                output_names_expr = repr(tuple(out.name for out in unit.outputs))
                if unit.node_id in _guard_specs:
                    if available_args:
                        guard_expr = (
                            f"lambda: _sciona_check_confidence({unit.node_id!r}, "
                            f"{unit.declaration_name}, {ordered_names_expr}, {available_args})"
                        )
                    else:
                        guard_expr = (
                            f"lambda: _sciona_check_confidence({unit.node_id!r}, "
                            f"{unit.declaration_name}, {ordered_names_expr})"
                        )
                    lines.append(
                        f"    return _sciona_probe({unit.node_id!r}, lambda: {call_expr}, "
                        f"output_names={output_names_expr}, guard_fn={guard_expr})"
                    )
                else:
                    lines.append(
                        f"    return _sciona_probe({unit.node_id!r}, lambda: {call_expr}, output_names={output_names_expr})"
                    )
            else:
                if unit.node_id in _guard_specs:
                    if available_args:
                        lines.append(
                            f"    _sciona_check_confidence({unit.node_id!r}, "
                            f"{unit.declaration_name}, {ordered_names_expr}, {available_args})"
                        )
                    else:
                        lines.append(
                            f"    _sciona_check_confidence({unit.node_id!r}, "
                            f"{unit.declaration_name}, {ordered_names_expr})"
                        )
                lines.append(f"    return {call_expr}")
            lines.append("")
            lines.append("")

        # Emit cycle while-loop (if any)
        _cycle_ids = cycle_node_ids or set()
        _fp_bodies = fp_bodies or {}
        node_map_py = {n.node_id: n for n in root_nodes}
        for n in root_nodes:
            node_map_py[n.node_id] = n

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            lines.append(f"# Composition: {root.name} ({root.node_id})")

            # Build composition function with appropriate params from root inputs
            params = []
            for inp in root.inputs:
                if inp.type_desc:
                    params.append(f"{inp.name}: {_python_annotation_expr(inp.type_desc)}")
                else:
                    params.append(inp.name)
            param_str = ", ".join(params) if params else ""

            ret_type = ""
            if len(root.outputs) == 1:
                ret_type = _python_annotation_expr(root.outputs[0].type_desc)
            elif len(root.outputs) > 1:
                out_types = ", ".join(
                    _python_annotation_expr(o.type_desc) for o in root.outputs
                )
                ret_type = f"tuple[{out_types}]"
            ret_str = f" -> {ret_type}" if ret_type else ""

            lines.append(f"def {rname}_composition({param_str}){ret_str}:")
            if root.type_signature:
                lines.append(f'    """Compose: {root.type_signature}"""')
            else:
                lines.append('    """Compose the resolved child pipeline."""')

            # Emit FIXED_POINT body if this root is a FIXED_POINT
            if root.concept_type == ConceptType.FIXED_POINT and root.node_id in _fp_bodies:
                body_ids = _fp_bodies[root.node_id]
                body_unit_map = {u.node_id: u for u in units}
                body_units = [body_unit_map[bid] for bid in body_ids if bid in body_unit_map]
                if body_units:
                    fp_lines = self._emit_fixed_point_python(root, body_units, glue_edges)
                    lines.extend(fp_lines)
                    if body_units:
                        last = unit_names[body_units[-1].node_id]
                        lines.append(f"    return {last}_result")
                    lines.append("")
                    lines.append("")
                    continue

            if root.concept_type == ConceptType.MAP_OVER and root.node_id in _fp_bodies:
                body_ids = _fp_bodies[root.node_id]
                body_unit_map = {u.node_id: u for u in units}
                body_units = [body_unit_map[bid] for bid in body_ids if bid in body_unit_map]
                if body_units:
                    map_lines = self._emit_map_over_python(root, body_units, glue_edges)
                    lines.extend(map_lines)
                    lines.append("")
                    lines.append("")
                    continue

            composition = self._compose_python(
                units,
                glue_edges,
                root,
                decomposed_map=node_map_py,
                unit_names=unit_names,
            )

            # If there are cycle nodes inside this root's children,
            # inject a while-loop for them
            if _cycle_ids:
                child_ids = set(root.children) if root.children else {u.node_id for u in units}
                root_cycle_ids = _cycle_ids & child_ids
                if root_cycle_ids:
                    cycle_units = [u for u in units if u.node_id in root_cycle_ids]
                    # Emit non-cycle composition first
                    non_cycle = [line for line in composition if not any(
                        sanitize_name(u.name) in line for u in cycle_units
                    )]
                    lines.extend(non_cycle)
                    lines.extend(self._emit_cycle_python(cycle_units, glue_edges))
                    if cycle_units:
                        last = sanitize_name(cycle_units[-1].name)
                        lines.append(f"    return {last}_result")
                    lines.append("")
                    lines.append("")
                    continue

            lines.extend(composition)
            # Count NotImplementedError stubs
            for line in composition:
                if "NotImplementedError" in line:
                    sorry_count += 1
            lines.append("")
            lines.append("")

        return "\n".join(lines), sorry_count

    def _emit_coq(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
        cycle_node_ids: set[str] | None = None,
        fp_bodies: dict[str, list[str]] | None = None,
    ) -> tuple[str, int]:
        """Generate Coq source. Returns (source_code, sorry_count)."""
        if telemetry:
            raise NotImplementedError(
                "Coq telemetry instrumentation is not yet implemented. "
                "Coq's `Time` vernacular command will be integrated here."
            )
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("(*")
        lines.append("  AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"  Goal: {goal}")
        lines.append(f"  Generated: {metadata.get('timestamp', '')}")
        lines.append("*)")
        lines.append("")

        # Emit atomic leaf definitions
        for unit in units:
            sname = sanitize_name(unit.name)
            lines.append(f"(* Node: {unit.name} ({unit.node_id}) *)")
            lines.append(f"Check @{unit.declaration_name}.")
            if unit.type_signature:
                lines.append(
                    f"Definition {sname} : {unit.type_signature} := @{unit.declaration_name}."
                )
            else:
                lines.append(f"Definition {sname} := @{unit.declaration_name}.")
            lines.append("")

        # Emit cycle placeholder (if any)
        _cycle_ids = cycle_node_ids or set()
        if _cycle_ids:
            cycle_units = [u for u in units if u.node_id in _cycle_ids]
            if cycle_units:
                cycle_lines = self._emit_cycle_coq(cycle_units)
                lines.extend(cycle_lines)
                for line in cycle_lines:
                    if "Admitted" in line:
                        sorry_count += 1
                lines.append("")

        # Emit FIXED_POINT blocks
        _fp_bodies = fp_bodies or {}
        node_map_coq = {n.node_id: n for n in root_nodes}
        for fp_id in _fp_bodies:
            fp_node = node_map_coq.get(fp_id)
            if fp_node is None:
                continue
            if fp_node.concept_type == ConceptType.FIXED_POINT:
                fp_lines = self._emit_fixed_point_coq(fp_node)
                lines.extend(fp_lines)
                for line in fp_lines:
                    if "Admitted" in line:
                        sorry_count += 1
                lines.append("")
            elif fp_node.concept_type == ConceptType.MAP_OVER:
                map_lines = self._emit_map_over_coq(fp_node)
                lines.extend(map_lines)
                for line in map_lines:
                    if "Admitted" in line:
                        sorry_count += 1
                lines.append("")

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"(* Composition: {root.name} ({root.node_id}) *)")
                lines.append(f"Lemma {rname}_composition : {root.type_signature}.")
                composition = self._compose_coq(units, glue_edges, root)
                lines.extend(composition)
                for line in composition:
                    if "Admitted" in line:
                        sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count
