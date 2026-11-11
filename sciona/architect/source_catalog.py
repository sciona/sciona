"""Derive architect primitives from configured atom source registrations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import ast
import importlib
import inspect
import json
import logging
import sqlite3
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, get_args, get_origin

from sciona.architect.catalog import CatalogReport, PrimitiveCatalog
from sciona.architect.models import (
    AlgorithmicPrimitive,
    CommonPattern,
    ConceptType,
    IOSpec,
    PortTypeSpec,
    StatePortSpec,
)
from sciona.license_policy import load_license_policy_from_env
from sciona.sources import (
    AtomSource,
    SourcesConfig,
    discover_cdgs,
    excluded_import_prefixes,
    import_atoms,
    is_excluded_module,
    load_sources,
    resolve_package_root,
)
from sciona.synthesizer.uncertainty import load_uncertainty_json

if TYPE_CHECKING:
    from sciona.architect.embedder import SkillIndex

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _bump_source_breakdown(
    report: CatalogReport | None,
    source_name: str,
    key: str,
    amount: int = 1,
) -> None:
    if report is None or not source_name:
        return
    row = report.source_breakdown.setdefault(source_name, {})
    row[key] = int(row.get(key, 0) or 0) + amount


@dataclass
class _AtomicNodeMeta:
    node_id: str
    name: str
    description: str
    concept_type: ConceptType
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    type_signature: str
    matched_primitive: str
    phase: str = "stateless"
    state_ports: list[StatePortSpec] = field(default_factory=list)
    semantic_role: str = ""
    time_complexity: str = ""
    space_complexity: str = ""
    aliases: list[str] = field(default_factory=list)
    common_patterns: list[CommonPattern] = field(default_factory=list)


def _annotation_to_text(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if annotation is None:
        return "None"
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, "__module__", "") == "builtins":
        return getattr(annotation, "__name__", repr(annotation))
    rendered = repr(annotation)
    for prefix in ("typing.", "collections.abc."):
        if rendered.startswith(prefix):
            rendered = rendered[len(prefix) :]
    return rendered


def _ports_from_callable(func: Any) -> tuple[list[IOSpec], list[IOSpec]]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [IOSpec(name="result", type_desc="Any")]

    inputs: list[IOSpec] = []
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.name in {"self", "cls"}:
            continue
        required = param.default is inspect._empty
        default_value_repr = "" if required else repr(param.default)
        inputs.append(
            IOSpec(
                name=param.name,
                type_desc=_annotation_to_text(param.annotation),
                required=required,
                default_value_repr=default_value_repr,
            )
        )

    return_annotation = signature.return_annotation
    origin = get_origin(return_annotation)
    args = get_args(return_annotation)
    if origin is tuple and args and args[-1] is not Ellipsis:
        outputs = [
            IOSpec(name=f"result_{idx + 1}", type_desc=_annotation_to_text(arg))
            for idx, arg in enumerate(args)
        ]
    else:
        outputs = [IOSpec(name="result", type_desc=_annotation_to_text(return_annotation))]
    return inputs, outputs


def _signature_from_ports(inputs: list[IOSpec], outputs: list[IOSpec]) -> str:
    in_sig = ", ".join(_format_port_signature(port) for port in inputs) or "void"
    if len(outputs) == 1:
        out_sig = outputs[0].type_desc
    else:
        out_sig = "tuple[" + ", ".join(port.type_desc for port in outputs) + "]"
    return f"({in_sig}) -> {out_sig}"


def _manifest_signature_from_ports(inputs: list[IOSpec], outputs: list[IOSpec]) -> str:
    in_sig = ", ".join(_format_port_signature(port) for port in inputs) or "void"
    if not outputs:
        out_sig = "Any"
    elif len(outputs) == 1:
        out_sig = outputs[0].type_desc
    else:
        out_sig = "tuple[" + ", ".join(port.type_desc for port in outputs) + "]"
    return f"({in_sig}) -> {out_sig}"


def _format_port_signature(port: IOSpec) -> str:
    rendered = f"{port.name}: {port.type_desc}"
    if not port.required:
        rendered += "?"
        if port.default_value_repr:
            rendered += f" = {port.default_value_repr}"
    return rendered


def _iter_registry_modules() -> Iterable[Any]:
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            registry = getattr(module, "REGISTRY")
        except Exception:
            continue
        if isinstance(registry, dict):
            yield module


def _live_registry_entries_for_source(
    source: AtomSource,
    *,
    package_root: Path,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    import_atoms(source, base_dir)
    try:
        importlib.import_module(f"{source.package}.ghost.registry")
    except Exception:
        pass

    excluded = excluded_import_prefixes(source, base_dir)
    for registry_module in _iter_registry_modules():
        registry = getattr(registry_module, "REGISTRY", {})
        for name, meta in registry.items():
            impl = meta.get("impl")
            witness = meta.get("witness")
            module_name = getattr(impl, "__module__", "") or getattr(witness, "__module__", "")
            if is_excluded_module(module_name, excluded) or is_excluded_module(str(name), excluded):
                continue
            file_path = None
            for fn in (impl, witness):
                try:
                    if fn is not None:
                        file_path = inspect.getsourcefile(fn) or inspect.getfile(fn)
                        if file_path:
                            break
                except Exception:
                    continue
            if not _module_belongs_to_source(
                module_name,
                file_path,
                source=source,
                package_root=package_root,
            ):
                continue
            entries.setdefault(str(name), meta)
    return entries


def _package_python_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*.py") if path.name != "__init__.py")


def _module_belongs_to_source(
    module_name: str,
    path: str | None,
    *,
    source: AtomSource,
    package_root: Path,
) -> bool:
    if path:
        try:
            resolved_path = Path(path).resolve()
            if resolved_path.is_relative_to(package_root):
                return True
            if package_root.exists():
                return False
        except Exception:
            pass
    if module_name == source.package or module_name.startswith(source.package + "."):
        return True
    return False


def _enrich_ports_with_witness(
    ports: list[IOSpec],
    witness_ports: list[IOSpec],
) -> list[IOSpec]:
    """Attach structured ``PortTypeSpec`` using concrete port types + witness tags."""
    enriched: list[IOSpec] = []
    witness_by_name = {w.name: w for w in witness_ports}
    for idx, port in enumerate(ports):
        w_port = witness_by_name.get(port.name)
        if w_port is None and idx < len(witness_ports):
            w_port = witness_ports[idx]
        w_type = w_port.type_desc if w_port is not None else ""
        spec = PortTypeSpec.from_type_desc(
            port.type_desc,
            constraints=port.constraints,
            data_kind=port.data_kind,
            required=port.required,
            default_value_repr=port.default_value_repr,
            witness_type_desc=w_type,
        )
        enriched.append(port.model_copy(update={"port_type": spec}))
    return enriched


def _load_atomic_node_index(
    source: AtomSource, *, base_dir: Path | None = None
) -> tuple[dict[str, list[_AtomicNodeMeta]], dict[str, str]]:
    """Load atomic node metadata and topo-hashes from CDG files.

    Supports both canonical ``nodes``/``edges`` and legacy ``atomic_nodes``
    CDG files. Returns ``(atom_index, topo_hashes)``.
    """
    from sciona.graph_store import _topo_hash

    index: dict[str, list[_AtomicNodeMeta]] = defaultdict(list)
    topo_hashes: dict[str, str] = {}

    for cdg_path in discover_cdgs(source, base_dir):
        try:
            with open(cdg_path) as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                continue
        except Exception:
            continue

        all_nodes_raw = payload.get("nodes")
        legacy_atomic = False
        if all_nodes_raw is None and isinstance(payload.get("atomic_nodes"), list):
            all_nodes_raw = payload["atomic_nodes"]
            legacy_atomic = True
        all_nodes_raw = all_nodes_raw or []
        all_edges_raw = payload.get("edges", []) or []

        # Compute topo_hashes for decomposed parent nodes.
        for raw in all_nodes_raw:
            if not isinstance(raw, dict):
                continue
            children = raw.get("children", []) or []
            status = raw.get("status", "")
            if status == "decomposed" and len(children) >= 2:
                node_id = str(raw.get("node_id", ""))
                nodes_dicts = [
                    {"node_id": str(n.get("node_id", "")), "parent_id": str(n.get("parent_id", ""))}
                    for n in all_nodes_raw
                    if isinstance(n, dict)
                ]
                edges_dicts = [
                    {
                        "source_id": str(e.get("source_id") or e.get("source") or ""),
                        "target_id": str(e.get("target_id") or e.get("target") or ""),
                    }
                    for e in all_edges_raw
                    if isinstance(e, dict)
                ]
                try:
                    h = _topo_hash(nodes_dicts, edges_dicts, node_id)
                    if h:
                        topo_hashes[h] = source.name
                except Exception:
                    pass

        for raw in all_nodes_raw:
            if not isinstance(raw, dict):
                continue
            if not legacy_atomic and raw.get("status") != "atomic":
                continue
            node_id = str(raw.get("node_id") or raw.get("name") or "")
            name = str(raw.get("name") or node_id)
            semantic_role = str(raw.get("semantic_role", "") or "")
            description = str(raw.get("description") or semantic_role or "")
            raw_ct = raw.get("concept_type")
            try:
                concept_type = (
                    ConceptType(raw_ct)
                    if raw_ct
                    else _infer_concept_type(
                        name=name or node_id,
                        module=source.package,
                        description=description,
                    )
                )
            except ValueError:
                concept_type = ConceptType.CUSTOM
            inputs = _enrich_ports_with_witness(
                [IOSpec.model_validate(item) for item in (raw.get("inputs") or []) if isinstance(item, dict)],
                [],
            )
            outputs = _enrich_ports_with_witness(
                [IOSpec.model_validate(item) for item in (raw.get("outputs") or []) if isinstance(item, dict)],
                [],
            )
            state_ports = [
                StatePortSpec.model_validate(sp)
                for sp in (raw.get("state_ports") or [])
                if isinstance(sp, dict)
            ]
            common_patterns = [
                CommonPattern.model_validate(cp)
                for cp in (raw.get("common_patterns") or [])
                if isinstance(cp, dict)
            ]
            meta = _AtomicNodeMeta(
                node_id=node_id,
                name=name,
                description=description,
                concept_type=concept_type,
                inputs=inputs,
                outputs=outputs,
                type_signature=str(raw.get("type_signature", "")),
                matched_primitive=str(raw.get("matched_primitive", "") or ""),
                phase=str(raw.get("phase") or "stateless"),
                state_ports=state_ports,
                semantic_role=semantic_role,
                time_complexity=str(raw.get("time_complexity") or ""),
                space_complexity=str(raw.get("space_complexity") or ""),
                aliases=[str(a) for a in (raw.get("aliases") or []) if a],
                common_patterns=common_patterns,
            )
            for key in {
                _normalize(meta.node_id),
                _normalize(meta.name),
                _normalize(meta.matched_primitive),
            }:
                if key:
                    index[key].append(meta)
    return index, topo_hashes


def _expr_to_text(expr: ast.expr | None) -> str:
    if expr is None:
        return "Any"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    try:
        return ast.unparse(expr)
    except Exception:
        return "Any"


def _literal_default_repr(expr: ast.expr | None) -> str:
    if expr is None:
        return ""
    try:
        return ast.unparse(expr)
    except Exception:
        return ""


def _ports_from_ast_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[IOSpec], list[IOSpec]]:
    posonly = list(func.args.posonlyargs)
    regular = list(func.args.args)
    if regular and regular[0].arg in {"self", "cls"}:
        regular = regular[1:]
    ordered = posonly + regular
    defaults = list(func.args.defaults)
    required_prefix = len(ordered) - len(defaults)

    inputs: list[IOSpec] = []
    for idx, arg in enumerate(ordered):
        default_expr = defaults[idx - required_prefix] if idx >= required_prefix else None
        required = default_expr is None
        inputs.append(
            IOSpec(
                name=arg.arg,
                type_desc=_expr_to_text(arg.annotation),
                required=required,
                default_value_repr=_literal_default_repr(default_expr),
            )
        )

    for arg, default_expr in zip(func.args.kwonlyargs, func.args.kw_defaults):
        if arg.arg in {"self", "cls"}:
            continue
        required = default_expr is None
        inputs.append(
            IOSpec(
                name=arg.arg,
                type_desc=_expr_to_text(arg.annotation),
                required=required,
                default_value_repr=_literal_default_repr(default_expr),
            )
        )

    return_annotation = func.returns
    if isinstance(return_annotation, ast.Subscript) and _expr_to_text(return_annotation.value) == "tuple":
        slice_value = return_annotation.slice
        tuple_elts = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
        outputs = [
            IOSpec(name=f"result_{idx + 1}", type_desc=_expr_to_text(item))
            for idx, item in enumerate(tuple_elts)
        ]
    else:
        outputs = [IOSpec(name="result", type_desc=_expr_to_text(return_annotation))]
    return inputs, outputs


def _ports_are_uninformative(inputs: list[IOSpec], outputs: list[IOSpec]) -> bool:
    if not inputs:
        return True
    if all(port.type_desc == "Any" for port in inputs) and all(
        port.type_desc == "Any" for port in outputs
    ):
        return True
    return False


_KEYWORD_TYPES: list[tuple[tuple[str, ...], ConceptType]] = [
    (("sort",), ConceptType.SORTING),
    (("search", "lookup"), ConceptType.SEARCHING),
    (("graph", "laplacian", "dijkstra", "shortest_path"), ConceptType.GRAPH_TRAVERSAL),
    (("filter", "sosfilt", "lfilter", "butter", "cheby", "freqz"), ConceptType.SIGNAL_FILTER),
    (("fft", "dct", "spectrum", "spectral"), ConceptType.SIGNAL_TRANSFORM),
    (("particle", "posterior", "prior", "mcmc", "resample"), ConceptType.SAMPLER),
    (("kalman",), ConceptType.CONJUGATE_UPDATE),
    (("state", "initialize", "bootstrap"), ConceptType.STATE_INIT),
    (("assemble", "adapter", "builder"), ConceptType.DATA_ASSEMBLY),
    (("progress", "diagnostic", "plot", "render", "visual"), ConceptType.OBSERVABILITY),
]


def _infer_concept_type(*, name: str, module: str, description: str) -> ConceptType:
    haystack = f"{name} {module} {description}".lower()
    for words, concept_type in _KEYWORD_TYPES:
        if any(word in haystack for word in words):
            return concept_type
    return ConceptType.CUSTOM


def _ast_register_atom_spec(decorator: ast.AST) -> tuple[str, str] | None:
    if isinstance(decorator, ast.Call):
        target = decorator.func
        if isinstance(target, ast.Name) and target.id == "register_atom":
            witness_name = ""
            if decorator.args:
                first = decorator.args[0]
                if isinstance(first, ast.Name):
                    witness_name = first.id
                elif isinstance(first, ast.Attribute):
                    witness_name = first.attr
            for kw in decorator.keywords:
                if kw.arg == "name":
                    try:
                        value = ast.literal_eval(kw.value)
                    except Exception:
                        value = None
                    if isinstance(value, str) and value:
                        return value, witness_name
            return "", witness_name
    elif isinstance(decorator, ast.Name) and decorator.id == "register_atom":
        return "", ""
    return None


def _extract_ast_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract docstring, including docstrings displaced below leading imports."""
    doc = ast.get_docstring(node)
    if doc:
        return doc
    idx = 0
    while idx < len(node.body) and isinstance(node.body[idx], (ast.Import, ast.ImportFrom)):
        idx += 1
    if idx > 0 and idx < len(node.body):
        stmt = node.body[idx]
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return inspect.cleandoc(stmt.value.value)
    return ""


def _parse_register_atom_functions(
    package_root: Path,
    package_name: str,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    pkg_parts = [p for p in package_name.split(".") if p]
    depth = max(1, len(pkg_parts))
    try:
        import_root = package_root.parents[depth - 1]
    except IndexError:
        import_root = package_root.parent

    for py_file in _package_python_files(package_root):
        try:
            module_name = ".".join(py_file.relative_to(import_root).with_suffix("").parts)
        except Exception:
            module_name = ".".join(py_file.relative_to(package_root.parent).with_suffix("").parts)
        if is_excluded_module(module_name, excluded_prefixes):
            continue
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except Exception:
            continue
        function_defs = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            reg_spec = None
            for decorator in node.decorator_list:
                reg_spec = _ast_register_atom_spec(decorator)
                if reg_spec is not None:
                    break
            if reg_spec is None:
                continue
            reg_name, witness_name = reg_spec
            atom_name = reg_name or node.name
            if is_excluded_module(atom_name, excluded_prefixes):
                continue
            inputs, outputs = _ports_from_ast_function(node)
            doc = _extract_ast_docstring(node)
            witness_doc = ""
            witness_inputs: list[IOSpec] = []
            witness_outputs: list[IOSpec] = []
            doc_source = "impl"
            signature_source = "impl"
            witness_func = function_defs.get(witness_name)
            if witness_func is not None:
                witness_doc = _extract_ast_docstring(witness_func)
                witness_inputs, witness_outputs = _ports_from_ast_function(witness_func)
                if not doc and witness_doc:
                    doc_source = "witness"
                if _ports_are_uninformative(inputs, outputs) and not _ports_are_uninformative(
                    witness_inputs, witness_outputs
                ):
                    inputs = witness_inputs
                    outputs = witness_outputs
                    signature_source = "witness"
            inputs = _enrich_ports_with_witness(inputs, witness_inputs)
            outputs = _enrich_ports_with_witness(outputs, witness_outputs)
            parsed[atom_name] = {
                "impl": None,
                "witness": None,
                "doc": doc or witness_doc,
                "doc_source": doc_source if (doc or witness_doc) else "",
                "module": module_name,
                "name": atom_name,
                "ast_inputs": inputs,
                "ast_outputs": outputs,
                "signature_source": signature_source,
                "witness_name": witness_name,
            }
    return parsed


def _match_atomic_meta(
    name: str,
    impl: Any,
    atom_index: dict[str, list[_AtomicNodeMeta]],
) -> _AtomicNodeMeta | None:
    candidates = [
        _normalize(name),
        _normalize(name.split(".")[-1]),
        _normalize(getattr(impl, "__name__", "")),
    ]
    for key in candidates:
        if not key:
            continue
        matches = atom_index.get(key)
        if matches:
            return matches[0]
    return None


def _registration_to_primitive(
    source: AtomSource,
    name: str,
    meta: dict[str, Any],
    atom_index: dict[str, list[_AtomicNodeMeta]],
    *,
    report: CatalogReport | None = None,
) -> tuple[AlgorithmicPrimitive, list[str]] | None:
    impl = meta.get("impl")
    witness = meta.get("witness")
    module_name = getattr(impl, "__module__", "") or getattr(witness, "__module__", "")
    meta_doc = str(meta.get("doc", "") or "").strip()
    impl_doc = str(getattr(impl, "__doc__", "") or "").strip()
    witness_doc = str(getattr(witness, "__doc__", "") or "").strip()
    description = str(meta_doc or impl_doc or witness_doc).strip()
    if report is not None:
        if meta.get("doc_source") == "witness" or (not meta_doc and not impl_doc and witness_doc):
            report.source_witness_doc_fallbacks += 1
        if meta.get("signature_source") == "witness":
            report.source_witness_signature_fallbacks += 1

    matched = _match_atomic_meta(name, impl, atom_index)
    if matched is not None:
        if report is not None:
            report.source_cdg_metadata_matches += 1
        def _merge_ast_port_enrichment(
            cdg_ports: list[IOSpec], ast_ports: list[IOSpec]
        ) -> list[IOSpec]:
            if not ast_ports or len(ast_ports) != len(cdg_ports):
                return cdg_ports
            merged: list[IOSpec] = []
            for c_port, a_port in zip(cdg_ports, ast_ports):
                a_pt = a_port.resolved_port_type()
                c_pt = c_port.resolved_port_type()
                new_tags = list(dict.fromkeys([*c_pt.domain_tags, *a_pt.domain_tags]))
                new_ndim = c_pt.ndim if c_pt.ndim is not None else a_pt.ndim
                new_dtype = c_pt.dtype or a_pt.dtype
                updated_pt = c_pt.model_copy(
                    update={
                        "domain_tags": new_tags,
                        "ndim": new_ndim,
                        "dtype": new_dtype,
                    }
                )
                merged.append(c_port.model_copy(update={"port_type": updated_pt}))
            return merged

        cdg_inputs = _merge_ast_port_enrichment(list(matched.inputs), list(meta.get("ast_inputs") or []))
        cdg_outputs = _merge_ast_port_enrichment(list(matched.outputs), list(meta.get("ast_outputs") or []))
        primitive = AlgorithmicPrimitive(
            name=name,
            source=source.name,
            category=matched.concept_type,
            description=matched.description or description or name,
            inputs=cdg_inputs,
            outputs=cdg_outputs,
            type_signature=matched.type_signature or _signature_from_ports(cdg_inputs, cdg_outputs),
            phase=matched.phase,
            state_ports=matched.state_ports,
            semantic_role=matched.semantic_role,
            time_complexity=matched.time_complexity,
            space_complexity=matched.space_complexity,
            aliases=matched.aliases,
            common_patterns=matched.common_patterns,
        )
        aliases = [
            alias
            for alias in {matched.node_id, matched.name, matched.matched_primitive, *matched.aliases}
            if alias and alias != name
        ]
        return primitive, aliases

    if meta.get("ast_inputs") is not None:
        inputs = meta.get("ast_inputs", [])
        outputs = meta.get("ast_outputs", []) or [IOSpec(name="result", type_desc="Any")]
    else:
        impl_inputs, impl_outputs = _ports_from_callable(impl) if impl is not None else ([], [])
        witness_inputs, witness_outputs = _ports_from_callable(witness) if witness is not None else ([], [])
        if impl is not None and not _ports_are_uninformative(impl_inputs, impl_outputs):
            inputs = _enrich_ports_with_witness(impl_inputs, witness_inputs)
            outputs = _enrich_ports_with_witness(impl_outputs, witness_outputs)
        elif witness is not None and not _ports_are_uninformative(witness_inputs, witness_outputs):
            inputs = _enrich_ports_with_witness(witness_inputs, witness_inputs)
            outputs = _enrich_ports_with_witness(witness_outputs, witness_outputs)
            if report is not None:
                report.source_witness_signature_fallbacks += 1
        else:
            fallback_fn = impl or witness
            raw_in, raw_out = _ports_from_callable(fallback_fn)
            inputs = _enrich_ports_with_witness(raw_in, witness_inputs)
            outputs = _enrich_ports_with_witness(raw_out, witness_outputs)
    primitive = AlgorithmicPrimitive(
        name=name,
        source=source.name,
        category=_infer_concept_type(
            name=name,
            module=module_name or str(meta.get("module", "")),
            description=description,
        ),
        description=description or f"Registered atom from {module_name or source.package}",
        inputs=inputs,
        outputs=outputs,
        type_signature=_signature_from_ports(inputs, outputs),
    )
    aliases = sorted(
        _alias_candidates(
            name,
            impl,
            witness=witness,
            witness_name=str(meta.get("witness_name", "") or ""),
            module_name=module_name or str(meta.get("module", "") or ""),
        )
    )
    return primitive, aliases


def _alias_candidates(
    name: str,
    impl: Any,
    *,
    witness: Any = None,
    witness_name: str = "",
    module_name: str = "",
) -> set[str]:
    aliases = {
        alias
        for alias in {
            getattr(impl, "__name__", ""),
            getattr(witness, "__name__", ""),
            witness_name,
            name.split(".")[-1],
        }
        if alias and alias != name
    }
    expanded: set[str] = set()
    for alias in aliases:
        expanded.add(alias)
        if alias.startswith("witness_") and len(alias) > len("witness_"):
            expanded.add(alias[len("witness_") :])
    aliases = expanded
    parts = [part for part in name.split(".") if part]
    if len(parts) >= 2:
        suffix = parts[-2:]
        aliases.add(".".join(suffix))
        aliases.add(" ".join(suffix))
        aliases.add("_".join(suffix))
    module_parts = [part for part in module_name.split(".") if part]
    leaf_name = parts[-1] if parts else name
    if leaf_name:
        if module_parts:
            module_leaf = module_parts[-1]
            aliases.add(f"{module_leaf}.{leaf_name}")
            aliases.add(f"{module_leaf} {leaf_name}")
            aliases.add(f"{module_leaf}_{leaf_name}")
            aliases.add(f"{module_name}.{leaf_name}")
        if len(module_parts) >= 2:
            module_suffix = module_parts[-2:]
            dotted = ".".join([*module_suffix, leaf_name])
            spaced = " ".join([*module_suffix, leaf_name])
            underscored = "_".join([*module_suffix, leaf_name])
            aliases.add(dotted)
            aliases.add(spaced)
            aliases.add(underscored)
    return {alias for alias in aliases if alias and alias != name}


def _add_primitive_with_aliases(
    catalog: PrimitiveCatalog,
    primitive: AlgorithmicPrimitive,
    aliases: list[str],
    *,
    skill_index: SkillIndex | None = None,
    dedup_threshold: float = 0.85,
    report: CatalogReport | None = None,
) -> bool:
    """Insert *primitive* via dedup, register aliases.  Returns True if added."""
    result = catalog.add_with_dedup(
        primitive, skill_index, dedup_threshold, report
    )
    canonical = result.incumbent_name if result.is_duplicate else primitive.name
    for alias in aliases:
        try:
            catalog.add_alias(alias, canonical)
        except KeyError:
            pass
    if skill_index is not None and not result.is_duplicate:
        skill_index.add_primitive(primitive)
    return not result.is_duplicate


def _sqlite_table_names(con: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _sqlite_table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {
            str(row[1])
            for row in con.execute(f"PRAGMA table_info({table})").fetchall()
        }
    except sqlite3.Error:
        return set()


def _manifest_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _manifest_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _manifest_text(value).lower() in {"1", "true", "t", "yes", "y", "on"}


def _manifest_license_allowed(row: sqlite3.Row | dict[str, Any]) -> bool:
    policy = load_license_policy_from_env(
        developer_mode=False,
        default_allow_unknown=True,
        default_enforce_status=False,
    )
    if not policy.enabled:
        return True
    keys = row.keys() if hasattr(row, "keys") else set(row)
    if "license_expression" not in keys and "license_status" not in keys and "license_family" not in keys:
        return True
    return policy.permits(
        license_expression=_manifest_text(row["license_expression"]) if "license_expression" in keys else "",
        license_status=_manifest_text(row["license_status"]) if "license_status" in keys else "unknown",
        license_family=_manifest_text(row["license_family"]) if "license_family" in keys else "unknown",
    )


def _manifest_description_map(
    con: sqlite3.Connection,
    tables: set[str],
) -> dict[str, str]:
    if "descriptions" not in tables:
        return {}

    descriptions: dict[str, str] = {}
    try:
        rows = con.execute(
            """
            SELECT atom_id, content
            FROM descriptions
            WHERE content IS NOT NULL AND content != ''
            ORDER BY atom_id ASC,
                     CASE WHEN kind = 'dejargonized' THEN 0 ELSE 1 END,
                     reviewed DESC,
                     updated_at DESC,
                     description_id ASC
            """
        ).fetchall()
    except sqlite3.Error:
        return {}

    for row in rows:
        atom_id = _manifest_text(row["atom_id"])
        content = _manifest_text(row["content"])
        if atom_id and content:
            descriptions.setdefault(atom_id, content)
    return descriptions


def _manifest_io_specs_map(
    con: sqlite3.Connection,
    tables: set[str],
) -> dict[str, tuple[list[IOSpec], list[IOSpec]]]:
    if "io_specs" not in tables:
        return {}

    io_columns = _sqlite_table_columns(con, "io_specs")
    has_port_type = "port_type_json" in io_columns
    select_cols = [
        "atom_id",
        "port_name",
        "direction",
        "ordinal",
        "type_desc",
        "constraints",
        "data_kind",
        "required",
        "default_value_repr",
    ]
    if has_port_type:
        select_cols.append("port_type_json")

    grouped: dict[str, tuple[list[IOSpec], list[IOSpec]]] = {}
    try:
        rows = con.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM io_specs
            ORDER BY atom_id ASC, COALESCE(ordinal, 0) ASC, port_name ASC
            """
        ).fetchall()
    except sqlite3.Error:
        return {}

    for row in rows:
        atom_id = _manifest_text(row["atom_id"])
        if not atom_id:
            continue
        parsed_port_type: PortTypeSpec | None = None
        if has_port_type:
            raw_pt = _manifest_text(row["port_type_json"])
            if raw_pt:
                try:
                    parsed_port_type = PortTypeSpec.model_validate_json(raw_pt)
                except Exception:
                    parsed_port_type = None
        port = IOSpec(
            name=_manifest_text(row["port_name"]),
            type_desc=_manifest_text(row["type_desc"]) or "Any",
            constraints=_manifest_text(row["constraints"]),
            data_kind=_manifest_text(row["data_kind"]),
            required=_manifest_bool(row["required"]),
            default_value_repr=_manifest_text(row["default_value_repr"]),
            port_type=parsed_port_type,
        )
        inputs, outputs = grouped.setdefault(atom_id, ([], []))
        direction = _manifest_text(row["direction"]).lower()
        if direction.startswith("out"):
            outputs.append(port)
        else:
            inputs.append(port)
    return grouped


def _manifest_concept_type(
    *,
    fqdn: str,
    description: str,
    domain_tags: str,
    source_kind: str,
    source_package: str,
    source_module_path: str,
    namespace_root: str,
    source_symbol: str,
) -> ConceptType:
    module_hint = ".".join(
        part for part in (namespace_root, source_package, source_module_path) if part
    )
    semantic_hints = " ".join(
        part
        for part in (
            domain_tags,
            source_kind,
            source_package,
            source_module_path,
            source_symbol,
        )
        if part
    )
    return _infer_concept_type(
        name=fqdn,
        module=module_hint,
        description=f"{description} {semantic_hints}".strip(),
    )


def seed_catalog_from_manifest_sqlite(
    catalog: PrimitiveCatalog,
    manifest_path: Path,
    *,
    skip_locally_installed: bool = True,
    allowed_namespaces: Sequence[str] | None = None,
    dedup_threshold: float = 0.85,
    skill_index: SkillIndex | None = None,
    report: CatalogReport | None = None,
) -> int:
    """Seed the catalog from approved atoms in a manifest SQLite snapshot."""
    path = Path(manifest_path)
    if not path.is_file():
        logger.warning("Manifest SQLite not found: %s", path)
        return 0

    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("Failed to open manifest SQLite %s: %s", path, exc)
        return 0

    con.row_factory = sqlite3.Row
    try:
        tables = _sqlite_table_names(con)
        if "atoms" not in tables:
            logger.warning("Manifest SQLite missing atoms table: %s", path)
            return 0

        descriptions_by_atom = _manifest_description_map(con, tables)
        io_specs_by_atom = _manifest_io_specs_map(con, tables)
        atom_columns = _sqlite_table_columns(con, "atoms")
        select_columns = [
            "atom_id",
            "fqdn",
            "status",
            "domain_tags",
            "description",
            "visibility_tier",
            "source_kind",
            "namespace_root",
            "source_repo_id",
            "source_package",
            "source_module_path",
            "source_symbol",
        ]
        for optional in (
            "license_expression",
            "license_status",
            "license_family",
            "concept_type",
            "phase",
            "type_signature",
            "state_ports_json",
            "time_complexity",
            "space_complexity",
        ):
            if optional in atom_columns:
                select_columns.append(optional)
        rows = con.execute(
            f"""
            SELECT {", ".join(select_columns)}
            FROM atoms
            WHERE status = 'approved'
            ORDER BY fqdn ASC
            """
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Failed to read manifest SQLite %s: %s", path, exc)
        return 0
    finally:
        con.close()

    normalized_namespaces = tuple(
        ns.strip().rstrip(".")
        for ns in (allowed_namespaces or ())
        if ns and ns.strip()
    )

    added = 0
    for row in rows:
        fqdn = _manifest_text(row["fqdn"])
        if not fqdn:
            continue
        if normalized_namespaces:
            ns_root = _manifest_text(row["namespace_root"])
            src_pkg = _manifest_text(row["source_package"])
            candidates = [fqdn]
            if src_pkg:
                candidates.append(src_pkg)
                if ns_root and not src_pkg.startswith(ns_root):
                    candidates.append(f"{ns_root}.{src_pkg}")
            if not any(
                cand == ns or cand.startswith(f"{ns}.")
                for cand in candidates
                for ns in normalized_namespaces
            ):
                continue
        if not _manifest_license_allowed(row):
            continue
        if skip_locally_installed and catalog.get(fqdn) is not None:
            continue

        atom_id = _manifest_text(row["atom_id"])
        description = (
            descriptions_by_atom.get(atom_id)
            or _manifest_text(row["description"])
            or fqdn
        )
        inputs, outputs = io_specs_by_atom.get(atom_id, ([], []))
        source_repo_id = _manifest_text(row["source_repo_id"])
        source_package = _manifest_text(row["source_package"])
        row_keys = set(row.keys())
        explicit_concept: ConceptType | None = None
        if "concept_type" in row_keys:
            raw_ct = _manifest_text(row["concept_type"])
            if raw_ct:
                try:
                    explicit_concept = ConceptType(raw_ct)
                except ValueError:
                    explicit_concept = None
        category = explicit_concept or _manifest_concept_type(
            fqdn=fqdn,
            description=description,
            domain_tags=_manifest_text(row["domain_tags"]),
            source_kind=_manifest_text(row["source_kind"]),
            source_package=source_package,
            source_module_path=_manifest_text(row["source_module_path"]),
            namespace_root=_manifest_text(row["namespace_root"]),
            source_symbol=_manifest_text(row["source_symbol"]),
        )
        phase = (
            _manifest_text(row["phase"])
            if "phase" in row_keys and _manifest_text(row["phase"])
            else "stateless"
        )
        state_ports: list[StatePortSpec] = []
        if "state_ports_json" in row_keys:
            raw_sp = _manifest_text(row["state_ports_json"])
            if raw_sp:
                try:
                    parsed_sp = json.loads(raw_sp)
                    if isinstance(parsed_sp, list):
                        state_ports = [
                            StatePortSpec.model_validate(item)
                            for item in parsed_sp
                            if isinstance(item, dict)
                        ]
                except Exception:
                    state_ports = []
        stored_sig = (
            _manifest_text(row["type_signature"])
            if "type_signature" in row_keys
            else ""
        )
        time_complexity = (
            _manifest_text(row["time_complexity"])
            if "time_complexity" in row_keys
            else ""
        )
        space_complexity = (
            _manifest_text(row["space_complexity"])
            if "space_complexity" in row_keys
            else ""
        )
        primitive = AlgorithmicPrimitive(
            name=fqdn,
            source=(
                f"manifest:{source_repo_id}"
                if source_repo_id
                else f"manifest:{source_package}"
                if source_package
                else "manifest"
            ),
            category=category,
            description=description,
            inputs=inputs,
            outputs=outputs,
            type_signature=stored_sig or _manifest_signature_from_ports(inputs, outputs),
            phase=phase,
            state_ports=state_ports,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
        )

        if report is not None:
            report.total_candidates += 1
            _bump_source_breakdown(
                report,
                source_repo_id or source_package or "manifest",
                "manifest_candidates",
            )

        if skill_index is not None:
            result = catalog.add_with_dedup(
                primitive,
                skill_index,
                dedup_threshold,
                report=None,
            )
            added_here = not result.is_duplicate
            if report is not None:
                if added_here:
                    report.added += 1
                else:
                    report.merged += 1
                    report.merge_details.append(
                        (
                            primitive.name,
                            result.incumbent_name or primitive.name,
                            result.similarity,
                        )
                    )
        else:
            catalog.add(primitive)
            added_here = True
            if report is not None:
                report.added += 1

        if added_here:
            added += 1
            if report is not None:
                _bump_source_breakdown(
                    report,
                    source_repo_id or source_package or "manifest",
                    "added",
                )

    return added


def seed_catalog_from_sources(
    catalog: PrimitiveCatalog,
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
    include_live_registries: bool = True,
    skill_index: SkillIndex | None = None,
    dedup_threshold: float = 0.85,
    report: CatalogReport | None = None,
) -> int:
    """Add architect primitives derived from configured source registrations."""
    if config is None:
        config = load_sources()

    added = 0
    seen_topo_hashes: dict[str, str] = {}  # topo_hash -> first source name

    for source in config.sources:
        package_root = resolve_package_root(source, base_dir).expanduser().resolve()
        atom_index, source_topo_hashes = _load_atomic_node_index(
            source, base_dir=base_dir
        )

        # Structural de-duplication: skip CDG subtrees already seen.
        structural_skips: set[str] = set()
        for h, src_name in source_topo_hashes.items():
            if h in seen_topo_hashes and seen_topo_hashes[h] != src_name:
                logger.debug(
                    "Structural duplicate: topo_hash %s from '%s' "
                    "already seen from '%s'",
                    h,
                    src_name,
                    seen_topo_hashes[h],
                )
                structural_skips.add(h)
                if report is not None:
                    report.structural_skips += 1
            else:
                seen_topo_hashes[h] = src_name

        ast_entries = _parse_register_atom_functions(
            package_root,
            source.package,
            excluded_prefixes=excluded_import_prefixes(source, base_dir),
        )
        seen_names: set[str] = set()
        if include_live_registries:
            for name, meta in _live_registry_entries_for_source(
                source, package_root=package_root, base_dir=base_dir
            ).items():
                if name in seen_names:
                    continue
                if report is not None:
                    report.source_live_registry_candidates += 1
                _bump_source_breakdown(report, source.name, "live_registry_candidates")
                built = _registration_to_primitive(
                    source,
                    name,
                    meta,
                    atom_index,
                    report=report,
                )
                if built is None:
                    continue
                primitive, aliases = built
                if _add_primitive_with_aliases(
                    catalog,
                    primitive,
                    aliases,
                    skill_index=skill_index,
                    dedup_threshold=dedup_threshold,
                    report=report,
                ):
                    added += 1
                    _bump_source_breakdown(report, source.name, "added")
                seen_names.add(name)

        for name, meta in ast_entries.items():
            if name in seen_names:
                continue
            if report is not None:
                report.source_ast_candidates += 1
            _bump_source_breakdown(report, source.name, "ast_candidates")
            if catalog.get(name) is not None:
                continue
            built = _registration_to_primitive(
                source,
                name,
                meta,
                atom_index,
                report=report,
            )
            if built is None:
                continue
            primitive, aliases = built
            if _add_primitive_with_aliases(
                catalog,
                primitive,
                aliases,
                skill_index=skill_index,
                dedup_threshold=dedup_threshold,
                report=report,
            ):
                added += 1
                _bump_source_breakdown(report, source.name, "added")
            seen_names.add(name)

        # Load uncertainty.json files from this source
        if report is not None:
            for uj_path in package_root.rglob("uncertainty.json"):
                atom_name, est = load_uncertainty_json(uj_path)
                if atom_name and est.mode != "unknown":
                    report.uncertainty_estimates[atom_name] = est
    return added


def audit_source_registration_alignment(
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare AST-discovered registrations with live registry registrations."""
    if config is None:
        config = load_sources()

    rows: list[dict[str, Any]] = []
    for source in config.sources:
        registry_error = ""
        ast_names: set[str] = set()
        live_names: set[str] = set()
        try:
            package_root = resolve_package_root(source, base_dir).expanduser().resolve()
            ast_names = set(
                _parse_register_atom_functions(
                    package_root,
                    source.package,
                    excluded_prefixes=excluded_import_prefixes(source, base_dir),
                )
            )
            try:
                live_names = set(
                    _live_registry_entries_for_source(
                        source, package_root=package_root, base_dir=base_dir
                    )
                )
            except Exception as exc:
                registry_error = str(exc)
        except Exception as exc:
            registry_error = str(exc)

        registry_only = sorted(live_names - ast_names)
        ast_only = sorted(ast_names - live_names)
        severity = "healthy"
        if registry_error:
            severity = "critical"
        elif registry_only:
            severity = "high"
        elif ast_only:
            severity = "medium"
        rows.append(
            {
                "source": source.name,
                "ast_count": len(ast_names),
                "live_registry_count": len(live_names),
                "matched_count": len(ast_names & live_names),
                "registry_only_count": len(registry_only),
                "ast_only_count": len(ast_only),
                "registry_only_examples": registry_only[:5],
                "ast_only_examples": ast_only[:5],
                "registry_error": registry_error,
                "severity": severity,
            }
        )

    drift_sources = sorted(
        row["source"]
        for row in rows
        if row["registry_error"]
        or int(row["registry_only_count"]) > 0
        or int(row["ast_only_count"]) > 0
    )
    severity_counts = {"healthy": 0, "medium": 0, "high": 0, "critical": 0}
    for row in rows:
        severity = str(row.get("severity", "healthy") or "healthy")
        if severity not in severity_counts:
            continue
        severity_counts[severity] += 1
    highest_severity = "healthy"
    for level in ("critical", "high", "medium", "healthy"):
        if severity_counts[level] > 0:
            highest_severity = level
            break
    return {
        "rows": rows,
        "source_count": len(rows),
        "matched_total": sum(int(row["matched_count"]) for row in rows),
        "registry_only_total": sum(int(row["registry_only_count"]) for row in rows),
        "ast_only_total": sum(int(row["ast_only_count"]) for row in rows),
        "drift_sources": drift_sources,
        "registry_error_sources": sorted(row["source"] for row in rows if row["registry_error"]),
        "severity_counts": severity_counts,
        "highest_severity": highest_severity,
    }
