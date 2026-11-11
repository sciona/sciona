"""Two-plane compiled local catalog, PEP 420 namespace scoping, and FTS5 index.

Plane 1 (in-git, per PEP 420 provider wheel):
  Static code metadata — normalized ``cdg.json`` files, ``@register_atom``
  registrations, concrete Python type annotations, ``PortTypeSpec``s,
  ``StatePortSpec``s, and ``ConceptType`` categories.

Plane 2 (out-of-git, dynamic empirical metadata):
  Dataset benchmarks, multi-regime ``AtomUncertaintyProfile``s, dejargonized
  descriptions, and audit rollups synced from the platform or local evaluation
  harnesses into ``~/.sciona/manifest.sqlite`` without requiring git commits or
  code reviews in atom repositories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from sciona.api.snapshot import (
    _insert_benchmark,
    _insert_description,
    _insert_manifest_metadata,
    _insert_uncertainty_profile,
    _manifest_content_hash,
    generate_manifest_sqlite,
    rebuild_catalog_fts,
)
from sciona.architect.catalog import CatalogReport, PrimitiveCatalog
from sciona.architect.hyperparams import load_manifest as load_hyperparams_manifest
from sciona.architect.models import (
    AlgorithmicPrimitive,
    IOSpec,
    PortTypeSpec,
    PrimitiveParamSpec,
    split_top_level_type_args,
)
from sciona.architect.source_catalog import (
    _sqlite_table_columns,
    _sqlite_table_names,
    seed_catalog_from_sources,
)
from sciona.sources import (
    SourcesConfig,
    load_sources,
    resolve_package_root,
    resolve_source,
)
from sciona.types import Declaration, Prover

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def discover_installed_atom_namespaces(
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
    include_sources_yml: bool = True,
) -> tuple[str, ...]:
    """Return installed PEP 420 ``sciona.atoms.<ns>`` namespace prefixes.

    Inspects both ``sys.path`` (via ``importlib.util.find_spec("sciona.atoms")``)
    and optionally locally present repositories configured in ``sources.yml``.
    """
    discovered: set[str] = set()
    try:
        spec = importlib.util.find_spec("sciona.atoms")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None

    if spec is not None and spec.submodule_search_locations:
        for raw_loc in spec.submodule_search_locations:
            loc = Path(raw_loc)
            if not loc.is_dir():
                continue
            for child in loc.iterdir():
                name = child.name
                if name.startswith("_") or name.startswith("."):
                    continue
                if child.is_dir():
                    discovered.add(f"sciona.atoms.{name}")
                elif child.suffix == ".py" and child.stem not in {"__init__", "provider_inventory"}:
                    discovered.add(f"sciona.atoms.{child.stem}")

    if include_sources_yml:
        try:
            cfg = config if config is not None else load_sources()
            for source in cfg.sources:
                pkg_dir = resolve_package_root(source, base_dir)
                if not pkg_dir.is_dir() or not source.package:
                    continue
                pkg_name = source.package.strip().rstrip(".")
                if pkg_name == "sciona.atoms":
                    for child in pkg_dir.iterdir():
                        cname = child.name
                        if cname.startswith("_") or cname.startswith("."):
                            continue
                        if child.is_dir():
                            discovered.add(f"sciona.atoms.{cname}")
                        elif child.suffix == ".py" and child.stem not in {"__init__", "provider_inventory"}:
                            discovered.add(f"sciona.atoms.{child.stem}")
                else:
                    discovered.add(pkg_name)
        except Exception:
            pass

    return tuple(sorted(discovered))


def atom_matches_namespaces(
    fqdn: str,
    namespaces: Sequence[str],
    *,
    namespace_root: str = "",
    source_package: str = "",
) -> bool:
    """Return True if *fqdn* belongs to any prefix in *namespaces*."""
    clean_ns = [ns.strip().rstrip(".") for ns in namespaces if ns and ns.strip()]
    if not clean_ns:
        return True
    candidates = [fqdn.strip()]
    pkg = source_package.strip().rstrip(".")
    root = namespace_root.strip().rstrip(".")
    if pkg:
        candidates.append(pkg)
        if root and not pkg.startswith(root):
            candidates.append(f"{root}.{pkg}")
    return any(
        cand == ns or cand.startswith(f"{ns}.")
        for cand in candidates
        if cand
        for ns in clean_ns
    )


def prune_manifest_to_namespaces(
    manifest_path: Path,
    namespaces: Sequence[str],
) -> dict[str, int]:
    """Scope *manifest_path* in-place to atoms matching *namespaces*."""
    path = Path(manifest_path)
    clean_ns = [ns.strip().rstrip(".") for ns in namespaces if ns and ns.strip()]
    if not path.is_file() or not clean_ns:
        return {"kept_atoms": 0, "pruned_atoms": 0}

    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        tables = _sqlite_table_names(con)
        if "atoms" not in tables:
            return {"kept_atoms": 0, "pruned_atoms": 0}

        atom_cols = _sqlite_table_columns(con, "atoms")
        has_ns_root = "namespace_root" in atom_cols
        has_src_pkg = "source_package" in atom_cols
        rows = con.execute(
            f"""
            SELECT atom_id, fqdn,
                   {"namespace_root" if has_ns_root else "'' AS namespace_root"},
                   {"source_package" if has_src_pkg else "'' AS source_package"}
            FROM atoms
            """
        ).fetchall()

        keep_ids: set[str] = set()
        keep_fqdns: set[str] = set()
        prune_ids: list[str] = []
        prune_fqdns: list[str] = []

        for row in rows:
            aid = str(row["atom_id"] or "")
            fqdn = str(row["fqdn"] or "")
            if atom_matches_namespaces(
                fqdn,
                clean_ns,
                namespace_root=str(row["namespace_root"] or ""),
                source_package=str(row["source_package"] or ""),
            ):
                keep_ids.add(aid)
                keep_fqdns.add(fqdn)
            else:
                if aid:
                    prune_ids.append(aid)
                if fqdn:
                    prune_fqdns.append(fqdn)

        if prune_ids:
            placeholders = ",".join("?" for _ in prune_ids)
            con.execute(f"DELETE FROM atoms WHERE atom_id IN ({placeholders})", prune_ids)
            for dep_table in ("io_specs", "hyperparams", "audit_rollups", "descriptions"):
                if dep_table in tables:
                    con.execute(
                        f"DELETE FROM {dep_table} WHERE atom_id IN ({placeholders})",
                        prune_ids,
                    )

        if prune_fqdns:
            fqdn_placeholders = ",".join("?" for _ in prune_fqdns)
            for fqdn_table in ("benchmarks", "uncertainty_profiles"):
                if fqdn_table in tables:
                    con.execute(
                        f"DELETE FROM {fqdn_table} WHERE atom_fqdn IN ({fqdn_placeholders})",
                        prune_fqdns,
                    )

        if "manifest_metadata" in tables:
            _insert_manifest_metadata(
                con,
                {
                    "scoped_namespaces": ",".join(sorted(set(clean_ns))),
                    "content_hash": _manifest_content_hash(
                        [{"fqdn": fqdn} for fqdn in sorted(keep_fqdns)]
                    ),
                },
            )

        rebuild_catalog_fts(con)
        con.commit()
        return {"kept_atoms": len(keep_fqdns), "pruned_atoms": len(prune_fqdns)}
    finally:
        con.close()


def _ensure_uncertainty_profiles_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS uncertainty_profiles (
            atom_fqdn TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL DEFAULT 2,
            profile_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (atom_fqdn, content_hash)
        )
        """
    )


def upsert_plane2_metadata(
    manifest_path: Path,
    *,
    benchmarks: Sequence[Mapping[str, Any]] = (),
    uncertainty_profiles: Sequence[Mapping[str, Any]] = (),
    descriptions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, int]:
    """Upsert dynamic Plane-2 metadata into *manifest_path* with zero git churn."""
    path = Path(manifest_path)
    if not path.is_file():
        con = generate_manifest_sqlite([], [], output_path=path)
        con.close()

    con = sqlite3.connect(str(path))
    try:
        _ensure_uncertainty_profiles_table(con)
        for bm in benchmarks:
            _insert_benchmark(con, bm)
        for up in uncertainty_profiles:
            _insert_uncertainty_profile(con, up)
        for desc in descriptions:
            _insert_description(con, desc)
        if descriptions:
            rebuild_catalog_fts(con)
        con.commit()
        return {
            "benchmarks_upserted": len(benchmarks),
            "uncertainty_profiles_upserted": len(uncertainty_profiles),
            "descriptions_upserted": len(descriptions),
        }
    finally:
        con.close()


def load_uncertainty_profiles_from_manifest(
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load Plane-2 ``AtomUncertaintyProfile`` payloads keyed by ``atom_fqdn``."""
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = _sqlite_table_names(con)
        if "uncertainty_profiles" not in tables:
            return {}
        rows = con.execute(
            """
            SELECT atom_fqdn, profile_json
            FROM uncertainty_profiles
            ORDER BY atom_fqdn ASC, updated_at DESC, content_hash DESC
            """
        ).fetchall()
        profiles: dict[str, dict[str, Any]] = {}
        for row in rows:
            fqdn = str(row["atom_fqdn"] or "").strip()
            raw = str(row["profile_json"] or "").strip()
            if not fqdn or not raw or fqdn in profiles:
                continue
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    profiles[fqdn] = parsed
            except json.JSONDecodeError:
                continue
        return profiles
    finally:
        con.close()


def _read_existing_plane2_rows(db_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    empty: dict[str, list[dict[str, Any]]] = {
        "benchmarks": [],
        "uncertainty_profiles": [],
        "descriptions": [],
        "rollups": [],
    }
    if db_path is None or not Path(db_path).is_file():
        return empty
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = _sqlite_table_names(con)
        if "benchmarks" in tables:
            empty["benchmarks"] = [
                dict(r) for r in con.execute("SELECT * FROM benchmarks").fetchall()
            ]
        if "uncertainty_profiles" in tables:
            empty["uncertainty_profiles"] = [
                dict(r)
                for r in con.execute("SELECT * FROM uncertainty_profiles").fetchall()
            ]
        if "descriptions" in tables:
            empty["descriptions"] = [
                dict(r) for r in con.execute("SELECT * FROM descriptions").fetchall()
            ]
        if "audit_rollups" in tables:
            empty["rollups"] = [
                dict(r) for r in con.execute("SELECT * FROM audit_rollups").fetchall()
            ]
        return empty
    except sqlite3.Error:
        return empty
    finally:
        con.close()


def _deterministic_atom_id(fqdn: str) -> str:
    return f"local:{hashlib.sha256(fqdn.encode('utf-8')).hexdigest()[:16]}"


def _resolve_manifest_fqdn(prim: AlgorithmicPrimitive, catalog: PrimitiveCatalog) -> str:
    if "." in prim.name:
        return prim.name
    aliases = [
        alias
        for alias, target in getattr(catalog, "_aliases", {}).items()
        if target == prim.name and "." in alias
    ]
    sciona_aliases = [a for a in aliases if a.startswith("sciona.atoms.")]
    if sciona_aliases:
        return max(sciona_aliases, key=len)
    if aliases:
        return max(aliases, key=len)
    return prim.name


def _primitive_to_manifest_rows(
    prim: AlgorithmicPrimitive,
    tunables: Sequence[PrimitiveParamSpec],
    *,
    fqdn_override: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fqdn = fqdn_override or prim.name
    atom_id = _deterministic_atom_id(fqdn)
    parts = [p for p in fqdn.split(".") if p]
    namespace_root = "sciona.atoms" if fqdn.startswith("sciona.atoms.") else (parts[0] if parts else "")
    source_package = ".".join(parts[:3]) if fqdn.startswith("sciona.atoms.") and len(parts) >= 3 else ".".join(parts[:-1])
    source_symbol = parts[-1] if parts else fqdn
    domain_tags: set[str] = {prim.category.value}
    for port in [*prim.inputs, *prim.outputs]:
        pt = port.resolved_port_type()
        domain_tags.update(pt.domain_tags)

    atom_row = {
        "atom_id": atom_id,
        "fqdn": fqdn,
        "status": "approved",
        "domain_tags": sorted(t for t in domain_tags if t),
        "description": prim.description,
        "visibility_tier": "general",
        "source_kind": "local_compiled",
        "stateful_kind": "explicit_state" if prim.state_ports else ("phased" if prim.phase != "stateless" else "none"),
        "namespace_root": namespace_root,
        "source_repo_id": prim.source,
        "source_package": source_package,
        "source_symbol": source_symbol,
        "is_publishable": True,
        "concept_type": prim.category.value,
        "phase": prim.phase,
        "type_signature": prim.type_signature,
        "state_ports": [sp.model_dump(mode="json") for sp in prim.state_ports],
        "time_complexity": prim.time_complexity,
        "space_complexity": prim.space_complexity,
    }

    io_rows: list[dict[str, Any]] = []
    for idx, port in enumerate(prim.inputs):
        pt = port.resolved_port_type()
        io_rows.append(
            {
                "io_spec_id": f"{atom_id}:in:{port.name}",
                "atom_id": atom_id,
                "port_name": port.name,
                "direction": "input",
                "type_desc": port.type_desc,
                "constraints": port.constraints,
                "data_kind": port.data_kind,
                "required": port.required,
                "default_value_repr": port.default_value_repr,
                "ordinal": idx,
                "port_type": pt,
            }
        )
    for idx, port in enumerate(prim.outputs):
        pt = port.resolved_port_type()
        io_rows.append(
            {
                "io_spec_id": f"{atom_id}:out:{port.name}",
                "atom_id": atom_id,
                "port_name": port.name,
                "direction": "output",
                "type_desc": port.type_desc,
                "constraints": port.constraints,
                "data_kind": port.data_kind,
                "required": port.required,
                "default_value_repr": port.default_value_repr,
                "ordinal": idx,
                "port_type": pt,
            }
        )

    hp_rows: list[dict[str, Any]] = []
    for hp in tunables:
        hp_rows.append(
            {
                "hp_id": f"{atom_id}:hp:{hp.name}",
                "atom_id": atom_id,
                "name": hp.name,
                "kind": hp.kind,
                "default_value": hp.default,
                "min_value": hp.min_value,
                "max_value": hp.max_value,
                "step_value": hp.step,
                "log_scale": hp.log_scale,
                "choices_json": hp.choices if hp.choices else None,
                "constraints_json": hp.constraints if hp.constraints else None,
                "semantic_role": hp.semantic_role,
                "status": "approved",
            }
        )

    return atom_row, io_rows, hp_rows


def _discover_static_uncertainty_seeds(
    cfg: SourcesConfig,
    base_dir: Path | None,
) -> list[dict[str, Any]]:
    """Collect optional static ``uncertainty.json`` seed files from local repos."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in cfg.sources:
        pkg_root = resolve_package_root(source, base_dir)
        if not pkg_root.is_dir():
            continue
        for unc_file in sorted(pkg_root.rglob("uncertainty.json")):
            try:
                payload = json.loads(unc_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            version = int(payload.get("version", 1) or 1)
            atom_key = str(payload.get("atom") or "").strip()
            if not atom_key:
                rel = unc_file.parent.relative_to(pkg_root)
                atom_key = ".".join([source.package, *rel.parts]).strip(".")
            elif not atom_key.startswith(source.package) and "." not in atom_key:
                rel = unc_file.parent.relative_to(pkg_root)
                atom_key = ".".join([source.package, *rel.parts, atom_key]).strip(".")
            if not atom_key or atom_key in seen:
                continue
            seen.add(atom_key)
            rows.append(
                {
                    "atom_fqdn": atom_key,
                    "content_hash": "plane1_seed",
                    "version": version,
                    "profile": payload,
                    "updated_at": "",
                }
            )
    return rows


def compile_local_manifest_sqlite(
    output_path: Path,
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
    plane2_source_db: Path | None = None,
    installed_only: bool = True,
    include_live_registries: bool = False,
) -> dict[str, Any]:
    """Compile Plane-1 PEP 420 sources + Plane-2 dynamic overlays into SQLite."""
    out = Path(output_path)
    cfg = config if config is not None else load_sources()
    catalog = PrimitiveCatalog()
    report = CatalogReport()

    seed_catalog_from_sources(
        catalog,
        config=cfg,
        base_dir=base_dir,
        include_live_registries=include_live_registries,
        report=report,
    )

    namespaces = (
        discover_installed_atom_namespaces(
            config=cfg,
            base_dir=base_dir,
            include_sources_yml=True,
        )
        if installed_only
        else ()
    )

    tunables_by_atom: dict[str, list[PrimitiveParamSpec]] = {}
    for source in cfg.sources:
        src_root = resolve_source(source, base_dir)
        hp_manifest = src_root / "data" / "hyperparams" / "manifest.json"
        if hp_manifest.is_file():
            try:
                tunables_by_atom.update(load_hyperparams_manifest(hp_manifest))
            except Exception:
                pass

    atoms_rows: list[dict[str, Any]] = []
    io_rows: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    kept_fqdns: set[str] = set()
    kept_ids: set[str] = set()

    for prim in sorted(catalog.all_primitives(), key=lambda p: p.name):
        fqdn = _resolve_manifest_fqdn(prim, catalog)
        if namespaces and not atom_matches_namespaces(fqdn, namespaces):
            continue
        tunables = tunables_by_atom.get(fqdn) or tunables_by_atom.get(prim.name, ())
        atom_row, prim_ios, prim_hps = _primitive_to_manifest_rows(
            prim,
            tunables,
            fqdn_override=fqdn,
        )
        atoms_rows.append(atom_row)
        io_rows.extend(prim_ios)
        hp_rows.extend(prim_hps)
        kept_fqdns.add(fqdn)
        kept_ids.add(str(atom_row["atom_id"]))

    overlay_path = plane2_source_db if plane2_source_db is not None else (out if out.is_file() else None)
    plane2 = _read_existing_plane2_rows(overlay_path)

    static_unc = _discover_static_uncertainty_seeds(cfg, base_dir)
    merged_unc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in static_unc:
        fqdn = str(row["atom_fqdn"])
        if not namespaces or atom_matches_namespaces(fqdn, namespaces):
            merged_unc[(fqdn, str(row.get("content_hash", "")))] = row
    for row in plane2["uncertainty_profiles"]:
        fqdn = str(row.get("atom_fqdn", ""))
        if fqdn and (not namespaces or atom_matches_namespaces(fqdn, namespaces)):
            merged_unc[(fqdn, str(row.get("content_hash", "")))] = row

    benchmarks = [
        row
        for row in plane2["benchmarks"]
        if not namespaces or atom_matches_namespaces(str(row.get("atom_fqdn", "")), namespaces)
    ]
    descriptions = [
        row
        for row in plane2["descriptions"]
        if not kept_ids or str(row.get("atom_id", "")) in kept_ids
    ]
    rollups = [
        row
        for row in plane2["rollups"]
        if not kept_ids or str(row.get("atom_id", "")) in kept_ids
    ]

    manifest_data: dict[str, Any] = {
        "atoms": atoms_rows,
        "hyperparams": hp_rows,
        "io_specs": io_rows,
        "benchmarks": benchmarks,
        "uncertainty_profiles": list(merged_unc.values()),
        "descriptions": descriptions,
        "rollups": rollups,
        "manifest_metadata": {
            "compiled_plane": "two_plane_local",
            "installed_only": str(bool(installed_only)).lower(),
            "scoped_namespaces": ",".join(namespaces),
        },
    }

    con = generate_manifest_sqlite(manifest_data, output_path=out)
    con.close()

    return {
        "output_path": str(out),
        "atoms_compiled": len(atoms_rows),
        "io_specs_compiled": len(io_rows),
        "hyperparams_compiled": len(hp_rows),
        "benchmarks_merged": len(benchmarks),
        "uncertainty_profiles_merged": len(merged_unc),
        "scoped_namespaces": list(namespaces),
    }


def _parse_query_port_specs(
    type_signature: str,
) -> tuple[list[PortTypeSpec], list[PortTypeSpec]] | None:
    """Parse ``(x: T1, y: T2) -> T3`` or ``T1 -> T2`` into ``PortTypeSpec`` lists."""
    sig = type_signature.strip()
    if "->" not in sig:
        return None
    lhs, rhs = sig.split("->", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    in_specs: list[PortTypeSpec] = []
    if lhs and lhs != "()":
        inner = lhs[1:-1].strip() if lhs.startswith("(") and lhs.endswith(")") else lhs
        if inner:
            for item in split_top_level_type_args(inner):
                type_part = item.split(":", 1)[1].strip() if ":" in item else item.strip()
                in_specs.append(PortTypeSpec.from_type_desc(type_part))

    out_specs: list[PortTypeSpec] = []
    if rhs and rhs != "()":
        inner_out = rhs[1:-1].strip() if rhs.startswith("(") and rhs.endswith(")") else rhs
        if inner_out:
            for item in split_top_level_type_args(inner_out):
                type_part = item.split(":", 1)[1].strip() if ":" in item else item.strip()
                out_specs.append(PortTypeSpec.from_type_desc(type_part))

    return in_specs, out_specs


class ManifestFTSIndex:
    """Fast local ``SemanticIndex`` backed by SQLite FTS5 and ``PortTypeSpec``s.

    Operates directly against ``manifest.sqlite`` in ``<5 ms`` per query with
    zero PyTorch/FAISS/network startup overhead.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        allowed_namespaces: Sequence[str] | None = None,
    ) -> None:
        self._path = Path(manifest_path)
        self._allowed_namespaces = tuple(
            ns.strip().rstrip(".")
            for ns in (allowed_namespaces or ())
            if ns and ns.strip()
        )
        self._con: sqlite3.Connection | None = None
        self._has_fts = False
        self._declarations: dict[str, Declaration] = {}
        self._ports: dict[
            str,
            tuple[tuple[IOSpec, ...], tuple[IOSpec, ...], str],
        ] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.is_file():
            return
        con = sqlite3.connect(
            f"file:{self._path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        con.row_factory = sqlite3.Row
        self._con = con
        tables = _sqlite_table_names(con)
        self._has_fts = "catalog_fts" in tables
        if "atoms" not in tables:
            return

        atom_cols = _sqlite_table_columns(con, "atoms")
        io_cols = _sqlite_table_columns(con, "io_specs") if "io_specs" in tables else set()
        has_pt_json = "port_type_json" in io_cols

        io_by_atom: dict[str, tuple[list[IOSpec], list[IOSpec]]] = {}
        if "io_specs" in tables:
            select_io = (
                "atom_id, port_name, direction, type_desc, constraints, data_kind, "
                "required, default_value_repr"
                + (", port_type_json" if has_pt_json else "")
            )
            for r in con.execute(
                f"SELECT {select_io} FROM io_specs ORDER BY atom_id ASC, COALESCE(ordinal, 0) ASC"
            ).fetchall():
                aid = str(r["atom_id"] or "")
                pt: PortTypeSpec | None = None
                if has_pt_json and r["port_type_json"]:
                    try:
                        pt = PortTypeSpec.model_validate_json(str(r["port_type_json"]))
                    except Exception:
                        pt = None
                spec = IOSpec(
                    name=str(r["port_name"] or ""),
                    type_desc=str(r["type_desc"] or "Any"),
                    constraints=str(r["constraints"] or ""),
                    data_kind=str(r["data_kind"] or ""),
                    required=bool(r["required"]),
                    default_value_repr=str(r["default_value_repr"] or ""),
                    port_type=pt,
                )
                ins, outs = io_by_atom.setdefault(aid, ([], []))
                if str(r["direction"] or "").lower().startswith("out"):
                    outs.append(spec)
                else:
                    ins.append(spec)

        desc_by_atom: dict[str, str] = {}
        if "descriptions" in tables:
            for r in con.execute(
                """
                SELECT atom_id, content FROM descriptions
                WHERE content IS NOT NULL AND content != ''
                ORDER BY atom_id ASC, CASE WHEN kind = 'dejargonized' THEN 0 ELSE 1 END, reviewed DESC
                """
            ).fetchall():
                desc_by_atom.setdefault(str(r["atom_id"] or ""), str(r["content"] or ""))

        select_atom = [
            "atom_id",
            "fqdn",
            "description",
            "namespace_root",
            "source_repo_id",
            "source_package",
            "source_symbol",
        ]
        for opt in ("concept_type", "phase", "type_signature"):
            if opt in atom_cols:
                select_atom.append(opt)

        for r in con.execute(
            f"SELECT {', '.join(select_atom)} FROM atoms WHERE status = 'approved' ORDER BY fqdn ASC"
        ).fetchall():
            fqdn = str(r["fqdn"] or "").strip()
            if not fqdn:
                continue
            ns_root = str(r["namespace_root"] or "")
            src_pkg = str(r["source_package"] or "")
            if self._allowed_namespaces and not atom_matches_namespaces(
                fqdn,
                self._allowed_namespaces,
                namespace_root=ns_root,
                source_package=src_pkg,
            ):
                continue

            aid = str(r["atom_id"] or "")
            ins, outs = io_by_atom.get(aid, ([], []))
            row_keys = set(r.keys())
            sig = str(r["type_signature"] or "") if "type_signature" in row_keys else ""
            if not sig:
                in_str = f"({', '.join(f'{p.name}: {p.type_desc}' for p in ins)})" if ins else "()"
                out_str = (
                    outs[0].type_desc
                    if len(outs) == 1
                    else (f"({', '.join(p.type_desc for p in outs)})" if outs else "Any")
                )
                sig = f"{in_str} -> {out_str}"
            doc = desc_by_atom.get(aid) or str(r["description"] or "")
            phase = str(r["phase"] or "stateless") if "phase" in row_keys else "stateless"
            concept = str(r["concept_type"] or "") if "concept_type" in row_keys else ""

            decl = Declaration(
                name=fqdn,
                type_signature=sig,
                docstring=doc,
                conceptual_summary=concept,
                source_lib=str(r["source_repo_id"] or src_pkg or "manifest"),
                prover=Prover.PYTHON,
                import_module=src_pkg,
                import_symbol=str(r["source_symbol"] or fqdn.split(".")[-1]),
            )
            self._declarations[fqdn] = decl
            self._ports[fqdn] = (tuple(ins), tuple(outs), phase)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "ManifestFTSIndex":
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_declaration(self, name: str) -> Declaration | None:
        self._ensure_loaded()
        return self._declarations.get(name)

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Execute ``<5 ms`` SQLite FTS5 BM25 lexical search over ``catalog_fts``."""
        self._ensure_loaded()
        if k <= 0 or not self._declarations:
            return []
        tokens = [tok.lower() for tok in _FTS_TOKEN_RE.findall(query_text) if len(tok) >= 2]
        if not tokens:
            return []

        if self._con is not None and self._has_fts:
            fts_expr = " OR ".join(f'"{tok}"*' for tok in dict.fromkeys(tokens))
            try:
                rows = self._con.execute(
                    """
                    SELECT fqdn, bm25(catalog_fts, 3.0, 1.5, 1.2, 1.0, 0.8) AS rank_score
                    FROM catalog_fts
                    WHERE catalog_fts MATCH ?
                    ORDER BY rank_score ASC
                    LIMIT ?
                    """,
                    (fts_expr, max(k * 3, 20)),
                ).fetchall()
                results: list[tuple[Declaration, float]] = []
                for row in rows:
                    fqdn = str(row["fqdn"] or "")
                    decl = self._declarations.get(fqdn)
                    if decl is None:
                        continue
                    raw_bm25 = abs(float(row["rank_score"] or 0.0))
                    norm_score = raw_bm25 / (1.0 + raw_bm25)
                    results.append((decl, norm_score))
                    if len(results) >= k:
                        break
                if results:
                    return results
            except sqlite3.Error:
                pass

        # Pure in-memory token overlap fallback if FTS5 was unavailable
        scored: list[tuple[float, Declaration]] = []
        unique_tokens = list(dict.fromkeys(tokens))
        for fqdn, decl in self._declarations.items():
            haystack = f"{fqdn} {decl.docstring} {decl.conceptual_summary} {decl.type_signature}".lower()
            hits = sum(1 for tok in unique_tokens if tok in haystack)
            if hits > 0:
                scored.append((hits / len(unique_tokens), decl))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [(decl, score) for score, decl in scored[:k]]

    def search_compatible_ports(
        self,
        inputs: Sequence[IOSpec | PortTypeSpec],
        outputs: Sequence[IOSpec | PortTypeSpec],
        *,
        phase: str | None = None,
        query_text: str = "",
        k: int = 10,
    ) -> list[tuple[Declaration, float]]:
        """Rank atoms whose ``PortTypeSpec``s are structurally compatible."""
        self._ensure_loaded()
        if k <= 0 or not self._declarations:
            return []

        expected_in = [
            p.resolved_port_type() if isinstance(p, IOSpec) else p for p in inputs
        ]
        expected_out = [
            p.resolved_port_type() if isinstance(p, IOSpec) else p for p in outputs
        ]
        fts_boost: dict[str, float] = {}
        if query_text.strip():
            for decl, score in self.search_by_embedding(query_text, k=max(k * 4, 40)):
                fts_boost[decl.name] = score

        matches: list[tuple[float, Declaration]] = []
        for fqdn, (cand_ins, cand_outs, cand_phase) in self._ports.items():
            if phase is not None and cand_phase != phase:
                continue
            required_cand_ins = [p for p in cand_ins if p.required]
            if not (len(required_cand_ins) <= len(expected_in) <= len(cand_ins)):
                continue
            if expected_out and len(cand_outs) != len(expected_out):
                continue

            compatible = True
            exact_hits = 0
            total_ports = len(expected_in) + len(expected_out)
            for exp_pt, cand_port in zip(expected_in, cand_ins):
                cand_pt = cand_port.resolved_port_type()
                if not exp_pt.is_compatible_with(cand_pt):
                    compatible = False
                    break
                if exp_pt.kind == cand_pt.kind and exp_pt.dtype == cand_pt.dtype:
                    exact_hits += 1

            if not compatible:
                continue
            for cand_port, exp_pt in zip(cand_outs, expected_out):
                cand_pt = cand_port.resolved_port_type()
                if not cand_pt.is_compatible_with(exp_pt):
                    compatible = False
                    break
                if cand_pt.kind == exp_pt.kind and cand_pt.dtype == exp_pt.dtype:
                    exact_hits += 1

            if not compatible:
                continue

            type_score = (exact_hits / total_ports) if total_ports > 0 else 0.5
            score = 0.6 * type_score + 0.4 * fts_boost.get(fqdn, 0.0)
            matches.append((score, self._declarations[fqdn]))

        matches.sort(key=lambda item: (-item[0], item[1].name))
        return [(decl, score) for score, decl in matches[:k]]

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        """Search by ``PortTypeSpec`` structural compatibility or signature text."""
        self._ensure_loaded()
        if k <= 0 or not self._declarations:
            return []
        parsed = _parse_query_port_specs(type_signature)
        if parsed is not None:
            in_specs, out_specs = parsed
            compat = self.search_compatible_ports(in_specs, out_specs, k=k)
            if compat:
                return [decl for decl, _ in compat]

        needle = type_signature.strip().lower()
        if not needle:
            return []
        exact_or_sub = [
            decl
            for decl in self._declarations.values()
            if needle in decl.type_signature.lower()
        ]
        if exact_or_sub:
            return exact_or_sub[:k]
        return [decl for decl, _ in self.search_by_embedding(type_signature, k=k)]
