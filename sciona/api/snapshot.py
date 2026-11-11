"""SQLite snapshot generation from Supabase-backed catalog data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import httpx

from sciona.license_policy import (
    LicensePolicy,
    load_license_policy_from_env,
    summarize_license_rows,
)

DEFAULT_PAGE_SIZE = 1000
DEFAULT_FILTER_BATCH_SIZE = 10
DEFAULT_MANIFEST_TIER = "general"
DEVELOPER_MANIFEST_TIER = "developer"
DEVELOPER_MANIFEST_INCLUDED_TIERS: tuple[str, ...] = ("general", "early_access", "internal")
DEFAULT_MANIFEST_VISIBILITY_TIER = "all"
MANIFEST_TIERS: dict[str, tuple[str, ...]] = {
    "general": ("general",),
    "early_access": ("general", "early_access"),
    "internal": ("general", "early_access", "internal"),
}


def manifest_artifact_key(tier: str) -> str:
    """Return the published object key for a manifest tier."""
    return f"manifests/manifest-{tier}.sqlite"


def _auth_headers(access_token: str) -> dict[str, str]:
    return {
        "apikey": access_token,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _domain_tags_to_text(tags: Any) -> str:
    if isinstance(tags, list):
        return ",".join(str(tag) for tag in tags)
    if tags is None:
        return ""
    return str(tags)


def _chunk(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[i : i + size]) for i in range(0, len(values), size)]


def _postgrest_in_filter(values: Sequence[str]) -> str:
    quoted = ",".join(json.dumps(str(value)) for value in values)
    return f"in.({quoted})"


def _normalize_visibility_tiers(
    visibility_tiers: Sequence[str] | None,
) -> list[str]:
    if visibility_tiers is None:
        return []
    raw = (
        [visibility_tiers]
        if isinstance(visibility_tiers, str)
        else [str(tier) for tier in visibility_tiers]
    )
    normalized: list[str] = []
    for tier in raw:
        tier = str(tier).strip()
        if tier and tier not in normalized:
            normalized.append(tier)
    return normalized


def _manifest_generator_version() -> str:
    try:
        return f"sciona {package_version('sciona')}"
    except PackageNotFoundError:
        return "sciona unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_content_hash(atoms: Sequence[Mapping[str, Any]]) -> str:
    fqdns = sorted(
        str(row.get("fqdn", "")).strip()
        for row in atoms
        if str(row.get("fqdn", "")).strip()
    )
    payload = "\n".join(fqdns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_io_spec_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Supabase or manifest-style IO spec rows to the SQLite shape."""
    data = dict(row)
    port_name = data.get("port_name")
    if not port_name:
        port_name = data.get("name", "")
    port_type_raw = data.get("port_type_json")
    if not port_type_raw and "port_type" in data and data["port_type"] is not None:
        pt: Any = data["port_type"]
        if hasattr(pt, "model_dump"):
            port_type_raw = json.dumps(pt.model_dump(mode="json"), sort_keys=True)
        elif isinstance(pt, Mapping):
            port_type_raw = json.dumps(dict(pt), sort_keys=True)
    return {
        "io_spec_id": data.get("io_spec_id", ""),
        "atom_id": data.get("atom_id", ""),
        "version_id": data.get("version_id"),
        "port_name": port_name,
        "direction": data.get("direction", ""),
        "type_desc": data.get("type_desc", "Any") or "Any",
        "constraints": data.get("constraints", ""),
        "data_kind": data.get("data_kind", ""),
        "required": data.get("required", True),
        "default_value_repr": data.get("default_value_repr", ""),
        "ordinal": data.get("ordinal", 0),
        "port_type_json": _stringify(port_type_raw),
    }


def _normalize_benchmark_row(
    row: Mapping[str, Any],
    *,
    atom_fqdn: str | None = None,
) -> dict[str, Any]:
    benchmark_id = row.get("benchmark_id") or row.get("benchmark_name") or ""
    benchmark_name = row.get("benchmark_name") or benchmark_id
    return {
        "atom_fqdn": atom_fqdn or str(row.get("atom_fqdn", "")),
        "content_hash": str(row.get("content_hash", "")),
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "metric_name": str(row.get("metric_name", "")),
        "metric_value": row.get("metric_value"),
        "dataset_tag": str(row.get("dataset_tag", "")),
        "measured_at": _stringify(row.get("measured_at")),
    }


def _normalize_license_row(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    return {
        "version_id": _stringify(data.get("version_id")),
        "license_expression": _stringify(data.get("license_expression")),
        "license_status": _stringify(data.get("license_status")) or "unknown",
        "license_family": _stringify(data.get("license_family")) or "unknown",
        "license_source_kind": _stringify(data.get("license_source_kind")),
        "license_source_path": _stringify(data.get("license_source_path")),
        "upstream_license_expression": _stringify(data.get("upstream_license_expression")),
        "license_confidence": _stringify(data.get("license_confidence")) or "unknown",
        "license_notes": _stringify(data.get("license_notes")),
    }


def _filter_atoms_by_license_policy(
    atoms: Sequence[dict[str, Any]],
    *,
    policy: LicensePolicy | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    license_rows = [_normalize_license_row(atom) for atom in atoms]
    summary = {
        "license_policy_enforced": bool(policy and policy.enabled and license_rows),
        "license_metadata_present": bool(license_rows),
        "license_row_count": len(license_rows),
        "license_status_counts": summarize_license_rows(license_rows),
    }
    if policy is None or not policy.enabled or not atoms or not license_rows:
        summary["excluded_atoms"] = 0
        return list(atoms), summary

    filtered: list[dict[str, Any]] = []
    excluded = 0
    for atom in atoms:
        if policy.permits(
            license_expression=_stringify(atom.get("license_expression")),
            license_status=_stringify(atom.get("license_status")),
            license_family=_stringify(atom.get("license_family")),
        ):
            filtered.append(atom)
        else:
            excluded += 1
    summary["excluded_atoms"] = excluded
    return filtered, summary


async def _fetch_all_rows(
    base_url: str,
    access_token: str,
    table: str,
    *,
    select: str = "*",
    filters: Mapping[str, str] | None = None,
    order: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    created_client = client is None
    http_client = client or httpx.AsyncClient(timeout=60.0)
    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, str] = {
                "select": select,
                "limit": str(page_size),
                "offset": str(offset),
            }
            if order:
                params["order"] = order
            if filters:
                params.update(filters)

            response = await http_client.get(
                f"{base_url.rstrip('/')}/rest/v1/{table}",
                params=params,
                headers=_auth_headers(access_token),
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise TypeError(f"Supabase table {table!r} did not return a list")
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return rows
    finally:
        if created_client:
            await http_client.aclose()


async def _call_rpc(
    base_url: str,
    access_token: str,
    rpc_name: str,
    payload: Mapping[str, Any] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> Any:
    created_client = client is None
    http_client = client or httpx.AsyncClient(timeout=60.0)
    try:
        response = await http_client.post(
            f"{base_url.rstrip('/')}/rest/v1/rpc/{rpc_name}",
            json=dict(payload or {}),
            headers=_auth_headers(access_token),
        )
        response.raise_for_status()
        return response.json()
    finally:
        if created_client:
            await http_client.aclose()


async def fetch_manifest_data(
    base_url: str,
    access_token: str,
    *,
    visibility_tiers: Sequence[str] | None = None,
    include_unpublished: bool = False,
    license_policy: LicensePolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all manifest inputs from Supabase REST endpoints."""
    requested_tiers = _normalize_visibility_tiers(visibility_tiers)
    atom_filters = {
        "status": "eq.approved",
    }
    if not include_unpublished:
        atom_filters["is_publishable"] = "eq.true"
    if requested_tiers:
        atom_filters["visibility_tier"] = _postgrest_in_filter(requested_tiers)

    atoms = await _fetch_all_rows(
        base_url,
        access_token,
        "atoms",
        select=(
            "atom_id,fqdn,status,domain_tags,description,visibility_tier,"
            "source_kind,stateful_kind,is_stochastic,is_ffi,namespace_root,"
            "namespace_path,source_repo_id,source_package,source_module_path,"
            "source_symbol,is_publishable"
        ),
        filters=atom_filters,
        order="fqdn.asc",
        client=client,
    )

    version_rows = await _fetch_all_rows(
        base_url,
        access_token,
        "atom_versions",
        select="version_id,atom_id,content_hash,is_latest,semver",
        order="atom_id.asc,is_latest.desc,version_id.asc",
        client=client,
    )
    latest_versions_by_atom: dict[str, dict[str, Any]] = {}
    for row in version_rows:
        atom_id = str(row.get("atom_id", "") or "")
        if not atom_id or atom_id in latest_versions_by_atom:
            continue
        latest_versions_by_atom[atom_id] = row

    license_rows: list[dict[str, Any]] = []
    try:
        source_rows = await _fetch_all_rows(
            base_url,
            access_token,
            "atom_version_license_metadata",
            select=(
                "version_id,license_expression,license_status,license_family,"
                "license_source_kind,license_source_path,upstream_license_expression,"
                "license_confidence,license_notes"
            ),
            order="version_id.asc",
            client=client,
        )
        license_rows = [_normalize_license_row(row) for row in source_rows]
    except httpx.HTTPStatusError:
        license_rows = []

    license_by_version = {
        str(row["version_id"]): row
        for row in license_rows
        if str(row.get("version_id", "")).strip()
    }
    atoms_with_versions: list[dict[str, Any]] = []
    for row in atoms:
        atom_id = str(row.get("atom_id", "") or "")
        version = latest_versions_by_atom.get(atom_id, {})
        license_row = license_by_version.get(str(version.get("version_id", "") or ""), {})
        atoms_with_versions.append(
            {
                **row,
                "content_hash": _stringify(version.get("content_hash")),
                "version_id": _stringify(version.get("version_id")),
                "semver": _stringify(version.get("semver")),
                "license_expression": _stringify(license_row.get("license_expression")),
                "license_status": _stringify(license_row.get("license_status")) or "unknown",
                "license_family": _stringify(license_row.get("license_family")) or "unknown",
                "license_source_kind": _stringify(license_row.get("license_source_kind")),
                "license_source_path": _stringify(license_row.get("license_source_path")),
                "upstream_license_expression": _stringify(
                    license_row.get("upstream_license_expression")
                ),
                "license_confidence": _stringify(license_row.get("license_confidence")) or "unknown",
                "license_notes": _stringify(license_row.get("license_notes")),
            }
        )
    atoms, license_summary = _filter_atoms_by_license_policy(
        atoms_with_versions,
        policy=license_policy,
    )

    atom_ids = [str(row["atom_id"]) for row in atoms if row.get("atom_id")]
    atom_id_set = set(atom_ids)
    atom_fqdns = {str(row["fqdn"]) for row in atoms if row.get("fqdn")}
    hyperparams: list[dict[str, Any]] = []
    rollups: list[dict[str, Any]] = []
    descriptions: list[dict[str, Any]] = []
    io_specs: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []

    if not atom_ids:
        return {
            "atoms": atoms,
            "hyperparams": hyperparams,
            "benchmarks": benchmarks,
            "rollups": rollups,
            "descriptions": descriptions,
            "io_specs": io_specs,
            "manifest_metadata": license_summary,
        }

    hyperparam_rows = await _fetch_all_rows(
        base_url,
        access_token,
        "hyperparams",
        select=(
            "hp_id,atom_id,name,kind,default_value,min_value,max_value,"
            "step_value,log_scale,choices_json,constraints_json,"
            "semantic_role,status"
        ),
        filters={"status": "eq.approved"},
        order="atom_id.asc,name.asc",
        client=client,
    )
    hyperparams.extend(
        row for row in hyperparam_rows if str(row.get("atom_id", "")) in atom_id_set
    )

    rollup_rows = await _fetch_all_rows(
        base_url,
        access_token,
        "atom_audit_rollups",
        select=(
            "atom_id,overall_verdict,structural_status,runtime_status,"
            "semantic_status,developer_semantics_status,risk_tier,"
            "risk_score,risk_dimensions,risk_reasons,acceptability_score,"
            "acceptability_band,parity_coverage_level,parity_test_status,"
            "parity_fixture_count,parity_case_count,review_status,"
            "review_semantic_verdict,review_developer_semantics_verdict,"
            "review_limitations,review_required_actions,trust_readiness,"
            "trust_blockers,updated_at"
        ),
        order="atom_id.asc",
        client=client,
    )
    rollups.extend(
        row for row in rollup_rows if str(row.get("atom_id", "")) in atom_id_set
    )

    description_rows = await _fetch_all_rows(
        base_url,
        access_token,
        "atom_descriptions",
        select=(
            "description_id,atom_id,kind,content,language,generated_by,"
            "reviewed,jargon_score,created_at,updated_at"
        ),
        filters={
            "kind": "eq.dejargonized",
            "language": "eq.en",
        },
        order="atom_id.asc,updated_at.desc",
        client=client,
    )
    descriptions.extend(
        row for row in description_rows if str(row.get("atom_id", "")) in atom_id_set
    )

    spec_rows = await _fetch_all_rows(
        base_url,
        access_token,
        "atom_io_specs",
        select=(
            "io_spec_id,atom_id,version_id,direction,name,type_desc,"
            "constraints,required,default_value_repr,ordinal"
        ),
        order="atom_id.asc,direction.asc,ordinal.asc,name.asc",
        client=client,
    )
    io_specs.extend(
        _normalize_io_spec_row(row)
        for row in spec_rows
        if str(row.get("atom_id", "")) in atom_id_set
    )

    try:
        rpc_rows = await _call_rpc(
            base_url,
            access_token,
            "get_manifest_benchmarks",
            client=client,
        )
        if isinstance(rpc_rows, list):
            source_rows = rpc_rows
        elif isinstance(rpc_rows, dict) and "data" in rpc_rows:
            source_rows = list(rpc_rows["data"] or [])
        else:
            source_rows = []
        for row in source_rows:
            atom_fqdn = str(row.get("atom_fqdn", ""))
            if atom_fqdns and atom_fqdn not in atom_fqdns:
                continue
            benchmarks.append(_normalize_benchmark_row(row, atom_fqdn=atom_fqdn))
    except httpx.HTTPStatusError:
        source_rows = await _fetch_all_rows(
            base_url,
            access_token,
            "atom_benchmarks",
            select=(
                "benchmark_id,version_id,benchmark_name,metric_name,metric_value,"
                "dataset_tag,measured_at"
            ),
            order="benchmark_name.asc,metric_name.asc",
            client=client,
        )
        version_by_id = {
            str(row["version_id"]): row for row in version_rows if row.get("version_id")
        }
        atom_rows = {str(row["atom_id"]): row for row in atoms if row.get("atom_id")}
        for row in source_rows:
            version = version_by_id.get(str(row.get("version_id", "")), {})
            atom = atom_rows.get(str(version.get("atom_id", "")), {})
            atom_fqdn = str(atom.get("fqdn", ""))
            if atom_fqdns and atom_fqdn not in atom_fqdns:
                continue
            benchmarks.append(
                _normalize_benchmark_row(
                    {
                        **row,
                        "atom_fqdn": atom_fqdn,
                        "content_hash": version.get("content_hash", ""),
                    },
                    atom_fqdn=atom_fqdn,
                )
            )

    return {
        "atoms": atoms,
        "hyperparams": hyperparams,
        "benchmarks": benchmarks,
        "rollups": rollups,
        "descriptions": descriptions,
        "io_specs": io_specs,
        "manifest_metadata": license_summary,
    }


def _coerce_manifest_data(
    atoms_or_data: list[dict[str, Any]] | Mapping[str, list[dict[str, Any]]],
    hyperparams: list[dict[str, Any]] | None,
    benchmarks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if isinstance(atoms_or_data, Mapping) and hyperparams is None and benchmarks is None:
        data = {
            "atoms": list(atoms_or_data.get("atoms", [])),
            "hyperparams": list(atoms_or_data.get("hyperparams", [])),
            "benchmarks": list(atoms_or_data.get("benchmarks", [])),
            "uncertainty_profiles": list(atoms_or_data.get("uncertainty_profiles", [])),
            "rollups": list(atoms_or_data.get("rollups", [])),
            "descriptions": list(atoms_or_data.get("descriptions", [])),
            "io_specs": [
                _normalize_io_spec_row(row) for row in atoms_or_data.get("io_specs", [])
            ],
        }
        if "manifest_metadata" in atoms_or_data:
            data["manifest_metadata"] = dict(atoms_or_data.get("manifest_metadata") or {})
        return data

    return {
        "atoms": list(atoms_or_data),
        "hyperparams": list(hyperparams or []),
        "benchmarks": list(benchmarks or []),
        "uncertainty_profiles": [],
        "rollups": [],
        "descriptions": [],
        "io_specs": [],
    }


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS catalog_fts;
        DROP TABLE IF EXISTS atoms;
        DROP TABLE IF EXISTS hyperparams;
        DROP TABLE IF EXISTS io_specs;
        DROP TABLE IF EXISTS benchmarks;
        DROP TABLE IF EXISTS uncertainty_profiles;
        DROP TABLE IF EXISTS audit_rollups;
        DROP TABLE IF EXISTS descriptions;
        DROP TABLE IF EXISTS manifest_metadata;

        CREATE TABLE atoms (
            atom_id TEXT PRIMARY KEY,
            fqdn TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'approved',
            domain_tags TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            visibility_tier TEXT NOT NULL DEFAULT 'general',
            source_kind TEXT NOT NULL DEFAULT 'hand_written',
            stateful_kind TEXT NOT NULL DEFAULT 'none',
            is_stochastic INTEGER NOT NULL DEFAULT 0,
            is_ffi INTEGER NOT NULL DEFAULT 0,
            namespace_root TEXT NOT NULL DEFAULT 'sciona.atoms',
            namespace_path TEXT NOT NULL DEFAULT '',
            source_repo_id TEXT NOT NULL DEFAULT '',
            source_package TEXT NOT NULL DEFAULT '',
            source_module_path TEXT NOT NULL DEFAULT '',
            source_symbol TEXT NOT NULL DEFAULT '',
            is_publishable INTEGER NOT NULL DEFAULT 0,
            version_id TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            semver TEXT NOT NULL DEFAULT '',
            license_expression TEXT NOT NULL DEFAULT '',
            license_status TEXT NOT NULL DEFAULT 'unknown',
            license_family TEXT NOT NULL DEFAULT 'unknown',
            license_source_kind TEXT NOT NULL DEFAULT '',
            license_source_path TEXT NOT NULL DEFAULT '',
            upstream_license_expression TEXT NOT NULL DEFAULT '',
            license_confidence TEXT NOT NULL DEFAULT 'unknown',
            license_notes TEXT NOT NULL DEFAULT '',
            concept_type TEXT NOT NULL DEFAULT '',
            phase TEXT NOT NULL DEFAULT 'stateless',
            type_signature TEXT NOT NULL DEFAULT '',
            state_ports_json TEXT NOT NULL DEFAULT '[]',
            time_complexity TEXT NOT NULL DEFAULT '',
            space_complexity TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE hyperparams (
            hp_id TEXT PRIMARY KEY,
            atom_id TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            default_value TEXT,
            min_value TEXT,
            max_value TEXT,
            step_value TEXT,
            log_scale INTEGER NOT NULL DEFAULT 0,
            choices_json TEXT,
            constraints_json TEXT,
            semantic_role TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'approved',
            UNIQUE (atom_id, name)
        );

        CREATE TABLE io_specs (
            atom_id TEXT NOT NULL,
            port_name TEXT NOT NULL,
            direction TEXT NOT NULL,
            type_desc TEXT NOT NULL DEFAULT 'Any',
            constraints TEXT NOT NULL DEFAULT '',
            data_kind TEXT NOT NULL DEFAULT '',
            required INTEGER NOT NULL DEFAULT 1,
            default_value_repr TEXT NOT NULL DEFAULT '',
            ordinal INTEGER NOT NULL DEFAULT 0,
            port_type_json TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (atom_id, direction, port_name)
        );

        CREATE TABLE benchmarks (
            atom_fqdn TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            benchmark_id TEXT NOT NULL,
            benchmark_name TEXT NOT NULL DEFAULT '',
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            dataset_tag TEXT NOT NULL DEFAULT '',
            measured_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (atom_fqdn, content_hash, benchmark_id, metric_name)
        );

        CREATE TABLE uncertainty_profiles (
            atom_fqdn TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL DEFAULT 2,
            profile_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (atom_fqdn, content_hash)
        );

        CREATE TABLE audit_rollups (
            atom_id TEXT PRIMARY KEY,
            overall_verdict TEXT NOT NULL DEFAULT 'unknown',
            structural_status TEXT NOT NULL DEFAULT 'unknown',
            runtime_status TEXT NOT NULL DEFAULT 'unknown',
            semantic_status TEXT NOT NULL DEFAULT 'unknown',
            developer_semantics_status TEXT NOT NULL DEFAULT 'unknown',
            risk_tier TEXT NOT NULL DEFAULT 'medium',
            risk_score INTEGER NOT NULL DEFAULT 0,
            risk_dimensions TEXT NOT NULL DEFAULT '{}',
            risk_reasons TEXT NOT NULL DEFAULT '[]',
            acceptability_score INTEGER NOT NULL DEFAULT 0,
            acceptability_band TEXT NOT NULL DEFAULT 'unknown',
            parity_coverage_level TEXT NOT NULL DEFAULT 'unknown',
            parity_test_status TEXT NOT NULL DEFAULT 'unknown',
            parity_fixture_count INTEGER NOT NULL DEFAULT 0,
            parity_case_count INTEGER NOT NULL DEFAULT 0,
            review_status TEXT NOT NULL DEFAULT 'missing',
            review_semantic_verdict TEXT NOT NULL DEFAULT 'unknown',
            review_developer_semantics_verdict TEXT NOT NULL DEFAULT 'unknown',
            review_limitations TEXT NOT NULL DEFAULT '[]',
            review_required_actions TEXT NOT NULL DEFAULT '[]',
            trust_readiness TEXT NOT NULL DEFAULT 'not_ready',
            trust_blockers TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE descriptions (
            description_id TEXT PRIMARY KEY,
            atom_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'en',
            generated_by TEXT NOT NULL DEFAULT '',
            reviewed INTEGER NOT NULL DEFAULT 0,
            jargon_score REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE manifest_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    try:
        con.execute(
            """
            CREATE VIRTUAL TABLE catalog_fts USING fts5(
                fqdn,
                description,
                domain_tags,
                concept_type,
                type_signature,
                tokenize='unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        pass


def _insert_atom(con: sqlite3.Connection, atom: Mapping[str, Any]) -> None:
    state_ports_raw = atom.get("state_ports_json")
    if state_ports_raw is None and atom.get("state_ports") is not None:
        state_ports_raw = [
            sp.model_dump(mode="json") if hasattr(sp, "model_dump") else dict(sp)
            for sp in atom.get("state_ports", [])
        ]
    con.execute(
        """
        INSERT OR REPLACE INTO atoms (
            atom_id, fqdn, status, domain_tags, description, visibility_tier,
            source_kind, stateful_kind, is_stochastic, is_ffi, namespace_root,
            namespace_path, source_repo_id, source_package, source_module_path,
            source_symbol, is_publishable, version_id, content_hash, semver,
            license_expression, license_status, license_family,
            license_source_kind, license_source_path, upstream_license_expression,
            license_confidence, license_notes, concept_type, phase,
            type_signature, state_ports_json, time_complexity, space_complexity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stringify(atom.get("atom_id")),
            atom.get("fqdn", ""),
            atom.get("status", "approved"),
            _domain_tags_to_text(atom.get("domain_tags", [])),
            atom.get("description", ""),
            atom.get("visibility_tier", "general"),
            atom.get("source_kind", "hand_written"),
            atom.get("stateful_kind", "none"),
            int(bool(atom.get("is_stochastic", False))),
            int(bool(atom.get("is_ffi", False))),
            atom.get("namespace_root", "sciona.atoms"),
            atom.get("namespace_path", ""),
            _stringify(atom.get("source_repo_id")),
            atom.get("source_package", ""),
            atom.get("source_module_path", ""),
            atom.get("source_symbol", ""),
            int(bool(atom.get("is_publishable", False))),
            _stringify(atom.get("version_id")),
            _stringify(atom.get("content_hash")),
            _stringify(atom.get("semver")),
            _stringify(atom.get("license_expression")),
            _stringify(atom.get("license_status")) or "unknown",
            _stringify(atom.get("license_family")) or "unknown",
            _stringify(atom.get("license_source_kind")),
            _stringify(atom.get("license_source_path")),
            _stringify(atom.get("upstream_license_expression")),
            _stringify(atom.get("license_confidence")) or "unknown",
            _stringify(atom.get("license_notes")),
            _stringify(atom.get("concept_type")),
            _stringify(atom.get("phase")) or "stateless",
            _stringify(atom.get("type_signature")),
            _stringify(state_ports_raw if state_ports_raw is not None else []),
            _stringify(atom.get("time_complexity")),
            _stringify(atom.get("space_complexity")),
        ),
    )


def _insert_hyperparam(con: sqlite3.Connection, hp: Mapping[str, Any]) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO hyperparams (
            hp_id, atom_id, name, kind, default_value, min_value, max_value,
            step_value, log_scale, choices_json, constraints_json,
            semantic_role, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stringify(hp.get("hp_id")),
            _stringify(hp.get("atom_id")),
            hp.get("name", ""),
            hp.get("kind", ""),
            _stringify(hp.get("default_value")) if hp.get("default_value") is not None else None,
            _stringify(hp.get("min_value")) if hp.get("min_value") is not None else None,
            _stringify(hp.get("max_value")) if hp.get("max_value") is not None else None,
            _stringify(hp.get("step_value")) if hp.get("step_value") is not None else None,
            int(bool(hp.get("log_scale", False))),
            _stringify(hp.get("choices_json")) if hp.get("choices_json") is not None else None,
            _stringify(hp.get("constraints_json")) if hp.get("constraints_json") is not None else None,
            hp.get("semantic_role", ""),
            hp.get("status", "approved"),
        ),
    )


def _insert_io_spec(con: sqlite3.Connection, io_spec: Mapping[str, Any]) -> None:
    data = _normalize_io_spec_row(io_spec)
    con.execute(
        """
        INSERT OR REPLACE INTO io_specs (
            atom_id, port_name, direction, type_desc, constraints, data_kind,
            required, default_value_repr, ordinal, port_type_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stringify(data.get("atom_id")),
            _stringify(data.get("port_name")),
            _stringify(data.get("direction")),
            _stringify(data.get("type_desc") or "Any"),
            _stringify(data.get("constraints")),
            _stringify(data.get("data_kind")),
            int(bool(data.get("required", True))),
            _stringify(data.get("default_value_repr")),
            int(data.get("ordinal", 0)),
            _stringify(data.get("port_type_json")),
        ),
    )


def _insert_benchmark(con: sqlite3.Connection, bm: Mapping[str, Any]) -> None:
    benchmark_id = bm.get("benchmark_id") or bm.get("benchmark_name") or ""
    benchmark_name = bm.get("benchmark_name") or benchmark_id
    con.execute(
        """
        INSERT OR REPLACE INTO benchmarks (
            atom_fqdn, content_hash, benchmark_id, benchmark_name, metric_name,
            metric_value, dataset_tag, measured_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bm.get("atom_fqdn", ""),
            bm.get("content_hash", ""),
            benchmark_id,
            benchmark_name,
            bm.get("metric_name", ""),
            bm.get("metric_value", 0.0),
            bm.get("dataset_tag", ""),
            _stringify(bm.get("measured_at")),
        ),
    )


def _insert_uncertainty_profile(con: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    profile_raw = row.get("profile_json")
    if not profile_raw and row.get("profile") is not None:
        profile_raw = row.get("profile")
    con.execute(
        """
        INSERT OR REPLACE INTO uncertainty_profiles (
            atom_fqdn, content_hash, version, profile_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            _stringify(row.get("atom_fqdn")),
            _stringify(row.get("content_hash")),
            int(row.get("version", 2) or 2),
            _stringify(profile_raw if profile_raw is not None else {}),
            _stringify(row.get("updated_at")),
        ),
    )


def rebuild_catalog_fts(con: sqlite3.Connection) -> None:
    """Populate or refresh the SQLite FTS5 lexical/type search index."""
    tables = {
        str(row[0])
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "atoms" not in tables:
        return
    if "catalog_fts" not in tables:
        try:
            con.execute(
                """
                CREATE VIRTUAL TABLE catalog_fts USING fts5(
                    fqdn,
                    description,
                    domain_tags,
                    concept_type,
                    type_signature,
                    tokenize='unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            return
    else:
        try:
            con.execute("DELETE FROM catalog_fts")
        except sqlite3.OperationalError:
            return

    atom_cols = {
        str(r[1]) for r in con.execute("PRAGMA table_info(atoms)").fetchall()
    }
    has_concept = "concept_type" in atom_cols
    has_sig = "type_signature" in atom_cols

    desc_by_atom: dict[str, str] = {}
    if "descriptions" in tables:
        for atom_id, content in con.execute(
            """
            SELECT atom_id, content
            FROM descriptions
            WHERE content IS NOT NULL AND content != ''
            ORDER BY atom_id ASC,
                     CASE WHEN kind = 'dejargonized' THEN 0 ELSE 1 END,
                     reviewed DESC,
                     updated_at DESC
            """
        ).fetchall():
            desc_by_atom.setdefault(str(atom_id), str(content))

    ports_by_atom: dict[str, tuple[list[str], list[str]]] = {}
    if "io_specs" in tables:
        for atom_id, port_name, direction, type_desc in con.execute(
            """
            SELECT atom_id, port_name, direction, type_desc
            FROM io_specs
            ORDER BY atom_id ASC, COALESCE(ordinal, 0) ASC, port_name ASC
            """
        ).fetchall():
            ins, outs = ports_by_atom.setdefault(str(atom_id), ([], []))
            label = f"{port_name}: {type_desc or 'Any'}" if port_name else str(type_desc or "Any")
            if str(direction).lower().startswith("out"):
                outs.append(label)
            else:
                ins.append(label)

    select_cols = ["atom_id", "fqdn", "description", "domain_tags"]
    select_cols.append("concept_type" if has_concept else "'' AS concept_type")
    select_cols.append("type_signature" if has_sig else "'' AS type_signature")

    fts_rows: list[tuple[str, str, str, str, str]] = []
    for atom_id, fqdn, raw_desc, domain_tags, concept_type, type_sig in con.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM atoms
        WHERE status = 'approved'
        ORDER BY fqdn ASC
        """
    ).fetchall():
        aid = str(atom_id or "")
        desc = desc_by_atom.get(aid) or str(raw_desc or "")
        sig = str(type_sig or "")
        if not sig and aid in ports_by_atom:
            ins, outs = ports_by_atom[aid]
            in_str = f"({', '.join(ins)})" if ins else "()"
            out_str = outs[0] if len(outs) == 1 else (f"({', '.join(outs)})" if outs else "Any")
            sig = f"{in_str} -> {out_str}"
        fts_rows.append(
            (
                str(fqdn or ""),
                desc,
                str(domain_tags or ""),
                str(concept_type or ""),
                sig,
            )
        )

    if fts_rows:
        con.executemany(
            """
            INSERT INTO catalog_fts (fqdn, description, domain_tags, concept_type, type_signature)
            VALUES (?, ?, ?, ?, ?)
            """,
            fts_rows,
        )


def _insert_rollup(con: sqlite3.Connection, rollup: Mapping[str, Any]) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO audit_rollups (
            atom_id, overall_verdict, structural_status, runtime_status,
            semantic_status, developer_semantics_status, risk_tier, risk_score,
            risk_dimensions, risk_reasons, acceptability_score,
            acceptability_band, parity_coverage_level, parity_test_status,
            parity_fixture_count, parity_case_count, review_status,
            review_semantic_verdict, review_developer_semantics_verdict,
            review_limitations, review_required_actions, trust_readiness,
            trust_blockers, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stringify(rollup.get("atom_id")),
            rollup.get("overall_verdict", "unknown"),
            rollup.get("structural_status", "unknown"),
            rollup.get("runtime_status", "unknown"),
            rollup.get("semantic_status", "unknown"),
            rollup.get("developer_semantics_status", "unknown"),
            rollup.get("risk_tier", "medium"),
            int(rollup.get("risk_score", 0)),
            _stringify(rollup.get("risk_dimensions", {})),
            _stringify(rollup.get("risk_reasons", [])),
            int(rollup.get("acceptability_score", 0)),
            rollup.get("acceptability_band", "unknown"),
            rollup.get("parity_coverage_level", "unknown"),
            rollup.get("parity_test_status", "unknown"),
            int(rollup.get("parity_fixture_count", 0)),
            int(rollup.get("parity_case_count", 0)),
            rollup.get("review_status", "missing"),
            rollup.get("review_semantic_verdict", "unknown"),
            rollup.get("review_developer_semantics_verdict", "unknown"),
            _stringify(rollup.get("review_limitations", [])),
            _stringify(rollup.get("review_required_actions", [])),
            rollup.get("trust_readiness", "not_ready"),
            _stringify(rollup.get("trust_blockers", [])),
            _stringify(rollup.get("updated_at")),
        ),
    )


def _insert_description(con: sqlite3.Connection, desc: Mapping[str, Any]) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO descriptions (
            description_id, atom_id, kind, content, language, generated_by,
            reviewed, jargon_score, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stringify(desc.get("description_id")),
            _stringify(desc.get("atom_id")),
            desc.get("kind", ""),
            desc.get("content", ""),
            desc.get("language", "en"),
            desc.get("generated_by", ""),
            int(bool(desc.get("reviewed", False))),
            float(desc.get("jargon_score", 1.0)),
            _stringify(desc.get("created_at")),
            _stringify(desc.get("updated_at")),
        ),
    )


def _insert_manifest_metadata(
    con: sqlite3.Connection,
    metadata: Mapping[str, Any],
) -> None:
    for key, value in sorted(metadata.items()):
        con.execute(
            """
            INSERT OR REPLACE INTO manifest_metadata (key, value)
            VALUES (?, ?)
            """,
            (str(key), _stringify(value)),
        )


def generate_manifest_sqlite(
    atoms_or_data: list[dict[str, Any]] | Mapping[str, list[dict[str, Any]]],
    hyperparams: list[dict[str, Any]] | None = None,
    benchmarks: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
) -> sqlite3.Connection:
    """Generate a manifest.sqlite from Supabase-fetched data."""
    data = _coerce_manifest_data(atoms_or_data, hyperparams, benchmarks)
    explicit_metadata = dict(data.get("manifest_metadata") or {})
    metadata = {
        "generated_at": explicit_metadata.get("generated_at") or _utc_now_iso(),
        "generator_version": explicit_metadata.get("generator_version")
        or _manifest_generator_version(),
        "visibility_tier": explicit_metadata.get("visibility_tier")
        or DEFAULT_MANIFEST_VISIBILITY_TIER,
        "content_hash": _manifest_content_hash(data["atoms"]),
    }
    for key, value in explicit_metadata.items():
        metadata.setdefault(str(key), value)

    db_str = str(output_path) if output_path else ":memory:"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_str)
    _create_schema(con)

    for atom in data["atoms"]:
        _insert_atom(con, atom)
    for hp in data["hyperparams"]:
        _insert_hyperparam(con, hp)
    for io_spec in data["io_specs"]:
        _insert_io_spec(con, io_spec)
    for bm in data["benchmarks"]:
        _insert_benchmark(con, bm)
    for up in data.get("uncertainty_profiles", []):
        _insert_uncertainty_profile(con, up)
    for rollup in data["rollups"]:
        _insert_rollup(con, rollup)
    for desc in data["descriptions"]:
        _insert_description(con, desc)
    _insert_manifest_metadata(con, metadata)
    rebuild_catalog_fts(con)

    con.commit()
    return con


async def export_tiered_manifests(
    base_url: str,
    access_token: str,
    output_dir: Path,
    *,
    include_developer_manifest: bool = False,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Path]:
    """Export one manifest SQLite per configured tier."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_license_policy = load_license_policy_from_env(developer_mode=False)
    developer_license_policy = load_license_policy_from_env(developer_mode=True)

    outputs: dict[str, Path] = {}
    for tier_name, included_tiers in MANIFEST_TIERS.items():
        data = await fetch_manifest_data(
            base_url,
            access_token,
            visibility_tiers=included_tiers,
            include_unpublished=False,
            license_policy=public_license_policy,
            client=client,
        )
        data["manifest_metadata"] = {
            **dict(data.get("manifest_metadata") or {}),
            "visibility_tier": tier_name,
        }
        output_path = output_dir / Path(manifest_artifact_key(tier_name)).name
        con = generate_manifest_sqlite(data, output_path=output_path)
        con.close()
        outputs[tier_name] = output_path

    if include_developer_manifest:
        data = await fetch_manifest_data(
            base_url,
            access_token,
            visibility_tiers=DEVELOPER_MANIFEST_INCLUDED_TIERS,
            include_unpublished=True,
            license_policy=developer_license_policy,
            client=client,
        )
        data["manifest_metadata"] = {
            **dict(data.get("manifest_metadata") or {}),
            "visibility_tier": DEVELOPER_MANIFEST_TIER,
            "publication_scope": "developer",
        }
        output_path = output_dir / Path(manifest_artifact_key(DEVELOPER_MANIFEST_TIER)).name
        con = generate_manifest_sqlite(data, output_path=output_path)
        con.close()
        outputs[DEVELOPER_MANIFEST_TIER] = output_path

    return outputs
