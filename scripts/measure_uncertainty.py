#!/usr/bin/env python3
"""Empirical v2 mixture-of-regimes & spectral channel calibration CLI.

Usage:
    python scripts/measure_uncertainty.py --atoms-root ../sciona-atoms/src --atom kalman_update
    python scripts/measure_uncertainty.py --atoms-root ../sciona-atoms/src --domain state_estimation
    python scripts/measure_uncertainty.py --all

Discovers callable atoms under ``--atoms-root`` (or configured ``sciona-atoms*``
provider roots), runs ``sciona.uncertainty.calibration.try_calibrate_atom``, and
writes v2 ``uncertainty.json`` artifacts (preserving backward-compatible v1
``estimates`` entries).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sciona.uncertainty.calibration import CalibrationConfig, try_calibrate_atom  # noqa: E402
from sciona.uncertainty.models import AtomUncertaintyProfile  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredAtom:
    """Metadata for a discovered callable atom."""

    name: str
    module_dir: Path
    output_path: Path
    fn: Any


def _default_provider_roots() -> list[Path]:
    """Discover sibling ``sciona-atoms*`` source roots when ``--atoms-root`` is omitted."""
    workspace = REPO_ROOT.parent
    roots: list[Path] = []
    for candidate in sorted(workspace.glob("sciona-atoms*")):
        src_dir = candidate / "src"
        if src_dir.is_dir():
            roots.append(src_dir)
    return roots


def _load_module_from_file(py_file: Path, root: Path) -> ModuleType | None:
    """Import ``py_file`` via package path relative to ``root`` or file spec fallback."""
    root_resolved = root.resolve()
    file_resolved = py_file.resolve()
    if str(root_resolved) not in sys.path:
        sys.path.insert(0, str(root_resolved))

    try:
        rel = file_resolved.relative_to(root_resolved).with_suffix("")
        mod_name = ".".join(rel.parts)
        return importlib.import_module(mod_name)
    except Exception:
        pass

    fallback_name = f"_sciona_cal_{abs(hash(str(file_resolved)))}"
    spec = importlib.util.spec_from_file_location(fallback_name, file_resolved)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        logger.debug("Failed to load module %s: %s", file_resolved, exc)
        return None
    return mod


def _is_candidate_atom_fn(fn: Any, mod: ModuleType) -> bool:
    if not inspect.isfunction(fn):
        return False
    if getattr(fn, "__module__", None) != mod.__name__:
        return False
    annotations = getattr(fn, "__annotations__", {})
    if not annotations:
        return False
    return any(
        any(tok in str(ann).lower() for tok in ("ndarray", "array", "float", "sequence"))
        for ann in annotations.values()
    )


def discover_atoms_in_roots(
    roots: list[Path],
    *,
    atom_name: str | None = None,
    domain: str | None = None,
) -> list[DiscoveredAtom]:
    """Discover calibratable atom callables across ``roots``."""
    discovered: list[DiscoveredAtom] = []
    for root in roots:
        if not root.exists():
            continue
        candidate_files = sorted(root.rglob("atoms.py"))
        if not candidate_files:
            candidate_files = [
                p
                for p in sorted(root.rglob("*.py"))
                if p.name not in {"__init__.py", "witnesses.py"}
                and not p.name.endswith("_witnesses.py")
                and not p.name.startswith("test_")
            ]

        for py_file in candidate_files:
            mod_dir = py_file.parent
            if any(part in {"ghost", "__pycache__", "_artifacts", "tests"} for part in mod_dir.parts):
                continue
            if domain and not any(p.name == domain for p in (mod_dir, *mod_dir.parents)):
                continue

            mod = _load_module_from_file(py_file, root)
            if mod is None:
                continue

            local_atoms: list[tuple[str, Any]] = []
            for attr_name in sorted(dir(mod)):
                if attr_name.startswith("_"):
                    continue
                fn = getattr(mod, attr_name, None)
                if not _is_candidate_atom_fn(fn, mod):
                    continue
                local_atoms.append((attr_name, fn))

            multi_atom_dir = len(local_atoms) > 1
            for attr_name, fn in local_atoms:
                if atom_name and attr_name != atom_name:
                    continue
                if multi_atom_dir:
                    out_path = mod_dir / "_artifacts" / attr_name / "uncertainty.json"
                else:
                    out_path = mod_dir / "uncertainty.json"
                discovered.append(
                    DiscoveredAtom(
                        name=attr_name,
                        module_dir=mod_dir,
                        output_path=out_path,
                        fn=fn,
                    )
                )
    return discovered


def write_v2_uncertainty_json(
    profile: AtomUncertaintyProfile,
    output_path: Path,
) -> Path:
    """Write a v2 ``AtomUncertaintyProfile`` preserving existing non-empirical v1 entries."""
    payload = profile.to_dict()
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if isinstance(existing, dict):
                kept = [
                    e
                    for e in existing.get("estimates", [])
                    if isinstance(e, dict) and e.get("mode") != "empirical"
                ]
                payload["estimates"] = kept + list(payload.get("estimates", []))
        except (json.JSONDecodeError, OSError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def run_measurement(
    roots: list[Path],
    *,
    atom_name: str | None = None,
    domain: str | None = None,
    config: CalibrationConfig | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Discover and calibrate matching atoms across ``roots``."""
    atoms = discover_atoms_in_roots(roots, atom_name=atom_name, domain=domain)
    stats = {"discovered": len(atoms), "calibrated": 0, "skipped": 0}
    cfg = config or CalibrationConfig()

    for item in atoms:
        profile = try_calibrate_atom(item.name, item.fn, config=cfg)
        if profile is None:
            stats["skipped"] += 1
            continue
        if not dry_run:
            write_v2_uncertainty_json(profile, item.output_path)
        stats["calibrated"] += 1
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--atom", help="Calibrate a specific atom by function name")
    group.add_argument("--domain", help="Calibrate all atoms under a domain directory")
    group.add_argument("--all", action="store_true", help="Calibrate all discovered atoms")
    parser.add_argument(
        "--atoms-root",
        type=Path,
        action="append",
        default=None,
        help="Atom source root(s) to scan (defaults to sibling sciona-atoms* src dirs)",
    )
    parser.add_argument("--n-trials", type=int, default=300, help="Total multi-regime trials per atom")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Fixed mixture component count K (default: joint BIC selection over 1..3)",
    )
    parser.add_argument("--trust-tier", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true", help="Calibrate without writing files")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    roots = list(args.atoms_root) if args.atoms_root else _default_provider_roots()
    if not roots:
        logger.error("No atom provider roots found. Pass --atoms-root explicitly.")
        sys.exit(1)

    cfg = CalibrationConfig(
        n_trials=args.n_trials,
        n_components=args.n_components,
        trust_tier=args.trust_tier,
    )
    stats = run_measurement(
        roots,
        atom_name=args.atom,
        domain=args.domain,
        config=cfg,
        dry_run=args.dry_run,
    )
    if stats["discovered"] == 0:
        print("No matching atoms found.")
        sys.exit(1)
    print(
        f"Done: {stats['calibrated']} calibrated, {stats['skipped']} skipped "
        f"(out of {stats['discovered']} discovered)."
    )


if __name__ == "__main__":
    main()
