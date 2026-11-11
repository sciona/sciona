"""Multi-repo atom source management via sources.yml."""

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import subprocess
import sys
from pathlib import Path
import types
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_DEFAULT_CDG_GLOB = "**/*cdg*.json"
_CACHE_DIR = ".sciona_cache"


class AtomSource(BaseModel):
    """A single atom repository source."""

    name: str
    package: str
    path: str | None = None
    python_path: str | None = None
    git: str | None = None
    ref: str = "main"
    cdg_glob: str = _DEFAULT_CDG_GLOB
    exclude_packages: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_path_or_git(self) -> "AtomSource":
        if not self.path and not self.git:
            raise ValueError(
                f"Source '{self.name}' must specify either 'path' or 'git'"
            )
        return self


class SourcesConfig(BaseModel):
    """Top-level sources.yml schema."""

    sources: list[AtomSource] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SourcesConfig":
        """Load from a YAML file.  Falls back to an empty config if missing."""
        if path is None:
            path = Path("sources.yml")
        path = Path(path)
        if not path.exists():
            logger.info("No sources.yml found at %s, using empty config", path)
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


def load_sources(path: str | Path | None = None) -> SourcesConfig:
    """Convenience wrapper for ``SourcesConfig.load``."""
    return SourcesConfig.load(path)


def resolve_source(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Return the local filesystem root for *source*.

    For ``path`` sources the path is resolved relative to *base_dir*
    (defaulting to cwd).  For ``git`` sources the repo is cloned /
    fetched into ``.sciona_cache/<name>``.
    """
    if source.path:
        base = base_dir or Path.cwd()
        return (base / Path(source.path).expanduser()).resolve()

    # Git source — clone or fetch
    assert source.git is not None
    cache_root = (base_dir or Path.cwd()) / _CACHE_DIR
    repo_dir = cache_root / source.name

    if repo_dir.exists():
        _git_fetch_checkout(repo_dir, source.ref)
    else:
        _git_clone(source.git, repo_dir, source.ref)

    return repo_dir


def resolve_import_root(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Return the Python import root for *source*.

    For standard repo layouts this is the same as :func:`resolve_source`.
    For ``src/`` layouts a source can set ``python_path`` to the directory
    within the repo that should be added to ``sys.path``.
    """
    root = resolve_source(source, base_dir)
    if source.python_path:
        return (root / Path(source.python_path).expanduser()).resolve()
    return root


def resolve_package_root(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Return the filesystem root for ``source.package`` within *source*."""
    return resolve_import_root(source, base_dir).joinpath(*source.package.split("."))


def excluded_import_prefixes(
    source: AtomSource, base_dir: Path | None = None
) -> tuple[str, ...]:
    """Return excluded module prefixes for *source* (explicit + pyproject.toml)."""
    prefixes: list[str] = [
        p.strip().rstrip(".") for p in source.exclude_packages if p and p.strip()
    ]
    if source.path and tomllib is not None:
        try:
            pyproject_path = resolve_source(source, base_dir) / "pyproject.toml"
            if pyproject_path.is_file():
                data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                provider_cfg = (
                    data.get("tool", {})
                    .get("sciona", {})
                    .get("provider", {})
                )
                for raw in provider_cfg.get("excluded-import-prefixes", []) or []:
                    token = str(raw or "").strip().rstrip(".")
                    if token and token not in prefixes:
                        prefixes.append(token)
        except Exception:
            pass
    return tuple(prefixes)


def is_excluded_module(modname: str, excluded_prefixes: tuple[str, ...]) -> bool:
    """Return True if *modname* falls under any excluded module prefix."""
    for prefix in excluded_prefixes:
        if modname == prefix or modname.startswith(prefix + "."):
            return True
    return False


def sync_source(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Fetch / update a git source.  No-op for local-path sources."""
    if source.path:
        base = base_dir or Path.cwd()
        resolved = (base / Path(source.path).expanduser()).resolve()
        logger.info("Source '%s' is a local path: %s", source.name, resolved)
        return resolved
    return resolve_source(source, base_dir)


def discover_cdgs(source: AtomSource, base_dir: Path | None = None) -> list[Path]:
    """Return CDG JSON files found in *source* using its ``cdg_glob``."""
    root = resolve_source(source, base_dir)
    if not root.exists():
        logger.warning("Source root does not exist: %s", root)
        return []
    return sorted(root.glob(source.cdg_glob))


def find_cdg(name: str, config: SourcesConfig | None = None, base_dir: Path | None = None) -> Path | None:
    """Search all sources for a CDG file whose stem contains *name*.

    Returns the first match or ``None``.
    """
    if config is None:
        config = load_sources()
    for source in config.sources:
        for cdg_path in discover_cdgs(source, base_dir):
            if name in cdg_path.stem:
                return cdg_path
    return None


def import_atoms(source: AtomSource, base_dir: Path | None = None) -> None:
    """Import the Python package for *source*, triggering ``@register_atom``.

    For ``path`` sources the Python import root is inserted into ``sys.path``
    so that the package can be found even if it isn't installed.
    """
    if source.path:
        import_root = resolve_import_root(source, base_dir)
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
        _prepare_package_import_paths(source.package, import_root)

    pkg = None
    pkg_path = None
    fallback_scan_root: Path | None = None
    try:
        pkg = importlib.import_module(source.package)
        pkg_path = getattr(pkg, "__path__", None)
    except Exception:
        if not source.path:
            logger.warning(
                "Could not import package '%s' for source '%s'",
                source.package,
                source.name,
            )
            return

        package_root = resolve_package_root(source, base_dir)
        if not package_root.exists():
            logger.warning(
                "Could not import package '%s' for source '%s'",
                source.package,
                source.name,
            )
            return

        # Fallback for broken top-level __init__.py files: treat the package as a
        # namespace package so submodules can still be imported and register atoms.
        pkg = _ensure_namespace_package(source.package, package_root)
        sys.modules[source.package] = pkg
        pkg_path = pkg.__path__
        fallback_scan_root = package_root

    excluded = excluded_import_prefixes(source, base_dir)

    # Walk submodules to trigger @register_atom decorators
    if fallback_scan_root is not None:
        import_root = resolve_import_root(source, base_dir)
        for py_file in sorted(fallback_scan_root.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_parts = py_file.relative_to(import_root).with_suffix("").parts
            modname = ".".join(module_parts)
            if is_excluded_module(modname, excluded):
                continue
            _import_module_from_file(modname, py_file)
        return

    if pkg_path is None:
        return

    import pkgutil

    walk_paths = list(pkg_path)
    if source.path:
        local_pkg_root = resolve_package_root(source, base_dir)
        if local_pkg_root.is_dir():
            walk_paths = [str(local_pkg_root)]

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        walk_paths,
        prefix=source.package + ".",
        onerror=lambda name: logger.debug(
            "Failed to enumerate %s for source %s", name, source.name, exc_info=True
        ),
    ):
        if is_excluded_module(modname, excluded):
            continue
        try:
            importlib.import_module(modname)
        except Exception:
            logger.debug("Failed to import %s, skipping", modname, exc_info=True)


def _import_module_from_file(modname: str, path: Path) -> None:
    """Import *modname* directly from *path*, creating namespace parents as needed."""
    import importlib.util

    parent_parts = modname.split(".")[:-1]
    for idx in range(len(parent_parts)):
        pkg_name = ".".join(parent_parts[: idx + 1])
        if pkg_name in sys.modules:
            continue
        pkg_path = path.parents[len(parent_parts) - idx - 1]
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.debug("Failed to import %s from %s, skipping", modname, path, exc_info=True)


def _prepare_package_import_paths(package_name: str, root: Path) -> None:
    """Expose namespace-style package paths to the import system for *root*."""
    importlib.invalidate_caches()

    parts = [part for part in package_name.split(".") if part]
    for idx in range(1, len(parts) + 1):
        prefix_name = ".".join(parts[:idx])
        prefix_path = root.joinpath(*parts[:idx])
        if not prefix_path.exists():
            break

        module = sys.modules.get(prefix_name)
        if module is not None:
            _append_module_search_path(module, prefix_path)
            continue

        try:
            spec = importlib.util.find_spec(prefix_name)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            spec = None
        if spec is not None and spec.submodule_search_locations is not None:
            try:
                module = importlib.import_module(prefix_name)
            except Exception:
                module = None
            if isinstance(module, types.ModuleType):
                _append_module_search_path(module, prefix_path)
                continue

        if prefix_path.joinpath("__init__.py").exists():
            continue

        module = _ensure_namespace_package(prefix_name, prefix_path)
        sys.modules[prefix_name] = module
        if idx > 1:
            parent_name = ".".join(parts[: idx - 1])
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, parts[idx - 1], module)


def _append_module_search_path(module: types.ModuleType, package_path: Path) -> None:
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return

    package_path_str = str(package_path)
    try:
        if package_path_str in module_path:
            return
        module_path.append(package_path_str)
    except Exception:
        normalized = list(module_path)
        if package_path_str not in normalized:
            normalized.append(package_path_str)
            module.__path__ = normalized  # type: ignore[attr-defined]

    spec = getattr(module, "__spec__", None)
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is None:
        return
    if package_path_str not in search_locations:
        search_locations.append(package_path_str)


def _ensure_namespace_package(package_name: str, package_path: Path) -> types.ModuleType:
    module = sys.modules.get(package_name)
    if isinstance(module, types.ModuleType):
        _append_module_search_path(module, package_path)
        return module

    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    module.__package__ = package_name
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_path)]
    module.__spec__ = spec
    return module


def import_all_sources(
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
) -> None:
    """Import atom packages from all configured sources."""
    if config is None:
        config = load_sources()
    for source in config.sources:
        import_atoms(source, base_dir)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_clone(url: str, dest: Path, ref: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s -> %s (ref=%s)", url, dest, ref)
    subprocess.check_call(
        ["git", "clone", "--branch", ref, "--single-branch", url, str(dest)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _git_fetch_checkout(repo: Path, ref: str) -> None:
    logger.info("Fetching and checking out ref=%s in %s", ref, repo)
    subprocess.check_call(
        ["git", "fetch", "origin", ref],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "checkout", f"origin/{ref}"],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
