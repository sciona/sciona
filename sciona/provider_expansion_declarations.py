"""Load provider-owned expansion declaration maps from configured atom repos."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

from sciona.atom_identity import candidate_atom_provider_roots


def _module_candidates(family: str) -> tuple[tuple[str, ...], ...]:
    base = tuple(part for part in str(family or "").strip().split(".") if part)
    if not base:
        return tuple()
    module_name = base[-1]
    return (
        ("src", "sciona", "atoms", "expansion", *base[:-1], f"{module_name}.py"),
        ("src", "sciona", "atoms", "expansion", *base[:-1], f"{module_name}_registry.py"),
        ("sciona", "atoms", "expansion", *base[:-1], f"{module_name}.py"),
        ("sciona", "atoms", "expansion", *base[:-1], f"{module_name}_registry.py"),
    )


def _ensure_package_chain(pkg_name: str, pkg_dir: Path) -> None:
    """Ensure parent namespace package entries exist in ``sys.modules`` for relative imports."""
    parts = pkg_name.split(".")
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        ancestor_dir = pkg_dir
        for _ in range(len(parts) - depth):
            ancestor_dir = ancestor_dir.parent
        existing = sys.modules.get(prefix)
        if existing is None:
            ns_mod = types.ModuleType(prefix)
            setattr(ns_mod, "__path__", [str(ancestor_dir)])
            ns_mod.__package__ = prefix
            ns_mod.__spec__ = importlib.machinery.ModuleSpec(
                prefix, loader=None, is_package=True
            )
            sys.modules[prefix] = ns_mod
        else:
            mod_path = getattr(existing, "__path__", None)
            if mod_path is not None and str(ancestor_dir) not in mod_path:
                try:
                    mod_path.append(str(ancestor_dir))
                except AttributeError:
                    setattr(existing, "__path__", [*list(mod_path), str(ancestor_dir)])


def _dotted_module_name(path: Path) -> tuple[str, Path | None]:
    parts = path.with_suffix("").parts
    if "sciona" in parts:
        idx = parts.index("sciona")
        dotted = ".".join(parts[idx:])
        return dotted, path.parent
    fallback = "_sciona_provider_expansion_" + path.as_posix().replace("/", "_").replace(".", "_")
    return fallback, None


def _load_module(path: Path) -> object | None:
    module_name, pkg_dir = _dotted_module_name(path)
    if pkg_dir is not None and "." in module_name:
        parent_pkg = module_name.rsplit(".", 1)[0]
        _ensure_package_chain(parent_pkg, pkg_dir)
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=None,
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    if "." in module_name:
        module.__package__ = module_name.rsplit(".", 1)[0]
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        return None
    return module


@lru_cache(maxsize=32)
def load_provider_expansion_declarations(
    family: str,
    declaration_attr: str,
) -> dict[str, tuple[str, str, str]]:
    """Return provider-owned expansion declarations for a family.

    No matcher-local fallback is used here. If the configured atom providers do
    not expose the requested declaration map, an empty dict is returned.
    """
    family_text = str(family or "").strip()
    attr_text = str(declaration_attr or "").strip()
    if not family_text or not attr_text:
        return {}

    for provider_root in candidate_atom_provider_roots():
        root = Path(provider_root).expanduser().resolve()
        for relative in _module_candidates(family_text):
            module_path = root.joinpath(*relative)
            if not module_path.exists():
                continue
            module = _load_module(module_path)
            if module is None:
                continue
            declarations = getattr(module, attr_text, None)
            if isinstance(declarations, dict) and declarations:
                return dict(declarations)
    return {}


def clear_provider_expansion_declaration_caches() -> None:
    load_provider_expansion_declarations.cache_clear()

