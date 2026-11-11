"""Regression checks for PEP 420 domain runtime co-location and clean core namespace."""
# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

from sciona.provider_release import validate_provider_release

MATCHER_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCIONA_PKG = MATCHER_ROOT / "sciona"
WORKSPACE_ROOT = MATCHER_ROOT.parent


def test_sciona_root_contains_only_core_framework_modules() -> None:
    loose_files = sorted(p.name for p in SCIONA_PKG.iterdir() if p.is_file())
    non_py = [f for f in loose_files if not f.endswith(".py")]
    assert non_py == [], f"Unexpected non-.py files in sciona/: {non_py}"
    # Competition runtime and historical ingestion files must not live in sciona-matcher/sciona
    assert len(loose_files) <= 65, f"Expected <=65 core root modules, found {len(loose_files)}"
    forbidden_prefixes = (
        "amex_",
        "aptos_",
        "adversarial_",
        "bengali_",
        "cassava_",
        "contrails_",
        "decomposed_",
        "dfdc_",
        "otto_",
        "plasticc_",
        "trackml_",
        "webtraffic_",
        "wheat_",
    )
    leaked = [f for f in loose_files if f.startswith(forbidden_prefixes)]
    assert leaked == [], f"Competition modules leaked into sciona/: {leaked}"


def test_zero_dangling_top_level_sciona_imports_across_installed_sources() -> None:
    valid_top = (
        {p.stem for p in SCIONA_PKG.glob("*.py")}
        | {p.name for p in SCIONA_PKG.iterdir() if p.is_dir() and not p.name.startswith("__")}
        | {"atoms", "probes", "_runtime"}
    )
    scan_roots = [SCIONA_PKG]
    for repo in sorted(WORKSPACE_ROOT.glob("sciona-atoms*")):
        if (repo / "src").is_dir():
            scan_roots.append(repo / "src")

    dangling: list[tuple[str, int, str]] = []
    for root in scan_roots:
        for py_file in sorted(root.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        if len(parts) >= 2 and parts[0] == "sciona" and parts[1] not in valid_top:
                            dangling.append((str(py_file), node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    parts = node.module.split(".")
                    if parts[0] == "sciona":
                        if len(parts) >= 2 and parts[1] not in valid_top:
                            dangling.append((str(py_file), node.lineno, node.module))
                        elif len(parts) == 1:
                            for alias in node.names:
                                if alias.name not in valid_top:
                                    dangling.append(
                                        (str(py_file), node.lineno, f"sciona.{alias.name}")
                                    )
    assert dangling == [], f"Found dangling sciona imports: {dangling[:10]}"


@pytest.mark.parametrize("repo_name", ["sciona-atoms-ml", "sciona-atoms-dl", "sciona-atoms-physics"])
def test_domain_provider_pep420_layout_and_standalone_runtime(repo_name: str) -> None:
    repo_root = WORKSPACE_ROOT / repo_name
    if not repo_root.is_dir():
        pytest.skip(f"Sibling repo {repo_name} not present in workspace")

    report = validate_provider_release(repo_root, build=False)
    assert report.ok, f"Provider release static check failed for {repo_name}: {report.issues}"
    assert not (repo_root / "src" / "sciona" / "_runtime" / "__init__.py").exists()

    domain = repo_name.removeprefix("sciona-atoms-")
    base_atoms = WORKSPACE_ROOT / "sciona-atoms" / "src"
    pythonpath_entries = [str(MATCHER_ROOT), str(repo_root / "src")]
    if base_atoms.is_dir():
        pythonpath_entries.insert(1, str(base_atoms))

    code = (
        f"import importlib\n"
        f"rt = importlib.import_module('sciona._runtime.{domain}')\n"
        f"assert rt is not None\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": ":".join(pythonpath_entries)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
