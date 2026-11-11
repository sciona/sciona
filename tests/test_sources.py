"""Tests for sciona.sources — multi-repo atom source management."""

from __future__ import annotations

import json
import textwrap
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from sciona.sources import (
    AtomSource,
    SourcesConfig,
    discover_cdgs,
    find_cdg,
    import_atoms,
    load_sources,
    resolve_package_root,
    resolve_source,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestAtomSource:
    def test_valid_path_source(self):
        src = AtomSource(name="foo", package="foo_pkg", path="../foo")
        assert src.name == "foo"
        assert src.package == "foo_pkg"
        assert src.path == "../foo"
        assert src.git is None

    def test_valid_src_layout_source(self):
        src = AtomSource(
            name="demo",
            package="sciona.atoms.demo",
            path="../sciona-atoms",
            python_path="src",
        )
        assert src.python_path == "src"
        assert src.package == "sciona.atoms.demo"

    def test_valid_git_source(self):
        src = AtomSource(
            name="bar",
            package="bar_pkg",
            git="https://github.com/org/bar.git",
            ref="develop",
        )
        assert src.git == "https://github.com/org/bar.git"
        assert src.ref == "develop"

    def test_missing_path_and_git_raises(self):
        with pytest.raises(ValueError, match="must specify either"):
            AtomSource(name="bad", package="bad_pkg")

    def test_default_cdg_glob(self):
        src = AtomSource(name="x", package="x", path=".")
        assert src.cdg_glob == "**/*cdg*.json"

    def test_custom_cdg_glob(self):
        src = AtomSource(name="x", package="x", path=".", cdg_glob="data/*.json")
        assert src.cdg_glob == "data/*.json"


class TestSourcesConfig:
    def test_load_from_yaml(self, tmp_path: Path):
        yml = tmp_path / "sources.yml"
        yml.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "name": "a",
                            "package": "a_pkg",
                            "path": "../a",
                            "python_path": "src",
                        },
                        {
                            "name": "b",
                            "package": "b_pkg",
                            "git": "https://github.com/x/b.git",
                        },
                    ]
                }
            )
        )
        cfg = SourcesConfig.load(yml)
        assert len(cfg.sources) == 2
        assert cfg.sources[0].name == "a"
        assert cfg.sources[0].python_path == "src"
        assert cfg.sources[1].git == "https://github.com/x/b.git"

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        cfg = SourcesConfig.load(tmp_path / "nonexistent.yml")
        assert cfg.sources == []

    def test_load_empty_file(self, tmp_path: Path):
        yml = tmp_path / "sources.yml"
        yml.write_text("")
        cfg = SourcesConfig.load(yml)
        assert cfg.sources == []


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolveSource:
    def test_resolve_path_source(self, tmp_path: Path):
        repo = tmp_path / "my-atoms"
        repo.mkdir()
        src = AtomSource(name="my-atoms", package="my", path="my-atoms")
        resolved = resolve_source(src, base_dir=tmp_path)
        assert resolved == repo.resolve()

    def test_resolve_git_source_clones(self, tmp_path: Path):
        src = AtomSource(
            name="remote",
            package="remote_pkg",
            git="https://github.com/org/remote.git",
        )
        with patch("sciona.sources._git_clone") as mock_clone:
            resolved = resolve_source(src, base_dir=tmp_path)
            mock_clone.assert_called_once()
            assert ".sciona_cache" in str(resolved)
            assert "remote" in str(resolved)

    def test_resolve_git_source_fetches_existing(self, tmp_path: Path):
        cache_dir = tmp_path / ".sciona_cache" / "remote"
        cache_dir.mkdir(parents=True)
        src = AtomSource(
            name="remote",
            package="remote_pkg",
            git="https://github.com/org/remote.git",
        )
        with patch("sciona.sources._git_fetch_checkout") as mock_fetch:
            resolved = resolve_source(src, base_dir=tmp_path)
            mock_fetch.assert_called_once_with(cache_dir, "main")
            assert resolved == cache_dir

    def test_resolve_path_source_expands_user(self, tmp_path: Path):
        home_repo = Path.home() / "codex_test_atoms_repo"
        src = AtomSource(name="home-repo", package="home_pkg", path="~/codex_test_atoms_repo")
        resolved = resolve_source(src, base_dir=tmp_path)
        assert resolved == home_repo.resolve()

    def test_resolve_package_root_uses_python_path_for_src_layout(self, tmp_path: Path):
        repo = tmp_path / "sciona-atoms"
        package_root = repo / "src" / "sciona" / "atoms" / "demo"
        package_root.mkdir(parents=True)

        src = AtomSource(
            name="demo",
            package="sciona.atoms.demo",
            path="sciona-atoms",
            python_path="src",
        )
        resolved = resolve_package_root(src, base_dir=tmp_path)
        assert resolved == package_root.resolve()


# ---------------------------------------------------------------------------
# CDG discovery
# ---------------------------------------------------------------------------


class TestDiscoverCdgs:
    def test_finds_cdg_files(self, tmp_path: Path):
        repo = tmp_path / "atoms"
        repo.mkdir()
        (repo / "pipeline_cdg.json").write_text("{}")
        (repo / "sub").mkdir()
        (repo / "sub" / "ecg_cdg.json").write_text("{}")
        (repo / "not_a_cdg.txt").write_text("")

        src = AtomSource(name="atoms", package="a", path="atoms")
        cdgs = discover_cdgs(src, base_dir=tmp_path)
        stems = {p.stem for p in cdgs}
        assert "pipeline_cdg" in stems
        assert "ecg_cdg" in stems
        assert "not_a_cdg" not in stems

    def test_missing_root_returns_empty(self, tmp_path: Path):
        src = AtomSource(name="gone", package="g", path="gone")
        cdgs = discover_cdgs(src, base_dir=tmp_path)
        assert cdgs == []


class TestFindCdg:
    def test_finds_cdg_by_name(self, tmp_path: Path):
        repo = tmp_path / "atoms"
        repo.mkdir()
        (repo / "ecg_cdg.json").write_text("{}")

        cfg = SourcesConfig(
            sources=[AtomSource(name="atoms", package="a", path="atoms")]
        )
        result = find_cdg("ecg", config=cfg, base_dir=tmp_path)
        assert result is not None
        assert "ecg_cdg" in result.stem

    def test_returns_none_when_not_found(self, tmp_path: Path):
        repo = tmp_path / "atoms"
        repo.mkdir()

        cfg = SourcesConfig(
            sources=[AtomSource(name="atoms", package="a", path="atoms")]
        )
        assert find_cdg("nonexistent", config=cfg, base_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# Atom import
# ---------------------------------------------------------------------------


class TestImportAtoms:
    def test_import_atoms_adds_sys_path(self, tmp_path: Path):
        """Verify that sys.path is extended for path sources."""
        repo = tmp_path / "myrepo"
        repo.mkdir()
        pkg = repo / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("LOADED = True\n")

        src = AtomSource(name="myrepo", package="mypkg", path="myrepo")
        import_atoms(src, base_dir=tmp_path)

        import sys

        assert str(repo.resolve()) in sys.path

        # Clean up
        sys.path.remove(str(repo.resolve()))
        import importlib

        if "mypkg" in sys.modules:
            del sys.modules["mypkg"]

    def test_import_atoms_supports_src_layout_namespace_package(self, tmp_path: Path):
        repo = tmp_path / "sciona-atoms"
        mod_dir = repo / "src" / "sciona" / "atoms" / "demo"
        mod_dir.mkdir(parents=True)
        (mod_dir / "atoms.py").write_text("LOADED = True\n")

        src = AtomSource(
            name="demo",
            package="sciona.atoms.demo",
            path="sciona-atoms",
            python_path="src",
        )

        import sciona as matcher_sciona

        before_sciona_path = list(matcher_sciona.__path__)
        before_sciona_spec_path = list(
            matcher_sciona.__spec__.submodule_search_locations or []
        )
        before_modules = set(sys.modules)
        try:
            import_atoms(src, base_dir=tmp_path)

            import importlib

            module = importlib.import_module("sciona.atoms.demo.atoms")
            assert getattr(module, "LOADED", False) is True
            assert str((repo / "src").resolve()) in sys.path
        finally:
            for name in set(sys.modules) - before_modules:
                if name == "sciona" or name.startswith("sciona."):
                    sys.modules.pop(name, None)
            matcher_sciona.__path__[:] = before_sciona_path
            matcher_sciona.__spec__.submodule_search_locations[:] = before_sciona_spec_path
            sys.path = [entry for entry in sys.path if entry != str((repo / "src").resolve())]

    def test_import_missing_package_warns(self, tmp_path: Path, caplog):
        src = AtomSource(name="missing", package="no_such_pkg_12345", path=".")
        import_atoms(src, base_dir=tmp_path)
        assert any("Could not import" in r.message for r in caplog.records)

    def test_import_atoms_skips_syntax_error_submodule(self, tmp_path: Path):
        repo = tmp_path / "brokenrepo"
        pkg = repo / "brokenpkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "broken.py").write_text("def nope(:\n")

        src = AtomSource(name="brokenrepo", package="brokenpkg", path="brokenrepo")
        import_atoms(src, base_dir=tmp_path)

    def test_import_atoms_falls_back_when_package_init_is_broken(self, tmp_path: Path):
        repo = tmp_path / "fallbackrepo"
        pkg = repo / "fallbackpkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("from fallbackpkg import missing_submodule\n")
        (pkg / "atoms.py").write_text("LOADED = True\n")

        src = AtomSource(name="fallbackrepo", package="fallbackpkg", path="fallbackrepo")
        import_atoms(src, base_dir=tmp_path)

        import importlib

        atoms = importlib.import_module("fallbackpkg.atoms")
        assert getattr(atoms, "LOADED", False) is True


# ---------------------------------------------------------------------------
# load_sources convenience
# ---------------------------------------------------------------------------


class TestLoadSources:
    def test_roundtrip(self, tmp_path: Path):
        data = {
            "sources": [
                {"name": "a", "package": "apkg", "path": "../a"},
            ]
        }
        yml = tmp_path / "sources.yml"
        yml.write_text(yaml.dump(data))
        cfg = load_sources(yml)
        assert len(cfg.sources) == 1
        assert cfg.sources[0].package == "apkg"


def test_live_state_estimation_source_exposes_snake_case_kalman_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_sources(repo_root / "sources.yml")
    source = next(
        src for src in config.sources if src.name == "sciona-atoms"
    )

    import sciona as matcher_sciona

    before_sciona_path = list(matcher_sciona.__path__)
    before_sciona_spec_path = list(
        matcher_sciona.__spec__.submodule_search_locations or []
    )
    before_modules = set(sys.modules)
    try:
        import_atoms(source, base_dir=repo_root)

        import importlib

        module = importlib.import_module(
            "sciona.atoms.state_estimation.kalman_filters.filter_rs"
        )
        assert hasattr(module, "initialize_kalman_state_model")
        assert hasattr(module, "evaluate_measurement_oracle")
        assert module.initializekalmanstatemodel is module.initialize_kalman_state_model
        assert module.evaluatemeasurementoracle is module.evaluate_measurement_oracle
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "sciona" or name.startswith("sciona."):
                sys.modules.pop(name, None)
        matcher_sciona.__path__[:] = before_sciona_path
        matcher_sciona.__spec__.submodule_search_locations[:] = before_sciona_spec_path
