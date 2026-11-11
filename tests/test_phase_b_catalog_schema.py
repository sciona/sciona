"""Phase B verification tests: PortTypeSpec, per-repo normalization, and two-plane manifest."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

import pytest

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.compiled_catalog import (
    ManifestFTSIndex,
    compile_local_manifest_sqlite,
    load_uncertainty_profiles_from_manifest,
    prune_manifest_to_namespaces,
    upsert_plane2_metadata,
)
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
    NodeStatus,
    PortTypeSpec,
    StatePortSpec,
)
from sciona.architect.source_catalog import (
    seed_catalog_from_manifest_sqlite,
    seed_catalog_from_sources,
)
from sciona.judge.python_env import PythonEnvironment, _count_params_from_signature
from sciona.principal.variant_mutation import _is_primitive_structurally_compatible
from sciona.protocols import SemanticIndex
from sciona.sources import AtomSource, SourcesConfig, load_sources


# ---------------------------------------------------------------------------
# 1. PortTypeSpec & Variant Compatibility
# ---------------------------------------------------------------------------


class TestPortTypeSpec:
    def test_parses_arrays_generics_tuples_and_witness_tags(self) -> None:
        arr = PortTypeSpec.from_type_desc("NDArray[np.float64]")
        assert arr.kind == "array"
        assert arr.dtype == "float64"

        sig = PortTypeSpec.from_type_desc("AbstractSignal")
        assert sig.kind == "array"
        assert sig.ndim == 1
        assert "signal" in sig.domain_tags

        mat = PortTypeSpec.from_type_desc("AbstractMatrix")
        assert mat.kind == "array"
        assert mat.ndim == 2

        tup = PortTypeSpec.from_type_desc("tuple[np.ndarray, Optional[int]]")
        assert tup.kind == "tuple"
        assert len(tup.tuple_elements) == 2
        assert tup.tuple_elements[0].kind == "array"
        assert tup.tuple_elements[1].kind == "scalar"
        assert tup.tuple_elements[1].dtype == "int"
        assert tup.tuple_elements[1].optional is True

    def test_compatibility_rules(self) -> None:
        generic_arr = PortTypeSpec.from_type_desc("np.ndarray")
        typed_arr = PortTypeSpec.from_type_desc("NDArray[np.float64]")
        assert generic_arr.is_compatible_with(typed_arr)
        assert typed_arr.is_compatible_with(generic_arr)

        int_scalar = PortTypeSpec.from_type_desc("int")
        float_scalar = PortTypeSpec.from_type_desc("float")
        assert int_scalar.is_compatible_with(float_scalar)
        assert not float_scalar.is_compatible_with(int_scalar)

        arr_1d = PortTypeSpec.from_type_desc("AbstractSignal")
        arr_2d = PortTypeSpec.from_type_desc("AbstractMatrix")
        assert not arr_1d.is_compatible_with(arr_2d)
        assert not int_scalar.is_compatible_with(arr_1d)

    def test_variant_structural_compatibility_handles_equivalent_arrays_and_defaults(self) -> None:
        node = AlgorithmicNode(
            node_id="n_filter",
            name="Bandpass Filter",
            description="Filter 1D signal",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
            phase="transform",
        )
        compatible_cand = AlgorithmicPrimitive(
            name="sciona.atoms.synthetic.fir_bandpass",
            source="synthetic",
            category=ConceptType.SIGNAL_FILTER,
            description="FIR bandpass filter",
            inputs=[
                IOSpec(name="x", type_desc="NDArray[np.float64]"),
                IOSpec(
                    name="order",
                    type_desc="int",
                    required=False,
                    default_value_repr="64",
                ),
            ],
            outputs=[IOSpec(name="y", type_desc="NDArray[np.float64]")],
            phase="transform",
        )
        wrong_phase_cand = compatible_cand.model_copy(update={"phase": "fit"})
        incompatible_port_cand = AlgorithmicPrimitive(
            name="sciona.atoms.synthetic.scalar_op",
            source="synthetic",
            category=ConceptType.SIGNAL_FILTER,
            description="Scalar transform",
            inputs=[IOSpec(name="x", type_desc="float")],
            outputs=[IOSpec(name="y", type_desc="np.ndarray")],
            phase="transform",
        )

        assert _is_primitive_structurally_compatible(node, compatible_cand) is True
        assert _is_primitive_structurally_compatible(node, wrong_phase_cand) is False
        assert _is_primitive_structurally_compatible(node, incompatible_port_cand) is False


# ---------------------------------------------------------------------------
# 2. PythonEnvironment Signature & Concurrency Soundness
# ---------------------------------------------------------------------------


class TestPythonEnvSoundness:
    def test_count_params_handles_nested_generic_commas(self) -> None:
        assert _count_params_from_signature("(cfg: dict[str, int], bounds: tuple[float, float]) -> bool") == 2
        assert _count_params_from_signature("(fn: Callable[[int, float], float]) -> float") == 1
        assert _count_params_from_signature("() -> np.ndarray") == 0

    @pytest.mark.asyncio
    async def test_check_term_rejects_under_arity_and_isolates_concurrent_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg_dir = tmp_path / "synth_env_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dir / "ops.py").write_text(
            "def unary_op(x: int) -> int:\n"
            "    return x + 1\n\n"
            "def binary_op(x: int, y: int) -> int:\n"
            "    return x + y\n",
            encoding="utf-8",
        )

        import os

        existing_pp = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv(
            "PYTHONPATH",
            f"{tmp_path}{os.pathsep}{existing_pp}" if existing_pp else str(tmp_path),
        )
        env = PythonEnvironment()
        try:
            ok_under, msg_under = await env.check_term(
                "@synth_env_pkg.ops.unary_op",
                "(a: int, b: int) -> int",
            )
            assert ok_under is False
            assert "arity mismatch" in msg_under.lower()

            ok_exact, _ = await env.check_term(
                "@synth_env_pkg.ops.binary_op",
                "(a: int, b: int) -> int",
            )
            assert ok_exact is True

            # Concurrent _run_file invocations should never clobber one another
            results = await asyncio.gather(
                *[
                    env._run_file(f"assert {i} + {i} == {2 * i}\nprint('OK_{i}')\n")
                    for i in range(6)
                ]
            )
            for idx, fb in enumerate(results):
                assert fb.success is True
                assert f"OK_{idx}" in fb.raw_output
        finally:
            await env.close()


# ---------------------------------------------------------------------------
# 3. Plane-1 Per-Repo Normalization & Two-Plane Compiled SQLite Catalog
# ---------------------------------------------------------------------------


def _create_synthetic_provider_repo(root: Path, pkg_leaf: str, fn_suffix: str) -> AtomSource:
    repo_dir = root / f"sciona-atoms-{pkg_leaf}"
    pkg_dir = repo_dir / "src" / "sciona" / "atoms" / pkg_leaf
    sub_dir = pkg_dir / "kalman_block"
    sub_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (sub_dir / "__init__.py").write_text("", encoding="utf-8")

    fn_name = f"kalman_update_{fn_suffix}"
    fqdn = f"sciona.atoms.{pkg_leaf}.kalman_block.atoms.{fn_name}"

    (sub_dir / "atoms.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from sciona.ghost.registry import register_atom\n\n"
        f"def witness_{fn_name}(prior: 'AbstractDistribution', obs: 'AbstractSignal') -> 'AbstractDistribution':\n"
        "    \"\"\"Witness docstring.\"\"\"\n"
        "    return prior\n\n"
        f"@register_atom(witness_{fn_name}, name={fqdn!r})\n"
        f"def {fn_name}(prior: np.ndarray, obs: np.ndarray, gain: float = 0.5) -> np.ndarray:\n"
        "    \"\"\"Update Gaussian state estimate from 1D observation.\"\"\"\n"
        "    return prior + gain * obs\n",
        encoding="utf-8",
    )

    # Canonical normalized cdg.json with explicit non-heuristic ConceptType, phase, and state_ports
    cdg_payload = {
        "nodes": [
            {
                "node_id": f"{fn_name}_node",
                "parent_id": None,
                "name": fn_name,
                "description": f"Bayesian Kalman measurement update step ({pkg_leaf}).",
                "concept_type": "conjugate_update",
                "status": "atomic",
                "phase": "update",
                "inputs": [
                    {"name": "prior", "type_desc": "np.ndarray"},
                    {"name": "obs", "type_desc": "np.ndarray"},
                    {
                        "name": "gain",
                        "type_desc": "float",
                        "required": False,
                        "default_value_repr": "0.5",
                    },
                ],
                "outputs": [{"name": "posterior", "type_desc": "np.ndarray"}],
                "state_ports": [
                    {
                        "name": "covariance_state",
                        "artifact_kind": "matrix",
                        "Persist": False,
                    }
                ],
                "time_complexity": "O(d^2)",
                "space_complexity": "O(d^2)",
                "matched_primitive": fqdn,
            }
        ],
        "edges": [],
    }
    (sub_dir / "cdg.json").write_text(json.dumps(cdg_payload, indent=2), encoding="utf-8")
    return AtomSource(
        name=f"synth-{pkg_leaf}",
        package=f"sciona.atoms.{pkg_leaf}",
        path=str(repo_dir),
        python_path="src",
    )


class TestTwoPlaneCompiledCatalog:
    def test_compile_roundtrip_plane2_survival_scoping_and_fts5_latency(
        self, tmp_path: Path
    ) -> None:
        src_alpha = _create_synthetic_provider_repo(tmp_path, "synth_alpha", "alpha")
        src_beta = _create_synthetic_provider_repo(tmp_path, "synth_beta", "beta")
        cfg = SourcesConfig(sources=[src_alpha, src_beta])

        # Verify Plane-1 seeding preserves concrete Python types + witness domain tags + state_ports
        cat_plane1 = PrimitiveCatalog()
        added = seed_catalog_from_sources(
            cat_plane1,
            config=cfg,
            base_dir=tmp_path,
            include_live_registries=False,
        )
        assert added == 2
        alpha_fqdn = "sciona.atoms.synth_alpha.kalman_block.atoms.kalman_update_alpha"
        prim = cat_plane1.get(alpha_fqdn)
        assert prim is not None
        assert prim.category == ConceptType.CONJUGATE_UPDATE
        assert prim.phase == "update"
        assert len(prim.state_ports) == 1
        assert prim.state_ports[0].artifact_kind == "matrix"
        # Concrete Python np.ndarray beat witness AbstractSignal, while inheriting witness signal tag
        obs_port = prim.inputs[1]
        assert obs_port.type_desc == "np.ndarray"
        assert obs_port.resolved_port_type().kind == "array"
        assert "signal" in obs_port.resolved_port_type().domain_tags

        # Compile Two-Plane manifest.sqlite
        manifest_path = tmp_path / "manifest.sqlite"
        summary1 = compile_local_manifest_sqlite(
            manifest_path,
            config=cfg,
            base_dir=tmp_path,
            installed_only=True,
        )
        assert summary1["atoms_compiled"] == 2

        # Upsert dynamic Plane-2 benchmarks + multi-regime uncertainty profile with ZERO code changes
        upsert_stats = upsert_plane2_metadata(
            manifest_path,
            benchmarks=[
                {
                    "atom_fqdn": alpha_fqdn,
                    "content_hash": "eval_run_1",
                    "benchmark_id": "synthetic_suite_01",
                    "benchmark_name": "synthetic_suite_01",
                    "metric_name": "rmse",
                    "metric_value": 0.031,
                    "dataset_tag": "opaque_synth_v1",
                    "measured_at": "2026-11-10T00:00:00Z",
                }
            ],
            uncertainty_profiles=[
                {
                    "atom_fqdn": alpha_fqdn,
                    "content_hash": "cal_v2",
                    "version": 2,
                    "profile": {
                        "version": 2,
                        "atom": alpha_fqdn,
                        "default_regime": {"scalar_factor": 1.08, "trust_tier": "empirical"},
                    },
                    "updated_at": "2026-11-10T00:00:00Z",
                }
            ],
        )
        assert upsert_stats["benchmarks_upserted"] == 1
        assert upsert_stats["uncertainty_profiles_upserted"] == 1

        # Recompile Plane-1 sources into the same manifest.sqlite -> Plane-2 rows MUST survive!
        summary2 = compile_local_manifest_sqlite(
            manifest_path,
            config=cfg,
            base_dir=tmp_path,
            installed_only=True,
        )
        assert summary2["benchmarks_merged"] == 1
        assert summary2["uncertainty_profiles_merged"] == 1
        loaded_profiles = load_uncertainty_profiles_from_manifest(manifest_path)
        assert loaded_profiles[alpha_fqdn]["default_regime"]["scalar_factor"] == pytest.approx(1.08)

        # Round-trip back into PrimitiveCatalog: explicit ConceptType, phase, state_ports, PortTypeSpec survive!
        cat_roundtrip = PrimitiveCatalog()
        seeded = seed_catalog_from_manifest_sqlite(
            cat_roundtrip,
            manifest_path,
            skip_locally_installed=False,
        )
        assert seeded == 2
        rt_prim = cat_roundtrip.get(alpha_fqdn)
        assert rt_prim is not None
        assert rt_prim.category == ConceptType.CONJUGATE_UPDATE
        assert rt_prim.phase == "update"
        assert len(rt_prim.state_ports) == 1
        assert "signal" in rt_prim.inputs[1].resolved_port_type().domain_tags

        # ManifestFTSIndex satisfies SemanticIndex and runs FTS5 + PortTypeSpec search in <5 ms
        with ManifestFTSIndex(manifest_path) as fts_idx:
            assert isinstance(fts_idx, SemanticIndex)
            t0 = time.perf_counter()
            # Query specifically mentioning synth_alpha ranks alpha_fqdn first
            hits = fts_idx.search_by_embedding("Bayesian Kalman measurement synth alpha", k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert len(hits) == 2
            assert hits[0][0].name == alpha_fqdn
            assert elapsed_ms < 25.0  # typically <1 ms; generous CI guard

            type_hits = fts_idx.search_by_type(
                "(prior: NDArray[np.float64], obs: NDArray[np.float64]) -> NDArray[np.float64]",
                k=5,
            )
            assert any(d.name == alpha_fqdn for d in type_hits)

        # Prune manifest to a single installed PEP 420 namespace
        prune_stats = prune_manifest_to_namespaces(
            manifest_path,
            ["sciona.atoms.synth_alpha"],
        )
        assert prune_stats == {"kept_atoms": 1, "pruned_atoms": 1}
        cat_scoped = PrimitiveCatalog()
        assert (
            seed_catalog_from_manifest_sqlite(
                cat_scoped, manifest_path, skip_locally_installed=False
            )
            == 1
        )
        assert cat_scoped.get(alpha_fqdn) is not None

    def test_workspace_sources_yml_is_deduplicated_and_normalized(self) -> None:
        from sciona.architect.compiled_catalog import discover_installed_atom_namespaces

        matcher_root = Path(__file__).resolve().parents[1]
        cfg = load_sources(matcher_root / "sources.yml")
        names = [s.name for s in cfg.sources]
        paths = [s.path for s in cfg.sources]
        assert len(names) == len(set(names)) == 10, "Duplicate source names in sources.yml"
        assert len(paths) == len(set(paths)) == 10, "Duplicate repo paths in sources.yml"
        namespaces = discover_installed_atom_namespaces(
            config=cfg,
            base_dir=matcher_root,
            include_sources_yml=True,
        )
        assert "sciona.atoms" not in namespaces  # must expand to concrete PEP 420 subpackages
        assert "sciona.atoms.ml" in namespaces
        assert "sciona.atoms.signal_processing" in namespaces
        assert "sciona.atoms.state_estimation" in namespaces
