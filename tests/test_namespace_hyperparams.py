from __future__ import annotations

from pathlib import Path

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.hyperparams import load_manifest
from sciona.architect.models import AlgorithmicPrimitive, ConceptType
from sciona.sources import load_sources, resolve_source


def _primitive(name: str) -> AlgorithmicPrimitive:
    return AlgorithmicPrimitive(
        name=name,
        source="sciona-atoms-signal",
        category=ConceptType.SIGNAL_TRANSFORM,
        description=f"{name} description",
    )


def test_live_namespace_hyperparams_manifest_loads_and_attaches_signal_tunables() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_sources(repo_root / "sources.yml")
    source = next(
        src for src in config.sources if src.name == "sciona-atoms-signal"
    )

    source_root = resolve_source(source, base_dir=repo_root)
    tunables_map = load_manifest(source_root / "data" / "hyperparams" / "manifest.json")

    assert sorted(tunables_map) == [
        "heart_rate_computation_median_smoothed",
        "peak_correction",
        "reject_outlier_intervals",
    ]
    assert [spec.name for spec in tunables_map["peak_correction"]] == ["tol"]
    assert [spec.name for spec in tunables_map["reject_outlier_intervals"]] == [
        "mad_scale",
        "min_interval_s",
        "max_interval_s",
    ]
    assert [
        spec.name for spec in tunables_map["heart_rate_computation_median_smoothed"]
    ] == ["smoothing_window"]

    catalog = PrimitiveCatalog()
    catalog.add(_primitive("peak_correction"))
    catalog.add(_primitive("reject_outlier_intervals"))
    catalog.add(_primitive("heart_rate_computation_median_smoothed"))

    attached = catalog.attach_tunables(tunables_map)

    assert attached == 3
    assert [spec.name for spec in catalog.get("peak_correction").tunable_params] == [
        "tol"
    ]
    assert [
        spec.name
        for spec in catalog.get("heart_rate_computation_median_smoothed").tunable_params
    ] == ["smoothing_window"]
