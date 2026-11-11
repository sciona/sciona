"""CLI handler for ``sciona proto`` — fast in-process CDG prototyping & refinement loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport, load_json
from sciona.principal.expansion import ExpansionEngine
from sciona.principal.expansion_rules import default_rule_sets
from sciona.prototyping.harness import (
    ProblemHarness,
    ResidentSliceHarness,
    SyntheticSuiteHarness,
    infer_cdg_root_ports,
)
from sciona.prototyping.session import PrototypingSession


def load_slice_harness(
    data_path: Path,
    cdg: CDGExport,
    *,
    metric: str = "rmse",
    target_key: str | None = None,
) -> ProblemHarness:
    """Load a ``ResidentSliceHarness`` from JSON, NPZ, NPY, CSV, or Parquet."""
    root_in, root_out = infer_cdg_root_ports(cdg)
    return ResidentSliceHarness.from_dataset_file(
        data_path,
        root_inputs=root_in,
        root_outputs=root_out,
        target_key=target_key or "expected",
        metric=metric,
    )


async def _cmd_proto(args: argparse.Namespace) -> None:
    """Execute ``sciona proto``."""
    cdg_arg = getattr(args, "cdg", None)
    goal_arg = getattr(args, "goal", None) or ""
    if not cdg_arg and not goal_arg:
        print("Error: provide either --cdg <cdg.json> or --goal <problem>", file=sys.stderr)
        sys.exit(1)

    catalog: PrimitiveCatalog | None = None
    cat_arg = getattr(args, "catalog", None)
    if cat_arg:
        catalog = PrimitiveCatalog.load(Path(cat_arg))
    else:
        try:
            from sciona.architect.catalog import seed_builtin_primitives

            catalog = PrimitiveCatalog()
            seed_builtin_primitives(catalog)
        except Exception:
            catalog = None

    if cdg_arg:
        cdg = load_json(Path(cdg_arg))
    else:
        from sciona.sdk import Sciona

        sdk = await Sciona.create()
        catalog = sdk.catalog
        prop = await sdk.propose(problem=goal_arg)
        cdg = prop.cdg

    metric_name = str(getattr(args, "metric", "rmse") or "rmse")
    data_arg = getattr(args, "data", None)
    target_key = getattr(args, "target_key", None)
    harness: ProblemHarness
    if data_arg:
        harness = load_slice_harness(
            Path(data_arg),
            cdg,
            metric=metric_name,
            target_key=target_key,
        )
    else:
        harness = SyntheticSuiteHarness.from_cdg(cdg)

    manifest_arg = getattr(args, "manifest", None)
    manifest_path: Path | None = Path(manifest_arg) if manifest_arg else None
    if manifest_path is None:
        default_manifest = Path.home() / ".sciona" / "manifest.sqlite"
        if default_manifest.is_file():
            manifest_path = default_manifest

    session = PrototypingSession(
        cdg,
        goal=goal_arg or str(cdg.metadata.get("goal") or "sciona proto"),
        harness=harness,
        catalog=catalog,
        manifest_sqlite_path=manifest_path,
        expansion_engine=ExpansionEngine(default_rule_sets()),
        hpo_trials_per_rung=int(getattr(args, "hpo_trials", 8)),
    )

    summary = await session.run(
        max_iterations=int(getattr(args, "max_iterations", 6)),
        time_budget_s=float(getattr(args, "budget", 15.0)),
        target_loss=getattr(args, "target_loss", None),
        max_rung=int(getattr(args, "max_rung", 4)),
        verify_holdout=not bool(getattr(args, "no_holdout", False)),
    )

    exported_cdg_path: str | None = None
    out_cdg = getattr(args, "output_cdg", None)
    if out_cdg:
        exported_cdg_path = str(session.export_cdg(out_cdg))

    exported_py_path: str | None = None
    out_py = getattr(args, "export", None)
    if out_py:
        exported_py_path = str(session.export_python_module(out_py))

    payload = summary.to_dict()
    if exported_cdg_path:
        payload["exported_cdg"] = exported_cdg_path
    if exported_py_path:
        payload["exported_module"] = exported_py_path

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return

    print(
        f"Prototyping complete: initial_loss={summary.initial_loss:.6g} -> "
        f"best_loss={summary.best_loss:.6g} "
        f"({summary.iterations_completed} iter(s), {summary.total_trials_evaluated} eval(s), "
        f"{summary.total_elapsed_ms:.1f} ms, cache_hit_rate={summary.cache_stats.get('hit_rate', 0.0):.1%})"
    )
    for rec in summary.history:
        status_mark = "OK" if rec.improved else "--"
        print(
            f"  [iter {rec.iteration}] [{status_mark}] {rec.rung}: {rec.rationale} "
            f"({rec.elapsed_ms:.1f} ms)"
        )
    if summary.holdout_result is not None:
        h_state = "PASS" if summary.holdout_result.passed else "FAIL"
        print(f"  [Tier 3 Holdout] [{h_state}] loss={summary.holdout_result.loss:.6g}")
    if exported_cdg_path:
        print(f"  Exported CDG: {exported_cdg_path}")
    if exported_py_path:
        print(f"  Exported Python module: {exported_py_path}")
