"""Phase E verification: Live visualizer prototyping API, real diffs, UI assets, and benchmark gate."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sciona.architect.handoff import CDGExport
from sciona.commands.proto_cmds import _cmd_proto
from sciona.prototyping.benchmark_gate import (
    GateSLOs,
    run_prototyping_benchmark_suite,
)
from sciona.prototyping.benchmarks import (
    BENCHMARK_PRESETS,
    build_benchmark_session,
    list_benchmark_presets,
    resolve_preset_id,
)
from sciona.visualizer.app import create_app


@pytest.fixture()
def proto_client() -> Iterator[TestClient]:
    """Create a FastAPI TestClient without requiring a live Neo4j/Memgraph instance."""
    app = create_app()

    @asynccontextmanager
    async def _test_lifespan(a: FastAPI) -> AsyncIterator[None]:
        a.state.driver = AsyncMock()
        yield

    app.router.lifespan_context = _test_lifespan
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_cross_domain_presets_and_repo_aliases_are_purely_synthetic() -> None:
    presets = list_benchmark_presets()
    preset_ids = {p["preset_id"] for p in presets}
    assert preset_ids == {
        "waveform_event_cadence",
        "stateful_tabular_classifier",
        "calibrated_linear_solver",
        "novel_nonlinear_micro_atom",
    }
    assert resolve_preset_id("showcase/public-waveform-rate") == "waveform_event_cadence"
    assert (
        resolve_preset_id("showcase/public-tabular-classification")
        == "stateful_tabular_classifier"
    )
    assert resolve_preset_id("showcase/public-calibrated-solver") == "calibrated_linear_solver"
    assert (
        resolve_preset_id("showcase/public-nonlinear-calibration")
        == "novel_nonlinear_micro_atom"
    )


def test_proto_api_interactive_start_step_get_and_export(proto_client: TestClient) -> None:
    presets_resp = proto_client.get("/api/cdg/proto/presets")
    assert presets_resp.status_code == 200
    assert len(presets_resp.json()["presets"]) == 4

    start_resp = proto_client.post(
        "/api/cdg/proto/start",
        json={"preset_id": "waveform_event_cadence"},
    )
    assert start_resp.status_code == 200
    started = start_resp.json()
    session_id = started["session_id"]
    assert started["trace"]["schema_version"] == "1.0"
    assert len(started["trace"]["versions"]) == 1
    v1_loss = started["best_loss"]
    assert v1_loss > 100.0

    # Step 1: Rung 1 node HPO
    step1_resp = proto_client.post(
        "/api/cdg/proto/step",
        json={"session_id": session_id},
    )
    assert step1_resp.status_code == 200
    step1 = step1_resp.json()
    assert step1["step"]["rung"] == "rung1_node_hpo"
    assert step1["step"]["improved"] is True
    assert step1["best_loss"] < v1_loss
    assert len(step1["trace"]["versions"]) == 2
    assert len(step1["trace"]["transitions"]) == 1

    # Step 2: Rung 2 sibling swap (mean -> median cadence)
    step2_resp = proto_client.post(
        "/api/cdg/proto/step",
        json={"session_id": session_id},
    )
    assert step2_resp.status_code == 200
    step2 = step2_resp.json()
    assert step2["step"]["rung"] == "rung2_sibling_swap"
    assert step2["step"]["improved"] is True
    assert step2["best_loss"] < 2.0
    assert len(step2["trace"]["versions"]) == 3

    # Inspect session state endpoint
    get_resp = proto_client.get(f"/api/cdg/proto/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id

    # Export standalone Python module
    exp_resp = proto_client.post(
        "/api/cdg/proto/export",
        json={"session_id": session_id},
    )
    assert exp_resp.status_code == 200
    exported = exp_resp.json()
    assert "def run_pipeline(" in exported["python_source"]
    ast.parse(exported["python_source"], mode="exec")


def test_proto_api_full_run_stateful_tabular_and_telemetry(proto_client: TestClient) -> None:
    run_resp = proto_client.post(
        "/api/cdg/proto/run",
        json={"repo": "showcase/public-tabular-classification", "verify_holdout": True},
    )
    assert run_resp.status_code == 200
    data = run_resp.json()
    summary = data["summary"]
    assert summary["holdout"]["passed"] is True
    assert summary["best_loss"] < summary["initial_loss"]
    assert summary["cache_stats"]["hits"] >= 1

    trace = data["trace"]
    assert trace["schema_version"] == "1.0"
    assert len(trace["versions"]) >= 2
    latest_graph = trace["versions"][-1]["graph"]
    atomic_nodes = [n for n in latest_graph["nodes"] if n.get("status") == "atomic"]
    assert any(
        isinstance(n.get("telemetry"), dict) and n["telemetry"].get("status") == "ok"
        for n in atomic_nodes
    )


def test_delta_planner_apply_fix_emits_real_unified_diff_and_topology_delta(
    proto_client: TestClient,
) -> None:
    before_cdg: dict[str, Any] = {
        "nodes": [
            {
                "node_id": "feat_ext",
                "name": "Extract Raw Features",
                "description": "Compute unclipped features",
                "concept_type": "data_extraction",
                "status": "atomic",
                "matched_primitive": "sciona.prototyping.benchmarks.tabular_raw_features",
                "inputs": [{"name": "raw_x", "type_desc": "np.ndarray"}],
                "outputs": [{"name": "features", "type_desc": "np.ndarray"}],
                "depth": 1,
            },
            {
                "node_id": "fit_cls",
                "name": "Fit Ridge Logit",
                "description": "Fit linear classifier",
                "concept_type": "ml_model_selection",
                "status": "atomic",
                "matched_primitive": "sciona.prototyping.benchmarks.tabular_fit_ridge_logit",
                "inputs": [
                    {"name": "features", "type_desc": "np.ndarray"},
                    {"name": "train_y", "type_desc": "np.ndarray"},
                ],
                "outputs": [{"name": "weights", "type_desc": "np.ndarray"}],
                "depth": 1,
            },
        ],
        "edges": [
            {
                "source_id": "feat_ext",
                "target_id": "fit_cls",
                "output_name": "features",
                "input_name": "features",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            }
        ],
        "metadata": {"goal": "Tabular classification"},
    }
    after_cdg = CDGExport.model_validate(
        {
            "nodes": [
                before_cdg["nodes"][0],
                {
                    "node_id": "clip_spikes",
                    "name": "Clip Outlier Spikes",
                    "description": "Winsorize extreme outlier spikes",
                    "concept_type": "data_assembly",
                    "status": "atomic",
                    "matched_primitive": "sciona.prototyping.benchmarks.tabular_outlier_clipper",
                    "inputs": [{"name": "features", "type_desc": "np.ndarray"}],
                    "outputs": [{"name": "features", "type_desc": "np.ndarray"}],
                    "depth": 1,
                },
                before_cdg["nodes"][1],
            ],
            "edges": [
                {
                    "source_id": "feat_ext",
                    "target_id": "clip_spikes",
                    "output_name": "features",
                    "input_name": "features",
                    "source_type": "np.ndarray",
                    "target_type": "np.ndarray",
                },
                {
                    "source_id": "clip_spikes",
                    "target_id": "fit_cls",
                    "output_name": "features",
                    "input_name": "features",
                    "source_type": "np.ndarray",
                    "target_type": "np.ndarray",
                },
            ],
            "metadata": {"goal": "Tabular classification"},
        }
    )

    with patch("sciona.principal.expansion.ExpansionEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine._rule_index = {"insert_outlier_clipper": MagicMock()}
        mock_res = MagicMock()
        mock_res.is_failure = False
        mock_res.unwrap.return_value = after_cdg
        mock_engine._rewriter.apply_rule.return_value = mock_res
        mock_engine_cls.return_value = mock_engine

        resp = proto_client.post(
            "/api/delta_planner/apply_fix",
            json={"rule_name": "insert_outlier_clipper", "cdg": before_cdg},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "--- base_pipeline.py" in body["diff"]
        assert "+++ outlier_clipper.py" in body["diff"]
        assert "+    # Stage: clip_spikes | Clip Outlier Spikes" in body["diff"]
        assert "+    _kwargs['features'] = node_outs['clip_spikes']['features']" in body["diff"]
        assert "tabular_outlier_clipper" in body["diff"]
        assert "nodes 2 -> 3" in body["logs"]
        assert "clip_spikes" in body["logs"]


def test_static_app_js_confidentiality_and_live_proto_wiring() -> None:
    app_js_path = Path(__file__).resolve().parents[1] / "sciona" / "static" / "app.js"
    js_text = app_js_path.read_text(encoding="utf-8")

    assert "synthetic-waveform-cadence-v1" in js_text
    assert "loss: null" not in js_text
    assert "/api/cdg/proto/run" in js_text
    assert "/api/cdg/proto/step" in js_text
    assert "function buildTutorialCEvolution(" in js_text


@pytest.mark.asyncio
async def test_cross_domain_prototyping_benchmark_gate_passes_all_slos() -> None:
    report = await run_prototyping_benchmark_suite(slos=GateSLOs())
    assert report.passed is True, f"SLO gate failed with violations: {report.violations}"
    assert report.violations == []
    assert report.total_subprocess_spawns == 0
    assert report.total_wall_clock_ms < 3000.0
    assert report.zero_llm_step_ratio >= 0.75
    assert report.aggregate_cache_hit_rate >= 0.45
    assert report.token_reduction_vs_iterative_pct >= 85.0
    assert len(report.tasks) == len(BENCHMARK_PRESETS)

    for task in report.tasks:
        assert task.wall_clock_ms < 1000.0
        assert task.holdout_passed is True
        assert task.exported_module_valid is True
        assert task.holdout_loss < task.single_shot_baseline.holdout_loss


@pytest.mark.asyncio
async def test_cli_proto_benchmark_gate_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        benchmark_gate=True,
        preset="waveform_event_cadence",
        cdg=None,
        harness=None,
        max_iterations=6,
        hpo_trials=10,
        export_py=None,
        export_cdg=None,
        no_holdout=False,
        json=True,
    )
    await _cmd_proto(args)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["passed"] is True
    assert payload["total_subprocess_spawns"] == 0
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["preset_id"] == "waveform_event_cadence"


@pytest.mark.asyncio
async def test_calibrated_linear_solver_session_repairs_risk_budget_violation() -> None:
    session = build_benchmark_session("calibrated_linear_solver")
    summary = await session.run(verify_holdout=True)
    assert summary.initial_loss == float("inf")
    assert summary.best_loss < 0.08
    assert summary.holdout_result is not None
    assert summary.holdout_result.passed is True
