"""Live in-memory Prototyping API for the CDG Visualizer.

Backs interactive CDG prototyping, stepping, multi-rung refinement, evolution
trace streaming, and standalone Python/CDG export over HTTP without subprocesses.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.handoff import CDGExport
from sciona.principal.expansion import ExpansionEngine
from sciona.principal.expansion_rules import default_rule_sets
from sciona.prototyping.benchmarks import (
    BENCHMARK_PRESETS,
    build_benchmark_session,
    list_benchmark_presets,
    resolve_preset_id,
)
from sciona.prototyping.harness import (
    EvalCase,
    ProblemHarness,
    ResidentSliceHarness,
    SyntheticSuiteHarness,
    infer_cdg_root_ports,
)
from sciona.prototyping.session import PrototypingSession

router = APIRouter()

_MAX_ACTIVE_SESSIONS = 64
_ACTIVE_PROTO_SESSIONS: OrderedDict[str, PrototypingSession] = OrderedDict()


def _store_session(session: PrototypingSession) -> None:
    sid = session.session_id
    if sid in _ACTIVE_PROTO_SESSIONS:
        _ACTIVE_PROTO_SESSIONS.move_to_end(sid)
    _ACTIVE_PROTO_SESSIONS[sid] = session
    while len(_ACTIVE_PROTO_SESSIONS) > _MAX_ACTIVE_SESSIONS:
        _ACTIVE_PROTO_SESSIONS.popitem(last=False)


def _get_session(session_id: str) -> PrototypingSession:
    session = _ACTIVE_PROTO_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prototyping session {session_id!r} not found.",
        )
    _ACTIVE_PROTO_SESSIONS.move_to_end(session_id)
    return session


def _coerce_json_value(val: Any) -> Any:
    if isinstance(val, list):
        return np.asarray(val)
    return val


def _build_custom_session(
    cdg: CDGExport,
    *,
    session_id: str,
    goal: str = "",
    cases: list[dict[str, Any]] | None = None,
    metric: str = "rmse",
    hpo_trials: int = 8,
) -> PrototypingSession:
    catalog = PrimitiveCatalog()
    try:
        seed_builtin_primitives(catalog)
    except Exception:
        pass

    harness: ProblemHarness
    if cases:
        eval_cases: list[EvalCase] = []
        for idx, item in enumerate(cases):
            raw_inputs = item.get("inputs", {})
            inputs = {str(k): _coerce_json_value(v) for k, v in raw_inputs.items()}
            exp = item.get("expected")
            if isinstance(exp, list):
                exp = np.asarray(exp)
            eval_cases.append(
                EvalCase(
                    case_id=str(item.get("case_id") or f"case_{idx}"),
                    inputs=inputs,
                    expected=exp,
                )
            )
        r_in, r_out = infer_cdg_root_ports(cdg)
        harness = ResidentSliceHarness(
            eval_cases,
            root_inputs=r_in,
            root_outputs=r_out,
            metric=metric,
        )
    else:
        harness = SyntheticSuiteHarness.from_cdg(cdg)

    return PrototypingSession(
        cdg,
        goal=goal or str((cdg.metadata or {}).get("goal") or "Visualizer Prototyping Session"),
        session_id=session_id,
        harness=harness,
        catalog=catalog,
        expansion_engine=ExpansionEngine(default_rule_sets()),
        hpo_trials_per_rung=hpo_trials,
    )


class ProtoStartRequest(BaseModel):
    preset_id: str | None = None
    repo: str | None = None
    cdg: CDGExport | None = None
    goal: str = ""
    cases: list[dict[str, Any]] | None = None
    metric: str = "rmse"
    hpo_trials: int = Field(default=8, ge=1, le=64)


class ProtoStepRequest(BaseModel):
    session_id: str
    max_rung: int = Field(default=4, ge=1, le=4)


class ProtoRunRequest(BaseModel):
    session_id: str | None = None
    preset_id: str | None = None
    repo: str | None = None
    cdg: CDGExport | None = None
    goal: str = ""
    cases: list[dict[str, Any]] | None = None
    metric: str = "rmse"
    hpo_trials: int = Field(default=8, ge=1, le=64)
    max_iterations: int = Field(default=6, ge=1, le=20)
    time_budget_s: float = Field(default=15.0, gt=0.0, le=120.0)
    target_loss: float | None = None
    max_rung: int = Field(default=4, ge=1, le=4)
    verify_holdout: bool = True


class ProtoExportRequest(BaseModel):
    session_id: str
    format: Literal["python", "cdg", "both"] = "both"


def _resolve_or_create_session(
    *,
    session_id: str | None = None,
    preset_id: str | None = None,
    repo: str | None = None,
    cdg: CDGExport | None = None,
    goal: str = "",
    cases: list[dict[str, Any]] | None = None,
    metric: str = "rmse",
    hpo_trials: int = 8,
) -> tuple[PrototypingSession, str]:
    if session_id:
        return _get_session(session_id), metric

    new_sid = f"proto-{uuid.uuid4().hex[:8]}"
    preset_Lookup = preset_id or repo
    if preset_Lookup:
        try:
            canonical = resolve_preset_id(preset_Lookup)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        sess = build_benchmark_session(canonical, session_id=new_sid)
        obj = BENCHMARK_PRESETS[canonical].objective
        _store_session(sess)
        return sess, obj

    if cdg is not None:
        sess = _build_custom_session(
            cdg,
            session_id=new_sid,
            goal=goal,
            cases=cases,
            metric=metric,
            hpo_trials=hpo_trials,
        )
        _store_session(sess)
        return sess, metric

    raise HTTPException(
        status_code=422,
        detail="Provide session_id, preset_id, repo, or cdg.",
    )


@router.get("/api/cdg/proto/presets")
async def get_proto_presets() -> dict[str, Any]:
    """List live executable cross-domain synthetic prototyping presets."""
    return {"presets": list_benchmark_presets()}


@router.post("/api/cdg/proto/start")
async def start_proto_session(body: ProtoStartRequest) -> dict[str, Any]:
    """Initialize a stateful ``PrototypingSession`` and run Tier 0-2 initial evaluation."""
    session, objective = _resolve_or_create_session(
        preset_id=body.preset_id,
        repo=body.repo,
        cdg=body.cdg,
        goal=body.goal,
        cases=body.cases,
        metric=body.metric,
        hpo_trials=body.hpo_trials,
    )
    rec = session.evaluate_initial()
    return {
        "session_id": session.session_id,
        "goal": session.goal,
        "initial_step": rec.to_dict(),
        "best_loss": session.best_loss if math.isfinite(session.best_loss) else None,
        "cache_stats": session.runner.cache.stats(),
        "trace": session.to_evolution_trace(objective=objective),
    }


@router.post("/api/cdg/proto/step")
async def step_proto_session(body: ProtoStepRequest) -> dict[str, Any]:
    """Execute one multi-rung diagnostic & refinement iteration on an active session."""
    session = _get_session(body.session_id)
    rec = await session.step(max_rung=body.max_rung)
    trace = session.to_evolution_trace()
    latest_graph = (
        trace["versions"][-1]["graph"]
        if trace.get("versions")
        else session.best_cdg.model_dump(mode="json")
    )
    return {
        "session_id": session.session_id,
        "step": rec.to_dict(),
        "improved": rec.improved,
        "operation": rec.rung,
        "selection_reason": rec.rationale,
        "best_loss": session.best_loss if math.isfinite(session.best_loss) else None,
        "updated_cdg": latest_graph,
        "cache_stats": session.runner.cache.stats(),
        "trace": trace,
    }


@router.post("/api/cdg/proto/run")
async def run_proto_session(body: ProtoRunRequest) -> dict[str, Any]:
    """Run iterative multi-rung prototyping until convergence + Tier 3 holdout check."""
    session, objective = _resolve_or_create_session(
        session_id=body.session_id,
        preset_id=body.preset_id,
        repo=body.repo,
        cdg=body.cdg,
        goal=body.goal,
        cases=body.cases,
        metric=body.metric,
        hpo_trials=body.hpo_trials,
    )
    summary = await session.run(
        max_iterations=body.max_iterations,
        time_budget_s=body.time_budget_s,
        target_loss=body.target_loss,
        max_rung=body.max_rung,
        verify_holdout=body.verify_holdout,
    )
    return {
        "session_id": session.session_id,
        "summary": summary.to_dict(),
        "trace": session.to_evolution_trace(objective=objective),
    }


@router.get("/api/cdg/proto/sessions/{session_id}")
async def get_proto_session_state(session_id: str) -> dict[str, Any]:
    """Retrieve the current telemetry, history, and evolution trace of an active session."""
    session = _get_session(session_id)
    return {
        "session_id": session.session_id,
        "goal": session.goal,
        "initial_loss": session.initial_loss if math.isfinite(session.initial_loss) else None,
        "best_loss": session.best_loss if math.isfinite(session.best_loss) else None,
        "best_node_params": session.best_node_params,
        "history": [h.to_dict() for h in session.history],
        "cache_stats": session.runner.cache.stats(),
        "trace": session.to_evolution_trace(),
    }


@router.post("/api/cdg/proto/export")
async def export_proto_session(body: ProtoExportRequest) -> dict[str, Any]:
    """Export standalone runnable Python source and/or CDG JSON for an active session."""
    session = _get_session(body.session_id)
    result: dict[str, Any] = {"session_id": session.session_id}
    if body.format in ("python", "both"):
        result["python_source"] = session.render_python_source()
    if body.format in ("cdg", "both"):
        cdg_copy = session.best_cdg.model_copy(deep=True)
        cdg_copy.metadata = dict(cdg_copy.metadata or {})
        cdg_copy.metadata["prototyping_best_loss"] = (
            session.best_loss if math.isfinite(session.best_loss) else None
        )
        cdg_copy.metadata["best_node_params"] = dict(session.best_node_params)
        if session.synthesized_micro_atoms:
            cdg_copy.metadata["synthesized_micro_atoms"] = dict(
                session.synthesized_micro_atoms
            )
        result["cdg"] = cdg_copy.model_dump(mode="json")
    return result
