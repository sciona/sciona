"""Command for full orchestration: decompose -> match -> refine -> assemble."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sciona.commands.llm_helpers import _create_llm_router, _warm_llm_if_supported
from sciona.commands.routing_helpers import (
    _mode_feature_summary,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_retrieval_policy,
    _resolve_retrieval_policy,
    _routing_metadata_summary,
    _summarize_prompt_routing,
)
from sciona.commands.runtime_helpers import (
    _create_proof_env,
    _load_architect_catalog,
    _load_semantic_index,
    _load_skill_index_or_empty,
    _shutdown_telemetry_drain,
)
from sciona.commands.shared_context_helpers import (
    _create_shared_context,
    _print_shared_context_metrics,
    _shared_context_metadata,
    _write_shared_context_metrics_file,
)
from sciona.runtime_paths import (
    _build_rapid_direct_cdg,
    _run_rapid_direct_match,
    _run_structured_single_pass,
)

if TYPE_CHECKING:
    from sciona.types import Prover


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full orchestration loop: decompose -> match -> refine -> assemble."""
    from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
    from sciona.architect.checkpointer import create_checkpointer
    from sciona.architect.graph import DecompositionAgent
    from sciona.architect.handoff import save_json
    from sciona.config import AgeomConfig, resolve_execution_mode
    from sciona.hunter.graph import HunterAgent
    from sciona.judge.checker import VerificationOracleImpl
    from sciona.orchestrator import run_orchestration
    from sciona.services import (
        ArchitectService,
        HunterService,
        OrchestratorService,
        SingleAgentPlanner,
    )
    from sciona.services.catalog_artifact_retrieval import (
        build_default_macro_retriever,
    )
    from sciona.telemetry import (
        configure_dashboard_output,
        configure_postgres_telemetry,
        finish_run,
        get_event_log,
        merge_run_metadata,
        start_run,
        telemetry_scope,
        telemetry_stage,
        update_stage,
    )
    from sciona.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    prover = Prover(args.prover)
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_mode_summary("run", mode_settings)

    configure_dashboard_output(config.telemetry_runs_dir)

    # Postgres telemetry drain
    _telem_drain = None
    _telem_store = None
    if config.telemetry_backend != "file" and config.postgres_uri:
        try:
            from sciona.telemetry_store import PostgresTelemetryStore, TelemetryDrain

            _telem_store = PostgresTelemetryStore(config.postgres_uri)
            await _telem_store.setup()
            _telem_drain = TelemetryDrain(_telem_store)
            configure_postgres_telemetry(_telem_store, _telem_drain)
            await _telem_drain.start()
        except Exception:
            _telem_drain = None
            _telem_store = None

    event_log = get_event_log()
    event_log.configure_live_output(None)
    event_log.clear()
    if getattr(args, "trace", False):
        event_log.configure_live_output(output_dir / "trace.jsonl")
    catalog, catalog_alignment = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
        config=config,
    )
    _print_retrieval_policy(retrieval_policy)
    architect_routing = None
    if mode_settings.mode != "rapid":
        architect_routing = _summarize_prompt_routing(
            config,
            "architect",
            [
                "architect_strategy",
                "architect_decompose",
                "architect_critique",
                "orchestrator_refine",
            ],
            mode_settings.mode,
        )
    hunter_routing = _summarize_prompt_routing(
        config,
        "hunter",
        [
            "hunter_score",
            "hunter_reformulate",
            "hunter_analyze_failure",
        ],
        mode_settings.mode,
    )
    telemetry_run_id = start_run(
        "algorithm_creation",
        label=getattr(args, "label", ""),
        metadata={
            "command": "run",
            "goal": args.goal,
            "prover": prover.value,
            "max_rounds": int(args.max_rounds),
            "execution_mode": mode_settings.mode,
            "execution_path": (
                "rapid_direct"
                if mode_settings.mode == "rapid"
                else "structured_single_pass"
                if mode_settings.mode == "structured"
                else "single_agent_planner"
                if mode_settings.mode == "single_agent"
                else "verified_orchestration"
            ),
            "mode_features": _mode_feature_summary(mode_settings),
            "retrieval_policy": {
                "catalog_confidence": retrieval_policy.catalog_confidence,
                "confidence_band": retrieval_policy.confidence_band,
                "skill_index": retrieval_policy.skill_index_enabled,
                "graph_retrieval": retrieval_policy.graph_retrieval_enabled,
                "semantic_backend": retrieval_policy.semantic_index_backend_override
                or "default",
                "hunter_mode": retrieval_policy.hunter_mode,
            },
            "rapid_direct_path": mode_settings.mode == "rapid",
            "single_agent_mode": mode_settings.mode == "single_agent",
            "llm_routing": (
                {
                    "hunter": _routing_metadata_summary(hunter_routing),
                }
                if architect_routing is None
                else {
                    "architect": _routing_metadata_summary(architect_routing),
                    "hunter": _routing_metadata_summary(hunter_routing),
                }
            ),
            "catalog_alignment": catalog_alignment,
        },
    )

    architect_shared_metrics = None
    hunter_shared_metrics = None
    result = None
    planner_result = None
    try:
        with telemetry_scope(run_id=telemetry_run_id):
            update_stage(stage="setup", status="running", message="loading dependencies")

            skill_index = None
            llm = None
            if mode_settings.mode != "rapid":
                skill_index = _load_skill_index_or_empty(
                    config,
                    enabled=retrieval_policy.skill_index_enabled,
                )

                try:
                    from sciona.llm_router import (
                        ARCHITECT_CRITIQUE,
                        ARCHITECT_DECOMPOSE,
                        ARCHITECT_STRATEGY,
                        ORCHESTRATOR_REFINE,
                    )

                    architect_prompt_keys = [
                        ARCHITECT_STRATEGY,
                        ARCHITECT_DECOMPOSE,
                        ARCHITECT_CRITIQUE,
                        ORCHESTRATOR_REFINE,
                    ]
                    _print_prompt_routing_summary(
                        config,
                        "architect",
                        architect_prompt_keys,
                        mode_settings.mode,
                    )
                    llm = _create_llm_router(
                        args,
                        config,
                        "architect",
                        architect_prompt_keys,
                    )
                    await _warm_llm_if_supported(llm, "architect")
                except (ValueError, ImportError) as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)
            update_stage(stage="setup", status="completed")

            cdg = None
            architect_service = None
            if mode_settings.mode not in {"rapid", "single_agent"}:
                print(f"Decomposing: {args.goal}")

                retriever = None
                graph_store_ctx = None
                if retrieval_policy.graph_retrieval_enabled:
                    from sciona.architect.graph_retrieval import make_retriever
                    from sciona.graph_store import GraphStore as _GraphStore

                    graph_store_ctx = _GraphStore(
                        uri=config.memgraph_uri,
                        user=config.memgraph_user,
                        password=config.memgraph_password,
                    )
                architect_run_id = uuid.uuid4().hex
                architect_shared_context, architect_shared_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.architect_shared_context_enabled,
                )

                with telemetry_stage("architect_decompose", message="building initial CDG"):
                    async with create_checkpointer(config.postgres_uri) as checkpointer:
                        if graph_store_ctx is not None:
                            async with graph_store_ctx as gstore:
                                retriever = make_retriever(config, gstore, current_repo="")
                                architect = DecompositionAgent(
                                    catalog=catalog,
                                    skill_index=skill_index,
                                    llm=llm,
                                    checkpointer=checkpointer,
                                    graph_retriever=retriever,
                                    shared_context=architect_shared_context,
                                    shared_context_metrics=architect_shared_metrics,
                                    context_namespace=f"architect/{architect_run_id}",
                                    context_budget_chars=config.architect_shared_context_budget_chars,
                                    architect_critique_llm_enabled=config.architect_critique_llm_enabled,
                                )
                                architect_service = ArchitectService(architect)
                                cdg = await architect.decompose(args.goal)
                        else:
                            architect = DecompositionAgent(
                                catalog=catalog,
                                skill_index=skill_index,
                                llm=llm,
                                checkpointer=checkpointer,
                                shared_context=architect_shared_context,
                                shared_context_metrics=architect_shared_metrics,
                                context_namespace=f"architect/{architect_run_id}",
                                context_budget_chars=config.architect_shared_context_budget_chars,
                                architect_critique_llm_enabled=config.architect_critique_llm_enabled,
                            )
                            architect_service = ArchitectService(architect)
                            cdg = await architect.decompose(args.goal)

                print(f"  Decomposed: {len(cdg.nodes)} nodes, {len(cdg.edges)} edges")
            elif mode_settings.mode == "rapid":
                print(f"Rapid mode: matching goal directly without decomposition: {args.goal}")
            else:
                print("Single-agent mode: planner will attempt direct grounding before decomposition.")

                async def _architect_factory():
                    nonlocal architect_service, architect_shared_metrics
                    if architect_service is not None:
                        return architect_service

                    architect_run_id = uuid.uuid4().hex
                    architect_shared_context, architect_shared_metrics = await _create_shared_context(
                        config,
                        enabled=mode_settings.architect_shared_context_enabled,
                    )
                    class _LazyArchitectService:
                        async def decompose(self, request):
                            with telemetry_stage(
                                "architect_decompose",
                                message="building planner decomposition",
                            ):
                                async with create_checkpointer(config.postgres_uri) as checkpointer:
                                    architect = DecompositionAgent(
                                        catalog=catalog,
                                        skill_index=skill_index,
                                        llm=llm,
                                        checkpointer=checkpointer,
                                        shared_context=architect_shared_context,
                                        shared_context_metrics=architect_shared_metrics,
                                        context_namespace=f"architect/{architect_run_id}",
                                        context_budget_chars=config.architect_shared_context_budget_chars,
                                        architect_critique_llm_enabled=config.architect_critique_llm_enabled,
                                    )
                                    service = ArchitectService(architect)
                                    result = await service.decompose(request)
                                    built_cdg = result.cdg
                                    print(
                                        f"  Decomposed: {len(built_cdg.nodes)} nodes, {len(built_cdg.edges)} edges"
                                    )
                                    return result

                    architect_service = _LazyArchitectService()
                    return architect_service

            index_dir = config.index_dir
            if not index_dir.exists():
                print(
                    f"Error: index directory {index_dir} not found. Run 'sciona index build' first.",
                    file=sys.stderr,
                )
                finish_run(
                    telemetry_run_id,
                    status="failed",
                    error=f"missing index directory: {index_dir}",
                )
                sys.exit(1)

            env = None
            with telemetry_stage("hunter_setup", message="loading retrieval index"):
                index, index_mode = _load_semantic_index(
                    index_dir,
                    config,
                    backend_override=retrieval_policy.semantic_index_backend_override,
                )
                if index_mode == "lexical_fallback":
                    print(
                        "Warning: FAISS unavailable; using lexical fallback index for Hunter.",
                        file=sys.stderr,
                    )

                env = _create_proof_env(prover, config)
                if prover == Prover.LEAN4:
                    oracle = VerificationOracleImpl(lean_env=env)
                elif prover == Prover.PYTHON:
                    oracle = VerificationOracleImpl(python_env=env)
                else:
                    oracle = VerificationOracleImpl(coq_env=env)

                try:
                    from sciona.llm_router import (
                        HUNTER_ANALYZE_FAILURE,
                        HUNTER_REFORMULATE,
                        HUNTER_SCORE,
                    )

                    hunter_prompt_keys = [
                        HUNTER_SCORE,
                        HUNTER_REFORMULATE,
                        HUNTER_ANALYZE_FAILURE,
                    ]
                    _print_prompt_routing_summary(
                        config,
                        "hunter",
                        hunter_prompt_keys,
                        mode_settings.mode,
                    )
                    # Extract embedder from index for embedding-based reranking
                    _embedder = getattr(index, "_embedder", None)
                    hunter_llm = _create_llm_router(
                        args,
                        config,
                        "hunter",
                        hunter_prompt_keys,
                        embedder=_embedder,
                    )
                    await _warm_llm_if_supported(hunter_llm, "hunter")
                except (ValueError, ImportError) as exc:
                    print(f"Error setting up hunter LLM: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)

                run_id = uuid.uuid4().hex
                hunter_shared_context, hunter_shared_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.hunter_shared_context_enabled,
                )
                hunter = HunterAgent(
                    index=index,
                    oracle=oracle,
                    llm=hunter_llm,
                    max_iterations=config.hunter_max_iterations,
                    top_k_verify=config.hunter_top_k_verify,
                    search_k=config.hunter_search_k,
                    mode=retrieval_policy.hunter_mode,
                    use_gbnf=mode_settings.hunter_use_gbnf,
                    query_batch_size=config.hunter_query_batch_size,
                    top_k_per_query=config.hunter_top_k_per_query,
                    max_candidates_total=config.hunter_max_candidates_total,
                    live_catalog=catalog,
                    shared_context=hunter_shared_context,
                    shared_context_metrics=hunter_shared_metrics,
                    context_namespace="hunter",
                    run_id=run_id,
                    context_budget_chars=config.hunter_shared_context_budget_chars,
                )
                hunter_service = HunterService(hunter)

            try:
                if mode_settings.mode == "rapid":
                    with telemetry_stage(
                        "rapid_direct_match",
                        message="matching goal directly",
                    ):
                        result = await _run_rapid_direct_match(
                            args.goal,
                            prover=prover,
                            hunter=hunter,
                        )
                elif mode_settings.mode == "single_agent":
                    print("Running single-agent planner...")
                    artifact_retriever = await build_default_macro_retriever()

                    planner = SingleAgentPlanner(
                        hunter=hunter_service,
                        architect_factory=_architect_factory,
                        orchestrator=OrchestratorService(hunter, run_orchestration),
                        llm=llm,
                        prover=prover,
                        max_rounds=args.max_rounds,
                        hunter_concurrency=config.orchestrator_hunter_concurrency,
                        artifact_retriever=artifact_retriever,
                    )
                    with telemetry_stage(
                        "single_agent_planner",
                        message="tool-orchestrated direct->decompose->escalate planner",
                    ):
                        planner_result = await planner.run(args.goal)
                        result = planner_result.result
                    update_stage(
                        stage="single_agent_planner",
                        completed=planner_result.state.budget.steps_used,
                        total=planner_result.state.budget.max_steps,
                    )
                elif mode_settings.mode == "structured":
                    print("Running structured single-pass matching...")
                    with telemetry_stage(
                        "structured_match",
                        message="matching decomposed leaves once",
                    ):
                        result = await _run_structured_single_pass(
                            cdg,
                            prover=prover,
                            hunter=hunter,
                        )
                    update_stage(
                        stage="structured_match",
                        completed=len(result.match_results),
                        total=len(result.match_results),
                    )
                else:
                    print(f"Running orchestration (max {args.max_rounds} rounds)...")
                    with telemetry_stage(
                        "orchestration",
                        message="architect->hunter refine loop",
                        total=int(args.max_rounds),
                    ):
                        result = await run_orchestration(
                            cdg,
                            hunter_agent=hunter,
                            llm=llm,
                            prover=prover,
                            max_rounds=args.max_rounds,
                            hunter_concurrency=config.orchestrator_hunter_concurrency,
                        )
                    update_stage(
                        stage="orchestration",
                        completed=result.rounds_used,
                        total=int(args.max_rounds),
                    )
            finally:
                if env is not None:
                    await env.close()
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        await _shutdown_telemetry_drain(_telem_drain, _telem_store)
        raise
    else:
        # --- Auto-upsert solved runs into the graph store ---
        if config.auto_upsert_enabled and result is not None:
            try:
                from datetime import datetime, timezone

                from sciona.graph_store import GraphStore as _GraphStore
                from sciona.result_to_cdg import RunCDGMetadata, orchestrator_result_to_cdg
                from sciona.telemetry import log_event as _log_event

                _au_metadata = RunCDGMetadata(
                    run_id=telemetry_run_id,
                    goal=args.goal,
                    execution_path=mode_settings.mode,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    verified_leaf_coverage=0.0,
                )
                # orchestrator_result_to_cdg already calls sanitize_cdg internally
                _au_cdg_dict = orchestrator_result_to_cdg(result, _au_metadata)

                if _au_metadata.verified_leaf_coverage >= config.auto_upsert_min_coverage:
                    _au_store = _GraphStore(
                        uri=config.memgraph_uri,
                        user=config.memgraph_user,
                        password=config.memgraph_password,
                    )
                    async with _au_store as _gs:
                        _au_counts = await _gs.upsert_cdg(
                            repo=f"run/{telemetry_run_id}",
                            cdg_dict=_au_cdg_dict,
                            witness_meta={},
                            contract_meta={},
                        )
                    _log_event(
                        "run_cmds",
                        "auto_upsert",
                        "AUTO_UPSERT_COMPLETED",
                        payload={
                            "run_id": telemetry_run_id,
                            "coverage": _au_metadata.verified_leaf_coverage,
                            "upsert_counts": _au_counts,
                        },
                    )
                    print(f"  Auto-upsert: coverage={_au_metadata.verified_leaf_coverage:.2f}, upserted {_au_counts}")
                else:
                    _log_event(
                        "run_cmds",
                        "auto_upsert",
                        "AUTO_UPSERT_SKIPPED_LOW_COVERAGE",
                        payload={
                            "run_id": telemetry_run_id,
                            "coverage": _au_metadata.verified_leaf_coverage,
                            "min_coverage": config.auto_upsert_min_coverage,
                        },
                    )
            except Exception as _au_exc:
                import warnings

                warnings.warn(
                    f"Auto-upsert failed (non-fatal): {_au_exc}",
                    RuntimeWarning,
                    stacklevel=1,
                )
    finally:
        event_log.configure_live_output(None)

    # Output
    print("\nRun complete:")
    print(f"  Rounds used: {result.rounds_used}")
    print(
        f"  Matches: {sum(1 for mr in result.match_results if mr.success)}/{len(result.match_results)}"
    )
    if result.ungroundable:
        print(f"  Ungroundable: {result.ungroundable}")

    save_json(result.cdg, output_dir / "cdg.json")

    if result.match_results:

        matches_data = [mr.to_dict() for mr in result.match_results]
        with open(output_dir / "matches.json", "w") as f:
            json.dump(matches_data, f, indent=2)

        if result.all_matched:
            try:
                from sciona.synthesizer.assembler import Assembler, AssemblyError

                ext = (
                    "py"
                    if prover == Prover.PYTHON
                    else ("v" if prover == Prover.COQ else "lean")
                )
                skeleton = Assembler(prover).assemble(result.cdg, result.match_results)
                (output_dir / f"skeleton.{ext}").write_text(skeleton.source_code)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Round 3 skeleton assembly skipped in run: %s", exc)
    matches_path = output_dir / "matches.json"
    cdg_path = output_dir / "cdg.json"

    print(f"  Output: {output_dir}/")
    _print_shared_context_metrics("architect", architect_shared_metrics)
    _print_shared_context_metrics("hunter", hunter_shared_metrics)
    metrics_path = _write_shared_context_metrics_file(
        output_dir / "shared_context_metrics.json",
        {
            "architect": architect_shared_metrics,
            "hunter": hunter_shared_metrics,
        },
    )
    merge_run_metadata(
        {
            "shared_context": _shared_context_metadata(
                {
                    "architect": architect_shared_metrics,
                    "hunter": hunter_shared_metrics,
                },
                metrics_path=metrics_path,
            )
        },
        run_id=telemetry_run_id,
    )
    if planner_result is not None:
        planner_artifact_manifest_path = output_dir / "planner_artifacts.json"
        planner_artifact_manifest = {
            "execution_path": planner_result.execution_path,
            "termination_reason": planner_result.state.termination_reason,
            "verification_status": planner_result.state.verification_status,
            "escalation_events": [dict(event) for event in planner_result.state.escalation_events],
            "tool_metrics": {
                name: {
                    "dispatches": int(metrics.get("dispatches", 0) or 0),
                    "latency_ms_total": round(
                        float(metrics.get("latency_ms_total", 0.0) or 0.0), 4
                    ),
                    "avg_latency_ms": round(
                        float(metrics.get("avg_latency_ms", 0.0) or 0.0), 4
                    ),
                }
                for name, metrics in sorted(planner_result.state.tool_metrics.items())
            },
            "artifacts": {
                "cdg": {
                    "source": planner_result.state.artifacts.get("cdg", ""),
                    "path": str(cdg_path),
                    "exists": cdg_path.exists(),
                    "mutations": int(planner_result.state.artifact_mutations.get("cdg", 0) or 0),
                },
                "match_results": {
                    "source": planner_result.state.artifacts.get("match_results", ""),
                    "path": str(matches_path),
                    "exists": matches_path.exists(),
                    "mutations": int(
                        planner_result.state.artifact_mutations.get("match_results", 0) or 0
                    ),
                },
            },
            "attempt_history": list(planner_result.state.attempt_history),
            "steps": [
                {
                    "action": step.action,
                    "detail": step.detail,
                    "status": step.status,
                }
                for step in planner_result.steps
            ],
        }
        if "orchestration" in planner_result.state.artifacts:
            planner_artifact_manifest["artifacts"]["orchestration"] = {
                "source": planner_result.state.artifacts.get("orchestration", ""),
                "path": str(cdg_path),
                "exists": cdg_path.exists(),
                "mutations": int(
                    planner_result.state.artifact_mutations.get("orchestration", 0) or 0
                ),
            }
        planner_artifact_manifest_path.write_text(
            json.dumps(planner_artifact_manifest, indent=2),
            encoding="utf-8",
        )
        merge_run_metadata(
            {
                "execution_path": planner_result.execution_path,
                "single_agent": {
                    "tool_metrics": {
                        name: {
                            "dispatches": int(metrics.get("dispatches", 0) or 0),
                            "latency_ms_total": round(
                                float(metrics.get("latency_ms_total", 0.0) or 0.0), 4
                            ),
                            "avg_latency_ms": round(
                                float(metrics.get("avg_latency_ms", 0.0) or 0.0), 4
                            ),
                        }
                        for name, metrics in sorted(
                            planner_result.state.tool_metrics.items()
                        )
                    },
                    "tool_dispatch_count_total": sum(
                        int(metrics.get("dispatches", 0) or 0)
                        for metrics in planner_result.state.tool_metrics.values()
                    ),
                    "escalation_events": [
                        dict(event) for event in planner_result.state.escalation_events
                    ],
                    "tool_latency_ms_total": round(
                        sum(
                            float(metrics.get("latency_ms_total", 0.0) or 0.0)
                            for metrics in planner_result.state.tool_metrics.values()
                        ),
                        4,
                    ),
                    "policy": {
                        "direct_grounding_enabled": planner_result.state.policy.direct_grounding_enabled,
                        "decomposition_mode": planner_result.state.policy.decomposition_mode,
                        "retrieval_intensity": planner_result.state.policy.retrieval_intensity,
                        "escalation_enabled": planner_result.state.policy.escalation_enabled,
                        "repair_policy": planner_result.state.policy.repair_policy,
                        "partial_accept_enabled": planner_result.state.policy.partial_accept_enabled,
                        "selective_redecompose_enabled": planner_result.state.policy.selective_redecompose_enabled,
                    },
                    "termination_reason": planner_result.state.termination_reason,
                    "verification_status": planner_result.state.verification_status,
                    "step_budget": planner_result.state.budget.max_steps,
                    "steps_used": planner_result.state.budget.steps_used,
                    "open_failures": list(planner_result.state.open_failures),
                    "artifacts": dict(planner_result.state.artifacts),
                    "artifact_mutations": dict(planner_result.state.artifact_mutations),
                    "artifact_manifest_path": str(planner_artifact_manifest_path),
                    "concrete_artifacts": planner_artifact_manifest["artifacts"],
                    "attempt_history": list(planner_result.state.attempt_history),
                    "steps": [
                        {
                            "action": step.action,
                            "detail": step.detail,
                            "status": step.status,
                        }
                        for step in planner_result.steps
                    ]
                },
            },
            run_id=telemetry_run_id,
        )
    if metrics_path is not None:
        print(f"  Shared context metrics: {metrics_path}")

    if getattr(args, "trace", False):
        if len(event_log) > 0:
            event_log.save(output_dir / "trace.jsonl")
            print(f"  Trace: {output_dir / 'trace.jsonl'} ({len(event_log)} events)")

    finish_run(telemetry_run_id, status="completed")

    # Shut down Postgres telemetry drain
    await _shutdown_telemetry_drain(_telem_drain, _telem_store)
