"""Per-prompt LLM routing.

Wraps a default ``LLMClient`` with optional per-prompt overrides so that
different prompts in the pipeline can be dispatched to different providers
(e.g. local Qwen for ranking, Anthropic for decomposition).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any

from sciona.hunter.llm import LLMClient
from sciona.telemetry import (
    finish_prompt_dispatch,
    get_current_run_id,
    get_current_stage,
    log_event,
    start_prompt_dispatch,
    update_stage,
)

# ---------------------------------------------------------------------------
# Prompt key constants — one per LLM call site
# ---------------------------------------------------------------------------

# Architect
ARCHITECT_STRATEGY = "architect_strategy"
ARCHITECT_DECOMPOSE = "architect_decompose"
ARCHITECT_CRITIQUE = "architect_critique"

# Hunter
HUNTER_SCORE = "hunter_score"
HUNTER_REFORMULATE = "hunter_reformulate"
HUNTER_ANALYZE_FAILURE = "hunter_analyze_failure"

# Synthesizer
SYNTHESIZER_REPAIR = "synthesizer_repair"
SYNTHESIZER_TACTIC = "synthesizer_tactic"

# Ingester
INGESTER_CHUNK = "ingester_chunk"
INGESTER_HOIST_STATE = "ingester_hoist_state"
INGESTER_ABSTRACT = "ingester_abstract"
INGESTER_FIX_TYPE = "ingester_fix_type"
INGESTER_FIX_GHOST = "ingester_fix_ghost"
INGESTER_OPAQUE_WITNESS = "ingester_opaque_witness"
INGESTER_FIX_MESSAGE_CYCLE = "ingester_fix_message_cycle"
INGESTER_DECOMPOSE = "ingester_decompose"

# Orchestrator
ORCHESTRATOR_REFINE = "orchestrator_refine"

# Prototyping (Neuro-symbolic RevisionLadder navigator & micro-atom synthesizer)
PROTOTYPING_NAVIGATE = "prototyping_navigate"
PROTOTYPING_SYNTHESIZE = "prototyping_synthesize"

ALL_PROMPT_KEYS = [
    ARCHITECT_STRATEGY,
    ARCHITECT_DECOMPOSE,
    ARCHITECT_CRITIQUE,
    HUNTER_SCORE,
    HUNTER_REFORMULATE,
    HUNTER_ANALYZE_FAILURE,
    SYNTHESIZER_REPAIR,
    SYNTHESIZER_TACTIC,
    INGESTER_CHUNK,
    INGESTER_HOIST_STATE,
    INGESTER_ABSTRACT,
    INGESTER_FIX_TYPE,
    INGESTER_FIX_GHOST,
    INGESTER_OPAQUE_WITNESS,
    INGESTER_FIX_MESSAGE_CYCLE,
    INGESTER_DECOMPOSE,
    ORCHESTRATOR_REFINE,
    PROTOTYPING_NAVIGATE,
    PROTOTYPING_SYNTHESIZE,
]

PROMPT_TIMEOUTS_S: dict[str, float] = {
    ARCHITECT_STRATEGY: 20.0,
    ARCHITECT_DECOMPOSE: 45.0,
    ARCHITECT_CRITIQUE: 35.0,
    HUNTER_SCORE: 20.0,
    HUNTER_REFORMULATE: 30.0,
    HUNTER_ANALYZE_FAILURE: 20.0,
    INGESTER_CHUNK: 120.0,
    INGESTER_HOIST_STATE: 60.0,
    INGESTER_ABSTRACT: 60.0,
    INGESTER_DECOMPOSE: 90.0,
    INGESTER_FIX_TYPE: 60.0,
    PROTOTYPING_NAVIGATE: 45.0,
    PROTOTYPING_SYNTHESIZE: 60.0,
}


def prompt_timeout_seconds(prompt_key: str) -> float | None:
    """Resolve router-enforced timeout for a prompt key."""
    env_key = f"SCIONA_{prompt_key.upper()}_TIMEOUT_S"
    raw = os.getenv(env_key, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        return value if value > 0 else None
    default_raw = os.getenv("SCIONA_PROMPT_TIMEOUT_DEFAULT_S", "").strip()
    if default_raw:
        try:
            value = float(default_raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    value = PROMPT_TIMEOUTS_S.get(prompt_key, 0.0)
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class LLMRouter:
    """Routes LLM calls to per-prompt override clients or a shared default.

    Satisfies the ``LLMClient`` protocol so it can be used as a drop-in
    replacement everywhere the pipeline expects an ``LLMClient``.
    """

    def __init__(
        self,
        default: LLMClient,
        overrides: dict[str, LLMClient] | None = None,
    ) -> None:
        self._default = default
        self._overrides: dict[str, LLMClient] = overrides or {}

    def for_prompt(self, key: str) -> LLMClient:
        """Return the override client for *key*, or the default."""
        return self._overrides.get(key, self._default)

    # -- LLMClient protocol methods (delegate to default) ------------------

    async def complete(self, system: str, user: str) -> str:
        return await self._default.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self._default.complete_with_grammar(system, user, grammar)

    async def warmup(self) -> None:
        """Prewarm unique underlying clients when they expose a warmup hook."""
        seen: set[int] = set()
        for client in [self._default, *self._overrides.values()]:
            marker = id(client)
            if marker in seen:
                continue
            seen.add(marker)
            warm = getattr(client, "warmup", None)
            if callable(warm):
                await warm()


# ---------------------------------------------------------------------------
# Prompt-key wrapper for telemetry
# ---------------------------------------------------------------------------


class PromptKeyLLMClient:
    """LLMClient wrapper that attributes dispatches to one prompt key."""

    def __init__(self, base: LLMClient, prompt_key: str) -> None:
        self._base = base
        self._prompt_key = prompt_key
        self._heartbeat_interval_sec = 5.0

    async def _heartbeat_loop(self, dispatch_id: str, started_at: float) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_sec)
            elapsed = max(0.0, time.time() - started_at)
            stage = get_current_stage()
            if stage:
                update_stage(
                    stage=stage,
                    status="running",
                    message=f"waiting on {self._prompt_key} ({elapsed:.0f}s)",
                )
            log_event(
                "llm",
                phase=stage or "prompt_dispatch",
                event_type="PROMPT_DISPATCH_HEARTBEAT",
                stage=stage,
                prompt_key=self._prompt_key,
                dispatch_id=dispatch_id,
                payload={"elapsed_sec": round(elapsed, 2)},
            )

    async def _stop_heartbeat(
        self, task: asyncio.Task[None] | None
    ) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _completion_metadata(self) -> dict[str, Any]:
        getter = getattr(self._base, "get_last_completion_metadata", None)
        if not callable(getter):
            return {}
        if inspect.iscoroutinefunction(getter):
            return {}
        try:
            metadata = getter()
        except Exception:
            return {}
        if inspect.isawaitable(metadata):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _error_metadata(self, exc: Exception) -> dict[str, Any]:
        payload: dict[str, Any] = {"error_type": exc.__class__.__name__}
        getter = getattr(self._base, "get_last_error_metadata", None)
        if callable(getter):
            if inspect.iscoroutinefunction(getter):
                return payload
            try:
                metadata = getter()
            except Exception:
                metadata = {}
            if inspect.isawaitable(metadata):
                metadata = {}
            if isinstance(metadata, dict):
                payload.update(metadata)
        return payload

    async def complete(self, system: str, user: str) -> str:
        dispatch_id = start_prompt_dispatch(self._prompt_key, client=self._base)
        started_at = time.time()
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop(dispatch_id, started_at))
            if dispatch_id
            else None
        )
        timeout_s = prompt_timeout_seconds(self._prompt_key)
        try:
            if timeout_s:
                output = await asyncio.wait_for(
                    self._base.complete(system, user),
                    timeout=timeout_s,
                )
            else:
                output = await self._base.complete(system, user)
        except asyncio.TimeoutError as exc:
            await self._stop_heartbeat(heartbeat_task)
            error = f"{self._prompt_key} timed out after {timeout_s:.1f}s"
            finish_prompt_dispatch(
                dispatch_id,
                ok=False,
                error=error,
                payload={
                    "error_type": "TimeoutError",
                    "provider_error_phase": "router_timeout",
                    "prompt_timeout_s": timeout_s,
                },
            )
            raise RuntimeError(error) from exc
        except Exception as exc:
            await self._stop_heartbeat(heartbeat_task)
            finish_prompt_dispatch(
                dispatch_id,
                ok=False,
                error=str(exc),
                payload=self._error_metadata(exc),
            )
            raise
        await self._stop_heartbeat(heartbeat_task)
        finish_prompt_dispatch(dispatch_id, ok=True, payload=self._completion_metadata())
        return output

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        dispatch_id = start_prompt_dispatch(self._prompt_key, client=self._base)
        started_at = time.time()
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop(dispatch_id, started_at))
            if dispatch_id
            else None
        )
        timeout_s = prompt_timeout_seconds(self._prompt_key)
        try:
            if timeout_s:
                output = await asyncio.wait_for(
                    self._base.complete_with_grammar(system, user, grammar),
                    timeout=timeout_s,
                )
            else:
                output = await self._base.complete_with_grammar(system, user, grammar)
        except asyncio.TimeoutError as exc:
            await self._stop_heartbeat(heartbeat_task)
            error = f"{self._prompt_key} timed out after {timeout_s:.1f}s"
            finish_prompt_dispatch(
                dispatch_id,
                ok=False,
                error=error,
                payload={
                    "error_type": "TimeoutError",
                    "provider_error_phase": "router_timeout",
                    "prompt_timeout_s": timeout_s,
                },
            )
            raise RuntimeError(error) from exc
        except Exception as exc:
            await self._stop_heartbeat(heartbeat_task)
            finish_prompt_dispatch(
                dispatch_id,
                ok=False,
                error=str(exc),
                payload=self._error_metadata(exc),
            )
            raise
        await self._stop_heartbeat(heartbeat_task)
        finish_prompt_dispatch(dispatch_id, ok=True, payload=self._completion_metadata())
        return output


# ---------------------------------------------------------------------------
# select_llm helper
# ---------------------------------------------------------------------------


def select_llm(llm: Any, key: str) -> LLMClient:
    """Pick the right client for *key*.

    If *llm* is an ``LLMRouter``, delegates to ``for_prompt(key)``.
    Otherwise returns *llm* unchanged (plain ``LLMClient``).
    """
    base = llm.for_prompt(key) if isinstance(llm, LLMRouter) else llm

    # Wrap prompt clients whenever telemetry is active or a router timeout
    # needs to be enforced for this prompt, even outside telemetry scope.
    if not get_current_run_id() and prompt_timeout_seconds(key) is None:
        return base

    if isinstance(base, PromptKeyLLMClient):
        if base._prompt_key == key:
            return base
        # Avoid nested wrappers on prompt-key switches.
        return PromptKeyLLMClient(base._base, key)
    return PromptKeyLLMClient(base, key)
