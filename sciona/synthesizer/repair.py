"""Pydantic-graph repair state machine for skeleton compilation.

Graph topology:
    CompileCheck -> DeterministicFix -> CompileCheck
                 -> LLMRepair -> CompileCheck
                 -> SorryElimination -> CompileCheck
                 -> End[SkeletonFile]  (on success or budget exhausted)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sciona.json_utils import extract_json

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from sciona.hunter.llm import LLMClient
from sciona.judge.models import CompilerFeedback
from sciona.protocols import ProofEnvironment
from sciona.synthesizer.classifier import (
    ErrorCategory,
    classify_feedback,
    suggest_deterministic_fix,
)
from sciona.synthesizer.assembler import sanitize_python_source_annotations
from sciona.synthesizer.models import SkeletonFile
from sciona.synthesizer.patcher import (
    Patch,
    apply_patches,
    extract_error_context,
    find_sorry_locations,
)
from sciona.llm_router import SYNTHESIZER_REPAIR, SYNTHESIZER_TACTIC, select_llm
from sciona.shared_context import (
    SharedContextMetrics,
    SharedContextStore,
    format_context_block,
)
from sciona.synthesizer.prompts import (
    ANALYZE_ERROR_SYSTEM,
    ANALYZE_ERROR_SYSTEM_PYTHON,
    ANALYZE_ERROR_USER,
    GENERATE_IMPLEMENTATION_SYSTEM_PYTHON,
    GENERATE_TACTIC_SYSTEM,
    GENERATE_TACTIC_USER,
)

logger = logging.getLogger(__name__)

# Error category priority for LLM repair (lower = higher priority)
_ERROR_PRIORITY: dict[ErrorCategory, int] = {
    ErrorCategory.TYPE_MISMATCH: 0,
    ErrorCategory.UNIVERSE_MISMATCH: 1,
    ErrorCategory.MISSING_IMPORT: 2,
    ErrorCategory.SYNTAX: 3,
    ErrorCategory.UNKNOWN: 4,
    ErrorCategory.UNSOLVED_GOAL: 5,
}


@dataclass
class RepairState:
    """Mutable state threaded through the repair graph."""

    skeleton: SkeletonFile
    max_iterations: int = 10
    iteration: int = 0
    patches_applied: list[Patch] = field(default_factory=list)
    error_history: list[tuple[int, ErrorCategory, str]] = field(default_factory=list)
    sorry_remaining: int = 0
    compiled_ok: bool = False
    _last_feedback: CompilerFeedback | None = field(default=None, repr=False)

    # Resilience tracking (Issue 8)
    _source_snapshots: list[str] = field(default_factory=list, repr=False)
    _last_error_count: int = field(default=-1, repr=False)
    llm_attempts: int = 0
    llm_successes: int = 0  # produced usable patches
    best_source: str = ""
    best_error_count: int = field(default=999999, repr=False)


@dataclass
class RepairDeps:
    """External dependencies for the repair graph."""

    env: ProofEnvironment
    llm: LLMClient
    shared_context: SharedContextStore | None = None
    shared_context_metrics: SharedContextMetrics | None = None
    context_namespace: str = ""
    context_budget_chars: int = 900


async def _compile_source(env: ProofEnvironment, source: str) -> CompilerFeedback:
    """Compile source code via the proof environment."""
    if hasattr(env, "_run"):
        return await env._run(source)  # type: ignore[attr-defined]
    success, output = await env.check_proof("True", source)
    return CompilerFeedback(
        raw_output=output,
        errors=[] if success else [output],
    )


async def _search_context(
    deps: RepairDeps,
    *,
    channel: str,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return ""
    try:
        records = await store.search(f"{ns}/{channel}", query, limit=limit)
        return format_context_block(
            "Shared Context",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
    except Exception:
        return ""


async def _put_context(
    deps: RepairDeps,
    *,
    channel: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return
    try:
        await store.put(f"{ns}/{channel}", text, metadata=metadata)
    except Exception:
        return


@dataclass
class CompileCheck(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Compile the skeleton and route based on result."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> DeterministicFix | LLMRepair | SorryElimination | End[SkeletonFile]:
        state = ctx.state
        deps = ctx.deps

        if state.skeleton.prover == "python":
            state.skeleton.source_code = sanitize_python_source_annotations(
                state.skeleton.source_code
            )

        feedback = await _compile_source(deps.env, state.skeleton.source_code)
        state._last_feedback = feedback

        # Track error count for regression detection
        current_error_count = len(feedback.errors) if not feedback.success else 0

        # Regression detection: if errors increased, rollback
        if (
            state._last_error_count >= 0
            and current_error_count > state._last_error_count
            and state._source_snapshots
        ):
            logger.info(
                "Regression detected: errors %d -> %d, rolling back",
                state._last_error_count,
                current_error_count,
            )
            state.skeleton.source_code = state._source_snapshots[-1]
            # Re-compile after rollback
            feedback = await _compile_source(deps.env, state.skeleton.source_code)
            state._last_feedback = feedback
            current_error_count = len(feedback.errors) if not feedback.success else 0

        state._last_error_count = current_error_count

        # Track best source
        if current_error_count < state.best_error_count:
            state.best_error_count = current_error_count
            state.best_source = state.skeleton.source_code

        if feedback.success:
            remaining_sorries = find_sorry_locations(
                state.skeleton.source_code, state.skeleton.prover
            )
            if remaining_sorries:
                state.sorry_remaining = len(remaining_sorries)
                if state.iteration >= state.max_iterations:
                    if state.best_source and state.best_error_count < current_error_count:
                        state.skeleton.source_code = state.best_source
                    return End(state.skeleton)
                state._source_snapshots.append(state.skeleton.source_code)
                return SorryElimination()

            state.compiled_ok = True
            state.sorry_remaining = 0
            await _put_context(
                deps,
                channel="compile",
                text=(
                    f"Iteration: {state.iteration}\n"
                    "Compilation: success\n"
                    f"Patches: {len(state.patches_applied)}"
                ),
                metadata={"confidence": 0.95, "compiled_ok": True},
            )
            return End(state.skeleton)

        # Budget check — return best version if exhausted
        if state.iteration >= state.max_iterations:
            if state.best_source and state.best_error_count < current_error_count:
                state.skeleton.source_code = state.best_source
            return End(state.skeleton)

        # Snapshot current source before any modifications
        state._source_snapshots.append(state.skeleton.source_code)

        # Classify errors
        classified = classify_feedback(feedback)
        if classified:
            cat, text = classified[0]
            await _put_context(
                deps,
                channel="compile",
                text=(
                    f"Iteration: {state.iteration}\n"
                    f"Top error category: {cat.value}\n"
                    f"Error: {text}"
                ),
                metadata={"confidence": 0.45, "compiled_ok": False},
            )

        # Check if any have deterministic fixes
        has_deterministic = any(
            suggest_deterministic_fix(cat, text) is not None for cat, text in classified
        )
        if has_deterministic:
            return DeterministicFix()

        # Check if only unsolved goals remain (no hard errors)
        hard_errors = [
            (cat, text)
            for cat, text in classified
            if cat != ErrorCategory.UNSOLVED_GOAL
        ]
        if not hard_errors and classified:
            return SorryElimination()

        return LLMRepair()


def _find_header_insert_line(lines: list[str], prover: str) -> int:
    """Return 1-indexed line number *after* which top-level imports/opens should be inserted.

    Returns ``0`` when fixes should be prepended before line 1 (e.g. no header
    block comment or existing import block). Never splits ``/- ... -/``,
    ``(* ... *)``, or Python ``\"\"\" ... \"\"\"`` header comments.
    """
    if not lines:
        return 0
    idx = 0
    n = len(lines)
    while idx < n:
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if prover == "python":
            if stripped.startswith("#"):
                idx += 1
                continue
            if stripped.startswith(('"""', "'''")):
                quote = stripped[:3]
                if len(stripped) >= 6 and stripped.endswith(quote):
                    idx += 1
                    continue
                idx += 1
                while idx < n and quote not in lines[idx]:
                    idx += 1
                if idx < n:
                    idx += 1
                continue
            if stripped.startswith(("from __future__", "import ", "from ")):
                idx += 1
                continue
        elif prover == "coq":
            if stripped.startswith("(*"):
                while idx < n and "*)" not in lines[idx]:
                    idx += 1
                if idx < n:
                    idx += 1
                continue
            if stripped.startswith(("Require ", "From ", "Import ")):
                idx += 1
                continue
        else:  # lean4
            if stripped.startswith("--"):
                idx += 1
                continue
            if stripped.startswith("/-"):
                while idx < n and "-/" not in lines[idx]:
                    idx += 1
                if idx < n:
                    idx += 1
                continue
            if stripped.startswith("import "):
                idx += 1
                continue
        break
    return idx


@dataclass
class DeterministicFix(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Apply deterministic fixes (e.g. missing imports).

    Applies ALL deterministic fixes in one pass instead of one-per-iteration.
    """

    async def run(self, ctx: GraphRunContext[RepairState, RepairDeps]) -> CompileCheck:
        state = ctx.state

        if state._last_feedback is None:
            state.iteration += 1
            return CompileCheck()

        classified = classify_feedback(state._last_feedback)
        lines = state.skeleton.source_code.splitlines()

        # Collect ALL deterministic fixes in one pass
        fixes_to_apply: list[str] = []
        for cat, text in classified:
            fix = suggest_deterministic_fix(cat, text)
            if fix is not None:
                if fix not in fixes_to_apply:
                    fixes_to_apply.append(fix)
                state.error_history.append((state.iteration, cat, text))

        if fixes_to_apply:
            combined_fix = "\n".join(fixes_to_apply)
            insert_idx = _find_header_insert_line(lines, state.skeleton.prover)
            if not lines:
                patch = Patch(
                    line_start=1,
                    line_end=1,
                    replacement=combined_fix,
                    description=f"Deterministic fixes: {len(fixes_to_apply)} fix(es)",
                )
            elif insert_idx == 0:
                patch = Patch(
                    line_start=1,
                    line_end=1,
                    replacement=combined_fix + "\n" + lines[0],
                    description=f"Deterministic fixes: {len(fixes_to_apply)} fix(es)",
                )
            else:
                patch = Patch(
                    line_start=insert_idx,
                    line_end=insert_idx,
                    replacement=lines[insert_idx - 1] + "\n" + combined_fix,
                    description=f"Deterministic fixes: {len(fixes_to_apply)} fix(es)",
                )
            state.skeleton.source_code = apply_patches(
                state.skeleton.source_code, [patch]
            )
            state.patches_applied.append(patch)

        state.iteration += 1
        return CompileCheck()


@dataclass
class LLMRepair(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Use LLM to generate a patch for the highest-priority error."""

    async def run(self, ctx: GraphRunContext[RepairState, RepairDeps]) -> CompileCheck:
        state = ctx.state
        deps = ctx.deps

        if state._last_feedback is None:
            state.iteration += 1
            return CompileCheck()

        classified = classify_feedback(state._last_feedback)
        classified.sort(key=lambda x: _ERROR_PRIORITY.get(x[0], 99))

        if not classified:
            state.iteration += 1
            return CompileCheck()

        cat, error_text = classified[0]
        state.error_history.append((state.iteration, cat, error_text))

        error_line = _extract_line_number(error_text)
        context = extract_error_context(
            state.skeleton.source_code, error_line, radius=3
        )

        user_msg = ANALYZE_ERROR_USER.format(
            source_code=state.skeleton.source_code,
            error_text=error_text,
            error_category=cat.value,
            error_context=context,
        )
        shared_block = await _search_context(
            deps,
            channel="repair",
            query=f"{cat.value} {error_text}",
            limit=3,
        )
        if shared_block:
            user_msg += f"\n\n{shared_block}"

        system_prompt = (
            ANALYZE_ERROR_SYSTEM_PYTHON
            if state.skeleton.prover == "python"
            else ANALYZE_ERROR_SYSTEM
        )

        state.llm_attempts += 1
        try:
            response = await select_llm(deps.llm, SYNTHESIZER_REPAIR).complete(
                system_prompt, user_msg
            )
            patch = _parse_patch_response(response)
            if patch is not None:
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
                state.llm_successes += 1
                await _put_context(
                    deps,
                    channel="repair",
                    text=(
                        f"Error category: {cat.value}\n"
                        f"Patch: {patch.description or '(none)'}\n"
                        f"Lines: {patch.line_start}-{patch.line_end}"
                    ),
                    metadata={
                        "confidence": 0.9,
                        "category": cat.value,
                        "line_start": patch.line_start,
                        "line_end": patch.line_end,
                    },
                )
            else:
                await _put_context(
                    deps,
                    channel="repair",
                    text=(
                        f"Error category: {cat.value}\n"
                        "Patch: parse failure"
                    ),
                    metadata={"confidence": 0.3, "category": cat.value},
                )
        except json.JSONDecodeError as exc:
            state.error_history.append(
                (
                    state.iteration,
                    ErrorCategory.UNKNOWN,
                    f"LLM_FAILURE: JSON parse error: {exc}",
                )
            )
            logger.warning("LLM repair JSON parse error: %s", exc)
        except ValueError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("LLM repair value error: %s", exc)
        except RuntimeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("LLM repair runtime error: %s", exc)

        state.iteration += 1
        return CompileCheck()


@dataclass
class SorryElimination(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Replace sorry/Admitted placeholders with LLM-generated tactic proofs."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> CompileCheck | LLMRepair:
        state = ctx.state
        deps = ctx.deps

        locations = find_sorry_locations(
            state.skeleton.source_code, state.skeleton.prover
        )
        if not locations:
            state.iteration += 1
            return LLMRepair()

        line_num, context = locations[0]  # Fix one sorry per iteration

        goal_type = _extract_goal_from_context(state.skeleton.source_code, line_num)

        user_msg = GENERATE_TACTIC_USER.format(
            goal_type=goal_type or "(unknown — see context above)",
            hypotheses=context,
            available_lemmas="(use standard Mathlib tactics)",
        )
        shared_block = await _search_context(
            deps,
            channel="tactic",
            query=f"{goal_type} {context}",
            limit=3,
        )
        if shared_block:
            user_msg += f"\n\n{shared_block}"

        system_prompt = (
            GENERATE_IMPLEMENTATION_SYSTEM_PYTHON
            if state.skeleton.prover == "python"
            else GENERATE_TACTIC_SYSTEM
        )

        state.llm_attempts += 1
        try:
            response = await select_llm(deps.llm, SYNTHESIZER_TACTIC).complete(
                system_prompt, user_msg
            )
            tactic_body = response.strip()
            if tactic_body and "sorry" not in tactic_body.lower():
                patch = Patch(
                    line_start=line_num,
                    line_end=line_num,
                    replacement=tactic_body,
                    description=f"Sorry elimination at line {line_num}",
                )
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
                state.llm_successes += 1
                await _put_context(
                    deps,
                    channel="tactic",
                    text=(
                        f"Goal: {goal_type or '(unknown)'}\n"
                        f"Tactic snippet: {tactic_body[:140]}"
                    ),
                    metadata={"confidence": 0.88, "line": line_num},
                )
        except json.JSONDecodeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"PARSE_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination parse error: %s", exc)
        except ValueError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination value error: %s", exc)
        except RuntimeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination runtime error: %s", exc)

        state.sorry_remaining = len(locations) - 1
        state.iteration += 1
        return CompileCheck()


# --- Helpers ---


def _extract_line_number(error_text: str) -> int:
    """Extract a line number from an error message, defaulting to 1."""
    m = re.search(r":(\d+):", error_text)
    if m:
        return int(m.group(1))
    m = re.search(r"line (\d+)", error_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1


def _parse_patch_response(response: str) -> Patch | None:
    """Parse an LLM JSON response into a Patch."""
    try:
        data = extract_json(response)
        return Patch(
            line_start=int(data["line_start"]),
            line_end=int(data["line_end"]),
            replacement=str(data["replacement"]),
            description=str(data.get("description", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _extract_goal_from_context(source: str, sorry_line: int) -> str:
    """Try to extract the theorem/lemma goal type from lines before a sorry."""
    lines = source.splitlines()
    for i in range(sorry_line - 1, max(sorry_line - 20, -1), -1):
        if i < 0 or i >= len(lines):
            continue
        line = lines[i]
        m = re.match(
            r"\s*(?:theorem|lemma|def|noncomputable\s+def)\s+\w+\s*.*?:\s*(.*)",
            line,
        )
        if not m:
            # Python: def func_name(params) -> ReturnType:
            m = re.match(
                r"\s*def\s+\w+\s*\(.*?\)\s*(?:->\s*(.+?))?\s*:",
                line,
            )
        if m:
            goal = m.group(1).strip()
            goal = re.sub(r"\s*:=\s*(by)?\s*$", "", goal)
            if goal:
                return goal
    return ""


repair_graph: Graph[RepairState, RepairDeps, SkeletonFile] = Graph(
    nodes=[CompileCheck, DeterministicFix, LLMRepair, SorryElimination]
)
