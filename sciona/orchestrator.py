"""Top-level orchestration loop: Architect -> Hunter -> (on failure) -> Architect.refine -> Hunter.

Replaces the one-shot CLI flow with a feedback loop that re-decomposes
atomic nodes when the Hunter fails to ground them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sciona.architect.handoff import CDGExport, to_pdg_nodes
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.llm_router import ORCHESTRATOR_REFINE, select_llm
from sciona.telemetry import log_event, telemetry_scope, telemetry_stage, update_stage
from sciona.types import (
    FailureAction,
    MatchFailureReport,
    MatchResult,
    Prover,
)

logger = logging.getLogger(__name__)

_REFINE_STOPWORDS = {
    "a",
    "an",
    "and",
    "apply",
    "by",
    "for",
    "from",
    "function",
    "helper",
    "implement",
    "into",
    "of",
    "on",
    "or",
    "return",
    "same",
    "step",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}
_CONNECTOR_PATTERN = re.compile(r"\b(?:and then|then|with|plus|before|after)\b", re.IGNORECASE)


def _find_cdg_node(cdg: CDGExport, node_id: str) -> AlgorithmicNode | None:
    for node in cdg.nodes:
        if node.node_id == node_id:
            return node
    return None


def _apply_split_subnodes(
    cdg: CDGExport,
    original: AlgorithmicNode,
    sub_nodes: list[dict[str, Any]],
) -> CDGExport:
    bounded = sub_nodes[:3]
    if not bounded:
        return cdg

    original.status = NodeStatus.DECOMPOSED
    original.children = []

    orig_inputs = [inp.model_copy(deep=True) for inp in original.inputs] or [
        IOSpec(name="input", type_desc="Any")
    ]
    orig_outputs = [out.model_copy(deep=True) for out in original.outputs] or [
        IOSpec(name="output", type_desc=orig_inputs[0].type_desc or "Any")
    ]

    created: list[AlgorithmicNode] = []
    n_subs = len(bounded)
    for i, sub in enumerate(bounded):
        sub_id = f"{original.node_id}_sub{i}"
        sig = str(sub.get("type_signature", "") or "")
        inferred_in_type = "Any"
        inferred_out_type = "Any"
        if "->" in sig:
            lhs, rhs = sig.rsplit("->", 1)
            inferred_in_type = lhs.strip(" ()") or "Any"
            inferred_out_type = rhs.strip() or "Any"

        if i == 0:
            sub_inputs = [inp.model_copy(deep=True) for inp in orig_inputs]
        else:
            prev_out = created[-1].outputs[0]
            sub_inputs = [
                IOSpec(
                    name=prev_out.name,
                    type_desc=(
                        prev_out.type_desc
                        if prev_out.type_desc != "Any"
                        else inferred_in_type
                    ),
                )
            ]

        if i == n_subs - 1:
            sub_outputs = [out.model_copy(deep=True) for out in orig_outputs]
        else:
            out_type = (
                inferred_out_type
                if inferred_out_type != "Any"
                else sub_inputs[0].type_desc
            )
            sub_outputs = [IOSpec(name=f"stage_{i}_out", type_desc=out_type)]

        sub_node = AlgorithmicNode(
            node_id=sub_id,
            parent_id=original.node_id,
            name=str(sub.get("name", f"sub_{i}")),
            description=str(sub.get("description", "")),
            concept_type=original.concept_type,
            status=NodeStatus.ATOMIC,
            depth=original.depth + 1,
            type_signature=sig or f"{sub_inputs[0].type_desc} -> {sub_outputs[0].type_desc}",
            inputs=sub_inputs,
            outputs=sub_outputs,
        )
        cdg.nodes.append(sub_node)
        original.children.append(sub_id)
        created.append(sub_node)

    # Rewire parent data-flow edges to first/last split sub-nodes
    first_sub = created[0]
    last_sub = created[-1]
    rewired_edges: list[DependencyEdge] = []
    for edge in cdg.edges:
        updated = edge.model_copy(deep=True)
        if updated.target_id == original.node_id:
            updated.target_id = first_sub.node_id
        if updated.source_id == original.node_id:
            updated.source_id = last_sub.node_id
        rewired_edges.append(updated)

    # Add internal sequential data-flow edges across split sub-nodes
    for idx in range(len(created) - 1):
        src_node = created[idx]
        dst_node = created[idx + 1]
        src_port = src_node.outputs[0]
        dst_port = dst_node.inputs[0]
        rewired_edges.append(
            DependencyEdge(
                source_id=src_node.node_id,
                target_id=dst_node.node_id,
                output_name=src_port.name,
                input_name=dst_port.name,
                source_type=src_port.type_desc,
                target_type=dst_port.type_desc,
            )
        )
    cdg.edges = rewired_edges
    return cdg


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [tok for tok in tokens if tok not in _REFINE_STOPWORDS and len(tok) >= 3]


def _title_from_clause(clause: str) -> str:
    tokens = _content_tokens(clause)[:4]
    if not tokens:
        return clause.strip().title() or "Refined Step"
    return " ".join(token.capitalize() for token in tokens)


def _split_on_connectors(text: str) -> list[dict[str, str]] | None:
    parts = [part.strip(" ,.;") for part in _CONNECTOR_PATTERN.split(text) if part.strip(" ,.;")]
    if len(parts) < 2 or len(parts) > 3:
        return None

    refined: list[dict[str, str]] = []
    for part in parts:
        tokens = _content_tokens(part)
        if len(tokens) < 2:
            return None
        refined.append(
            {
                "name": _title_from_clause(part),
                "description": part.strip(),
                "type_signature": "",
            }
        )
    return refined


_ALGORITHM_NAMES = re.compile(
    r"\b(?:dijkstra|bellman[ -]?ford|floyd[ -]?warshall|prim|kruskal"
    r"|cholesky|householder|givens|gram[ -]?schmidt|lanczos"
    r"|butterworth|chebyshev|bessel|elliptic"
    r"|viterbi|baum[ -]?welch|metropolis|gibbs"
    r"|newton|gauss|euler|runge[ -]?kutta"
    r"|levenberg[ -]?marquardt|nelder[ -]?mead"
    r"|cooley[ -]?tukey|bluestein"
    r"|karatsuba|strassen)\b",
    re.IGNORECASE,
)


def _generalize_description(description: str, error_summaries: list[str]) -> str:
    """Strip algorithm-specific names while keeping structural intent."""
    if not description:
        return description
    # Remove named-algorithm references
    generalized = _ALGORITHM_NAMES.sub("", description)
    # Collapse extra whitespace
    generalized = re.sub(r"\s{2,}", " ", generalized).strip()
    # If stripping removed too much, fall back to original
    if len(generalized) < 10:
        return description
    return generalized


def _load_split_patterns(path: str = "") -> list[dict]:
    """Load split patterns from JSON, falling back to built-in defaults."""
    if not path:
        default = Path(__file__).parent / "data" / "split_patterns.json"
        if default.exists():
            path = str(default)
    if path:
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("patterns", [])
        except Exception:
            logger.debug("Failed to load split patterns from %s", path)
    return []


_SPLIT_PATTERNS: list[dict] | None = None


def _get_split_patterns() -> list[dict]:
    global _SPLIT_PATTERNS
    if _SPLIT_PATTERNS is None:
        _SPLIT_PATTERNS = _load_split_patterns()
    return _SPLIT_PATTERNS


def _pattern_matches(conditions: dict, lowered: str) -> bool:
    """Check if a pattern's conditions match the lowered context string."""
    # "all" — every term must be present
    if "all" in conditions:
        if not all(term in lowered for term in conditions["all"]):
            return False

    # "any" — at least one term in the list must be present
    if "any" in conditions:
        if not any(term in lowered for term in conditions["any"]):
            # Unless there are "any_combo" alternatives
            if "any_combo" not in conditions:
                return False

    # "any_combo" — at least one group must fully match (each group is AND)
    if "any_combo" in conditions:
        has_any = "any" in conditions and any(term in lowered for term in conditions["any"])
        has_combo = any(
            all(term in lowered for term in group)
            for group in conditions["any_combo"]
        )
        if not has_any and not has_combo:
            # Also check any_padded (terms tested against " text ")
            has_padded = "any_padded" in conditions and any(
                term in f" {lowered} " for term in conditions["any_padded"]
            )
            if not has_padded:
                return False

    # "any_padded" — at least one padded term present (standalone check)
    if "any_padded" in conditions and "any_combo" not in conditions:
        if not any(term in f" {lowered} " for term in conditions["any_padded"]):
            return False

    # "any_phrase" — at least one phrase present
    if "any_phrase" in conditions:
        if not any(phrase in lowered for phrase in conditions["any_phrase"]):
            return False

    # "require_any" — at least one of these must also be present
    if "require_any" in conditions:
        if not any(term in lowered for term in conditions["require_any"]):
            return False

    # "any" with nested lists (e.g., [["ecg", "bandpass"], ["filter", "coeff"]])
    # = at least one sub-list must have ALL its terms present
    if "any" in conditions and conditions["any"] and isinstance(conditions["any"][0], list):
        if not any(
            all(term in lowered for term in group)
            for group in conditions["any"]
        ):
            return False

    return True


def _deterministic_split_subnodes(
    failure: MatchFailureReport,
    original: AlgorithmicNode | None,
) -> list[dict[str, str]] | None:
    description = (original.description if original is not None else "") or failure.pdg_node.informal_desc
    statement = failure.pdg_node.statement
    error_text = " ".join(failure.error_summaries[:3])
    candidate_names = " ".join(
        candidate.declaration.name for candidate in failure.best_candidates[:3]
    )
    combined = " ".join(part for part in [statement, description, error_text, candidate_names] if part)
    lowered = combined.lower()

    for pattern in _get_split_patterns():
        conditions = pattern.get("conditions", {})
        if _pattern_matches(conditions, lowered):
            return pattern["sub_nodes"]

    return _split_on_connectors(description or statement)


@dataclass
class OrchestratorResult:
    """Result of the full orchestration loop."""

    cdg: CDGExport
    match_results: list[MatchResult]
    rounds_used: int = 0
    failures: list[MatchFailureReport] = field(default_factory=list)
    ungroundable: list[str] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(mr.success for mr in self.match_results)


async def refine_on_failure(
    failure: MatchFailureReport,
    cdg: CDGExport,
    llm: Any,
    template_retriever: Any | None = None,
) -> CDGExport:
    """Refine the CDG based on a match failure.

    Uses the LLM to analyze the failure and either:
    - Split the atomic node into 2-3 finer sub-atoms
    - Replace it with an equivalent formulation
    - Mark it as UNGROUNDABLE

    Args:
        failure: The match failure report.
        cdg: The current CDG.
        llm: An LLM client for analysis.

    Returns:
        Updated CDG with the failed node refined.
    """
    node = failure.pdg_node
    action = failure.suggested_action

    if action == FailureAction.UNGROUNDABLE:
        # Mark the node and return unchanged CDG
        for n in cdg.nodes:
            if n.node_id == node.predicate_id:
                n.status = NodeStatus.REJECTED
                n.critic_notes = "Marked UNGROUNDABLE after match failure"
                break
        return cdg

    if action == FailureAction.SPLIT:
        original = _find_cdg_node(cdg, node.predicate_id)

        # Layer 1: Try retrieval-based refinement templates
        if template_retriever is not None and original is not None:
            try:
                failure_context_dict = {
                    "description": original.description or node.informal_desc,
                    "concept_type": original.concept_type.value if hasattr(original.concept_type, "value") else str(original.concept_type),
                    "error_summaries": failure.error_summaries[:3],
                    "statement": node.statement,
                }
                retrieval_matches = await template_retriever.find_refinement_templates(
                    original, failure_context_dict
                )
                # Use first match with confidence >= 0.7
                good_matches = [m for m in retrieval_matches if m.confidence >= 0.7]
                if good_matches:
                    best = good_matches[0]
                    retrieved_sub_nodes = [
                        {
                            "name": child.name,
                            "description": child.description,
                            "type_signature": getattr(child, "type_signature", ""),
                        }
                        for child in best.example.children
                    ]
                    if retrieved_sub_nodes:
                        _apply_split_subnodes(cdg, original, retrieved_sub_nodes)
                        logger.info(
                            "Retrieval-based split node %s into %d sub-nodes (confidence=%.2f)",
                            node.predicate_id,
                            len(retrieved_sub_nodes),
                            best.confidence,
                        )
                        log_event(
                            "orchestrator",
                            "refinement",
                            "SPLIT_RETRIEVAL",
                            payload={
                                "node_id": node.predicate_id,
                                "confidence": best.confidence,
                                "source": best.source,
                                "sub_node_count": len(retrieved_sub_nodes),
                            },
                        )
                        return cdg
            except Exception as exc:
                logger.debug(
                    "Retrieval-based refinement failed for %s: %s",
                    node.predicate_id,
                    exc,
                )

        # Layer 2: Deterministic pattern-based split
        deterministic_sub_nodes = _deterministic_split_subnodes(failure, original)
        if deterministic_sub_nodes and original is not None:
            _apply_split_subnodes(cdg, original, deterministic_sub_nodes)
            logger.info(
                "Deterministically split node %s into %d sub-nodes",
                node.predicate_id,
                len(deterministic_sub_nodes),
            )
            log_event(
                "orchestrator",
                "refinement",
                "SPLIT_DETERMINISTIC",
                payload={
                    "node_id": node.predicate_id,
                    "sub_node_count": len(deterministic_sub_nodes),
                },
            )
            return cdg

        # Layer 3: LLM-based split
        error_context = "\n".join(failure.error_summaries[:3])
        system_prompt = (
            "You are an algorithm decomposition expert. A predicate could not be "
            "grounded to a single library function. Suggest 2-3 finer-grained "
            "sub-predicates that together implement the same functionality. "
            "Reply with a JSON array of objects: "
            '[{"name": "...", "description": "...", "type_signature": "..."}]'
        )
        user_prompt = (
            f"Predicate: {node.statement}\n"
            f"Description: {node.informal_desc}\n"
            f"Match errors:\n{error_context}\n\n"
            "Split this into 2-3 finer sub-predicates."
        )

        try:
            response = await select_llm(llm, ORCHESTRATOR_REFINE).complete(
                system_prompt, user_prompt
            )

            # Parse JSON from response
            text = response.strip()
            if "```" in text:
                lines = text.splitlines()
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```") and not in_block:
                        in_block = True
                        continue
                    if line.strip() == "```" and in_block:
                        break
                    if in_block:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            sub_nodes = json.loads(text)
            if isinstance(sub_nodes, list) and sub_nodes and original is not None:
                _apply_split_subnodes(cdg, original, sub_nodes)

                logger.info(
                    "LLM split node %s into %d sub-nodes",
                    node.predicate_id,
                    len(sub_nodes[:3]),
                )
                log_event(
                    "orchestrator",
                    "refinement",
                    "SPLIT_LLM",
                    payload={
                        "node_id": node.predicate_id,
                        "sub_node_count": len(sub_nodes[:3]),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to split node %s: %s", node.predicate_id, exc)
            # Mark as ungroundable if splitting fails
            for n in cdg.nodes:
                if n.node_id == node.predicate_id:
                    n.status = NodeStatus.REJECTED
                    n.critic_notes = f"Split failed: {exc}"
                    break

    elif action == FailureAction.GENERALIZE:
        for n in cdg.nodes:
            if n.node_id == node.predicate_id:
                n.description = _generalize_description(
                    n.description, failure.error_summaries
                )
                n.type_signature = ""
                n.matched_primitive = None
                n.primitive_binding_confidence = 0.0
                n.critic_notes = "Re-formulated after match failure (GENERALIZE)"
                break
        log_event(
            "orchestrator",
            "refinement",
            "GENERALIZE_APPLIED",
            payload={"node_id": node.predicate_id},
        )

    return cdg


async def run_orchestration(
    cdg: CDGExport,
    *,
    hunter_agent: Any,
    llm: Any,
    prover: Prover = Prover.LEAN4,
    max_rounds: int = 3,
    hunter_concurrency: int = 1,
    template_retriever: Any | None = None,
) -> OrchestratorResult:
    """Run the full Architect -> Hunter feedback loop.

    Args:
        cdg: Initial CDG from the Architect.
        hunter_agent: A RetrievalAgent (has find_match method).
        llm: An LLM client for refinement.
        prover: Target prover.
        max_rounds: Maximum refinement rounds.
        hunter_concurrency: Maximum concurrent Hunter node matches per round.
        template_retriever: Optional TemplateRetriever for retrieval-based refinement.

    Returns:
        OrchestratorResult with final CDG and match results.
    """
    result = OrchestratorResult(cdg=cdg, match_results=[])

    architect_issues = cdg.architect_issues()
    if architect_issues:
        message = "; ".join(architect_issues)
        logger.warning("Architect produced blocked CDG: %s", message)
        update_stage(
            stage="orchestration",
            status="failed",
            message=message,
            completed=0,
            total=max_rounds,
        )
        log_event(
            "orchestrator",
            "round",
            "ARCHITECT_BLOCKED",
            payload={"issues": architect_issues},
        )
        return result

    for round_num in range(max_rounds):
        result.rounds_used = round_num + 1
        logger.info("Orchestration round %d/%d", round_num + 1, max_rounds)
        update_stage(
            stage="orchestration",
            status="running",
            message=f"round {round_num + 1}/{max_rounds}",
            completed=round_num,
            total=max_rounds,
        )
        log_event(
            "orchestrator",
            "round",
            "ROUND_START",
            payload={"round_num": round_num + 1, "max_rounds": max_rounds},
        )

        # Convert current CDG to PDG nodes
        try:
            pdg_nodes = to_pdg_nodes(cdg, prover=prover, strict=False)
        except ValueError as exc:
            logger.warning("CDG conversion failed: %s", exc)
            log_event(
                "orchestrator",
                "round",
                "CDG_CONVERSION_FAILED",
                payload={"error": str(exc)},
            )
            break

        if not pdg_nodes:
            logger.info("No atomic nodes to match")
            log_event("orchestrator", "round", "NO_ATOMIC_NODES")
            break

        # Filter out already-matched nodes
        matched_ids = {
            mr.pdg_node.predicate_id for mr in result.match_results if mr.success
        }
        pending_nodes = [n for n in pdg_nodes if n.predicate_id not in matched_ids]

        if not pending_nodes:
            logger.info("All nodes matched")
            log_event("orchestrator", "round", "ALL_MATCHED_EARLY")
            break

        # Run Hunter on pending nodes
        round_failures: list[MatchFailureReport] = []
        round_results: list[MatchResult] = []
        hunter_stage = f"hunter_round_{round_num + 1}"
        with telemetry_stage(
            hunter_stage,
            message=f"pending={len(pending_nodes)}",
            total=len(pending_nodes),
        ):
            with telemetry_scope(stage=hunter_stage):
                if hunter_concurrency <= 1 or len(pending_nodes) <= 1:
                    for i, pdg_node in enumerate(pending_nodes, start=1):
                        round_results.append(await hunter_agent.find_match(pdg_node))
                        update_stage(
                            stage=hunter_stage,
                            completed=i,
                            total=len(pending_nodes),
                        )
                else:
                    semaphore = asyncio.Semaphore(max(1, hunter_concurrency))

                    async def _run_hunter(pdg_node: Any) -> MatchResult:
                        async with semaphore:
                            return await hunter_agent.find_match(pdg_node)

                    round_results = list(
                        await asyncio.gather(
                            *[_run_hunter(node) for node in pending_nodes]
                        )
                    )
                    update_stage(
                        stage=hunter_stage,
                        completed=len(round_results),
                        total=len(pending_nodes),
                    )

        for match_result in round_results:
            pdg_node = match_result.pdg_node
            # Replace existing result for this node if any
            result.match_results = [
                mr
                for mr in result.match_results
                if mr.pdg_node.predicate_id != pdg_node.predicate_id
            ]
            result.match_results.append(match_result)

            if not match_result.success:
                failure = MatchFailureReport.from_match_result(match_result)
                round_failures.append(failure)
                result.failures.append(failure)

        log_event(
            "orchestrator",
            "hunter",
            "HUNTER_ROUND_DONE",
            payload={
                "round_num": round_num + 1,
                "pending_nodes": len(pending_nodes),
                "matches_succeeded": sum(1 for mr in round_results if mr.success),
                "matches_failed": len(round_failures),
            },
        )

        if not round_failures:
            logger.info("All nodes matched in round %d", round_num + 1)
            log_event(
                "orchestrator",
                "round",
                "ROUND_ALL_MATCHED",
                payload={"round_num": round_num + 1},
            )
            break

        # If this is the last round, don't refine
        if round_num >= max_rounds - 1:
            for f in round_failures:
                result.ungroundable.append(f.pdg_node.predicate_id)
            log_event(
                "orchestrator",
                "refine",
                "MAX_ROUNDS_REACHED",
                payload={"ungroundable": list(result.ungroundable)},
            )
            break

        # Refine CDG based on failures
        refine_stage = f"refine_round_{round_num + 1}"
        with telemetry_stage(
            refine_stage,
            message=f"failures={len(round_failures)}",
            total=len(round_failures),
        ):
            with telemetry_scope(stage=refine_stage):
                for i, failure in enumerate(round_failures, start=1):
                    cdg = await refine_on_failure(failure, cdg, llm, template_retriever=template_retriever)
                    update_stage(
                        stage=refine_stage,
                        completed=i,
                        total=len(round_failures),
                    )

        result.cdg = cdg
        decomposed_ids = {
            n.node_id for n in cdg.nodes if n.status == NodeStatus.DECOMPOSED
        }
        if decomposed_ids:
            result.match_results = [
                mr
                for mr in result.match_results
                if mr.pdg_node.predicate_id not in decomposed_ids
            ]

    update_stage(
        stage="orchestration",
        status="completed",
        completed=result.rounds_used,
        total=max_rounds,
    )
    log_event(
        "orchestrator",
        "run",
        "ORCHESTRATION_DONE",
        payload={
            "rounds_used": result.rounds_used,
            "matches_total": len(result.match_results),
            "matches_success": sum(1 for mr in result.match_results if mr.success),
            "ungroundable": list(result.ungroundable),
        },
    )
    return result
