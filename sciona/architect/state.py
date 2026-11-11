"""Decomposition engine state and dependency types for LangGraph."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from typing_extensions import TypedDict

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.embedder import SkillIndex
from sciona.architect.models import AlgorithmicNode, DependencyEdge
from sciona.architect.planning_contract import PlanningArtifact
from sciona.hunter.llm import LLMClient
from sciona.shared_context import SharedContextMetrics, SharedContextStore

if TYPE_CHECKING:
    from sciona.architect.graph_retrieval import CDGSubgraphRetriever
    from sciona.architect.template_retriever import TemplateRetriever


def _merge_nodes(
    existing: list[AlgorithmicNode], updates: list[AlgorithmicNode]
) -> list[AlgorithmicNode]:
    """Custom reducer: latest entry per node_id wins.

    This lets node functions update status (PENDING -> DECOMPOSED, PENDING -> REJECTED)
    by returning a copy of the node. Standard operator.add would create duplicates.
    """
    by_id: dict[str, AlgorithmicNode] = {}
    for node in existing:
        by_id[node.node_id] = node
    for node in updates:
        by_id[node.node_id] = node
    return list(by_id.values())


def _merge_edges(
    existing: list[DependencyEdge], updates: list[DependencyEdge]
) -> list[DependencyEdge]:
    """Custom reducer: deduplicate edges by (source_id, target_id, output_name, input_name)."""
    by_key: dict[tuple[str, str, str, str], DependencyEdge] = {}
    for edge in existing:
        key = (edge.source_id, edge.target_id, edge.output_name, edge.input_name)
        by_key[key] = edge
    for edge in updates:
        key = (edge.source_id, edge.target_id, edge.output_name, edge.input_name)
        by_key[key] = edge
    return list(by_key.values())


class DecompositionState(TypedDict):
    """LangGraph state for the decomposition cycle."""

    # Immutable inputs
    goal: str
    max_depth: int

    # CDG accumulation
    nodes: Annotated[list[AlgorithmicNode], _merge_nodes]
    edges: Annotated[list[DependencyEdge], _merge_edges]
    history: Annotated[list[dict], operator.add]
    planning_artifact: PlanningArtifact | dict[str, Any] | None
    skeleton_asset: dict[str, Any] | None

    # Per-iteration (overwrite)
    pending_node_ids: list[str]
    current_node_id: str
    paradigm: str
    skeleton_instantiated: bool

    # Critique state (overwrite per-iteration)
    critique_passed: bool
    critique_reason: str
    critique_retries: int

    # Termination flags
    done: bool
    error: str


@dataclass
class DecompositionDeps:
    """Dependencies injected into the decomposition graph via config."""

    catalog: PrimitiveCatalog
    skill_index: SkillIndex
    llm: LLMClient
    graph_retriever: CDGSubgraphRetriever | None = field(default=None)
    template_retriever: TemplateRetriever | None = field(default=None)
    shared_context: SharedContextStore | None = field(default=None)
    shared_context_metrics: SharedContextMetrics | None = field(default=None)
    context_namespace: str = field(default="")
    context_budget_chars: int = field(default=900)
    use_monadic_rewriter: bool = field(default=False)
    architect_critique_llm_enabled: bool = field(default=True)
    trick_context_novelty_assessment: Any = field(default=None)
    trick_context_missing_techniques: tuple[str, ...] = field(default_factory=tuple)
    trick_context_candidate_cdgs: tuple[str, ...] = field(default_factory=tuple)
    trick_context_families: tuple[str, ...] = field(default_factory=tuple)
    trick_context_tags: tuple[str, ...] = field(default_factory=tuple)
    trick_context_max_results: int = field(default=5)
