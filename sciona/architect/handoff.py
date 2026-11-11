"""CDG serialization and handoff to Round 2 (the Hunter agent).

Converts a completed Conceptual Dependency Graph into PDGNode objects
that the Round 2 matcher can ground into verified library functions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sciona.architect.models import (
    AlgorithmicNode,
    DependencyEdge,
    NodeStatus,
)
from sciona.architect.stage_resolution import is_non_atom_resolved_node
from sciona.types import PDGNode, Prover

import ast
import logging
import re

_logger = logging.getLogger(__name__)


class HandoffValidationError(ValueError):
    """Raised when a CDG fails strict handoff validation."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__(
            f"Handoff validation failed with {len(issues)} issue(s): "
            + "; ".join(issues)
        )


def _hierarchy_parent_ids(nodes: list[AlgorithmicNode]) -> set[str]:
    """Return node IDs that act as decomposition parents in the CDG hierarchy.

    Note: ``DependencyEdge`` represents *data-flow* between sibling/leaf stages,
    NOT parent-child decomposition hierarchy.
    """
    existing_ids = {n.node_id for n in nodes}
    parents: set[str] = set()
    for node in nodes:
        if node.children or node.status == NodeStatus.DECOMPOSED:
            parents.add(node.node_id)
        if node.parent_id and node.parent_id in existing_ids:
            parents.add(node.parent_id)
    return parents


class CDGExport(BaseModel):
    """A complete CDG ready for Round 2 handoff."""

    nodes: list[AlgorithmicNode]
    edges: list[DependencyEdge]
    planning_artifact: dict[str, Any] | None = None
    metadata: dict = Field(default_factory=dict)

    def leaf_nodes(self) -> list[AlgorithmicNode]:
        """Return all leaf (atomic) nodes in the CDG."""
        parent_ids = _hierarchy_parent_ids(self.nodes)
        return [
            n
            for n in self.nodes
            if n.status == NodeStatus.ATOMIC and n.node_id not in parent_ids
        ]

    def handoff_issues(self) -> list[str]:
        """Convenience: return validation issues for Round 2 handoff."""
        return validate_handoff(self)

    def non_atomic_leaves(self) -> list[AlgorithmicNode]:
        """Return leaf nodes that are NOT atomic (validation failures)."""
        parent_ids = _hierarchy_parent_ids(self.nodes)

        leaves = []
        for node in self.nodes:
            if not node.children and node.node_id not in parent_ids:
                if node.status != NodeStatus.ATOMIC:
                    leaves.append(node)
        return leaves

    def is_complete(self) -> bool:
        """Check that all leaf nodes are atomic."""
        return len(self.non_atomic_leaves()) == 0

    def blocked_nodes(self) -> list[AlgorithmicNode]:
        """Return nodes that terminated decomposition without a valid handoff."""
        return [n for n in self.nodes if n.status == NodeStatus.BLOCKED]

    def architect_issues(self) -> list[str]:
        """Return architect-level issues that should block handoff."""
        issues: list[str] = []
        blocked = self.blocked_nodes()
        if blocked:
            issues.append(
                "Architect blocked nodes: " + ", ".join(node.name for node in blocked)
            )
        pending = [n.name for n in self.nodes if n.status == NodeStatus.PENDING]
        if pending:
            issues.append("Pending nodes remain: " + ", ".join(pending[:10]))
        non_atomic = self.non_atomic_leaves()
        if non_atomic:
            issues.append(
                "Non-atomic leaves remain: "
                + ", ".join(node.name for node in non_atomic[:10])
            )
        metadata_error = str(self.metadata.get("architect_error", "")).strip()
        if metadata_error:
            issues.append(metadata_error)
        return issues

    def is_handoff_ready(self) -> bool:
        """Check whether the architect produced a handoff-safe graph."""
        return len(self.architect_issues()) == 0


def export_cdg(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
    *,
    goal: str = "",
    paradigm: str = "",
    planning_artifact: dict[str, Any] | None = None,
) -> CDGExport:
    """Build a CDGExport, validating that all leaves are atomic.

    Raises:
        ValueError: If any leaf node is not atomic.
    """
    cdg = CDGExport(
        nodes=nodes,
        edges=edges,
        planning_artifact=planning_artifact,
        metadata={
            "goal": goal,
            "paradigm": paradigm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            **(
                {"planning_artifact": planning_artifact}
                if planning_artifact is not None
                else {}
            ),
        },
    )

    non_atomic = cdg.non_atomic_leaves()
    if non_atomic:
        names = [n.name for n in non_atomic]
        raise ValueError(
            f"CDG has {len(non_atomic)} non-atomic leaf node(s): {names}. "
            "All leaves must be marked atomic before export."
        )

    return cdg


def validate_handoff(cdg: CDGExport) -> list[str]:
    """Check that every atomic leaf is ready for Round 2 handoff.

    Returns a list of human-readable issue strings (empty == valid).
    Checks:
    - Every atomic leaf must have a non-empty ``description``.
    - Every atomic leaf must have a non-empty ``type_signature``.
    - Flags non-atomic leaf nodes (they can't be handed off).
    """
    issues: list[str] = []
    parent_ids = _hierarchy_parent_ids(cdg.nodes)

    for node in cdg.nodes:
        is_leaf = not node.children and node.node_id not in parent_ids
        if not is_leaf:
            continue

        if node.status == NodeStatus.ATOMIC:
            if is_non_atom_resolved_node(node):
                continue
            if not node.description:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has empty description"
                )
            if not node.type_signature:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has empty type_signature"
                )
        else:
            issues.append(
                f"Leaf '{node.name}' ({node.node_id}) is {node.status.value}, not atomic"
            )

    return issues


def _validate_type_signature_syntax(
    sig: str, prover: Prover = Prover.LEAN4
) -> str | None:
    """Validate type signature syntax. Returns error string or None if OK."""
    if not sig.strip():
        return None  # empty is caught by basic validation
    sig = sig.strip()

    if prover == Prover.PYTHON:
        try:
            if "->" in sig:
                lhs, rhs = sig.rsplit("->", 1)
                lhs_clean = lhs.strip()
                rhs_clean = rhs.strip()
                if not rhs_clean:
                    return f"Invalid Python type annotation '{sig}': empty return type"
                if lhs_clean.startswith("(") and lhs_clean.endswith(")"):
                    ast.parse(f"def _sig{lhs_clean} -> {rhs_clean}: pass", mode="exec")
                else:
                    ast.parse(f"x: tuple[{lhs_clean}]", mode="exec")
                    ast.parse(f"y: {rhs_clean}", mode="exec")
            else:
                ast.parse(f"x: {sig}", mode="exec")
        except SyntaxError as exc:
            return f"Invalid Python type annotation '{sig}': {exc}"
        return None
    elif prover == Prover.LEAN4:
        # Lean 4: must contain arrow, colon, or be a single identifier
        if re.match(r"^[A-Za-z_]\w*$", sig):
            return None  # single identifier
        if "\u2192" in sig or "→" in sig or ":" in sig or "->" in sig:
            return None
        # Check for balanced parentheses
        depth = 0
        for ch in sig:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if depth < 0:
                return f"Unbalanced parentheses in Lean type '{sig}'"
        if depth != 0:
            return f"Unbalanced parentheses in Lean type '{sig}'"
        return None  # allow other forms
    elif prover == Prover.COQ:
        if re.match(r"^[A-Za-z_]\w*$", sig):
            return None
        if "->" in sig or "forall" in sig.lower() or ":" in sig:
            return None
        return None
    return None


def _check_edge_type_compatibility(edge: DependencyEdge) -> str | None:
    """Check if source_type and target_type are compatible.

    Returns a warning string or None if compatible.
    """
    src = edge.source_type.strip()
    tgt = edge.target_type.strip()

    if not src or not tgt:
        return None  # missing types are not a compatibility error

    if src == tgt:
        return None

    # Both contain ndarray → OK (shape mismatch caught by ghost sim)
    if "ndarray" in src and "ndarray" in tgt:
        return None

    # Both contain array → OK
    if "array" in src.lower() and "array" in tgt.lower():
        return None

    return (
        f"Edge {edge.source_id}->{edge.target_id}: "
        f"type mismatch '{src}' vs '{tgt}' "
        f"(output '{edge.output_name}' -> input '{edge.input_name}')"
    )


def _check_graph_connectivity(cdg: "CDGExport") -> list[str]:
    """Verify the CDG forms a connected DAG from root(s) to all atomic leaves.

    Returns list of issues (orphan nodes not reachable from any root).
    """
    if not cdg.nodes:
        return []

    # Build adjacency: parent -> children
    children_of: dict[str, set[str]] = {n.node_id: set() for n in cdg.nodes}
    for edge in cdg.edges:
        if edge.source_id in children_of:
            children_of[edge.source_id].add(edge.target_id)

    # Also use node.children
    for node in cdg.nodes:
        for child_id in node.children:
            children_of.setdefault(node.node_id, set()).add(child_id)

    # Find roots: nodes with no parent
    child_ids = set()
    for edge in cdg.edges:
        child_ids.add(edge.target_id)
    for node in cdg.nodes:
        if node.parent_id:
            child_ids.add(node.node_id)

    roots = [n.node_id for n in cdg.nodes if n.node_id not in child_ids]
    if not roots:
        # Fallback: treat DECOMPOSED nodes at depth 0 as roots
        roots = [n.node_id for n in cdg.nodes if n.depth == 0]

    # BFS from roots
    reachable: set[str] = set()
    queue = list(roots)
    while queue:
        nid = queue.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        for child in children_of.get(nid, set()):
            queue.append(child)

    # Check for orphans
    issues: list[str] = []
    all_ids = {n.node_id for n in cdg.nodes}
    orphans = all_ids - reachable - set(roots)
    for orphan_id in orphans:
        node = next((n for n in cdg.nodes if n.node_id == orphan_id), None)
        if node:
            issues.append(
                f"Orphan node '{node.name}' ({node.node_id}) is not reachable from any root"
            )

    return issues


def validate_handoff_strict(
    cdg: "CDGExport",
    prover: Prover = Prover.LEAN4,
) -> list[str]:
    """Strict validation of a CDG for Round 2 handoff.

    Runs all checks from ``validate_handoff`` plus:
    - Type signature syntax validation (per prover)
    - Edge type compatibility checks
    - Graph connectivity (no orphan nodes)
    - IOSpec arity (atoms must have inputs and outputs)

    Returns a list of issue strings (empty == valid).
    """
    issues = validate_handoff(cdg)

    # Type signature syntax validation
    for node in cdg.nodes:
        if node.status == NodeStatus.ATOMIC and node.type_signature:
            err = _validate_type_signature_syntax(node.type_signature, prover)
            if err:
                issues.append(f"Atomic leaf '{node.name}' ({node.node_id}): {err}")

    # Edge type compatibility
    for edge in cdg.edges:
        warning = _check_edge_type_compatibility(edge)
        if warning:
            _logger.warning("Handoff edge type warning: %s", warning)
            issues.append(warning)

    # Connectivity check
    connectivity_issues = _check_graph_connectivity(cdg)
    issues.extend(connectivity_issues)

    # IOSpec arity check: atomic nodes must have inputs and outputs
    for node in cdg.nodes:
        if node.status == NodeStatus.ATOMIC:
            if not node.inputs:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has no inputs"
                )
            if not node.outputs:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has no outputs"
                )

    return issues


def to_pdg_nodes(
    cdg: CDGExport,
    *,
    prover: Prover = Prover.LEAN4,
    strict: bool = True,
) -> list[PDGNode]:
    """Convert atomic leaf nodes of a CDG into PDGNode objects for the Round 2 Hunter.

    Each atomic leaf becomes a PDGNode with:
    - predicate_id: the node_id
    - statement: the type_signature (or description if no type sig)
    - informal_desc: the node description
    - context: parent chain, concept type, matched primitive

    Args:
        strict: When True (default), runs ``validate_handoff`` and raises
            ``HandoffValidationError`` on any issues. Set to False to skip.

    Raises:
        HandoffValidationError: If strict and the CDG has handoff issues.
        ValueError: If the CDG is not complete (has non-atomic leaves).
    """
    if strict:
        issues = validate_handoff(cdg)
        if issues:
            raise HandoffValidationError(issues)

    architect_issues = cdg.architect_issues()
    if architect_issues:
        raise ValueError("Cannot convert architect-blocked CDG. " + "; ".join(architect_issues))

    if not cdg.is_complete():
        non_atomic = cdg.non_atomic_leaves()
        names = [n.name for n in non_atomic]
        raise ValueError(f"Cannot convert incomplete CDG. Non-atomic leaves: {names}")

    pdg_nodes: list[PDGNode] = []
    for node in cdg.leaf_nodes():
        if is_non_atom_resolved_node(node):
            continue
        statement = node.type_signature or node.description
        context = {
            "concept_type": node.concept_type.value,
            "depth": str(node.depth),
        }
        if node.parent_id:
            context["parent_id"] = node.parent_id
        if node.matched_primitive:
            context["matched_primitive"] = node.matched_primitive
        if node.critic_notes:
            context["critic_notes"] = node.critic_notes

        pdg_node = PDGNode(
            predicate_id=node.node_id,
            statement=statement,
            informal_desc=node.description,
            prover=prover,
            context=context,
        )
        pdg_nodes.append(pdg_node)

    return pdg_nodes


def save_json(cdg: CDGExport, path: str | Path) -> None:
    """Save a CDGExport to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cdg.model_dump(), f, indent=2)


def load_json(path: str | Path) -> CDGExport:
    """Load a CDGExport from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return CDGExport.model_validate(data)


def find_cdg(name: str) -> Path | None:
    """Search all configured atom sources for a CDG file matching *name*.

    Delegates to :func:`sciona.sources.find_cdg`.  Returns ``None`` if
    no match is found or sources.yml is not present.
    """
    try:
        from sciona.sources import find_cdg as _find_cdg

        return _find_cdg(name)
    except Exception:
        _logger.debug("find_cdg failed", exc_info=True)
        return None
