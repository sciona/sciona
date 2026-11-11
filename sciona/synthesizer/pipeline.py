"""End-to-end orchestration for Phase 1: assemble and check."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING

from sciona.architect.handoff import CDGExport
from sciona.protocols import ProofEnvironment
from sciona.synthesizer.assembler import Assembler
from sciona.synthesizer.compiler import SkeletonCompiler
from sciona.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from sciona.synthesizer.models import AssemblyResult
from sciona.types import MatchResult
from sciona.uncertainty.conformal import TrustTierFilter

if TYPE_CHECKING:
    from sciona.synthesizer.uncertainty import UncertaintyBackend
    from sciona.uncertainty.models import AtomUncertaintyProfile

logger = logging.getLogger(__name__)


async def assemble_and_check(
    cdg: CDGExport,
    match_results: list[MatchResult],
    env: ProofEnvironment,
    *,
    skip_ghost_sim: bool = False,
    aot_target: str = "numpy",
    uncertainty_backend: UncertaintyBackend | None = None,
    uncertainty_profiles: Mapping[str, AtomUncertaintyProfile] | None = None,
    enable_confidence_guards: bool | None = None,
    trust_tier: TrustTierFilter = 1,
    confidence_alpha: float = 0.01,
    require_conformal: bool = True,
    risk_weights: Mapping[str, float] | None = None,
) -> AssemblyResult:
    """Assemble a CDG + match results into a skeleton, then compile it.

    Optionally runs a ghost witness simulation before assembly to catch
    structural mismatches early.  The simulation is best-effort — if
    ``sciona.atoms`` is not installed or no witnesses are available, it is
    silently skipped.

    Args:
        cdg: The Conceptual Dependency Graph from Round 1.
        match_results: Verified matches from Round 2.
        env: Proof environment with prover and compiler config.
        skip_ghost_sim: Set True to bypass the ghost simulation pass.

    Returns:
        AssemblyResult with the skeleton, compiler feedback,
        and whether compilation succeeded.
    """
    # Ghost simulation pass — runs before assembly
    ghost_report = GhostSimReport()
    if not skip_ghost_sim:
        ghost_report = run_ghost_simulation(cdg, match_results)
        if ghost_report.ran and not ghost_report.passed:
            logger.warning(
                "Ghost simulation detected issues at node '%s' (%s): %s",
                ghost_report.error_node,
                ghost_report.error_function,
                ghost_report.error,
            )

    assembler = Assembler(
        prover=env.prover_name,
        aot_target=aot_target,
        uncertainty_backend=uncertainty_backend,
        uncertainty_profiles=uncertainty_profiles,
        enable_confidence_guards=enable_confidence_guards,
        trust_tier=trust_tier,
        confidence_alpha=confidence_alpha,
        require_conformal=require_conformal,
    )
    skeleton = assembler.assemble(
        cdg,
        match_results,
        risk_weights=risk_weights,
    )

    # Attach ghost report to skeleton metadata
    if ghost_report.ran:
        skeleton.metadata["ghost_simulation"] = {
            "passed": ghost_report.passed,
            "node_count": ghost_report.node_count,
            "skipped_nodes": ghost_report.skipped_nodes,
            "trace": ghost_report.trace,
            "error": ghost_report.error,
        }

    compiler = SkeletonCompiler(env)
    result = await compiler.compile(skeleton)

    return result
