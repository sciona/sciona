"""High-level programmatic interface to the sciona framework.

Usage::

    from sciona.sdk import Sciona

    s = await Sciona.create(
        atom_repos=["~/personal/sciona-atoms", "~/personal/sciona-atoms-ml"],
        llm_provider="anthropic",
    )

    result = await s.propose(
        problem="Predict smartphone location from raw GNSS measurements",
        data="Pseudorange at 1Hz, IMU at 100Hz, ~500 traces",
        metric="50th percentile horizontal distance error (meters)",
    )
    print(result.grounding.grounding_rate)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sciona.api_models import (
    AtomMatch,
    AtomSearchResult,
    CDGInspection,
    GapReport,
    GeneratedCode,
    GroundingReport,
    ProposalResult,
    StageMatchResult,
)
from sciona.architect.catalog import (
    PrimitiveCatalog,
    seed_builtin_primitives,
    seed_solution_retrieval_aliases,
)
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    CommonPattern,
    ConceptType,
    IOSpec,
    StatePortSpec,
    format_canonical_type_signature,
)

if TYPE_CHECKING:
    from sciona.architect.handoff import CDGExport
    from sciona.config import AgeomConfig
    from sciona.synthesizer.models import (
        SkeletonFile,
        SynthesisResult,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog loading from repo paths
# ---------------------------------------------------------------------------


def _safe_concept_type(raw: str) -> ConceptType:
    try:
        return ConceptType(raw)
    except ValueError:
        return ConceptType.CUSTOM


def load_catalog_from_repos(
    repo_paths: list[str | Path],
) -> PrimitiveCatalog:
    """Load a full PrimitiveCatalog from a list of atom repo directories.

    Scans each repo for ``cdg.json`` files, extracts atomic nodes (supporting
    both canonical ``nodes`` and legacy ``atomic_nodes`` schemas), and
    registers them with their aliases and structured port types.
    """
    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)

    for repo_raw in repo_paths:
        repo = Path(repo_raw).expanduser().resolve()
        if not repo.exists():
            logger.warning("Atom repo not found, skipping: %s", repo)
            continue
        for cdg_path in repo.rglob("cdg.json"):
            if "solution_cdgs" in str(cdg_path):
                continue
            try:
                data = json.loads(cdg_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            raw_nodes = data.get("nodes")
            is_legacy_atomic = False
            if raw_nodes is None and isinstance(data.get("atomic_nodes"), list):
                raw_nodes = data["atomic_nodes"]
                is_legacy_atomic = True
            for node in raw_nodes or []:
                if not isinstance(node, dict):
                    continue
                if not is_legacy_atomic and node.get("status") != "atomic":
                    continue
                node_id = str(node.get("node_id") or node.get("name") or "").strip()
                if not node_id:
                    continue
                try:
                    inputs = [
                        IOSpec(**inp)
                        for inp in (node.get("inputs") or [])
                        if isinstance(inp, dict)
                    ]
                    for port in inputs:
                        if port.port_type is None:
                            port.port_type = port.resolved_port_type()
                    outputs = [
                        IOSpec(**out)
                        for out in (node.get("outputs") or [])
                        if isinstance(out, dict)
                    ]
                    for port in outputs:
                        if port.port_type is None:
                            port.port_type = port.resolved_port_type()
                    type_sig = str(node.get("type_signature", "") or "").strip()
                    if not type_sig and (inputs or outputs):
                        type_sig = format_canonical_type_signature(inputs, outputs)
                    desc = str(
                        node.get("description") or node.get("semantic_role") or ""
                    ).strip()
                    catalog.add(
                        AlgorithmicPrimitive(
                            name=node_id,
                            source=str(cdg_path.parent.relative_to(repo)),
                            category=_safe_concept_type(
                                str(node.get("concept_type", "custom") or "custom")
                            ),
                            description=desc,
                            inputs=inputs,
                            outputs=outputs,
                            type_signature=type_sig,
                            aliases=list(node.get("aliases") or []),
                            common_patterns=[
                                CommonPattern(**p)
                                for p in (node.get("common_patterns") or [])
                                if isinstance(p, dict)
                            ],
                            phase=str(node.get("phase") or "stateless"),
                            state_ports=[
                                StatePortSpec.model_validate(sp)
                                for sp in (node.get("state_ports") or [])
                                if isinstance(sp, dict)
                            ],
                            semantic_role=str(node.get("semantic_role") or ""),
                            time_complexity=str(node.get("time_complexity") or ""),
                            space_complexity=str(node.get("space_complexity") or ""),
                        )
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Skipping malformed atom %s in %s",
                        node_id,
                        cdg_path,
                    )

    seed_solution_retrieval_aliases(catalog)
    return catalog


# ---------------------------------------------------------------------------
# Default repo list
# ---------------------------------------------------------------------------

_DEFAULT_REPOS = [
    "sciona-atoms",
    "sciona-atoms-ml",
    "sciona-atoms-dl",
    "sciona-atoms-bio",
    "sciona-atoms-physics",
    "sciona-atoms-signal",
    "sciona-atoms-cs",
    "sciona-atoms-geo",
    "sciona-atoms-fintech",
    "sciona-atoms-robotics",
]


def _resolve_default_repos() -> list[Path]:
    """Auto-discover atom repos as siblings of sciona-matcher."""
    matcher_dir = Path(__file__).resolve().parent.parent
    base = matcher_dir.parent  # e.g., ~/personal
    repos = []
    for name in _DEFAULT_REPOS:
        candidate = base / name
        if candidate.is_dir():
            repos.append(candidate)
    return repos


# ---------------------------------------------------------------------------
# Sciona API
# ---------------------------------------------------------------------------


class Sciona:
    """High-level programmatic interface to the sciona framework.

    Holds shared state (catalog, indexes, LLM) and exposes every useful
    pipeline operation as a method.  Shared state is read-only after init;
    each method call is stateless.
    """

    # -- construction -------------------------------------------------------

    def __init__(
        self,
        catalog: PrimitiveCatalog,
        config: AgeomConfig | None = None,
        *,
        _llm: Any = None,
        _skill_index: Any = None,
        _semantic_index: Any = None,
        _proof_env: Any = None,
        _architect_agent: Any = None,
        _hunter_agent: Any = None,
        _solution_index: Any = None,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._llm = _llm
        self._skill_index = _skill_index
        self._semantic_index = _semantic_index
        self._proof_env = _proof_env
        self._architect_agent = _architect_agent
        self._hunter_agent = _hunter_agent
        self._solution_index = _solution_index

    @classmethod
    async def create(
        cls,
        atom_repos: list[str | Path] | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        prover: str = "python",
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> Sciona:
        """Async factory: load catalog, indexes, and LLM.

        Parameters
        ----------
        atom_repos:
            Paths to atom repo directories.  If *None*, auto-discovers
            repos as siblings of the sciona-matcher directory.
        llm_provider:
            LLM backend (``"anthropic"``, ``"openai"``, ``"llama_cpp"``).
        llm_model:
            Model name override.  If *None*, uses config default.
        prover:
            Proof environment (``"python"``, ``"lean4"``, ``"coq"``).
        config_overrides:
            Dict of ``AgeomConfig`` field overrides.
        """
        from sciona.config import AgeomConfig

        # -- config ---------------------------------------------------------
        overrides = config_overrides or {}
        if llm_provider:
            overrides.setdefault("llm_provider", llm_provider)
        if llm_model:
            overrides.setdefault("llm_model", llm_model)
        config = AgeomConfig(**overrides)  # type: ignore[arg-type]

        # -- catalog --------------------------------------------------------
        if atom_repos is not None:
            catalog = load_catalog_from_repos(atom_repos)
        else:
            repos = _resolve_default_repos()
            if repos:
                catalog = load_catalog_from_repos(repos)
            else:
                # Fall back to runtime_helpers loader
                args = argparse.Namespace(catalog=None, sources_only=False)
                from sciona.commands.runtime_helpers import (
                    _load_architect_catalog,
                )

                catalog, _ = _load_architect_catalog(args, config)

        logger.info("Catalog loaded: %d atoms", catalog.size)

        # -- optional indexes -----------------------------------------------
        skill_index = None
        semantic_index = None
        try:
            from sciona.commands.runtime_helpers import (
                _load_skill_index_or_empty,
            )

            skill_index = _load_skill_index_or_empty(config)
        except Exception:  # noqa: BLE001
            logger.debug("Skill index not available")

        try:
            from sciona.commands.runtime_helpers import _load_semantic_index

            semantic_index, _ = _load_semantic_index(
                config.index_dir, config
            )
        except Exception:  # noqa: BLE001
            logger.debug("Semantic index not available")

        # -- LLM ------------------------------------------------------------
        llm = None
        try:
            from sciona.commands.llm_helpers import _create_llm

            args = argparse.Namespace(
                mode=None,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                llm_max_tokens=config.llm_max_tokens,
            )
            llm = _create_llm(args, config, "architect")
        except Exception:  # noqa: BLE001
            logger.debug("LLM client not available")

        # -- proof environment ----------------------------------------------
        proof_env = None
        try:
            from sciona.commands.runtime_helpers import _create_proof_env
            from sciona.types import Prover as ProverEnum

            proof_env = _create_proof_env(ProverEnum(prover), config)
        except Exception:  # noqa: BLE001
            logger.debug("Proof environment not available for %s", prover)

        # -- architect agent ------------------------------------------------
        architect_agent = None
        if llm is not None:
            try:
                from sciona.architect.graph import DecompositionAgent

                architect_agent = DecompositionAgent(
                    catalog=catalog,
                    skill_index=skill_index,
                    llm=llm,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Architect agent not available")

        # -- hunter agent ---------------------------------------------------
        hunter_agent = None
        if llm is not None and semantic_index is not None and proof_env is not None:
            try:
                from sciona.hunter.graph import HunterAgent
                from sciona.judge.checker import VerificationOracleImpl
                from sciona.types import Prover as ProverEnum

                prover_enum = ProverEnum(prover)
                if prover_enum == ProverEnum.PYTHON:
                    oracle = VerificationOracleImpl(python_env=proof_env)
                elif prover_enum == ProverEnum.COQ:
                    oracle = VerificationOracleImpl(coq_env=proof_env)
                else:
                    oracle = VerificationOracleImpl(lean_env=proof_env)

                hunter_agent = HunterAgent(
                    index=semantic_index,
                    oracle=oracle,
                    llm=llm,
                    live_catalog=catalog,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Hunter agent not available")

        # -- solution template index -----------------------------------
        solution_index = None
        try:
            from sciona.architect.solution_index import SolutionTemplateIndex

            cdg_dir = Path.home() / "personal" / "sciona-atoms" / "data" / "solution_cdgs"
            if cdg_dir.is_dir():
                solution_index = SolutionTemplateIndex.from_directory(cdg_dir)
                logger.info("Solution index: %d templates", solution_index.size)
        except Exception:  # noqa: BLE001
            logger.debug("Solution template index not available")

        return cls(
            catalog=catalog,
            config=config,
            _llm=llm,
            _skill_index=skill_index,
            _semantic_index=semantic_index,
            _proof_env=proof_env,
            _architect_agent=architect_agent,
            _hunter_agent=hunter_agent,
            _solution_index=solution_index,
        )

    @classmethod
    def from_catalog(
        cls,
        catalog: PrimitiveCatalog,
        solution_cdg_dir: str | Path | None = None,
    ) -> Sciona:
        """Create a lightweight instance with only catalog (no LLM/indexes).

        Useful for retrieval-only workflows (search, inspect, gaps).
        """
        solution_index = None
        if solution_cdg_dir is None:
            solution_cdg_dir = (
                Path.home() / "personal" / "sciona-atoms" / "data" / "solution_cdgs"
            )
        cdg_dir = Path(solution_cdg_dir)
        if cdg_dir.is_dir():
            try:
                from sciona.architect.solution_index import SolutionTemplateIndex

                solution_index = SolutionTemplateIndex.from_directory(cdg_dir)
            except Exception:  # noqa: BLE001
                pass
        return cls(catalog=catalog, _solution_index=solution_index)

    # -- properties ---------------------------------------------------------

    @property
    def catalog(self) -> PrimitiveCatalog:
        """The loaded atom catalog."""
        return self._catalog

    @property
    def atom_count(self) -> int:
        return self._catalog.size

    # -- catalog & retrieval ------------------------------------------------

    def search_atoms(
        self, query: str, k: int = 10
    ) -> list[AtomSearchResult]:
        """Search the catalog by keyword similarity."""
        node = AlgorithmicNode(
            node_id=query.replace(" ", "_").lower(),
            name=query,
            description=query,
            concept_type=ConceptType.CUSTOM,
        )
        prims = self._catalog.find_matching_primitives(node, k=k)
        return [
            AtomSearchResult(
                atom_name=p.name,
                atom_fqdn=f"{p.source}.{p.name}" if p.source else p.name,
                source=p.source,
                category=p.category.value,
                description=p.description[:200],
            )
            for p in prims
        ]

    def find_matching_atoms(
        self,
        description: str,
        concept_type: str | None = None,
        k: int = 5,
    ) -> list[AtomMatch]:
        """Find atoms matching a stage description."""
        ct = _safe_concept_type(concept_type) if concept_type else ConceptType.CUSTOM
        node = AlgorithmicNode(
            node_id="query",
            name=description[:60],
            description=description,
            concept_type=ct,
        )
        prims = self._catalog.find_matching_primitives(node, k=k)
        return [
            AtomMatch(
                atom_name=p.name,
                atom_fqdn=f"{p.source}.{p.name}" if p.source else p.name,
                score=0.0,  # score not exposed by find_matching_primitives
                category=p.category.value,
                description=p.description[:200],
            )
            for p in prims
        ]

    def catalog_gaps(
        self, cdg: CDGExport, threshold: float = 0.5
    ) -> GapReport:
        """Detect atom coverage gaps in a CDG."""
        stages = cdg.nodes if hasattr(cdg, "nodes") else []
        gaps = []
        for node in stages:
            if getattr(node, "status", None) != "atomic":
                continue
            if self._catalog.get(node.name) is not None:
                continue
            if (
                node.matched_primitive
                and self._catalog.get(node.matched_primitive) is not None
            ):
                continue
            gaps.append(node.node_id)
        return GapReport(
            total_stages=len(stages),
            covered=len(stages) - len(gaps),
            gaps=gaps,
        )

    # -- decomposition (Round 1) --------------------------------------------

    async def decompose(
        self,
        goal: str,
        max_depth: int = 8,
        thread_id: str | None = None,
    ) -> CDGExport:
        """Decompose a high-level goal into a CDG."""
        if self._architect_agent is None:
            raise RuntimeError(
                "Architect agent not available. "
                "Call Sciona.create() with an LLM provider."
            )
        return await self._architect_agent.decompose(
            goal, thread_id=thread_id
        )

    def find_matching_templates(
        self,
        prompt: str,
        k: int = 10,
        *,
        dejargonize: bool = True,
    ) -> list[tuple[Any, float]]:
        """Find solution CDG templates matching a problem prompt.

        Uses dejargonized keyword search against the solution template index.
        Returns list of (SolutionTemplate, score) pairs.
        """
        if self._solution_index is None:
            return []

        from sciona.architect.dejargonizer import dejargonize_heuristic

        search_prompt = dejargonize_heuristic(prompt) if dejargonize else prompt
        return self._solution_index.search_dejargonized(
            dejargonized_prompt=search_prompt,
            original_prompt=prompt,
            k=k,
        )

    async def rerank_templates(
        self,
        prompt: str,
        candidates: list[Any],
        max_candidates: int = 5,
    ) -> Any:
        """Use LLM to semantically rerank template candidates.

        Requires an LLM client (call Sciona.create() with an LLM provider).
        """
        if self._llm is None:
            raise RuntimeError(
                "LLM not available for reranking. "
                "Call Sciona.create() with an LLM provider."
            )
        from sciona.architect.template_reranker import rerank_templates

        return await rerank_templates(prompt, candidates, self._llm, max_candidates)

    async def propose(
        self,
        problem: str,
        data: str = "",
        metric: str = "",
        constraints: str = "",
        domain_hints: list[str] | None = None,
    ) -> ProposalResult:
        """Compose a problem prompt, decompose, match, and ground.

        This is the main entry point for the Kaggle validation pipeline.
        Uses a three-phase approach:
        1. Dejargonize prompt + keyword match against solution templates
        2. If strong template match, use it directly
        3. Otherwise, decompose from scratch via architect
        """
        t0 = time.monotonic()

        # Format goal
        parts = [f"Problem: {problem}"]
        if data:
            parts.append(f"Data: {data}")
        if metric:
            parts.append(f"Metric: {metric}")
        if constraints:
            parts.append(f"Constraints: {constraints}")
        if domain_hints:
            parts.append(f"Domain: {', '.join(domain_hints)}")
        goal = "\n".join(parts)

        # Phase 1: Try template matching
        template_used = None
        template_score = 0.0
        top_cdg = None
        template_matches = self.find_matching_templates(goal, k=10)

        if template_matches:
            top_template, top_score = template_matches[0]
            template_used = top_template.name
            template_score = top_score
            top_cdg = getattr(top_template, "cdg", None)

        # Phase 2: Decompose (reuse high-confidence template or fallback when no LLM)
        if top_cdg is not None and (
            top_score >= 0.75 or self._architect_agent is None
        ):
            cdg = top_cdg.model_copy(deep=True) if hasattr(top_cdg, "model_copy") else top_cdg
        else:
            cdg = await self.decompose(goal)

        # Match stages
        matches = self._match_cdg_stages(cdg)

        # Build grounding report
        grounding = self._build_grounding_report(matches)

        return ProposalResult(
            cdg=cdg,
            template_used=template_used,
            template_match_score=template_score,
            grounding=grounding,
            matches=matches,
            wall_time_seconds=time.monotonic() - t0,
        )

    # -- matching (Round 2) -------------------------------------------------

    def _match_cdg_stages(self, cdg: CDGExport) -> list[StageMatchResult]:
        """Match all CDG stages to catalog atoms (keyword path)."""
        results = []
        nodes = cdg.nodes if hasattr(cdg, "nodes") else []
        for node in nodes:
            if getattr(node, "status", None) in (
                "decomposed",
                "rejected",
            ):
                continue

            # Check non-atom action classes
            action = getattr(node, "action_class", "") or ""
            if action in (
                "orchestration",
                "trivial_inline",
                "external_knowledge",
                "external_tool",
            ):
                results.append(
                    StageMatchResult(
                        stage_id=node.node_id,
                        stage_name=getattr(node, "name", node.node_id),
                        action_class=action,
                    )
                )
                continue

            # Try retrieval
            prims = self._catalog.find_matching_primitives(node, k=5)
            top = []
            for p in prims:
                top.append(
                    AtomMatch(
                        atom_name=p.name,
                        atom_fqdn=(
                            f"{p.source}.{p.name}" if p.source else p.name
                        ),
                        score=0.0,
                        category=p.category.value,
                        description=p.description[:200],
                    )
                )

            matched = top[0].atom_fqdn if top else None
            results.append(
                StageMatchResult(
                    stage_id=node.node_id,
                    stage_name=getattr(node, "name", node.node_id),
                    action_class="replace_stage",
                    matched_atom=matched,
                    top_candidates=top,
                )
            )
        return results

    # -- assembly & synthesis (Round 3) -------------------------------------

    def assemble(
        self,
        cdg: CDGExport,
        match_results: list[Any],
        *,
        prover: str | None = None,
    ) -> SkeletonFile:
        """Assemble a CDG + match results into a compilable skeleton."""
        from sciona.synthesizer.assembler import Assembler
        from sciona.types import Prover as ProverEnum

        resolved_prover = prover
        if resolved_prover is None:
            if match_results and hasattr(match_results[0], "pdg_node"):
                resolved_prover = getattr(
                    match_results[0].pdg_node.prover, "value", None
                )
            elif self._proof_env is not None:
                resolved_prover = getattr(self._proof_env, "prover_name", None)
        prover_enum = ProverEnum(resolved_prover or "python")

        assembler = Assembler(prover_enum)
        return assembler.assemble(cdg, match_results)

    async def synthesize(
        self,
        cdg: CDGExport,
        match_results: list[Any],
        max_iterations: int = 5,
    ) -> SynthesisResult:
        """Full assembly + compile + repair cycle."""
        from sciona.services import SynthesizerService
        from sciona.services.models import SynthesizerRepairRequest
        from sciona.synthesizer.agent import SynthesizerAgent
        from sciona.types import Prover as ProverEnum

        skeleton = self.assemble(cdg, match_results)

        if self._proof_env is None:
            raise RuntimeError(
                "Proof environment not available. "
                "Call Sciona.create() with a prover."
            )
        if self._llm is None:
            raise RuntimeError(
                "LLM client not available for repair. "
                "Call Sciona.create() with an LLM provider."
            )

        prover_enum = ProverEnum(skeleton.prover or "python")
        repair_agent = SynthesizerAgent(
            env=self._proof_env,
            llm=self._llm,
            max_iterations=max_iterations,
        )
        svc = SynthesizerService(prover=prover_enum, repair_agent=repair_agent)
        result = await svc.repair(SynthesizerRepairRequest(skeleton=skeleton))
        return result.result

    def generate_code(
        self,
        cdg: CDGExport,
        match_results: list[Any] | None = None,
    ) -> GeneratedCode:
        """Generate pure Python/NumPy code from a grounded CDG.

        Works without a proof environment — generates source code directly
        from the CDG and matched atoms.
        """
        from sciona.synthesizer.assembler import Assembler
        from sciona.types import (
            CandidateMatch,
            Declaration,
            MatchResult,
            PDGNode,
            Prover as ProverEnum,
            VerificationResult,
        )

        atoms_used: list[str] = []
        nodes = cdg.nodes if hasattr(cdg, "nodes") else []
        synth_matches: list[Any] = list(match_results) if match_results is not None else []
        build_synthetic = match_results is None

        for node in nodes:
            mp = getattr(node, "matched_primitive", None)
            if mp:
                atoms_used.append(mp)
                if build_synthetic and getattr(node, "status", None) != "decomposed":
                    decl = Declaration(
                        name=mp,
                        type_signature=getattr(node, "type_signature", "") or "",
                        docstring=getattr(node, "description", "") or "",
                        prover=ProverEnum.PYTHON,
                    )
                    cand = CandidateMatch(
                        declaration=decl, score=1.0, retrieval_method="matched_primitive"
                    )
                    vr = VerificationResult(candidate=cand, verified=True)
                    pdg = PDGNode(
                        predicate_id=node.node_id,
                        statement=getattr(node, "type_signature", "") or mp,
                        informal_desc=getattr(node, "description", "") or "",
                        prover=ProverEnum.PYTHON,
                    )
                    synth_matches.append(
                        MatchResult(
                            pdg_node=pdg,
                            verified_match=vr,
                            all_candidates=[cand],
                            all_verifications=[vr],
                        )
                    )

        if synth_matches:
            skeleton = Assembler(ProverEnum.PYTHON).assemble(cdg, synth_matches)
            return GeneratedCode(
                source=skeleton.source_code,
                atom_fqdns_used=atoms_used,
            )

        return GeneratedCode(
            source="# Ground at least one atomic leaf (matched_primitive or match_results) to generate code\n",
            atom_fqdns_used=atoms_used,
        )

    # -- end-to-end orchestration -------------------------------------------

    async def run(
        self,
        goal: str,
        mode: str = "single_agent",
        max_rounds: int = 3,
    ) -> Any:
        """Full pipeline: decompose, match, refine, assemble, compile.

        Returns the raw ``OrchestratorResult`` or ``PlannerRunResult``
        depending on the execution mode.
        """
        if mode == "single_agent":
            return await self._run_single_agent(goal, max_rounds=max_rounds)
        return await self._run_structured(goal)

    async def _run_single_agent(
        self, goal: str, max_rounds: int = 3
    ) -> Any:
        """Single-agent planner path."""
        if self._architect_agent is None or self._hunter_agent is None:
            raise RuntimeError(
                "Architect and Hunter agents required for run(). "
                "Call Sciona.create() with an LLM provider."
            )

        from sciona.services import (
            ArchitectService,
            HunterService,
            OrchestratorService,
            SingleAgentPlanner,
        )
        from sciona.types import Prover

        architect_svc = ArchitectService(agent=self._architect_agent)
        hunter_svc = HunterService(hunter=self._hunter_agent)

        async def architect_factory():
            return architect_svc

        # Orchestrator needs the orchestrate function
        from sciona.orchestrator import run_orchestration

        orchestrator_svc = OrchestratorService(
            hunter_agent=self._hunter_agent,
            orchestrate=run_orchestration,
        )

        planner = SingleAgentPlanner(
            hunter=hunter_svc,
            architect_factory=architect_factory,
            orchestrator=orchestrator_svc,
            llm=self._llm,
            prover=Prover.PYTHON,
            max_rounds=max_rounds,
            hunter_concurrency=1,
        )
        return await planner.run(goal)

    async def _run_structured(self, goal: str) -> ProposalResult:
        """Structured single-pass: decompose → match → ground."""
        return await self.propose(problem=goal)

    # -- inspection ---------------------------------------------------------

    def inspect_cdg(self, cdg: CDGExport) -> CDGInspection:
        """Inspect a CDG: stage count, grounding, topology."""
        nodes = cdg.nodes if hasattr(cdg, "nodes") else []
        edges = cdg.edges if hasattr(cdg, "edges") else []

        matches = self._match_cdg_stages(cdg)
        grounding = self._build_grounding_report(matches)

        concept_counts: dict[str, int] = {}
        max_depth = 0
        for node in nodes:
            ct = getattr(node, "concept_type", None)
            if ct:
                key = ct.value if hasattr(ct, "value") else str(ct)
                concept_counts[key] = concept_counts.get(key, 0) + 1
            depth = getattr(node, "depth", 0)
            if depth > max_depth:
                max_depth = depth

        return CDGInspection(
            total_stages=len(nodes),
            total_edges=len(edges),
            grounding=grounding,
            concept_types=concept_counts,
            max_depth=max_depth,
        )

    # -- fast prototyping loop ----------------------------------------------

    def proto(
        self,
        cdg: CDGExport,
        *,
        goal: str = "",
        harness: Any = None,
        manifest_sqlite_path: str | Path | None = None,
        hpo_trials_per_rung: int = 8,
        max_sibling_evals: int = 5,
        subdecompose_fn: Any = None,
        micro_atom_synthesizer: Any = None,
    ) -> Any:
        """Create an in-process ``PrototypingSession`` for iterative CDG refinement."""
        from sciona.principal.expansion import ExpansionEngine
        from sciona.principal.expansion_rules import default_rule_sets
        from sciona.prototyping.session import PrototypingSession

        m_path: Path | None = Path(manifest_sqlite_path) if manifest_sqlite_path else None
        if m_path is None:
            default_manifest = Path.home() / ".sciona" / "manifest.sqlite"
            if default_manifest.is_file():
                m_path = default_manifest

        exp_engine = ExpansionEngine(default_rule_sets())
        return PrototypingSession(
            cdg,
            goal=goal,
            harness=harness,
            catalog=self._catalog,
            manifest_sqlite_path=m_path,
            expansion_engine=exp_engine,
            subdecompose_fn=subdecompose_fn,
            micro_atom_synthesizer=micro_atom_synthesizer,
            hpo_trials_per_rung=hpo_trials_per_rung,
            max_sibling_evals=max_sibling_evals,
        )

    # -- source management --------------------------------------------------

    def list_sources(self) -> list[dict[str, Any]]:
        """List resolved atom sources from config."""
        if self._config is None:
            return []
        try:
            from sciona.commands.sources_cmds import _resolve_sources

            return _resolve_sources(self._config)
        except Exception:  # noqa: BLE001
            return []

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_grounding_report(
        matches: list[StageMatchResult],
    ) -> GroundingReport:
        bound_active = 0
        orchestration = 0
        trivial_inline = 0
        external_knowledge = 0
        external_tool = 0
        unbound = 0

        for m in matches:
            ac = m.action_class
            if ac == "orchestration":
                orchestration += 1
            elif ac == "trivial_inline":
                trivial_inline += 1
            elif ac == "external_knowledge":
                external_knowledge += 1
            elif ac == "external_tool":
                external_tool += 1
            elif m.matched_atom:
                bound_active += 1
            else:
                unbound += 1

        return GroundingReport(
            total_stages=len(matches),
            bound_active=bound_active,
            orchestration=orchestration,
            trivial_inline=trivial_inline,
            external_knowledge=external_knowledge,
            external_tool=external_tool,
            unbound=unbound,
        )

    def __repr__(self) -> str:
        parts = [f"Sciona(atoms={self.atom_count}"]
        if self._llm is not None:
            parts.append("llm=ready")
        if self._proof_env is not None:
            parts.append("prover=ready")
        parts.append(")")
        return ", ".join(parts)
