"""Deterministic lightweight replacement for the hunter score prompt."""

from __future__ import annotations

import json
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CANDIDATE_RE = re.compile(r"^\[(\d+)\]\s+(.+?)\s+:\s+(.+)$")


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace(".", " ")
    return set(_TOKEN_RE.findall(normalized))


def _extract_query(user: str) -> tuple[str, str]:
    statement = ""
    description = ""
    for line in user.splitlines():
        stripped = line.strip()
        if stripped.startswith("Statement:"):
            statement = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:"):
            description = stripped.split(":", 1)[1].strip()
    return statement, description


def _extract_candidates(user: str) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    for line in user.splitlines():
        match = _CANDIDATE_RE.match(line.strip())
        if not match:
            continue
        candidates.append((int(match.group(1)), match.group(2), match.group(3)))
    return candidates


class HeuristicCandidateRanker:
    """Lexical/type-aware ranker with LLM fallback on ambiguity."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "candidate_ranker_v1"

    def __init__(
        self,
        fallback: Any,
        *,
        min_score: float = 1.0,
        min_margin: float = 0.2,
        uncertainty_backend: Any | None = None,
        node_eval: Any | None = None,
    ) -> None:
        self._fallback = fallback
        self._min_score = min_score
        self._min_margin = min_margin
        self._uncertainty_backend = uncertainty_backend
        self._node_eval = node_eval
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        statement, description = _extract_query(user)
        candidates = _extract_candidates(user)
        ranked = self._rank(statement, description, candidates)
        if ranked is None:
            self._last_completion_metadata = {"ranking_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        ordered, best_score, margin = ranked
        self._last_completion_metadata = {
            "ranking_source": "deterministic",
            "ranking_best_score": round(best_score, 3),
            "ranking_margin": round(margin, 3),
        }
        self._last_error_metadata = {}
        return json.dumps(ordered)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)

    def _rank(
        self,
        statement: str,
        description: str,
        candidates: list[tuple[int, str, str]],
    ) -> tuple[list[int], float, float] | None:
        query_text = f"{statement} {description}".strip()
        query_tokens = _tokenize(query_text)
        if not query_tokens or not candidates:
            return None

        uncertainty_priors: dict[str, float] = {}
        if self._uncertainty_backend is not None or self._node_eval is not None:
            from sciona.uncertainty.gap_spec import uncertainty_priors_for_candidates

            uncertainty_priors = uncertainty_priors_for_candidates(
                [name for _i, name, _sig in candidates],
                node_eval=self._node_eval,
                backend=self._uncertainty_backend,
            )

        scored: list[tuple[float, int]] = []
        for idx, name, type_signature in candidates:
            name_tokens = _tokenize(name)
            type_tokens = _tokenize(type_signature.replace("->", " "))
            overlap_name = len(query_tokens & name_tokens)
            overlap_type = len(query_tokens & type_tokens)
            score = 1.0 * overlap_name + 0.35 * overlap_type

            # Prefer direct action/function alignment over generic helpers.
            if "filter" in query_tokens and "filter" in name_tokens:
                score += 1.5
            if "shortest" in query_tokens and "dijkstra" in name_tokens:
                score += 1.5
            if "subsequence" in query_tokens and "subsequence" in name_tokens:
                score += 1.5
            if "positive" in query_tokens and "cholesky" in name_tokens:
                score += 1.5
            if "signal" in description.lower() and "signal" in type_tokens:
                score += 0.5
            if "distances" in query_tokens and "distances" in type_tokens:
                score += 0.5
            if "solution" in query_tokens and "solution" in type_tokens:
                score += 0.5

            if score > 0.0 and name in uncertainty_priors:
                score += 0.6 * uncertainty_priors[name]

            scored.append((score, idx))

        scored.sort(key=lambda row: (-row[0], row[1]))
        best_score = scored[0][0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        if best_score < self._min_score or margin < self._min_margin:
            return None
        return [idx for _score, idx in scored if _score > 0], best_score, margin
