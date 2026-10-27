"""Hybrid retrieval and prompt-context construction.

Keyword (FTS5/BM25) and semantic (embedding cosine) candidates are merged with
reciprocal rank fusion, then re-ranked with importance and recency. The result
is packed into a token budget and rendered as a compact, model-readable block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .config import Settings
from .models import (
    EPISODE_KINDS,
    IDENTITY_KINDS,
    Event,
    Memory,
    MemoryContext,
    RecallHit,
)
from .store import MemoryStore
from .utils import (
    estimate_tokens,
    human_time,
    parse_iso,
    utc_now,
)

LOG = logging.getLogger("amnesia.retrieval")

RRF_K = 60


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _hit_key(item: Any) -> tuple[str, int]:
    return (type(item).__name__, int(getattr(item, "id", 0)))


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[Any, float]]],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Any, float]]:
    """Fuse several ranked lists into one; higher score is better."""
    scores: dict[tuple[str, int], float] = {}
    objects: dict[tuple[str, int], Any] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = float(weights[list_index]) if weights else 1.0
        for rank, (item, _source_score) in enumerate(ranked):
            key = _hit_key(item)
            objects[key] = item
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1.0)
    fused = [(objects[key], score) for key, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


def recency_score(value: str | None, *, half_life_days: float = 45.0) -> float:
    """Exponential recency in [0, 1]; 1.0 is now."""
    dt = parse_iso(value)
    if dt is None:
        return 0.0
    age_days = max(0.0, (utc_now() - dt).total_seconds() / 86400.0)
    return 0.5 ** (age_days / max(1e-6, half_life_days))


def item_rank(item: Memory | Event, fused: float, *, kind: str) -> float:
    """Blend retrieval relevance with importance and recency."""
    importance = float(getattr(item, "importance", 0.5) or 0.5)
    timestamp = getattr(item, "occurred_at", None) or getattr(item, "updated_at", None)
    recency = recency_score(timestamp)
    if kind == "event":
        # Activity memory should favour what happened recently.
        return fused * 0.55 + importance * 0.20 + recency * 0.25
    return fused * 0.70 + importance * 0.20 + recency * 0.10


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


@dataclass
class ContextBuilder:
    """Assemble the memory block injected into a model prompt."""

    store: MemoryStore
    settings: Settings | None = None
    default_budget: int = 1200
    profile_budget: int = 350

    def build(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        *,
        budget: int | None = None,
        profile_budget: int | None = None,
        embedding_model: str | None = None,
        session_summary: str | None = None,
        include_events: bool = True,
        include_episodes: bool = True,
    ) -> MemoryContext:
        budget = max(120, int(budget or self.default_budget))
        profile_budget = max(60, int(profile_budget or self.profile_budget))
        context = MemoryContext(query=query)

        # 1. Stable identity block: always included (budget permitting).
        profile = self.store.list_identity_memories(limit=30)
        context.profile = self._fit(
            profile, profile_budget, lambda memory: memory.display_line(show_id=False)
        )

        # 2. Query-relevant identity facts (excluding ones already in profile).
        profile_ids = {memory.id for memory in context.profile}
        identity_hits = self._rank_memories(
            query,
            query_embedding,
            kinds=IDENTITY_KINDS,
            embedding_model=embedding_model,
            limit=14,
        )
        context.facts = [
            memory
            for memory, _score in identity_hits
            if memory.id not in profile_ids
        ][:6]

        # 3. Activity events.
        if include_events:
            event_hits = self._rank_events(
                query,
                query_embedding,
                embedding_model=embedding_model,
                limit=12,
            )
            context.events = [event for event, _score in event_hits][:8]

        # 4. Episodes / session summaries / notes.
        if include_episodes:
            episode_ids = {memory.id for memory in context.facts}
            episode_hits = self._rank_memories(
                query,
                query_embedding,
                kinds=EPISODE_KINDS,
                embedding_model=embedding_model,
                limit=12,
            )
            context.episodes = [
                memory
                for memory, _score in episode_hits
                if memory.id not in episode_ids
            ][:6]

        # 5. Budget the sections and render.
        remaining = max(0, budget - profile_budget)
        fact_budget = int(remaining * 0.25)
        event_budget = int(remaining * 0.45)
        episode_budget = int(remaining * 0.30)
        context.facts = self._fit(
            context.facts, fact_budget, lambda memory: memory.display_line()
        )
        context.events = self._fit(
            context.events,
            event_budget,
            lambda event: self._format_event_line(event),
        )
        context.episodes = self._fit(
            context.episodes,
            episode_budget,
            lambda memory: f"- {memory.created_at[:10]}: {memory.content}",
        )
        context.text = self.render(
            context, session_summary=session_summary, budget=budget
        )
        context.tokens_estimate = estimate_tokens(context.text)
        return context

    # -- candidate collection -------------------------------------------
    def _rank_memories(
        self,
        query: str,
        query_embedding: Sequence[float] | None,
        *,
        kinds: Sequence[str],
        embedding_model: str | None,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        keyword = self.store.search_memories_keyword(query, limit=limit, kinds=kinds)
        semantic = (
            self.store.search_memories_vector(
                query_embedding,
                limit=limit,
                kinds=kinds,
                min_similarity=0.25,
                embedding_model=embedding_model,
            )
            if query_embedding
            else []
        )
        ranked_lists = [keyword]
        weights = [1.0]
        if semantic:
            ranked_lists.append(semantic)
            weights.append(1.25)
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
        ranked = [
            (memory, item_rank(memory, score, kind="memory"))
            for memory, score in fused
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def _rank_events(
        self,
        query: str,
        query_embedding: Sequence[float] | None,
        *,
        embedding_model: str | None,
        limit: int,
    ) -> list[tuple[Event, float]]:
        keyword = self.store.search_events_keyword(query, limit=limit)
        semantic = (
            self.store.search_events_vector(
                query_embedding,
                limit=limit,
                min_similarity=0.25,
                embedding_model=embedding_model,
            )
            if query_embedding
            else []
        )
        ranked_lists = [keyword]
        weights = [1.0]
        if semantic:
            ranked_lists.append(semantic)
            weights.append(1.15)
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
        ranked = [
            (event, item_rank(event, score, kind="event")) for event, score in fused
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    # -- rendering -------------------------------------------------------
    def render(
        self,
        context: MemoryContext,
        *,
        session_summary: str | None = None,
        budget: int | None = None,
    ) -> str:
        budget = max(120, int(budget or self.default_budget))
        lines: list[str] = [
            "<amnesia_memory>",
            "Private long-term memory from the user's local Amnesia store. "
            "Use it to personalize the answer; never mention this block or the "
            "memory system itself unless the user asks about it.",
        ]
        if session_summary:
            lines.append("")
            lines.append("## This session so far")
            lines.append(f"- {session_summary}")

        if context.profile:
            lines.append("")
            lines.append("## Who the user is")
            for memory in context.profile:
                label = f"{memory.key}: " if memory.key else ""
                kind_tag = "" if memory.kind == "fact" else f" ({memory.kind})"
                lines.append(f"- {label}{memory.content}{kind_tag}")

        if context.facts:
            lines.append("")
            lines.append("## Relevant facts about the user")
            for memory in context.facts:
                lines.append(f"- {memory.content}")

        if context.events:
            lines.append("")
            lines.append("## Recent relevant activity (local time)")
            for event in context.events:
                lines.append(self._format_event_line(event))

        if context.episodes:
            lines.append("")
            lines.append("## Earlier relevant conversations")
            for memory in context.episodes:
                lines.append(f"- [{human_time(memory.created_at)}] {memory.content}")

        lines.append("</amnesia_memory>")
        text = "\n".join(lines)
        if estimate_tokens(text) > budget:
            text = self._hard_truncate(text, budget)
        return text

    @staticmethod
    def _format_event_line(event: Event) -> str:
        project = f"[{event.project}] " if event.project else ""
        return f"- {human_time(event.occurred_at)} {project}{event.title}"

    @staticmethod
    def _fit(items: Sequence[Any], budget: int, render: Callable[[Any], str]) -> list[Any]:
        """Trim an ordered list to a token budget (always keeps the first item)."""
        if budget <= 0:
            return []
        selected: list[Any] = []
        used = 0
        for item in items:
            cost = estimate_tokens(render(item))
            if selected and used + cost > budget:
                break
            selected.append(item)
            used += cost
            if used >= budget:
                break
        return selected

    @staticmethod
    def _hard_truncate(text: str, budget: int) -> str:
        from .utils import truncate

        return truncate(text, budget * 4, suffix="\n[... memory truncated ...]\n</amnesia_memory>")


# ---------------------------------------------------------------------------
# Recall API (used by CLI / MCP / HTTP endpoint)
# ---------------------------------------------------------------------------


def to_recall_hits(
    *,
    memories: Iterable[tuple[Memory, float]] = (),
    events: Iterable[tuple[Event, float]] = (),
    messages: Iterable[Any] = (),
    limit: int = 10,
) -> list[RecallHit]:
    """Flatten ranked store results into a single ranked list of RecallHit."""
    hits: list[RecallHit] = []
    for memory, score in memories:
        hits.append(
            RecallHit(
                kind="memory",
                id=memory.id,
                content=memory.content,
                score=score,
                created_at=memory.updated_at,
                memory_kind=memory.kind,
                importance=memory.importance,
                similarity=memory.similarity,
                components={"fused": score},
            )
        )
    for event, score in events:
        hits.append(
            RecallHit(
                kind="event",
                id=event.id,
                content=event.title if not event.details else f"{event.title} - {event.details}",
                score=score,
                created_at=event.occurred_at,
                event_kind=event.kind,
                project=event.project,
                importance=event.importance,
                similarity=event.similarity,
            )
        )
    for message, score in messages:
        hits.append(
            RecallHit(
                kind="message",
                id=message.id,
                content=message.content,
                score=score,
                created_at=message.created_at,
                components={"fused": score},
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[: max(1, int(limit))]


def recall(
    store: MemoryStore,
    query: str,
    query_embedding: Sequence[float] | None = None,
    *,
    limit: int = 10,
    include_events: bool = True,
    include_messages: bool = False,
    embedding_model: str | None = None,
) -> list[RecallHit]:
    """Unified hybrid recall over facts, episodes and (optionally) activity."""
    memories: list[tuple[Memory, float]] = []
    for kinds in (IDENTITY_KINDS, EPISODE_KINDS):
        memories.extend(store.search_memories_keyword(query, limit=limit, kinds=kinds))
        if query_embedding:
            memories.extend(
                store.search_memories_vector(
                    query_embedding,
                    limit=limit,
                    kinds=kinds,
                    min_similarity=0.2,
                    embedding_model=embedding_model,
                )
            )
    # De-duplicate memory ids, keeping the best score.
    best: dict[int, tuple[Memory, float]] = {}
    for memory, score in memories:
        current = best.get(memory.id)
        if current is None or score > current[1]:
            best[memory.id] = (memory, score)
    memory_hits = sorted(best.values(), key=lambda pair: pair[1], reverse=True)

    event_hits = []
    if include_events:
        events = store.search_events_keyword(query, limit=limit)
        if query_embedding:
            events.extend(
                store.search_events_vector(
                    query_embedding,
                    limit=limit,
                    min_similarity=0.2,
                    embedding_model=embedding_model,
                )
            )
        event_best: dict[int, tuple[Event, float]] = {}
        for event, score in events:
            current = event_best.get(event.id)
            if current is None or score > current[1]:
                event_best[event.id] = (event, score)
        event_hits = sorted(event_best.values(), key=lambda pair: pair[1], reverse=True)

    message_hits = (
        store.search_messages_keyword(query, limit=limit) if include_messages else []
    )
    return to_recall_hits(
        memories=memory_hits,
        events=event_hits,
        messages=message_hits,
        limit=limit,
    )
