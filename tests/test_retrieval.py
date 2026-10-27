"""Unit tests for hybrid retrieval and prompt-context construction."""

from __future__ import annotations

from datetime import timedelta

import pytest

from amnesia.embeddings import HashEmbedder
from amnesia.models import Event, Memory
from amnesia.retrieval import (
    ContextBuilder,
    item_rank,
    recall,
    reciprocal_rank_fusion,
    recency_score,
    to_recall_hits,
)
from amnesia.utils import to_iso, utc_now


def _make_memory(memory_id: int, content: str, *, kind: str = "fact") -> Memory:
    now = utc_now().isoformat()
    return Memory(
        id=memory_id,
        kind=kind,
        content=content,
        created_at=now,
        updated_at=now,
    )


def _make_event(event_id: int, title: str, *, days_ago: int = 0) -> Event:
    now = utc_now()
    return Event(
        id=event_id,
        kind="project",
        title=title,
        occurred_at=to_iso(now - timedelta(days=days_ago)),
        created_at=to_iso(now - timedelta(days=days_ago)),
    )


# ---------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_prefers_consensus_items():
    shared = _make_memory(1, "appears in both lists")
    only_first = _make_memory(2, "first only")
    only_second = _make_memory(3, "second only")
    fused = reciprocal_rank_fusion(
        [
            [(only_first, 0.9), (shared, 0.8)],
            [(shared, 0.95), (only_second, 0.7)],
        ],
        weights=[1.0, 1.0],
    )
    assert fused[0][0].id == shared.id
    # First list's exclusive item outranks second list's exclusive item by rank.
    assert [memory.id for memory, _ in fused] == [1, 2, 3]


def test_reciprocal_rank_fusion_weights_change_order():
    first = _make_memory(1, "first")
    second = _make_memory(2, "second")
    fused_flat = reciprocal_rank_fusion([[(first, 1.0)], [(second, 1.0)]])
    assert fused_flat[0][0].id == 1
    fused_weighted = reciprocal_rank_fusion(
        [[(first, 1.0)], [(second, 1.0)]], weights=[0.1, 5.0]
    )
    assert fused_weighted[0][0].id == 2


def test_recency_score_decays():
    now_score = recency_score(to_iso(utc_now()))
    old_score = recency_score(to_iso(utc_now() - timedelta(days=90)))
    assert 0.9 <= now_score <= 1.0
    assert old_score < 0.3
    assert recency_score(None) == 0.0


def test_item_rank_blends_importance_and_recency():
    fresh = _make_memory(1, "fresh", kind="fact")
    fresh.importance = 0.9
    stale = _make_memory(2, "stale", kind="fact")
    stale.importance = 0.9
    stale.updated_at = to_iso(utc_now() - timedelta(days=365))
    assert item_rank(fresh, 0.5, kind="memory") > item_rank(stale, 0.5, kind="memory")


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def test_context_builder_always_includes_profile(store, settings):
    store.add_memory(kind="identity", content="Name: Mohamed El Alami", key="name")
    store.add_memory(kind="fact", content="Studies computer science", key="education")
    builder = ContextBuilder(store, settings)

    context = builder.build("hello there")
    assert not context.is_empty
    assert "Who the user is" in context.text
    assert "Mohamed El Alami" in context.text
    assert context.text.startswith("<amnesia_memory>")
    assert context.text.rstrip().endswith("</amnesia_memory>")
    assert context.tokens_estimate > 0
    assert len(context.profile) == 2


def test_context_includes_relevant_events_and_episodes(store, settings):
    session = store.create_session()
    store.add_event(
        title="Built the Amnesia memory proxy with FastAPI",
        kind="project",
        project="amnesia",
    )
    store.add_memory(
        kind="summary",
        content="Session summary: designed hybrid retrieval with FTS5 and embeddings",
        source="extraction",
    )
    builder = ContextBuilder(store, settings)

    context = builder.build("How did we design retrieval for the memory proxy?")
    assert any("FastAPI" in event.title for event in context.events)
    assert any("hybrid retrieval" in memory.content for memory in context.episodes)
    assert "Recent relevant activity" in context.text
    assert "Earlier relevant conversations" in context.text


def test_context_session_summary_section(store, settings):
    builder = ContextBuilder(store, settings)
    context = builder.build("anything", session_summary="User is building a memory system")
    assert "This session so far" in context.text
    assert "building a memory system" in context.text


def test_context_budget_keeps_output_bounded(store, settings):
    for index in range(60):
        store.add_memory(
            kind="fact",
            content=f"Long fact number {index}: " + ("detail " * 40),
            importance=0.5,
        )
    builder = ContextBuilder(store, settings)
    context = builder.build("fact", budget=300, profile_budget=80)
    assert context.tokens_estimate <= 300 * 1.3


def test_empty_store_context_is_empty_but_valid(store, settings):
    builder = ContextBuilder(store, settings)
    context = builder.build("nothing here yet")
    assert context.is_empty
    assert context.items_count == 0
    assert "<amnesia_memory>" in context.text


def test_context_deduplicates_profile_from_facts(store, settings):
    store.add_memory(kind="identity", content="Name: Mohamed", key="name")
    store.add_memory(kind="preference", content="Prefers Python examples", key="style")
    builder = ContextBuilder(store, settings)
    context = builder.build("What is my name and preferred style?")
    profile_ids = {memory.id for memory in context.profile}
    fact_ids = {memory.id for memory in context.facts}
    assert not profile_ids & fact_ids


# ---------------------------------------------------------------------------
# Unified recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_fuses_keyword_and_vector(store):
    embedder = HashEmbedder()
    vector_a = await embedder.embed_one("quantum trading algorithm design")
    vector_b = await embedder.embed_one("coffee preferences")
    store.add_memory(
        kind="fact",
        content="Building a quantum trading algorithm",
        embedding=vector_a,
        embedding_model=embedder.name,
    )
    store.add_memory(
        kind="preference",
        content="Prefers dark roast coffee",
        embedding=vector_b,
        embedding_model=embedder.name,
    )
    store.add_event(title="Started the quantum trading project", kind="project")

    query = "quantum trading algorithm"
    query_vector = await embedder.embed_one(query)
    hits = recall(
        store,
        query,
        query_vector,
        limit=5,
        include_events=True,
        embedding_model=embedder.name,
    )
    assert hits
    assert hits[0].score >= hits[-1].score
    contents = " ".join(hit.content for hit in hits)
    assert "quantum" in contents
    assert any(hit.kind == "event" for hit in hits)


def test_to_recall_hits_sorts_and_truncates():
    memories = [
        (_make_memory(1, "low"), 0.2),
        (_make_memory(2, "high"), 0.9),
    ]
    events = [(_make_event(10, "activity"), 0.5)]
    hits = to_recall_hits(memories=memories, events=events, limit=2)
    assert len(hits) == 2
    assert hits[0].content == "high"
    assert hits[0].kind == "memory"
    assert hits[1].kind == "event"


def test_recall_include_messages(store):
    session = store.create_session()
    store.add_message(session.id, "user", "the vector index needs rebuilding")
    hits = recall(store, "vector index", None, limit=5, include_messages=True)
    assert any(hit.kind == "message" for hit in hits)
