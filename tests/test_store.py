"""Unit tests for the synchronous memory store."""

from __future__ import annotations

from datetime import timedelta

import pytest

from amnesia.embeddings import pack_vector
from amnesia.models import IDENTITY_KINDS
from amnesia.utils import parse_iso, to_iso, utc_now


# ---------------------------------------------------------------------------
# Sessions & messages
# ---------------------------------------------------------------------------


def test_session_lifecycle_and_messages(store):
    session = store.create_session(model="dolphin3:latest", client="cli", title="Test session")
    assert session.id
    assert store.get_session(session.id).model == "dolphin3:latest"

    first = store.add_message(session.id, "user", "Hello memory")
    second = store.add_message(session.id, "assistant", "Hello human")
    assert first.id < second.id

    messages = store.session_messages(session.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert store.count_messages(session.id) == 2

    recent = store.recent_messages(session.id, limit=2)
    assert [message.role for message in recent] == ["user", "assistant"]

    refreshed = store.get_session(session.id)
    assert refreshed.message_count == 2

    store.end_session(session.id, summary="A short chat")
    ended = store.get_session(session.id)
    assert ended.ended_at is not None
    assert ended.summary == "A short chat"

    assert [item.id for item in store.list_sessions(limit=5)] == [session.id]
    assert store.latest_session() == ended


def test_session_filters_and_delete(tmp_path, store):
    old = store.create_session(
        model="dolphin3:latest",
        client="cli",
        started_at=to_iso(utc_now() - timedelta(days=3)),
    )
    new = store.create_session(model="other:latest", client="openai")
    assert store.latest_session().id == new.id
    assert [s.id for s in store.list_sessions(model="dolphin3:latest")] == [old.id]
    assert [s.id for s in store.list_sessions(client="openai")] == [new.id]

    assert store.delete_session(new.id) is True
    assert store.get_session(new.id) is None
    assert store.delete_session("does-not-exist") is False
    assert store.count_messages() == 0


def test_title_and_summary_updates(store):
    session = store.create_session()
    store.set_session_title(session.id, "New title")
    store.set_session_summary(session.id, "Summary text")
    updated = store.get_session(session.id)
    assert updated.title == "New title"
    assert updated.summary == "Summary text"


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


def test_add_and_get_memory(store):
    memory = store.add_memory(
        kind="fact",
        content="  Lives in Shenzhen  ",
        key="location",
        importance=0.8,
        confidence=0.9,
        tags=["profile", "geo"],
        embedding=[1.0, 0.0, 0.0],
        embedding_model="test:3",
    )
    assert memory.id > 0
    assert memory.content == "Lives in Shenzhen"
    assert memory.tags == ["profile", "geo"]
    assert memory.kind == "fact"
    assert memory.active is True

    fetched = store.get_memory(memory.id)
    assert fetched is not None
    assert fetched.key == "location"
    assert fetched.importance == pytest.approx(0.8)
    assert store.count_memories() == 1


def test_unknown_kind_is_coerced_to_note(store):
    memory = store.add_memory(kind="banana", content="Odd kind")
    assert memory.kind == "note"


def test_upsert_memory_by_key_merges(store):
    first = store.upsert_memory_by_key(
        kind="fact",
        key="editor",
        content="Uses VS Code",
        importance=0.5,
        tags=["tools"],
        embedding=[1.0, 0.0],
        embedding_model="test:2",
    )
    second = store.upsert_memory_by_key(
        kind="fact",
        key="editor",
        content="Uses VS Code with Vim keybindings",
        importance=0.9,
        tags=["tools", "vim"],
        embedding=[0.9, 0.1],
        embedding_model="test:2",
    )
    assert second.id == first.id
    assert second.content == "Uses VS Code with Vim keybindings"
    assert second.importance == pytest.approx(0.9)
    assert second.tags == ["tools", "vim"]
    assert store.count_memories() == 1


def test_find_memory_by_key_and_deactivate(store):
    memory = store.add_memory(kind="identity", content="Name: Mohamed", key="name")
    found = store.find_memory_by_key("identity", "name")
    assert found.id == memory.id

    assert store.deactivate_memory(memory.id) is True
    assert store.find_memory_by_key("identity", "name") is None
    assert store.get_memory(memory.id).active is False
    assert store.count_memories() == 0

    assert store.delete_memory(memory.id) is True
    assert store.get_memory(memory.id) is None
    assert store.delete_memory(99999) is False


def test_identity_list_orders_identity_then_importance(store):
    store.add_memory(kind="preference", content="Likes dark mode", importance=0.9)
    store.add_memory(kind="fact", content="Studies CS", importance=0.95)
    store.add_memory(kind="identity", content="Is Mohamed", importance=0.4)
    kinds = [memory.kind for memory in store.list_identity_memories()]
    assert kinds == ["identity", "fact", "preference"]
    assert set(kinds).issubset(IDENTITY_KINDS)


def test_update_memory_validates_fields_and_packs_embedding(store):
    memory = store.add_memory(kind="fact", content="Original")
    assert store.update_memory(memory.id, content="Changed", importance=2.0, unknown_field=1) is True
    updated = store.get_memory(memory.id)
    assert updated.content == "Changed"
    assert updated.importance == pytest.approx(1.0)  # clamped
    assert store.update_memory(memory.id) is False

    store.set_memory_embedding(memory.id, [0.0, 1.0], embedding_model="test:2")
    stored = store.db.query_one("SELECT embedding, embedding_dim FROM memories WHERE id = ?", (memory.id,))
    assert stored["embedding_dim"] == 2
    assert len(stored["embedding"]) == 8  # two float32s


def test_touch_memories_increments_access(store):
    first = store.add_memory(kind="fact", content="One")
    second = store.add_memory(kind="fact", content="Two")
    store.touch_memories([first.id, second.id, 99999])
    assert store.get_memory(first.id).access_count == 1
    assert store.get_memory(first.id).last_accessed_at is not None
    assert store.get_memory(second.id).access_count == 1


# ---------------------------------------------------------------------------
# Keyword search / FTS sync
# ---------------------------------------------------------------------------


def test_keyword_search_memories(store):
    store.add_memory(kind="fact", content="Building a quantum trading algorithm")
    store.add_memory(kind="fact", content="Prefers dark roast coffee")
    store.add_memory(kind="preference", content="Likes concise answers", tags=["style"])

    hits = store.search_memories_keyword("trading algorithm")
    assert hits
    assert "quantum" in hits[0][0].content
    assert hits[0][1] > 0

    filtered = store.search_memories_keyword("answers", kinds=["preference"])
    assert filtered and filtered[0][0].kind == "preference"
    assert store.search_memories_keyword("answers", kinds=["fact"]) == []


def test_fts_stays_in_sync_on_update_and_delete(store):
    memory = store.add_memory(kind="fact", content="Uses Postgres database")
    assert store.search_memories_keyword("Postgres")

    store.update_memory(memory.id, content="Uses SQLite database")
    assert store.search_memories_keyword("Postgres") == []
    assert store.search_memories_keyword("SQLite")

    store.delete_memory(memory.id)
    assert store.search_memories_keyword("SQLite") == []


def test_keyword_search_events_and_messages(store):
    session = store.create_session()
    store.add_message(session.id, "user", "How do I configure the vector index?")
    store.add_event(title="Set up FTS5 index", details="Configured triggers", project="amnesia")

    event_hits = store.search_events_keyword("FTS5 index")
    assert event_hits
    assert event_hits[0][0].project == "amnesia"

    message_hits = store.search_messages_keyword("vector index")
    assert message_hits
    assert message_hits[0][0].role == "user"


def test_keyword_search_handles_punctuation_only_query(store):
    store.add_memory(kind="note", content="Hello world")
    assert store.search_memories_keyword("???!!!") == []


# ---------------------------------------------------------------------------
# Vector search / dedupe primitives
# ---------------------------------------------------------------------------


def test_vector_search_orders_by_similarity(store):
    store.add_memory(
        kind="fact", content="Close vector", embedding=[1.0, 0.0, 0.0], embedding_model="test:3"
    )
    store.add_memory(
        kind="fact", content="Far vector", embedding=[0.0, 1.0, 0.0], embedding_model="test:3"
    )
    hits = store.search_memories_vector([1.0, 0.0, 0.0], limit=5, embedding_model="test:3")
    assert [memory.content for memory, _ in hits] == ["Close vector", "Far vector"]
    assert hits[0][1] == pytest.approx(1.0)

    # Vectors from another embedding model are not comparable and are ignored.
    assert store.search_memories_vector([1.0, 0.0, 0.0], embedding_model="other:3") == []


def test_find_similar_memories_threshold(store):
    target = store.add_memory(
        kind="fact", content="Lives in Shenzhen", embedding=[1.0, 0.0], embedding_model="test:2"
    )
    store.add_memory(
        kind="fact", content="Lives in Shenzhen, China", embedding=[1.0, 0.0], embedding_model="test:2"
    )
    store.add_memory(kind="fact", content="Unrelated", embedding=[0.0, 1.0], embedding_model="test:2")

    similar = store.find_similar_memories(
        [1.0, 0.0], threshold=0.95, kinds=["fact"], embedding_model="test:2"
    )
    assert len(similar) == 2
    assert target.id in {memory.id for memory, _ in similar}

    excluded = store.find_similar_memories(
        [1.0, 0.0], threshold=0.95, exclude_ids=[target.id], embedding_model="test:2"
    )
    assert len(excluded) == 1


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_filters_and_aggregations(store):
    now = utc_now()
    store.add_event(title="Built proxy", kind="project", project="amnesia", occurred_at=to_iso(now))
    store.add_event(
        title="Decided on SQLite",
        kind="decision",
        project="amnesia",
        occurred_at=to_iso(now - timedelta(days=3)),
    )
    store.add_event(
        title="Old idea",
        kind="note",
        occurred_at=to_iso(now - timedelta(days=40)),
    )

    assert len(store.list_events(days=7)) == 2
    assert len(store.list_events(days=60)) == 3
    assert [event.title for event in store.list_events(project="amnesia")][:1] == ["Built proxy"]
    assert [event.title for event in store.list_events(kind="decision")] == ["Decided on SQLite"]
    assert store.count_events() == 3

    projects = dict(store.event_projects())
    assert projects == {"amnesia": 2}

    by_day = store.events_by_day(days=7)
    assert sum(count for _day, count in by_day) == 2
    assert store.purge_old_events(days=30) == 1
    assert store.count_events() == 2


def test_unknown_event_kind_coerced(store):
    event = store.add_event(title="Something", kind="nonsense")
    assert event.kind == "other"


# ---------------------------------------------------------------------------
# Extraction bookkeeping & embedding backlog
# ---------------------------------------------------------------------------


def test_extraction_bookkeeping(store):
    session = store.create_session()
    store.add_message(session.id, "user", "one")
    store.add_message(session.id, "assistant", "two")
    store.add_message(session.id, "user", "three")

    assert store.count_turns_since_extraction(session.id) == 2
    store.record_extraction(kind="facts", session_id=session.id, items=3)
    assert store.count_turns_since_extraction(session.id) == 0
    assert store.last_extraction(session.id)["items"] == 3


def test_embedding_backlog(store):
    memory = store.add_memory(kind="fact", content="Needs a vector")
    event = store.add_event(title="Needs a vector too")
    assert [item.id for item in store.memories_missing_embeddings()] == [memory.id]
    assert [item.id for item in store.events_missing_embeddings()] == [event.id]

    store.set_memory_embedding(memory.id, [1.0, 0.0], embedding_model="test:2")
    store.update_event(event.id, embedding=[0.0, 1.0], embedding_model="test:2")
    assert store.memories_missing_embeddings() == []
    assert store.events_missing_embeddings() == []


# ---------------------------------------------------------------------------
# Stats, export/import, maintenance
# ---------------------------------------------------------------------------


def test_stats_and_meta(store):
    store.add_memory(kind="identity", content="Name: A")
    store.add_memory(kind="summary", content="Did some work")
    store.add_event(title="Worked")
    session = store.create_session()
    store.add_message(session.id, "user", "hi")
    assert store.db.get_meta("custom") is None
    store.db.set_meta("custom", "value")
    assert store.db.get_meta("custom") == "value"

    stats = store.stats(embed_model="hash:512", chat_model="dolphin3:latest")
    assert stats.memories == 2
    assert stats.identity_memories == 1
    assert stats.episode_memories == 1
    assert stats.events == 1
    assert stats.sessions == 1
    assert stats.messages == 1
    assert stats.db_size_bytes > 0
    assert stats.last_activity is not None
    assert stats.to_dict()["db_size_human"]


def test_export_import_roundtrip(store, tmp_path):
    store.add_memory(kind="identity", content="Name: Mohamed", key="name", importance=0.9)
    session = store.create_session(model="dolphin3:latest", client="cli", title="Chat")
    store.add_message(session.id, "user", "Remember this")
    store.add_event(title="Wrote tests", kind="task", project="amnesia")

    payload = store.export_data()
    assert payload["format"] == "amnesia-export"
    assert len(payload["memories"]) == 1

    store.purge_all()
    assert store.count_memories() == 0

    counts = store.import_data(payload)
    assert counts == {"sessions": 1, "messages": 1, "memories": 1, "events": 1}
    assert store.find_memory_by_key("identity", "name").content == "Name: Mohamed"
    assert store.list_sessions()[0].title == "Chat"
    assert store.count_messages() == 1


def test_import_merge_updates_existing_key(store):
    store.add_memory(kind="fact", content="Old content", key="editor")
    payload = {
        "format": "amnesia-export",
        "version": 1,
        "sessions": [],
        "messages": [],
        "memories": [
            {"kind": "fact", "key": "editor", "content": "New content", "importance": 0.9}
        ],
        "events": [],
    }
    counts = store.import_data(payload, merge=True)
    assert counts["memories"] == 1
    assert store.count_memories() == 1
    assert store.find_memory_by_key("fact", "editor").content == "New content"


def test_backup_creates_readable_copy(store, tmp_path):
    store.add_memory(kind="fact", content="Back me up")
    target = store.backup_to(tmp_path / "backup.db")
    assert target.exists()
    from amnesia.store import MemoryStore

    copy = MemoryStore(target).initialize()
    try:
        assert copy.count_memories() == 1
    finally:
        copy.close()


def test_fresh_database_has_schema_version_and_integrity(store):
    assert store.db.get_meta("schema_version") == "1"
    assert store.db.integrity_check() == "ok"
    assert store.db.fts_enabled is True
    assert store.db.get_meta("created_at")
