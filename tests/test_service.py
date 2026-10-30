"""Unit tests for the async Amnesia service facade."""

from __future__ import annotations

import json

import pytest

from amnesia.config import load_settings
from amnesia.embeddings import NullEmbedder
from amnesia.service import Amnesia


# ---------------------------------------------------------------------------
# Identity memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_and_profile(service):
    memory = await service.remember(
        "Prefers concise answers with code", kind="preference", importance=0.9
    )
    assert memory.id > 0
    profile = service.profile()
    assert [item.id for item in profile] == [memory.id]
    text = await service.profile_text()
    assert "concise answers" in text


@pytest.mark.asyncio
async def test_remember_empty_rejected(service):
    with pytest.raises(ValueError):
        await service.remember("   ")


@pytest.mark.asyncio
async def test_remember_with_key_upserts(service):
    first = await service.remember("Uses VS Code", kind="fact", key="editor")
    second = await service.remember("Uses VS Code + Vim keys", kind="fact", key="editor")
    assert first.id == second.id
    assert service.store.count_memories() == 1
    assert "Vim" in service.store.get_memory(first.id).content


@pytest.mark.asyncio
async def test_remember_dedupes_identical_content(service):
    first = await service.remember("Lives in Shenzhen")
    second = await service.remember("Lives in Shenzhen")
    assert first.id == second.id
    assert service.store.count_memories() == 1


@pytest.mark.asyncio
async def test_remember_redacts_secrets(service):
    memory = await service.remember("my api_key = sk-supersecretvalue12345 for testing")
    assert "sk-supersecret" not in memory.content
    assert "[REDACTED]" in memory.content


@pytest.mark.asyncio
async def test_forget_by_key_and_hard_delete(service):
    memory = await service.remember("Dark mode", kind="preference", key="theme")
    assert await service.forget(key="theme", kind="preference") is True
    assert service.store.get_memory(memory.id).active is False
    assert await service.forget(memory_id=memory.id, hard=True) is True
    assert service.store.get_memory(memory.id) is None
    assert await service.forget(memory_id=99999) is False


# ---------------------------------------------------------------------------
# Activity memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_event_and_timeline(service):
    event = await service.log_event(
        "Built the memory proxy", kind="project", project="amnesia", importance=0.8
    )
    assert event.id > 0
    assert event.embedding is None  # not returned by row mapper
    stored = service.store.get_event(event.id)
    assert stored.project == "amnesia"
    timeline = service.timeline(days=1)
    assert [item.id for item in timeline] == [event.id]
    assert "memory proxy" in service.timeline_text(days=1)


@pytest.mark.asyncio
async def test_log_event_rejects_empty_title(service):
    with pytest.raises(ValueError):
        await service.log_event("  ")


@pytest.mark.asyncio
async def test_log_event_redacts_details(service):
    event = await service.log_event(
        "Configured API", details="password: hunter2hunter2"
    )
    assert "hunter2" not in (event.details or "")


# ---------------------------------------------------------------------------
# Context building & prompt injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_for_includes_identity(service):
    await service.remember("Name: Mohamed El Alami", kind="identity", key="name")
    await service.log_event(
        "Built Amnesia memory proxy", kind="project", project="amnesia"
    )
    context = await service.context_for("What amnesia memory proxy work have I done?")
    assert "Mohamed El Alami" in context.text
    assert "Built Amnesia memory proxy" in context.text
    assert context.tokens_estimate > 0


@pytest.mark.asyncio
async def test_context_for_empty_when_memory_disabled(make_service):
    disabled = await make_service(memory_enabled=False)
    try:
        await disabled.remember("Name: Mohamed")
        context = await disabled.context_for("who am I?")
        assert context.is_empty
        assert context.text == ""
    finally:
        await disabled.close()


@pytest.mark.asyncio
async def test_context_for_respects_budget(service):
    for index in range(40):
        await service.remember(f"Fact {index}: " + ("word " * 30))
    context = await service.context_for("fact", budget=250)
    assert context.tokens_estimate <= 250 * 1.3


def test_augment_messages_prepends_without_system(service):
    messages = [{"role": "user", "content": "hello"}]
    augmented = service.augment_messages(messages, "<amnesia_memory>x</amnesia_memory>")
    assert augmented[0]["role"] == "system"
    assert "<amnesia_memory>" in augmented[0]["content"]
    assert augmented[1]["content"] == "hello"
    # Original list is not mutated.
    assert messages[0]["role"] == "user"


def test_augment_messages_appends_to_existing_system(service):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    augmented = service.augment_messages(messages, "<amnesia_memory>facts</amnesia_memory>")
    assert augmented[0]["content"].startswith("You are a helpful assistant.")
    assert "<amnesia_memory>" in augmented[0]["content"]
    assert len(augmented) == 2


def test_augment_messages_noop_for_empty_block(service):
    messages = [{"role": "user", "content": "hi"}]
    assert service.augment_messages(messages, "") == messages


# ---------------------------------------------------------------------------
# Sessions & transcripts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_turn_persists_and_titles_session(service):
    session = service.start_session(model="dolphin3:latest", client="cli")
    user, assistant = await service.add_turn(
        session.id, "How do I build a memory system?", "Use SQLite plus embeddings."
    )
    assert user.id < assistant.id
    assert service.store.get_session(session.id).title.startswith("How do I build")
    assert len(service.recent_turns(session.id)) == 2
    assert service.recent_turns(None) == []


@pytest.mark.asyncio
async def test_ensure_session_reuses_recent_and_respects_gap(service):
    first = await service.ensure_session(client="cli", model="dolphin3:latest")
    second = await service.ensure_session(client="cli", model="dolphin3:latest")
    assert first.id == second.id

    forced_new = await service.ensure_session(
        client="cli", model="dolphin3:latest", gap_minutes=0
    )
    assert forced_new.id != first.id

    explicit = await service.ensure_session(
        session_id="custom-session", client="cli", model="dolphin3:latest"
    )
    assert explicit.id == "custom-session"


@pytest.mark.asyncio
async def test_recall_returns_hits(service):
    await service.remember("Building a memory system for local models", kind="fact")
    await service.log_event("Decided to use SQLite", kind="decision")
    hits = await service.recall("memory system for local models", limit=5)
    assert hits
    assert any(hit.kind in {"memory", "event"} for hit in hits)


# ---------------------------------------------------------------------------
# Embeddings / backfill / health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_embeddings(service):
    memory = service.store.add_memory(kind="fact", content="Needs embedding later")
    event = service.store.add_event(title="Needs embedding too")
    counts = await service.backfill_embeddings(limit=10)
    assert counts["memories"] == 1
    assert counts["events"] == 1
    assert service.store.get_memory(memory.id) is not None
    assert service.store.memories_missing_embeddings() == []
    assert service.store.events_missing_embeddings() == []
    assert service.embedding_model_id == "hash:512"


@pytest.mark.asyncio
async def test_service_without_embeddings(make_service):
    svc = await make_service()
    svc._embedder = NullEmbedder()
    try:
        assert svc.embeddings_enabled is False
        assert svc.embedding_model_id is None
        assert await svc.embed("text") == []
        memory = await svc.remember("No vector here")
        assert memory.id > 0
        assert await svc.backfill_embeddings() == {"memories": 0, "events": 0}
        assert svc.stats().memories == 1
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_health_payload(service):
    await service.remember("Name: Tester", kind="identity")
    health = service.health()
    assert health["started"] is True
    assert health["db_integrity"] == "ok"
    assert health["embeddings"]["enabled"] is True
    assert health["ollama"]["available"] is True
    assert health["ollama"]["chat_model"] == "dolphin3:latest"
    assert health["counts"]["memories"] >= 1
    json.dumps(health)  # must be JSON serializable


@pytest.mark.asyncio
async def test_start_reports_unreachable_ollama(settings, store, fake_ollama):
    fake_ollama.fail_connection = True
    svc = Amnesia(settings, store=store, ollama=fake_ollama.ollama_client)
    try:
        await svc.start()
        assert svc.ollama_available is False
        assert svc.warnings
        health = svc.health()
        assert health["ollama"]["available"] is False
        # Memory still works without Ollama.
        memory = await svc.remember("Offline fact")
        assert memory.id > 0
    finally:
        await svc.close()


# ---------------------------------------------------------------------------
# Export / import / stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_import_via_service(service, tmp_path):
    await service.remember("Name: Exported", kind="identity", key="name")
    session = service.start_session(client="cli")
    await service.add_turn(session.id, "hi", "hello")
    path = service.export(tmp_path / "export.json")
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "amnesia-export"

    service.store.purge_all()
    counts = service.import_(path)
    assert counts == {"sessions": 1, "messages": 2, "memories": 1, "events": 0}
    assert service.store.count_memories() == 1


@pytest.mark.asyncio
async def test_stats(service):
    await service.remember("Name: Tester", kind="identity")
    stats = service.stats()
    assert stats.memories == 1
    assert stats.identity_memories == 1
    assert stats.embed_model == "hash:512"
    assert stats.chat_model == "dolphin3:latest"
    assert stats.db_size_bytes > 0

# ---------------------------------------------------------------------------
# Transcript redaction (secrets pasted into chat are never stored raw)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_turn_redacts_secrets_in_transcripts(service):
    session = service.start_session(client="cli")
    await service.add_turn(
        session.id,
        "connect with password=hunter2hunter2 please",
        "used token=ghp_abcdefghijklmnopqrstuvwxyz123456 to authenticate",
    )
    messages = service.store.session_messages(session.id)
    assert len(messages) == 2
    assert "hunter2" not in messages[0].content
    assert "ghp_" not in messages[1].content
    assert "[REDACTED]" in messages[0].content
    assert "[REDACTED]" in messages[1].content
    # Exports and session titles use the redacted transcript too.
    exported = service.store.export_data()
    assert "hunter2" not in str(exported)
    assert "hunter2" not in (service.store.get_session(session.id).title or "")
