"""Tests for the extraction/distillation pipeline (workstream A).

All tests use the shared ``fake_ollama`` fixture: no network, no real Ollama
daemon, no sleeps. Canned model replies exercise the happy path plus the usual
local-model failure modes (markdown fences, prose, truncated JSON).
"""

from __future__ import annotations

from typing import Any

import pytest

from amnesia.config import Settings
from amnesia.extraction import (
    ExtractionResult,
    ExtractionScheduler,
    MemoryExtractor,
)
from amnesia.service import Amnesia
from amnesia.store import MemoryStore

CANNED_EXTRACTION: dict[str, Any] = {
    "facts": [
        {
            "kind": "identity",
            "key": "Name",
            "content": "Is named Ahmed",
            "importance": 0.9,
            "confidence": 0.9,
        },
        {
            "kind": "preference",
            "content": "Prefers concise answers",
            "importance": 0.7,
            "confidence": 0.8,
        },
    ],
    "events": [
        {
            "kind": "project",
            "title": "Built the Amnesia memory layer",
            "details": "Added the extraction pipeline",
            "project": "Amnesia",
            "importance": 0.8,
            "occurred_at": "2024-03-05",
        },
        {
            "kind": "learning",
            "title": "Learned about SQLite FTS5 tokenizers",
            "importance": 0.6,
        },
    ],
    "summary": "The user discussed building Amnesia and its extraction pipeline.",
}

SUMMARY = CANNED_EXTRACTION["summary"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extraction_settings(settings: Settings, **overrides: Any) -> Settings:
    base = {
        "extraction_enabled": True,
        "extraction_every_n_turns": 1,
        "extraction_min_chars": 1,
        "summarization_enabled": True,
    }
    base.update(overrides)
    return settings.clone(**base)


async def seed_session(
    service: Amnesia,
    *turns: tuple[str, str],
) -> str:
    session = service.start_session()
    for user_text, assistant_text in turns:
        await service.add_turn(session.id, user_text, assistant_text)
    return session.id


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_extraction_result_defaults() -> None:
    result = ExtractionResult()
    assert result.facts == []
    assert result.events == []
    assert result.summary is None
    assert result.raw is None
    assert result.is_empty is True
    assert result.item_count == 0


async def test_default_model_resolution(service: Amnesia) -> None:
    assert MemoryExtractor(service).default_model == "dolphin3:latest"
    assert MemoryExtractor(service, model="custom:model").default_model == "custom:model"


# ---------------------------------------------------------------------------
# extract_exchange
# ---------------------------------------------------------------------------


async def test_extract_exchange_happy_path(
    service: Amnesia,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    fake_ollama.queue_json(CANNED_EXTRACTION)
    extractor = MemoryExtractor(service)

    result = await extractor.extract_exchange(
        "I'm Ahmed. I prefer concise answers.", "Got it, Ahmed."
    )

    assert isinstance(result, ExtractionResult)
    assert result.item_count == 4
    assert result.facts[0]["kind"] == "identity"
    assert result.facts[0]["key"] == "name"  # normalized key
    assert result.facts[1]["kind"] == "preference"
    assert result.summary == SUMMARY
    assert result.raw is not None

    # extract_exchange only extracts: no memories/events are stored.
    assert store.count_memories() == 0
    assert store.count_events() == 0

    payload = fake_ollama.last_chat_payload()
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["model"] == "dolphin3:latest"
    assert "JSON object" in payload["messages"][0]["content"]
    assert "Conversation exchange" in payload["messages"][1]["content"]
    assert "Ahmed" in payload["messages"][1]["content"]

    done = store.db.scalar(
        "SELECT count(*) FROM extractions WHERE kind = 'exchange' AND status = 'done'"
    )
    assert done == 1


async def test_fenced_json_is_parsed(service: Amnesia, fake_ollama: Any) -> None:
    import json

    fake_ollama.queue_reply(
        "Here you go:\n```json\n" + json.dumps(CANNED_EXTRACTION) + "\n```\n"
    )
    result = await MemoryExtractor(service).extract_exchange("hello", "hi")
    assert len(result.facts) == 2
    assert len(result.events) == 2
    assert result.summary == SUMMARY


async def test_invalid_json_is_tolerated(
    service: Amnesia,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    fake_ollama.queue_reply("I am not able to produce that.")
    result = await MemoryExtractor(service).extract_exchange("hello", "hi")

    assert result.facts == []
    assert result.events == []
    assert result.summary is None
    assert result.raw is None

    errors = store.db.scalar(
        "SELECT count(*) FROM extractions WHERE kind = 'exchange' AND status = 'error'"
    )
    assert errors == 1
    assert store.count_memories() == 0


async def test_truncated_json_is_salvaged(service: Amnesia, fake_ollama: Any) -> None:
    fake_ollama.queue_reply(
        '{"facts": [{"kind": "fact", "content": "Uses Python 3.11 for Amnesia", '
        '"importance": 0.7}], "events": ['
    )
    result = await MemoryExtractor(service).extract_exchange("we use python", "ok")
    assert len(result.facts) == 1
    assert result.facts[0]["content"] == "Uses Python 3.11 for Amnesia"
    assert result.raw is not None  # partial payload was salvaged


async def test_loose_model_shapes_are_coerced(
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    fake_ollama.queue_json(
        {
            "facts": [
                "Prefers tabs over spaces",
                {
                    "statement": "Works as a data engineer",
                    "kind": "identity",
                    "importance": "high",
                },
            ],
            "events": None,
            "summary": ["The user", "talked about work."],
        }
    )
    result = await MemoryExtractor(service).extract_exchange("a", "b")
    assert len(result.facts) == 2
    assert result.facts[0]["kind"] == "fact"
    assert result.facts[1]["kind"] == "identity"
    assert result.facts[1]["importance"] == pytest.approx(0.85)
    assert result.events == []
    assert result.summary == "The user talked about work."


async def test_zero_scores_fall_back_to_defaults(
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    # Real local models often emit 0.0 as a "no opinion" placeholder; storing
    # importance 0.0 would bury the fact at the bottom of every ranked list.
    fake_ollama.queue_json(
        {
            "facts": [
                {
                    "kind": "identity",
                    "content": "Is named Ahmed",
                    "importance": 0.0,
                    "confidence": 0,
                }
            ],
            "events": [{"title": "Finished the store", "importance": 0}],
            "summary": None,
        }
    )
    result = await MemoryExtractor(service).extract_exchange("a", "b")
    assert result.facts[0]["importance"] == pytest.approx(0.6)
    assert result.facts[0]["confidence"] == pytest.approx(0.75)
    assert result.events[0]["importance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# distill_session persistence
# ---------------------------------------------------------------------------


async def test_distill_session_stores_everything(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service,
        ("I'm Ahmed and I prefer short answers.", "Noted."),
        ("Today I built the Amnesia extraction pipeline.", "Nice."),
    )
    fake_ollama.queue_json(CANNED_EXTRACTION)

    result = await extractor.distill_session(session_id, force=True)

    assert result["facts"] == 2
    assert result["events"] == 2
    assert result["summary"] == SUMMARY
    assert result["skipped"] is False
    assert "error" not in result

    memories = store.list_identity_memories()
    assert len(memories) == 2
    keyed = next(memory for memory in memories if memory.key == "name")
    assert keyed.kind == "identity"
    assert keyed.content == "Is named Ahmed"
    assert keyed.source == "extraction"
    assert keyed.session_id == session_id
    preference = next(memory for memory in memories if memory.kind == "preference")
    assert preference.content == "Prefers concise answers"

    events = store.list_events(days=None)
    assert len(events) == 2
    titles = {event.title for event in events}
    assert titles == {
        "Built the Amnesia memory layer",
        "Learned about SQLite FTS5 tokenizers",
    }
    built = next(event for event in events if event.title.startswith("Built"))
    assert built.kind == "project"
    assert built.project == "Amnesia"
    assert built.details == "Added the extraction pipeline"
    assert built.occurred_at.startswith("2024-03-05")
    learned = next(event for event in events if event.title.startswith("Learned"))
    assert learned.kind == "learning"
    assert learned.occurred_at  # defaults to now, never empty

    session = store.get_session(session_id)
    assert session is not None
    assert session.summary == SUMMARY
    summaries = store.list_memories(kind="summary")
    assert len(summaries) == 1
    assert summaries[0].content == SUMMARY
    assert summaries[0].source == "extraction"
    assert summaries[0].importance >= 0.6

    audit = store.db.query_one(
        "SELECT * FROM extractions WHERE kind = 'distill' AND session_id = ?",
        (session_id,),
    )
    assert audit is not None
    assert audit["status"] == "done"
    assert audit["items"] == 5  # 2 facts + 2 events + 1 summary


async def test_extract_session_summary_persists_and_refreshes(
    service: Amnesia,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    session_id = await seed_session(
        service,
        ("Let's build the Amnesia extraction pipeline.", "Sure."),
    )
    extractor = MemoryExtractor(service)

    fake_ollama.queue_json({"summary": "The user is building Amnesia extraction."})
    first = await extractor.extract_session_summary(session_id)
    assert first == "The user is building Amnesia extraction."
    session = store.get_session(session_id)
    assert session is not None and session.summary == first

    fake_ollama.queue_json({"summary": "The user built the Amnesia extraction module."})
    second = await extractor.extract_session_summary(session_id)
    assert second == "The user built the Amnesia extraction module."
    summaries = store.list_memories(kind="summary")
    assert len(summaries) == 1  # keyed per session: refreshed, not duplicated
    assert summaries[0].content == second
    assert summaries[0].source == "extraction"


async def test_keyed_fact_upserts_and_duplicates_merge(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service,
        ("My name is Ahmed and I prefer concise answers.", "Noted."),
    )

    first_payload = {
        "facts": [
            {
                "kind": "identity",
                "key": "name",
                "content": "Is named Ahmed",
                "importance": 0.9,
                "confidence": 0.9,
            },
            {
                "kind": "preference",
                "content": "Prefers concise answers",
                "importance": 0.7,
                "confidence": 0.8,
            },
        ],
        "events": [],
        "summary": None,
    }
    fake_ollama.queue_json(first_payload)
    assert (await extractor.distill_session(session_id, force=True))["facts"] == 2
    assert len(store.list_identity_memories()) == 2

    second_payload = {
        "facts": [
            {
                "kind": "identity",
                "key": "Name",
                "content": "Is named Ahmed El Alami",
                "importance": 0.95,
                "confidence": 0.9,
            },
            {
                "kind": "preference",
                "content": "Prefers concise answers",
                "importance": 0.8,
                "confidence": 0.8,
            },
        ],
        "events": [],
        "summary": None,
    }
    fake_ollama.queue_json(second_payload)
    assert (await extractor.distill_session(session_id, force=True))["facts"] == 2

    memories = store.list_identity_memories()
    assert len(memories) == 2  # upsert by key + vector dedupe; no duplicates
    keyed = next(memory for memory in memories if memory.key == "name")
    assert "El Alami" in keyed.content
    # The known profile was injected into the second prompt to avoid duplicates.
    second_prompt = fake_ollama.last_chat_payload()["messages"][1]["content"]
    assert "Known profile" in second_prompt
    assert "Is named Ahmed" in second_prompt


async def test_event_kind_and_date_validation(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service,
        ("I finished the extraction module yesterday.", "Great work."),
    )
    fake_ollama.queue_json(
        {
            "facts": [],
            "events": [
                {
                    "kind": "finished",
                    "title": "Finished the extraction module",
                    "occurred_at": "2024-03-05",
                },
                {
                    "kind": "whatever",
                    "title": "Made up a date",
                    "occurred_at": "sometime last week",
                },
                {"details": "no title at all"},
            ],
            "summary": None,
        }
    )
    await extractor.distill_session(session_id, force=True)

    events = store.list_events(days=None)
    assert len(events) == 2  # the title-less candidate was dropped
    dated = next(event for event in events if event.title.startswith("Finished"))
    assert dated.kind == "note"  # unknown kind normalized
    assert dated.occurred_at.startswith("2024-03-05")
    invalid = next(event for event in events if event.title == "Made up a date")
    assert not invalid.occurred_at.startswith("sometime")
    assert invalid.occurred_at[:4].isdigit()


async def test_stored_facts_are_redacted(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service,
        ("My key is sk-abcdefghijklmnopqrstuvwx by the way.", "I won't store that."),
    )
    fake_ollama.queue_json(
        {
            "facts": [
                {
                    "kind": "fact",
                    "content": "Uses API key sk-abcdefghijklmnopqrstuvwx",
                }
            ],
            "events": [],
            "summary": None,
        }
    )
    await extractor.distill_session(session_id, force=True)

    content = store.list_identity_memories()[0].content
    assert "sk-" not in content
    assert "[REDACTED]" in content


# ---------------------------------------------------------------------------
# maybe_extract gating
# ---------------------------------------------------------------------------


async def test_maybe_extract_respects_cadence_and_records_audit(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    custom = extraction_settings(
        settings,
        extraction_every_n_turns=3,
        extraction_min_chars=40,
        summarization_enabled=False,
    )
    extractor = MemoryExtractor(service, settings=custom)
    user_text = "I am building a local memory layer for Ollama and testing extraction."
    session = service.start_session()
    await service.add_turn(session.id, user_text, "Sounds good.")
    await service.add_turn(session.id, user_text, "Sure thing.")

    await extractor.maybe_extract(session.id)
    assert fake_ollama.chat_payloads() == []  # cadence: 2 turns < 3

    fake_ollama.queue_json(CANNED_EXTRACTION)
    await service.add_turn(session.id, user_text, "Great.")
    await extractor.maybe_extract(session.id)

    assert len(fake_ollama.chat_payloads()) == 1
    assert store.count_events() == 2
    audit = store.db.scalar(
        "SELECT count(*) FROM extractions WHERE kind = 'distill' AND status = 'done'"
    )
    assert audit == 1

    # A finished run resets the cadence: another immediate call is a no-op.
    await extractor.maybe_extract(session.id)
    assert len(fake_ollama.chat_payloads()) == 1


async def test_maybe_extract_skips_short_text(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    custom = extraction_settings(
        settings,
        extraction_every_n_turns=1,
        extraction_min_chars=500,
        summarization_enabled=False,
    )
    extractor = MemoryExtractor(service, settings=custom)
    session_id = await seed_session(service, ("hi", "hello"))
    await extractor.maybe_extract(session_id)
    assert fake_ollama.chat_payloads() == []
    assert store.count_memories() == 0


async def test_disabled_extraction_is_a_noop(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    custom = settings.clone(
        extraction_enabled=False,
        extraction_every_n_turns=1,
        extraction_min_chars=1,
    )
    extractor = MemoryExtractor(service, settings=custom)
    session_id = await seed_session(
        service,
        ("I am building the Amnesia extraction pipeline today.", "Nice."),
    )
    await extractor.maybe_extract(session_id)
    assert fake_ollama.chat_payloads() == []
    assert store.count_memories() == 0
    assert store.count_events() == 0


async def test_distill_survives_transport_failure(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service,
        ("I am building the Amnesia extraction pipeline today.", "Nice."),
    )
    fake_ollama.fail_connection = True

    result = await extractor.distill_session(session_id, force=True)  # must not raise

    assert result["skipped"] is False
    assert result["facts"] == 0
    assert "error" in result
    assert store.db.scalar(
        "SELECT count(*) FROM extractions WHERE kind = 'distill' AND status = 'error'"
    ) == 1
    assert store.count_memories() == 0


# ---------------------------------------------------------------------------
# ExtractionScheduler
# ---------------------------------------------------------------------------


async def test_scheduler_drains_dedupes_and_closes(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    custom = extraction_settings(
        settings,
        extraction_every_n_turns=1,
        extraction_min_chars=1,
        summarization_enabled=False,
    )
    extractor = MemoryExtractor(service, settings=custom)
    first = await seed_session(
        service, ("I am building the Amnesia extraction pipeline today.", "ok")
    )
    second = await seed_session(
        service, ("I am testing the distillation scheduler today.", "ok")
    )
    fake_ollama.queue_json(CANNED_EXTRACTION)
    fake_ollama.queue_json(CANNED_EXTRACTION)

    scheduler = ExtractionScheduler(extractor, max_concurrency=1)
    try:
        scheduler.schedule(first)
        scheduler.schedule(first)  # in-flight dedupe
        scheduler.schedule(second)
        assert scheduler.pending == 2

        await scheduler.drain()
        assert scheduler.pending == 0
        assert len(fake_ollama.chat_payloads()) == 2
        assert store.count_events() == 4  # two events per distilled session
    finally:
        await scheduler.close()

    # Scheduling after close is a silent no-op.
    scheduler.schedule(first)
    assert scheduler.pending == 0
    assert len(fake_ollama.chat_payloads()) == 2


async def test_scheduler_close_cancels_pending_work(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(
        service, ("I am building the Amnesia extraction pipeline today.", "ok")
    )
    fake_ollama.queue_json(CANNED_EXTRACTION)

    scheduler = ExtractionScheduler(extractor, max_concurrency=2)
    scheduler.schedule(session_id)
    assert scheduler.pending == 1

    await scheduler.close()  # cancels before the task ever runs

    assert scheduler.closed is True
    assert scheduler.pending == 0
    assert scheduler.in_flight_sessions == set()
    assert fake_ollama.chat_payloads() == []

# ---------------------------------------------------------------------------
# Redaction of the returned summary (regression: CLI echo leaked secrets)
# ---------------------------------------------------------------------------


async def test_distill_returns_redacted_summary(
    service: Amnesia,
    settings: Settings,
    fake_ollama: Any,
    store: MemoryStore,
) -> None:
    extractor = MemoryExtractor(service, settings=extraction_settings(settings))
    session_id = await seed_session(service, ("hello there", "hi"))
    fake_ollama.queue_json(
        {
            "facts": [],
            "events": [],
            "summary": "User connected with password=hunter2hunter2 to the database.",
        }
    )

    result = await extractor.distill_session(session_id, force=True)

    assert result["summary"]
    assert "hunter2" not in result["summary"]
    assert "[REDACTED]" in result["summary"]
    session = store.get_session(session_id)
    assert session is not None and "hunter2" not in (session.summary or "")
