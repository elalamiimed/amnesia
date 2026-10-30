"""Cross-layer integration tests for Amnesia.

These tests deliberately cross module boundaries that the per-module suites keep
separate:

* Python API -> store -> retrieval (remember / profile / recall / context)
* HTTP proxy -> transcript persistence -> LLM extraction -> profile + timeline
* CLI (subprocess-style invocations against one temp home) -> store -> HTTP API
* MCP tools -> CLI and HTTP proxy over the same SQLite file
* export / purge / import round-trips, optional API-token auth, embedding
  backfill and session reuse/gap behaviour

Everything runs offline and deterministically: the ``fake_ollama`` fixture
serves every upstream HTTP call through ``httpx.MockTransport`` and each test
gets a fresh temp home.  No sleeps, no ports, no real daemon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import amnesia.cli as cli_module
from amnesia import mcp_server
from amnesia.config import Settings
from amnesia.embeddings import HashEmbedder
from amnesia.extraction import ExtractionScheduler, MemoryExtractor
from amnesia.proxy import create_app
from amnesia.service import Amnesia

MODEL = "dolphin3:latest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://amnesia.test",
    )


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What do I prefer?"}],
        "stream": False,
    }
    body.update(overrides)
    return body


def _assert_ok(result: Any) -> None:
    assert result.exit_code == 0, f"unexpected CLI failure:\n{result.output}"


def _json_stdout(result: Any) -> Any:
    return json.loads(result.stdout)


def _clear_amnesia_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real AMNESIA_* environment leak into tests."""
    for name in list(os.environ):
        if name.startswith("AMNESIA_"):
            monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class CliEnv:
    """Invoke the real Typer CLI against one temp home with an offline service."""

    def __init__(self, home: Path, build_service: Any) -> None:
        self.home = home
        self._build_service = build_service
        self.runner = CliRunner()

    def invoke(self, *args: str, input: str | None = None) -> Any:
        return self.runner.invoke(cli_module.app, list(args), input=input)

    def service(self, **overrides: Any) -> Amnesia:
        return self._build_service(**overrides)


@pytest.fixture()
def cli_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_home: Path,
    settings: Settings,
    fake_ollama: Any,
) -> CliEnv:
    """Real CLI invocations, fake Ollama, one temp ``AMNESIA_HOME``.

    ``amnesia.cli.make_service`` is monkeypatched (the CLI's documented test
    seam) so every invocation opens a *fresh* SQLite connection on the temp
    home while never touching the network or the developer's real store.
    """
    _clear_amnesia_env(monkeypatch)
    monkeypatch.setenv("AMNESIA_HOME", str(tmp_home))

    def build_service(**overrides: Any) -> Amnesia:
        merged: dict[str, Any] = {
            "home": tmp_home,
            "embed_model": "none",
            "extraction_enabled": False,
            "summarization_enabled": False,
        }
        merged.update(overrides)
        effective = settings.clone(**merged)
        return Amnesia(
            effective,
            embedder=HashEmbedder(),
            ollama=fake_ollama.ollama_client,
        )

    def fake_make_service(settings_arg: Settings | None = None) -> Amnesia:
        del settings_arg  # each CLI invocation loads its own Settings from env
        return build_service()

    monkeypatch.setattr(cli_module, "make_service", fake_make_service)
    return CliEnv(tmp_home, build_service)


# ---------------------------------------------------------------------------
# Python API -> store -> retrieval
# ---------------------------------------------------------------------------


async def test_remember_profile_recall_and_context(service: Amnesia) -> None:
    """The core memory loop: store a fact, see it in the profile, recall it."""
    memory = await service.remember(
        "User is Ada Lovelace",
        kind="identity",
        key="name",
        importance=0.9,
    )
    assert memory.kind == "identity"
    assert memory.key == "name"
    assert service.stats().memories == 1

    profile = service.profile()
    assert [fact.content for fact in profile] == ["User is Ada Lovelace"]

    # Re-remembering the same key updates the canonical statement in place.
    updated = await service.remember("User is Ada Byron", kind="identity", key="name")
    assert updated.id == memory.id
    assert [fact.content for fact in service.profile()] == ["User is Ada Byron"]

    hits = await service.recall("Ada")
    assert hits, "recall should find the stored identity fact"
    assert any("Ada" in hit.content for hit in hits)

    context = await service.context_for("What is my name?")
    assert "<amnesia_memory>" in context.text
    assert "Ada Byron" in context.text
    assert context.items_count >= 1


# ---------------------------------------------------------------------------
# HTTP proxy -> transcript -> extraction -> profile/timeline
# ---------------------------------------------------------------------------


async def test_proxy_chat_then_extraction_updates_profile_timeline_and_context(
    make_service: Any,
    fake_ollama: Any,
) -> None:
    """End-to-end: proxy transcript -> distill_session -> durable memory."""
    service = await make_service(extraction_enabled=True, summarization_enabled=True)
    fake_ollama.queue_reply("Noted, Ada - I will keep answers concise.")
    app = create_app(service, upstream_client=fake_ollama.upstream_client)

    async with _client(app) as client:
        chat = await client.post(
            "/api/chat",
            json=_chat_body(
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": "My name is Ada and I prefer concise answers.",
                    }
                ],
            ),
            headers={"X-Amnesia-Session": "integration-flow"},
        )
    assert chat.status_code == 200
    assert chat.headers["X-Amnesia-Session"] == "integration-flow"

    # 1. The proxy persisted the transcript before anything else happens.
    messages = service.store.session_messages("integration-flow")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content.startswith("My name is Ada")
    assert messages[1].content == "Noted, Ada - I will keep answers concise."

    # 2. Distillation consumes canned local-model JSON through the fake daemon.
    fake_ollama.queue_json(
        {
            "facts": [
                {
                    "kind": "identity",
                    "key": "name",
                    "content": "User's name is Ada",
                    "importance": 0.95,
                    "confidence": 0.9,
                },
                {
                    "kind": "preference",
                    "key": "answer_style",
                    "content": "Prefers concise answers",
                    "importance": 0.7,
                    "confidence": 0.8,
                },
            ],
            "events": [
                {
                    "kind": "project",
                    "title": "Built the Amnesia proxy",
                    "details": "Implemented the memory proxy workstream.",
                    "project": "amnesia",
                    "importance": 0.8,
                }
            ],
            "summary": "Ada introduced herself and asked for concise answers.",
        }
    )
    extractor = MemoryExtractor(service)
    result = await extractor.distill_session("integration-flow", force=True)
    assert result["skipped"] is False
    assert result["facts"] == 2
    assert result["events"] == 1
    assert result["summary"] == "Ada introduced herself and asked for concise answers."

    # 3. Profile and timeline are updated through the public API.
    profile = {memory.key: memory.content for memory in service.profile()}
    assert profile["name"] == "User's name is Ada"
    assert profile["answer_style"] == "Prefers concise answers"
    timeline = service.timeline(days=7)
    assert [(event.title, event.project) for event in timeline] == [
        ("Built the Amnesia proxy", "amnesia")
    ]

    # 4. Session summary + extraction audit trail are persisted.
    session = service.get_session("integration-flow")
    assert session is not None
    assert session.summary == "Ada introduced herself and asked for concise answers."
    audit = service.store.last_extraction("integration-flow")
    assert audit is not None
    assert audit["kind"] == "distill"
    assert audit["status"] == "done"
    assert audit["items"] == 4  # 2 facts + 1 event + 1 summary

    # 5. The extraction call used Ollama's JSON mode (not the chat proxy).
    extraction_payload = fake_ollama.chat_payloads()[-1]
    assert extraction_payload["format"] == "json"

    # 6. The refreshed summary is injected into the next prompt context.
    context = await service.context_for("What do I want?", session_id="integration-flow")
    assert "concise answers" in context.text


async def test_proxy_schedules_background_extraction_through_scheduler(
    make_service: Any,
    fake_ollama: Any,
) -> None:
    """The proxy's fire-and-forget scheduler runs the whole pipeline off-loop."""
    service = await make_service(
        extraction_enabled=True,
        summarization_enabled=True,
        extraction_every_n_turns=1,
        extraction_min_chars=0,
    )
    scheduler = ExtractionScheduler(MemoryExtractor(service))
    service.extraction_scheduler = scheduler  # type: ignore[attr-defined]

    fake_ollama.queue_reply("Dark mode it is.")
    fake_ollama.queue_json(
        {
            "facts": [
                {
                    "kind": "preference",
                    "content": "Prefers dark mode",
                    "importance": 0.8,
                }
            ],
            "events": [{"kind": "task", "title": "Configured dark mode"}],
            "summary": "Configured dark mode.",
        }
    )
    app = create_app(service, upstream_client=fake_ollama.upstream_client)
    try:
        async with _client(app) as client:
            chat = await client.post(
                "/api/chat",
                json=_chat_body(stream=False),
                headers={"X-Amnesia-Session": "scheduler-flow"},
            )
        assert chat.status_code == 200
        await scheduler.drain()
    finally:
        await scheduler.close()

    assert any("dark mode" in memory.content for memory in service.profile())
    assert any(event.title == "Configured dark mode" for event in service.timeline())
    assert service.get_session("scheduler-flow").summary == "Configured dark mode."


async def test_streaming_proxy_chat_assembles_transcript_across_chunk_splits(
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    """NDJSON chunks split mid-line must stream through and persist correctly."""
    await service.remember("Prefers streaming answers", kind="preference", importance=0.9)

    lines = [
        json.dumps(
            {
                "model": MODEL,
                "message": {"role": "assistant", "content": "Hello "},
                "done": False,
            }
        )
        + "\n",
        json.dumps(
            {
                "model": MODEL,
                "message": {"role": "assistant", "content": "world"},
                "done": False,
            }
        )
        + "\n",
        json.dumps(
            {
                "model": MODEL,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }
        )
        + "\n",
    ]
    blob = "".join(lines).encode("utf-8")
    offsets = [0, 13, 41, 97, len(blob)]  # deliberately splits JSON lines mid-way
    chunks = [blob[start:end] for start, end in zip(offsets, offsets[1:])]
    assert len(chunks) == 4

    class _ChunkedStream(httpx.AsyncByteStream):
        def __init__(self, pieces: list[bytes]) -> None:
            self.pieces = pieces
            self.yielded = 0

        async def __aiter__(self):
            for piece in self.pieces:
                self.yielded += 1
                yield piece

    stream = _ChunkedStream(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "application/x-ndjson"},
        )

    upstream = httpx.AsyncClient(
        base_url="http://chunked-ollama:11434",
        transport=httpx.MockTransport(handler),
    )
    try:
        app = create_app(service, upstream_client=upstream)
        async with _client(app) as client:
            response = await client.post(
                "/api/chat",
                json=_chat_body(stream=True),
                headers={"X-Amnesia-Session": "chunked-stream"},
            )
    finally:
        await upstream.aclose()

    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]
    assert response.headers["X-Amnesia-Memory-Injected"] == "1"
    decoded = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert "".join(chunk["message"]["content"] for chunk in decoded) == "Hello world"
    assert decoded[-1]["done"] is True
    assert stream.yielded == len(chunks)

    messages = service.store.session_messages("chunked-stream")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "Hello world"


# ---------------------------------------------------------------------------
# MCP -> CLI -> HTTP proxy share one store
# ---------------------------------------------------------------------------


async def test_mcp_cli_and_proxy_share_the_same_memory_store(
    cli_env: CliEnv,
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    written = await mcp_server.memory_remember(
        "User maintains the Amnesia HTTP proxy",
        kind="fact",
        key="project",
        service=service,
    )
    assert written["memory"]["key"] == "project"
    assert written["memory"]["source"] == "mcp"

    # A brand-new CLI invocation (new store connection) sees the MCP write.
    profile = _json_stdout(cli_env.invoke("profile", "--json"))
    assert any("Amnesia HTTP proxy" in fact["content"] for fact in profile["facts"])

    # The HTTP proxy also reads the same SQLite file.
    proxy_service = cli_env.service()
    await proxy_service.start()
    try:
        app = create_app(proxy_service, upstream_client=fake_ollama.upstream_client)
        async with _client(app) as client:
            response = await client.get("/amnesia/profile")
            stats = (await client.get("/amnesia/stats")).json()
    finally:
        await proxy_service.close()
    assert response.status_code == 200
    contents = [fact["content"] for fact in response.json()["facts"]]
    assert any("Amnesia HTTP proxy" in content for content in contents)
    assert stats["memories"] >= 1


# ---------------------------------------------------------------------------
# CLI end-to-end (real Typer app, temp AMNESIA_HOME, fresh store per command)
# ---------------------------------------------------------------------------


def test_cli_end_to_end_remember_note_export_purge_import(cli_env: CliEnv) -> None:
    remembered = cli_env.invoke(
        "remember",
        "User is Ada Lovelace",
        "--kind",
        "identity",
        "--key",
        "name",
        "--importance",
        "0.9",
    )
    _assert_ok(remembered)
    assert "Memory saved" in remembered.output
    _assert_ok(
        cli_env.invoke(
            "remember",
            "Prefers concise code-first answers",
            "--kind",
            "preference",
        )
    )

    # Each command is a separate invocation with its own SQLite connection.
    profile = _json_stdout(cli_env.invoke("profile", "--json"))
    assert profile["count"] == 2
    assert {fact["content"] for fact in profile["facts"]} == {
        "User is Ada Lovelace",
        "Prefers concise code-first answers",
    }

    recalled = _json_stdout(cli_env.invoke("recall", "Ada Lovelace", "--json"))
    assert recalled["query"] == "Ada Lovelace"
    assert recalled["count"] >= 1
    assert any("Ada Lovelace" in hit["content"] for hit in recalled["hits"])

    noted = cli_env.invoke(
        "note",
        "Shipped the QA workstream",
        "--kind",
        "project",
        "--project",
        "amnesia",
    )
    _assert_ok(noted)
    assert "Note saved" in noted.output
    timeline = _json_stdout(cli_env.invoke("timeline", "--json", "--days", "7"))
    assert timeline["count"] == 1
    assert timeline["events"][0]["title"] == "Shipped the QA workstream"
    assert timeline["events"][0]["project"] == "amnesia"
    assert (cli_env.home / "amnesia.db").exists()

    export_path = cli_env.home / "exports" / "roundtrip.json"
    exported = cli_env.invoke("export", str(export_path))
    _assert_ok(exported)
    assert export_path.exists()
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["format"] == "amnesia-export"
    assert {memory["content"] for memory in payload["memories"]} == {
        "User is Ada Lovelace",
        "Prefers concise code-first answers",
    }
    assert [event["title"] for event in payload["events"]] == ["Shipped the QA workstream"]

    purged = cli_env.invoke("purge", "--yes")
    _assert_ok(purged)
    assert "deleted" in purged.output.lower()
    assert _json_stdout(cli_env.invoke("profile", "--json"))["count"] == 0
    assert _json_stdout(cli_env.invoke("timeline", "--json"))["count"] == 0

    imported = cli_env.invoke("import", str(export_path))
    _assert_ok(imported)
    assert "Imported" in imported.output
    restored = _json_stdout(cli_env.invoke("profile", "--json"))
    assert {fact["content"] for fact in restored["facts"]} == {
        "User is Ada Lovelace",
        "Prefers concise code-first answers",
    }
    restored_timeline = _json_stdout(cli_env.invoke("timeline", "--json"))
    assert [event["title"] for event in restored_timeline["events"]] == [
        "Shipped the QA workstream"
    ]
    stats = _json_stdout(cli_env.invoke("stats", "--json"))
    assert stats["memories"] == 2
    assert stats["events"] == 1
    assert stats["db_size_bytes"] > 0


def test_serve_no_extraction_reaches_proxy_settings(
    cli_env: CliEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``serve --no-extraction`` must disable extraction in the proxy.

    The flag previously only skipped building the scheduler locally; the proxy
    still received settings with ``extraction_enabled=True`` and could rebuild
    its own pipeline on startup. The fix clones the settings with extraction off
    before calling ``create_app``.
    """
    import sys
    import types

    captured: dict[str, Any] = {}

    def fake_create_app(
        service: Amnesia,
        *,
        settings: Settings | None = None,
        extraction_scheduler: Any = None,
    ) -> Any:
        captured["service"] = service
        captured["settings"] = settings
        captured["scheduler"] = extraction_scheduler
        return object()

    class _FakeConfig:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            captured["config"] = kwargs

    class _FakeServer:
        def __init__(self, config: Any) -> None:
            captured["server_config"] = config

        async def serve(self) -> None:
            captured["served"] = True

    fake_proxy = types.SimpleNamespace(create_app=fake_create_app)
    monkeypatch.setattr(cli_module, "_load_proxy_module", lambda: fake_proxy)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(Config=_FakeConfig, Server=_FakeServer),
    )
    # The daemon's own settings have extraction enabled; the flag must win.
    monkeypatch.setattr(
        cli_module,
        "make_service",
        lambda settings_arg=None: cli_env.service(extraction_enabled=True),
    )

    result = cli_env.invoke(
        "serve", "--no-extraction", "--host", "127.0.0.1", "--port", "12345"
    )
    _assert_ok(result)
    assert captured["served"] is True
    effective = captured["settings"]
    assert effective is not None
    assert effective.extraction_enabled is False
    assert captured["scheduler"] is None
    # The daemon-side settings are untouched; only the proxy copy is adjusted.
    assert captured["service"].settings.extraction_enabled is True
    assert "extraction off" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# Session resolution: explicit headers, reuse, and idle-gap rollover
# ---------------------------------------------------------------------------


async def test_pinned_session_headers_are_reused(
    proxy_client: httpx.AsyncClient,
    fake_ollama: Any,
) -> None:
    fake_ollama.queue_reply("one")
    fake_ollama.queue_reply("two")
    for _ in range(2):
        response = await proxy_client.post(
            "/api/chat",
            json=_chat_body(stream=False),
            headers={"X-Amnesia-Session": "pinned-session"},
        )
        assert response.status_code == 200
        assert response.headers["X-Amnesia-Session"] == "pinned-session"

    sessions = (await proxy_client.get("/amnesia/sessions")).json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 4

    aliased = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False),
        headers={"X-Conversation-Id": "conv-alias"},
    )
    assert aliased.headers["X-Amnesia-Session"] == "conv-alias"
    detail = (await proxy_client.get("/amnesia/sessions/conv-alias")).json()
    assert detail["session"]["id"] == "conv-alias"
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]


async def test_new_session_starts_when_idle_gap_is_exceeded(
    make_service: Any,
    fake_ollama: Any,
) -> None:
    """A negative gap forces rollover, proving the gap is actually honoured."""
    service = await make_service(session_gap_minutes=-1)
    app = create_app(service, upstream_client=fake_ollama.upstream_client)
    fake_ollama.queue_reply("first")
    fake_ollama.queue_reply("second")

    async with _client(app) as client:
        first = await client.post("/api/chat", json=_chat_body(stream=False))
        second = await client.post("/api/chat", json=_chat_body(stream=False))
        sessions = (await client.get("/amnesia/sessions")).json()["sessions"]

    assert first.status_code == 200 and second.status_code == 200
    assert first.headers["X-Amnesia-Session"] != second.headers["X-Amnesia-Session"]
    assert len(sessions) == 2


# ---------------------------------------------------------------------------
# Double-injection guard
# ---------------------------------------------------------------------------


async def test_double_injection_guard_across_requests(
    proxy_client: httpx.AsyncClient,
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    await service.remember("Prefers dark mode", kind="preference", importance=0.9)
    fake_ollama.queue_reply("first reply")
    first = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False),
        headers={"X-Amnesia-Session": "double"},
    )
    assert first.status_code == 200
    injected_system = fake_ollama.chat_payloads()[-1]["messages"][0]["content"]
    assert injected_system.count("<amnesia_memory>") == 1

    # Some clients echo the injected prompt back on the next request.
    fake_ollama.queue_reply("second reply")
    second = await proxy_client.post(
        "/api/chat",
        json=_chat_body(
            stream=False,
            messages=[
                {"role": "system", "content": injected_system},
                {"role": "user", "content": "and now?"},
            ],
        ),
        headers={"X-Amnesia-Session": "double"},
    )
    assert second.status_code == 200
    forwarded = fake_ollama.chat_payloads()[-1]["messages"][0]["content"]
    assert forwarded.count("<amnesia_memory>") == 1
    assert forwarded.count("</amnesia_memory>") == 1
    assert "dark mode" in forwarded


async def test_double_injection_guard_strips_list_content_system_messages(
    proxy_client: httpx.AsyncClient,
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    """Regression: OpenAI-style list-content system parts are sanitised too.

    Some OpenAI clients send system content as a list of parts. A stale
    ``<amnesia_memory>`` block echoed back in such a part must be stripped before
    the fresh block is appended, otherwise the prompt carries two memory blocks.
    """
    await service.remember("Prefers dark mode", kind="preference", importance=0.9)
    stale = "You are helpful.\n<amnesia_memory>STALE OLD BLOCK</amnesia_memory>"
    bare_stale = "Also stale <amnesia_memory>BARE STALE BLOCK</amnesia_memory>"
    fake_ollama.queue_reply("reply")
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": stale},
                        bare_stale,
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        ),
    )
    assert response.status_code == 200
    forwarded = fake_ollama.chat_payloads()[-1]["messages"][0]["content"]
    assert isinstance(forwarded, str)
    assert forwarded.count("<amnesia_memory>") == 1
    assert forwarded.count("</amnesia_memory>") == 1
    assert "STALE OLD BLOCK" not in forwarded
    assert "BARE STALE BLOCK" not in forwarded
    assert "You are helpful." in forwarded
    assert "dark mode" in forwarded


# ---------------------------------------------------------------------------
# Secret redaction through proxy + extraction + export
# ---------------------------------------------------------------------------


async def test_proxy_redacts_secrets_before_storing_and_exporting(
    proxy_client: httpx.AsyncClient,
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    remembered = await proxy_client.post(
        "/amnesia/remember",
        json={
            "content": "OpenAI key sk-abcdefghijklmnop123456 and password=hunter2",
            "kind": "fact",
            "importance": 0.9,
        },
    )
    assert remembered.status_code == 200
    stored = remembered.json()["memory"]["content"]
    assert "[REDACTED]" in stored
    assert "sk-abcdefghijklmnop123456" not in stored
    assert "hunter2" not in stored

    # Model-extracted text goes through the same redaction path.
    service.start_session(session_id="redaction-extraction", model=MODEL)
    await service.add_turn("redaction-extraction", "remember my credentials", "ok")
    fake_ollama.queue_json(
        {
            "facts": [
                {
                    "kind": "fact",
                    "content": "Database password: hunter2secret",
                    "importance": 0.6,
                }
            ],
            "events": [
                {
                    "kind": "task",
                    "title": "Connected with api_key=supersecret123",
                }
            ],
            "summary": "User mentioned credentials.",
        }
    )
    result = await MemoryExtractor(service).distill_session(
        "redaction-extraction", force=True
    )
    assert result["facts"] == 1
    assert result["events"] == 1

    profile = (await proxy_client.get("/amnesia/profile")).json()["facts"]
    assert any("[REDACTED]" in fact["content"] for fact in profile)
    assert all("hunter2secret" not in fact["content"] for fact in profile)
    events = (await proxy_client.get("/amnesia/events", params={"days": 7})).json()["events"]
    assert any("[REDACTED]" in event["title"] for event in events)

    export = await proxy_client.get("/amnesia/export")
    assert export.status_code == 200
    assert "attachment" in export.headers.get("content-disposition", "")
    assert "sk-abcdefghijklmnop123456" not in export.text
    assert "hunter2secret" not in export.text
    assert "supersecret123" not in export.text


async def test_extraction_summary_is_redacted_before_persisting(
    service: Amnesia,
    fake_ollama: Any,
) -> None:
    """Regression: the extraction summary bypass path applies secret redaction."""
    service.start_session(session_id="summary-secret", model=MODEL)
    await service.add_turn("summary-secret", "please summarize our database setup", "ok")
    fake_ollama.queue_json(
        {
            "facts": [],
            "events": [],
            "summary": "User used password=hunter2summary to connect.",
        }
    )
    await MemoryExtractor(service).distill_session("summary-secret", force=True)
    session = service.get_session("summary-secret")
    assert session is not None
    assert session.summary is not None
    assert "hunter2summary" not in session.summary
    assert "[REDACTED]" in session.summary
    assert "User used" in session.summary  # the readable remainder is kept

    # The mirrored kind="summary" memory and the export are clean too.
    summary_memory = service.store.find_memory_by_key(
        "summary", "session-summary:summary-secret"
    )
    assert summary_memory is not None
    assert "hunter2summary" not in summary_memory.content
    assert "hunter2summary" not in json.dumps(service.store.export_data())


async def test_event_title_details_project_and_tags_are_redacted(service: Amnesia) -> None:
    """Regression: every persisted event text field goes through redaction."""
    event = await service.log_event(
        "Connected with api_key=supersecret123",
        details="DSN postgresql://admin:supersecret@db.internal/app",
        project="postgresql://admin:supersecret2@db.internal/app",
        tags="token=tok_abcdef123456, safe",
    )
    assert event.title == "Connected with api_key=[REDACTED]"
    assert "supersecret" not in (event.details or "")
    assert "[REDACTED]" in (event.details or "")
    assert "supersecret2" not in (event.project or "")
    assert (event.project or "").startswith("postgresql://admin:[REDACTED]@")
    assert "tok_abcdef123456" not in " ".join(event.tags)
    assert "safe" in event.tags

    export_text = json.dumps(service.store.export_data())
    assert "supersecret" not in export_text
    assert "tok_abcdef123456" not in export_text


async def test_string_tags_are_normalized_on_remember_and_dedupe_merge(
    service: Amnesia,
) -> None:
    """Regression: string tags are split correctly (never into characters)."""
    first = await service.remember("Prefers red themes in the editor", tags="theme, ui")
    assert first.tags == ["theme", "ui"]

    merged = await service.remember(
        "Prefers red themes in the editor", tags=["theme", "contrast"]
    )
    assert merged.id == first.id
    assert merged.tags == ["contrast", "theme", "ui"]

    # Keyed upsert keeps tags intact and redacts secret-bearing tags.
    keyed = await service.remember(
        "Uses VS Code", key="editor", tags="tools, token=tok_abcdef123456"
    )
    assert "tools" in keyed.tags
    assert all(len(tag) > 1 for tag in keyed.tags)
    assert "tok_abcdef123456" not in " ".join(keyed.tags)


# ---------------------------------------------------------------------------
# Optional API-token auth
# ---------------------------------------------------------------------------


async def test_optional_api_token_guards_memory_api_only(
    make_service: Any,
    fake_ollama: Any,
) -> None:
    token = "integration-token-123"
    service = await make_service(api_token=token)
    app = create_app(service, upstream_client=fake_ollama.upstream_client)

    async with _client(app) as client:
        unauthenticated = await client.get("/amnesia/profile")
        assert unauthenticated.status_code == 401
        assert unauthenticated.json()["error"]["type"] == "unauthorized"
        assert unauthenticated.headers["www-authenticate"] == "Bearer"

        # Health and dashboard stay public; passthrough chat stays open (local tool).
        assert (await client.get("/amnesia/health")).status_code == 200
        assert (await client.get("/amnesia/ui")).status_code == 200

        assert (
            await client.get(
                "/amnesia/profile", headers={"Authorization": f"Bearer {token}"}
            )
        ).status_code == 200
        assert (
            await client.get("/amnesia/profile", headers={"X-Amnesia-Token": token})
        ).status_code == 200
        assert (
            await client.get(
                "/amnesia/profile", headers={"Authorization": "Bearer wrong-token"}
            )
        ).status_code == 401
        assert (await client.get("/amnesia/profile", params={"token": token})).status_code == 200
        assert (await client.get("/amnesia/export")).status_code == 401

        fake_ollama.queue_reply("open surface")
        chat = await client.post(
            "/api/chat",
            json=_chat_body(stream=False),
            headers={
                "X-Amnesia-Session": "auth-open",
                # A client's own provider key must never be relayed upstream.
                "Authorization": "Bearer client-provider-secret",
            },
        )
        assert chat.status_code == 200

    # The Amnesia API token / client provider credentials must never reach Ollama.
    forwarded_headers = {
        key.lower(): value for key, value in fake_ollama.requests[-1]["headers"].items()
    }
    assert "authorization" not in forwarded_headers
    assert "x-amnesia-token" not in forwarded_headers
    assert "client-provider-secret" not in json.dumps(forwarded_headers)


# ---------------------------------------------------------------------------
# Embedding backfill
# ---------------------------------------------------------------------------


async def test_embedding_backfill_indexes_legacy_rows(service: Amnesia) -> None:
    store = service.store
    first = store.add_memory(kind="fact", content="Gamma builds rust services")
    second = store.add_memory(kind="fact", content="Delta writes python tooling")
    event = store.add_event(title="Migrated the rust services")
    assert len(store.memories_missing_embeddings(limit=10)) == 2
    assert len(store.events_missing_embeddings(limit=10)) == 1

    backfilled = await service.backfill_embeddings(limit=100)
    assert backfilled == {"memories": 2, "events": 1}
    assert store.memories_missing_embeddings(limit=10) == []
    assert store.events_missing_embeddings(limit=10) == []

    stats = service.stats()
    assert stats.embedded_memories == 2
    assert stats.embedded_events == 1
    for row_id, dim in ((first.id, 512), (second.id, 512)):
        row = store.db.query_one("SELECT embedding_dim FROM memories WHERE id = ?", (row_id,))
        assert row["embedding_dim"] == dim
    assert store.get_event(event.id) is not None

    hits = await service.recall("rust services")
    assert any("rust" in hit.content for hit in hits)

    # Idempotent: a second run has nothing left to embed.
    assert await service.backfill_embeddings(limit=100) == {"memories": 0, "events": 0}
