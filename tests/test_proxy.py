"""Tests for the Amnesia HTTP proxy (``amnesia/proxy.py``) and dashboard.

All tests run fully offline: requests go through ``httpx.ASGITransport`` to the
FastAPI app and upstream Ollama calls are served by the ``fake_ollama``
``MockTransport`` fixture defined in ``tests/conftest.py``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from amnesia.config import load_settings
from amnesia.proxy import create_app, get_app, openai_response_from_ollama
from amnesia.service import Amnesia
from amnesia.store import MemoryStore

MODEL = "dolphin3:latest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app(service: Amnesia, fake_ollama: Any, **kwargs: Any) -> FastAPI:
    """Build a proxy app wired to the fake Ollama transport."""
    return create_app(service, upstream_client=fake_ollama.upstream_client, **kwargs)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://amnesia.test",
    )


async def _remember(service: Amnesia, content: str, **kwargs: Any) -> Any:
    return await service.remember(content, kind=kwargs.pop("kind", "preference"), **kwargs)


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What do I prefer?"}],
        "stream": False,
    }
    body.update(overrides)
    return body


def _chat_payloads(fake_ollama: Any) -> list[dict[str, Any]]:
    return [request["json"] for request in fake_ollama.requests if request["path"] == "/api/chat"]


def _generate_payloads(fake_ollama: Any) -> list[dict[str, Any]]:
    return [
        request["json"]
        for request in fake_ollama.requests
        if request["path"] == "/api/generate"
    ]


def _has_memory_block(payload: dict[str, Any]) -> bool:
    return "<amnesia_memory>" in json.dumps(payload.get("messages") or payload)


class _RecordingScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def schedule(self, session_id: str) -> None:
        self.calls.append(session_id)


class _FailingScheduler:
    def schedule(self, session_id: str) -> None:
        raise RuntimeError("scheduler exploded")


# ---------------------------------------------------------------------------
# Dashboard & redirects
# ---------------------------------------------------------------------------


async def test_root_redirects_to_dashboard(proxy_client: httpx.AsyncClient) -> None:
    response = await proxy_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/amnesia/ui"


async def test_amnesia_root_redirects_to_dashboard(proxy_client: httpx.AsyncClient) -> None:
    response = await proxy_client.get("/amnesia", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/amnesia/ui"


async def test_dashboard_html_is_self_contained(proxy_client: httpx.AsyncClient) -> None:
    response = await proxy_client.get("/amnesia/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Amnesia" in html
    for tab in ("Profile", "Timeline", "Sessions", "Search", "Health"):
        assert tab in html
    lowered = html.lower()
    assert 'src="http' not in lowered
    assert 'href="http' not in lowered
    assert "cdn." not in lowered
    assert "/amnesia/remember" in html
    assert "/amnesia/events" in html


# ---------------------------------------------------------------------------
# Memory API endpoints
# ---------------------------------------------------------------------------


async def test_health_endpoint(proxy_client: httpx.AsyncClient) -> None:
    response = await proxy_client.get("/amnesia/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["started"] is True
    assert payload["db_path"]
    assert payload["embeddings"]["enabled"] is True
    assert payload["ollama"]["chat_model"] == MODEL
    assert payload["counts"]["memories"] == 0


async def test_profile_and_remember(proxy_client: httpx.AsyncClient) -> None:
    created = await proxy_client.post(
        "/amnesia/remember",
        json={
            "content": "Prefers dark mode editors",
            "kind": "preference",
            "key": "editor_theme",
            "importance": 0.9,
            "tags": ["ui", "preferences"],
        },
    )
    assert created.status_code == 200
    memory = created.json()["memory"]
    assert memory["kind"] == "preference"
    assert memory["key"] == "editor_theme"
    assert memory["importance"] == pytest.approx(0.9)
    assert memory["tags"] == ["ui", "preferences"]

    profile = await proxy_client.get("/amnesia/profile", params={"limit": 10})
    assert profile.status_code == 200
    payload = profile.json()
    assert payload["count"] >= 1
    assert any(fact["content"] == "Prefers dark mode editors" for fact in payload["facts"])
    assert "dark mode" in payload["text"]


async def test_remember_validation(proxy_client: httpx.AsyncClient) -> None:
    empty = await proxy_client.post("/amnesia/remember", json={"content": "   "})
    assert empty.status_code == 400

    bad_kind = await proxy_client.post(
        "/amnesia/remember", json={"content": "Something", "kind": "bogus"}
    )
    assert bad_kind.status_code == 400
    assert "bogus" in bad_kind.text

    missing = await proxy_client.post("/amnesia/remember", json={})
    assert missing.status_code == 422


async def test_recall_endpoint(proxy_client: httpx.AsyncClient) -> None:
    await proxy_client.post(
        "/amnesia/remember",
        json={"content": "Loves espresso in the morning", "kind": "preference"},
    )
    missing_query = await proxy_client.get("/amnesia/recall")
    assert missing_query.status_code == 400

    response = await proxy_client.get("/amnesia/recall", params={"q": "espresso", "limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "espresso"
    assert payload["count"] >= 1
    assert any("espresso" in hit["content"].lower() for hit in payload["hits"])


async def test_events_create_and_list(proxy_client: httpx.AsyncClient) -> None:
    created = await proxy_client.post(
        "/amnesia/events",
        json={
            "title": "Wrote proxy tests",
            "kind": "task",
            "project": "amnesia",
            "details": "Covered streaming and injection",
            "importance": 0.7,
        },
    )
    assert created.status_code == 200
    event = created.json()["event"]
    assert event["title"] == "Wrote proxy tests"
    assert event["kind"] == "task"
    assert event["project"] == "amnesia"

    listed = await proxy_client.get("/amnesia/events", params={"days": 7, "limit": 50})
    assert listed.status_code == 200
    events = listed.json()["events"]
    assert any(item["id"] == event["id"] for item in events)

    invalid = await proxy_client.post(
        "/amnesia/events", json={"title": "Nope", "kind": "unexpected"}
    )
    assert invalid.status_code == 400


async def test_forget_memory(proxy_client: httpx.AsyncClient) -> None:
    created = await proxy_client.post(
        "/amnesia/remember",
        json={"content": "A preference that will be forgotten", "kind": "preference"},
    )
    memory_id = created.json()["memory"]["id"]

    deleted = await proxy_client.delete(f"/amnesia/memories/{memory_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    profile = await proxy_client.get("/amnesia/profile")
    assert all(fact["id"] != memory_id for fact in profile.json()["facts"])

    again = await proxy_client.delete(f"/amnesia/memories/{memory_id}")
    assert again.status_code == 404

    missing = await proxy_client.delete("/amnesia/memories/999999")
    assert missing.status_code == 404


async def test_stats_and_export(proxy_client: httpx.AsyncClient) -> None:
    await proxy_client.post("/amnesia/remember", json={"content": "Uses PowerShell on Windows"})
    await proxy_client.post("/amnesia/events", json={"title": "Ran the test suite", "kind": "task"})

    stats = await proxy_client.get("/amnesia/stats")
    assert stats.status_code == 200
    payload = stats.json()
    assert payload["memories"] >= 1
    assert payload["events"] >= 1
    assert "db_size_human" in payload

    exported = await proxy_client.get("/amnesia/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    export_payload = exported.json()
    assert export_payload["format"] == "amnesia-export"
    assert any("PowerShell" in memory["content"] for memory in export_payload["memories"])


async def test_sessions_list_and_detail(proxy_client: httpx.AsyncClient) -> None:
    empty = await proxy_client.get("/amnesia/sessions")
    assert empty.status_code == 200
    assert empty.json()["sessions"] == []

    missing = await proxy_client.get("/amnesia/sessions/does-not-exist")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Ollama-native chat / generate: injection + transcript capture
# ---------------------------------------------------------------------------


async def test_ollama_chat_non_stream_injects_memory(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply("Sure - dark mode noted.")
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False),
        headers={"X-Amnesia-Session": "chat-nonstream"},
    )
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Sure - dark mode noted."
    assert response.headers["X-Amnesia-Memory-Injected"] == "1"

    payload = _chat_payloads(fake_ollama)[-1]
    assert payload["model"] == MODEL
    assert payload["stream"] is False
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert "<amnesia_memory>" in system["content"]
    assert "dark mode" in system["content"]
    assert payload["messages"][-1] == {"role": "user", "content": "What do I prefer?"}

    detail = await proxy_client.get("/amnesia/sessions/chat-nonstream")
    assert detail.status_code == 200
    messages = detail.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What do I prefer?"
    assert messages[1]["content"] == "Sure - dark mode noted."
    assert messages[0]["metadata"]["client"] == "ollama"
    assert messages[0]["metadata"]["route"] == "ollama"


async def test_ollama_chat_streaming(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.stream_split_words = True
    fake_ollama.queue_reply("Streaming reply with several words")
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=True),
        headers={"X-Amnesia-Session": "chat-stream"},
    )
    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert lines[-1]["done"] is True
    combined = "".join(
        line.get("message", {}).get("content", "") for line in lines if isinstance(line.get("message"), dict)
    )
    assert combined.strip() == "Streaming reply with several words"

    payload = _chat_payloads(fake_ollama)[-1]
    assert payload["stream"] is True
    assert "<amnesia_memory>" in payload["messages"][0]["content"]

    detail = await proxy_client.get("/amnesia/sessions/chat-stream")
    messages = detail.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"].strip() == "Streaming reply with several words"


async def test_session_header_is_honoured(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("hello")
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False),
        headers={"X-Conversation-Id": "conv-123"},
    )
    assert response.status_code == 200
    assert response.headers["X-Amnesia-Session"] == "conv-123"
    detail = await proxy_client.get("/amnesia/sessions/conv-123")
    assert detail.status_code == 200
    assert detail.json()["session"]["id"] == "conv-123"


async def test_sessions_are_reused_within_gap(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("one")
    fake_ollama.queue_reply("two")
    for _ in range(2):
        response = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
        assert response.status_code == 200

    sessions = (await proxy_client.get("/amnesia/sessions")).json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 4


async def test_generate_injects_memory_and_persists(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply("Generate reply")
    response = await proxy_client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "Tell me about my preferences", "stream": False},
        headers={"X-Amnesia-Session": "generate-session"},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "Generate reply"

    payload = _generate_payloads(fake_ollama)[-1]
    assert "<amnesia_memory>" in payload["system"]
    assert "dark mode" in payload["system"]

    detail = (await proxy_client.get("/amnesia/sessions/generate-session")).json()
    messages = detail["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Tell me about my preferences"
    assert messages[1]["content"] == "Generate reply"


async def test_generate_streaming(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("Generated stream")
    response = await proxy_client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "Hello", "stream": True},
        headers={"X-Amnesia-Session": "generate-stream"},
    )
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert lines[-1]["done"] is True
    combined = "".join(line.get("response", "") for line in lines)
    assert combined.strip() == "Generated stream"
    detail = (await proxy_client.get("/amnesia/sessions/generate-stream")).json()
    assert detail["messages"][1]["content"] == "Generated stream"


# ---------------------------------------------------------------------------
# Injection bypasses and the double-injection guard
# ---------------------------------------------------------------------------


async def test_injection_disabled_by_header(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply("no memory")
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False),
        headers={"X-Amnesia-Memory": "off"},
    )
    assert response.status_code == 200
    assert response.headers["X-Amnesia-Memory-Injected"] == "0"
    payload = _chat_payloads(fake_ollama)[-1]
    assert not _has_memory_block(payload)
    assert all(message["role"] != "system" for message in payload["messages"])


async def test_injection_disabled_by_metadata(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply("no memory")
    body = _chat_body(stream=False, metadata={"amnesia_memory": False})
    response = await proxy_client.post("/api/chat", json=body)
    assert response.status_code == 200
    payload = _chat_payloads(fake_ollama)[-1]
    assert not _has_memory_block(payload)


async def test_injection_skipped_for_extraction_model(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply('{"facts": []}')
    response = await proxy_client.post(
        "/api/chat",
        json=_chat_body(stream=False, model="amnesia-extract:latest"),
    )
    assert response.status_code == 200
    payload = _chat_payloads(fake_ollama)[-1]
    assert not _has_memory_block(payload)
    # Extraction traffic is not captured as a user conversation.
    sessions = (await proxy_client.get("/amnesia/sessions")).json()["sessions"]
    assert sessions == []


async def test_injection_disabled_by_settings(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    service.settings.inject_context = False
    fake_ollama.queue_reply("no memory")
    response = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert response.status_code == 200
    payload = _chat_payloads(fake_ollama)[-1]
    assert not _has_memory_block(payload)


async def test_no_injection_when_store_is_empty(
    proxy_client: httpx.AsyncClient, fake_ollama: Any
) -> None:
    fake_ollama.queue_reply("hi")
    response = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert response.status_code == 200
    payload = _chat_payloads(fake_ollama)[-1]
    assert not _has_memory_block(payload)
    assert all(message["role"] != "system" for message in payload["messages"])


async def test_double_injection_guard(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    stale_system = (
        "You are a helpful assistant.\n\n"
        "<amnesia_memory>STALE OLD MEMORY BLOCK</amnesia_memory>"
    )
    body = _chat_body(
        stream=False,
        messages=[
            {"role": "system", "content": stale_system},
            {"role": "user", "content": "What do I prefer?"},
        ],
    )
    response = await proxy_client.post("/api/chat", json=body)
    assert response.status_code == 200
    payload = _chat_payloads(fake_ollama)[-1]
    forwarded_system = payload["messages"][0]["content"]
    assert forwarded_system.count("<amnesia_memory>") == 1
    assert forwarded_system.count("</amnesia_memory>") == 1
    assert "STALE OLD MEMORY BLOCK" not in forwarded_system
    assert "You are a helpful assistant." in forwarded_system
    assert "dark mode" in forwarded_system


# ---------------------------------------------------------------------------
# Failure isolation: memory/persistence/extraction problems never fail chat
# ---------------------------------------------------------------------------


async def test_memory_context_failure_does_not_break_chat(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any, monkeypatch: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("memory backend exploded")

    monkeypatch.setattr(service, "context_for", boom)
    fake_ollama.queue_reply("still works")
    response = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "still works"
    assert response.headers["X-Amnesia-Memory-Injected"] == "0"


async def test_persistence_failure_does_not_break_chat(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any, monkeypatch: Any
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("database closed")

    monkeypatch.setattr(service, "add_turn", boom)
    fake_ollama.queue_reply("reply survives")
    response = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "reply survives"


async def test_extraction_scheduler_is_notified(service: Amnesia, fake_ollama: Any) -> None:
    scheduler = _RecordingScheduler()
    app = _app(service, fake_ollama, extraction_scheduler=scheduler)
    async with _client(app) as client:
        fake_ollama.queue_reply("remembered")
        response = await client.post(
            "/api/chat",
            json=_chat_body(stream=False),
            headers={"X-Amnesia-Session": "sched-session"},
        )
        assert response.status_code == 200
    assert scheduler.calls == ["sched-session"]


async def test_failing_extraction_scheduler_does_not_break_chat(
    service: Amnesia, fake_ollama: Any
) -> None:
    app = _app(service, fake_ollama, extraction_scheduler=_FailingScheduler())
    async with _client(app) as client:
        fake_ollama.queue_reply("fine")
        response = await client.post("/api/chat", json=_chat_body(stream=False))
        assert response.status_code == 200
        assert response.json()["message"]["content"] == "fine"


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


async def test_openai_chat_non_stream(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.queue_reply("OpenAI reply")
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What do I prefer?"},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    response = await proxy_client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "OpenAI reply"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["prompt_tokens"] >= 0

    forwarded = _chat_payloads(fake_ollama)[-1]
    assert forwarded["stream"] is False
    assert forwarded["options"]["temperature"] == 0.2
    assert forwarded["options"]["num_predict"] == 64
    assert forwarded["tools"][0]["function"]["name"] == "get_time"
    system = forwarded["messages"][0]["content"]
    assert "You are terse." in system
    assert "<amnesia_memory>" in system
    assert "dark mode" in system


async def test_openai_chat_streaming(
    proxy_client: httpx.AsyncClient, service: Amnesia, fake_ollama: Any
) -> None:
    await _remember(service, "Prefers dark mode and concise answers")
    fake_ollama.stream_split_words = True
    fake_ollama.queue_reply("Streamed openai answer")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    response = await proxy_client.post(
        "/v1/chat/completions",
        json=body,
        headers={"X-Amnesia-Session": "openai-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    )
    assert content.strip() == "Streamed openai answer"
    finish_reasons = [
        chunk["choices"][0]["finish_reason"]
        for chunk in chunks
        if chunk["choices"][0]["finish_reason"]
    ]
    assert finish_reasons == ["stop"]

    detail = (await proxy_client.get("/amnesia/sessions/openai-stream")).json()
    messages = detail["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"].strip() == "Streamed openai answer"
    assert messages[0]["metadata"]["route"] == "openai"
    assert messages[0]["metadata"]["client"] == "openai"


def test_openai_response_maps_tool_calls() -> None:
    data = {
        "model": MODEL,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 7,
    }
    converted = openai_response_from_ollama(data)
    choice = converted["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    assert converted["usage"]["prompt_tokens"] == 12
    assert converted["usage"]["completion_tokens"] == 7
    assert converted["usage"]["total_tokens"] == 19


async def test_openai_models(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    response = await proxy_client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = [model["id"] for model in payload["data"]]
    assert MODEL in ids
    assert "nomic-embed-text:latest" in ids
    assert all(model["object"] == "model" for model in payload["data"])


async def test_openai_embeddings(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    response = await proxy_client.post(
        "/v1/embeddings",
        json={"model": "nomic-embed-text", "input": ["alpha", "beta"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [item["index"] for item in payload["data"]] == [0, 1]
    assert len(payload["data"][0]["embedding"]) == fake_ollama.embed_dim
    assert payload["usage"]["prompt_tokens"] > 0

    bad = await proxy_client.post("/v1/embeddings", json={"model": "nomic-embed-text"})
    assert bad.status_code == 400


# ---------------------------------------------------------------------------
# Ollama passthrough
# ---------------------------------------------------------------------------


async def test_ollama_passthrough_routes(proxy_client: httpx.AsyncClient, fake_ollama: Any) -> None:
    tags = await proxy_client.get("/api/tags")
    assert tags.status_code == 200
    assert len(tags.json()["models"]) == 3

    version = await proxy_client.get("/api/version")
    assert version.status_code == 200
    assert version.json()["version"] == fake_ollama.version

    ps = await proxy_client.get("/api/ps")
    assert ps.status_code == 200
    assert ps.json()["models"] == []

    show = await proxy_client.post("/api/show", json={"model": MODEL})
    assert show.status_code == 200
    assert show.json()["capabilities"] == ["completion"]

    embed = await proxy_client.post("/api/embed", json={"model": "nomic-embed-text", "input": "hi"})
    assert embed.status_code == 200
    assert len(embed.json()["embeddings"]) == 1

    paths = {request["path"] for request in fake_ollama.requests}
    assert {"/api/tags", "/api/version", "/api/ps", "/api/show", "/api/embed"} <= paths


async def test_passthrough_preserves_upstream_errors(
    proxy_client: httpx.AsyncClient, fake_ollama: Any
) -> None:
    pull = await proxy_client.post("/api/pull", json={"model": "unknown:latest"})
    assert pull.status_code == 404
    assert "no route" in pull.json()["error"]

    delete = await proxy_client.post("/api/delete", json={"model": "unknown:latest"})
    assert delete.status_code == 404

    paths = {request["path"] for request in fake_ollama.requests}
    assert "/api/pull" in paths
    assert "/api/delete" in paths


async def test_upstream_unreachable_returns_502(
    proxy_client: httpx.AsyncClient, fake_ollama: Any
) -> None:
    fake_ollama.fail_connection = True
    chat = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert chat.status_code == 502
    payload = chat.json()
    assert payload["error"]["type"] == "upstream_unreachable"
    assert "ollama" in payload["error"]["message"].lower()

    tags = await proxy_client.get("/api/tags")
    assert tags.status_code == 502
    assert tags.json()["error"]["type"] == "upstream_unreachable"


async def test_upstream_server_error_becomes_502(
    proxy_client: httpx.AsyncClient, fake_ollama: Any
) -> None:
    fake_ollama.fail_status = 500
    chat = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert chat.status_code == 502
    error = chat.json()["error"]
    assert error["type"] == "upstream_error"
    assert error["upstream_status"] == 500
    assert "ollama" in error["message"].lower()

    openai_chat = await proxy_client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert openai_chat.status_code == 502
    assert openai_chat.json()["error"]["type"] == "upstream_error"

    tags = await proxy_client.get("/api/tags")
    assert tags.status_code == 502
    assert tags.json()["error"]["upstream_status"] == 500


async def test_upstream_client_error_status_is_preserved(
    proxy_client: httpx.AsyncClient, fake_ollama: Any
) -> None:
    fake_ollama.fail_status = 404
    chat = await proxy_client.post("/api/chat", json=_chat_body(stream=False))
    assert chat.status_code == 404
    assert chat.json()["error"] == "fake upstream failure"

    openai_chat = await proxy_client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert openai_chat.status_code == 404
    assert openai_chat.json()["error"]["type"] == "upstream_error"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


async def test_chat_rejects_invalid_body(proxy_client: httpx.AsyncClient) -> None:
    invalid_json = await proxy_client.post(
        "/api/chat",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert invalid_json.status_code == 400

    missing_messages = await proxy_client.post("/api/chat", json={"model": MODEL})
    assert missing_messages.status_code == 400

    openai_missing = await proxy_client.post(
        "/v1/chat/completions", json={"model": MODEL, "messages": []}
    )
    assert openai_missing.status_code == 400


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_auth_enforced_when_token_configured(
    proxy_client: httpx.AsyncClient, service: Amnesia
) -> None:
    service.settings.api_token = "s3cret-token"
    try:
        unauthorized = await proxy_client.get("/amnesia/profile")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        assert unauthorized.json()["error"]["type"] == "unauthorized"

        bearer = await proxy_client.get(
            "/amnesia/profile", headers={"Authorization": "Bearer s3cret-token"}
        )
        assert bearer.status_code == 200

        header_token = await proxy_client.get(
            "/amnesia/stats", headers={"X-Amnesia-Token": "s3cret-token"}
        )
        assert header_token.status_code == 200

        query_token = await proxy_client.get("/amnesia/stats?token=s3cret-token")
        assert query_token.status_code == 200

        wrong = await proxy_client.get(
            "/amnesia/profile", headers={"Authorization": "Bearer wrong-token"}
        )
        assert wrong.status_code == 401

        # Health and the dashboard itself stay open; passthrough routes are local-only.
        assert (await proxy_client.get("/amnesia/health")).status_code == 200
        assert (await proxy_client.get("/amnesia/ui")).status_code == 200
        assert (await proxy_client.get("/api/tags")).status_code == 200
    finally:
        service.settings.api_token = None


# ---------------------------------------------------------------------------
# Lifespan & module-level factory
# ---------------------------------------------------------------------------


async def test_lifespan_starts_passed_service(
    settings: Any, store: MemoryStore, hash_embedder: Any, fake_ollama: Any
) -> None:
    service = Amnesia(
        settings,
        store=store,
        embedder=hash_embedder,
        ollama=fake_ollama.ollama_client,
    )
    assert service.started is False
    app = _app(service, fake_ollama)
    async with app.router.lifespan_context(app):
        assert service.started is True
        assert app.state.service is service
        assert app.state.owns_service is False
    # A caller-provided service is not closed by the app.
    assert service.started is True


async def test_lifespan_builds_and_closes_owned_service(
    monkeypatch: Any, settings: Any, fake_ollama: Any
) -> None:
    from amnesia import proxy as proxy_module
    from amnesia.embeddings import HashEmbedder

    built_store = MemoryStore(settings.database).initialize()
    built = Amnesia(
        settings,
        store=built_store,
        embedder=HashEmbedder(),
        ollama=fake_ollama.ollama_client,
    )
    monkeypatch.setattr(proxy_module, "Amnesia", lambda _settings: built)

    app = proxy_module.create_app(settings=settings)
    assert app.state.service is None
    async with app.router.lifespan_context(app):
        assert app.state.service is built
        assert app.state.owns_service is True
        assert built.started is True
    assert built.started is False


async def test_lifespan_wires_extraction_scheduler_when_enabled(
    settings: Any, store: MemoryStore, hash_embedder: Any, fake_ollama: Any
) -> None:
    pytest.importorskip("amnesia.extraction")
    extraction_settings = settings.clone(extraction_enabled=True)
    service = Amnesia(
        extraction_settings,
        store=store,
        embedder=hash_embedder,
        ollama=fake_ollama.ollama_client,
    )
    app = create_app(service, upstream_client=fake_ollama.upstream_client)
    async with app.router.lifespan_context(app):
        scheduler = app.state.extraction_scheduler
        assert scheduler is not None
        assert hasattr(scheduler, "schedule")
        assert app.state.owns_scheduler is True
        assert getattr(service, "extraction_scheduler", None) is scheduler
    # Shutdown closed the scheduler the app created; no exception escaped.
    assert app.state.owns_scheduler is True


def test_create_app_with_extraction_enabled_does_not_crash(tmp_home: Any) -> None:
    settings = load_settings(
        home=tmp_home,
        chat_model=MODEL,
        extraction_enabled=True,
        proxy_port=0,
    )
    app = create_app(settings=settings)
    assert isinstance(app, FastAPI)
    assert app.state.service is None
    paths = {route.path for route in app.routes}
    assert "/api/chat" in paths
    assert "/amnesia/ui" in paths


def test_extraction_scheduler_builder_is_guarded(service: Amnesia) -> None:
    from amnesia.proxy import _build_extraction_scheduler

    scheduler = _build_extraction_scheduler(service)
    # When amnesia/extraction.py is available it must produce a scheduler with
    # the documented interface; otherwise the proxy degrades to None.
    assert scheduler is None or hasattr(scheduler, "schedule")


def test_get_app_module_factory() -> None:
    app = get_app()
    assert isinstance(app, FastAPI)
    paths = {route.path for route in app.routes}
    assert "/api/chat" in paths
    assert "/api/generate" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
    assert "/v1/embeddings" in paths
    assert "/amnesia/health" in paths
    assert "/amnesia/shutdown" in paths
    assert "/amnesia/ui" in paths


@pytest.mark.asyncio
async def test_root_head_heartbeat_for_ollama_cli(proxy_client):
    """Regression: `ollama run` probes HEAD / before chatting and aborts on 405."""
    response = await proxy_client.head("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Graceful shutdown endpoint (background `amnesia serve` / `amnesia stop`)
# ---------------------------------------------------------------------------


async def test_shutdown_endpoint_sets_flag_and_is_idempotent(
    service: Amnesia, fake_ollama: Any
) -> None:
    app = _app(service, fake_ollama)
    assert app.state.shutdown_requested is False
    async with _client(app) as client:
        first = await client.post("/amnesia/shutdown")
        assert first.status_code == 200
        assert first.json() == {"status": "shutting_down"}
        assert app.state.shutdown_requested is True

        second = await client.post("/amnesia/shutdown")
        assert second.status_code == 200
        assert second.json() == {"status": "shutting_down"}
        assert app.state.shutdown_requested is True


async def test_shutdown_rejects_non_loopback_clients(
    service: Amnesia, fake_ollama: Any
) -> None:
    app = _app(service, fake_ollama)
    transport = httpx.ASGITransport(app=app, client=("10.0.0.7", 12345))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://amnesia.test"
    ) as client:
        response = await client.post("/amnesia/shutdown")
    assert response.status_code == 403
    payload = response.json()
    assert payload["error"]["type"] == "forbidden"
    assert "loopback" in payload["error"]["message"].lower()
    assert app.state.shutdown_requested is False


async def test_shutdown_allows_other_loopback_127_addresses(
    service: Amnesia, fake_ollama: Any
) -> None:
    app = _app(service, fake_ollama)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.5", 4321))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://amnesia.test"
    ) as client:
        response = await client.post("/amnesia/shutdown")
    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"
    assert app.state.shutdown_requested is True


async def test_health_still_works_with_shutdown_route_registered(
    service: Amnesia, fake_ollama: Any
) -> None:
    app = _app(service, fake_ollama)
    async with _client(app) as client:
        response = await client.get("/amnesia/health")
        assert response.status_code == 200
        assert response.json()["started"] is True
    assert app.state.shutdown_requested is False


async def test_shutdown_requires_auth_when_token_configured(
    service: Amnesia, fake_ollama: Any
) -> None:
    service.settings.api_token = "s3cret-token"
    try:
        app = _app(service, fake_ollama)
        async with _client(app) as client:
            unauthorized = await client.post("/amnesia/shutdown")
            assert unauthorized.status_code == 401
            assert app.state.shutdown_requested is False

            authorized = await client.post(
                "/amnesia/shutdown",
                headers={"Authorization": "Bearer s3cret-token"},
            )
            assert authorized.status_code == 200
            assert authorized.json()["status"] == "shutting_down"
            assert app.state.shutdown_requested is True
    finally:
        service.settings.api_token = None
