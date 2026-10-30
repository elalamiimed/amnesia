"""Real-Ollama end-to-end smoke tests (opt-in).

These tests talk to the *actual* local Ollama daemon at
``http://127.0.0.1:11434``.  They are skipped unless ``AMNESIA_REAL_OLLAMA=1``
is set, so the default offline suite never touches the network::

    $env:AMNESIA_REAL_OLLAMA = "1"
    python -m pytest tests/test_e2e.py -q

Every test writes to a throwaway temp ``AMNESIA_HOME`` (never ``~/.amnesia``),
uses ``dolphin3:latest`` with tiny prompts (``num_predict`` <= 24, temperature
0), and skips gracefully when the daemon or the required models are missing.
The suite is intentionally a thin smoke layer: exhaustive behaviour is covered
offline by ``tests/test_proxy.py`` and ``tests/test_integration.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import amnesia.cli as cli_module
from amnesia.config import Settings, load_settings
from amnesia.ollama import OllamaClient, OllamaError
from amnesia.proxy import create_app
from amnesia.service import Amnesia

REAL_OLLAMA = os.environ.get("AMNESIA_REAL_OLLAMA", "").strip() == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not REAL_OLLAMA,
        reason="real-Ollama e2e tests are opt-in: set AMNESIA_REAL_OLLAMA=1",
    ),
]

OLLAMA_URL = os.environ.get("AMNESIA_E2E_OLLAMA_URL", "http://127.0.0.1:11434").strip()
CHAT_MODEL = "dolphin3:latest"
EMBED_MODEL = "nomic-embed-text:latest"
EMBED_DIM = 768
TIMEOUT = 240.0
TINY_OPTIONS = {"temperature": 0, "num_predict": 16}


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


async def _live_ollama() -> tuple[str, list[str]]:
    """Probe the daemon; ``pytest.skip`` when it is unreachable or incomplete."""
    client = OllamaClient(OLLAMA_URL, timeout=15.0)
    try:
        version = await client.version()
        names = await client.model_names()
    except OllamaError as exc:
        pytest.skip(f"real Ollama daemon unreachable at {OLLAMA_URL}: {exc}")
    finally:
        await client.close()

    def installed(model: str) -> bool:
        base = model.split(":")[0]
        return any(name == model or name.split(":")[0] == base for name in names)

    missing = [model for model in (CHAT_MODEL, EMBED_MODEL) if not installed(model)]
    if missing:
        pytest.skip(f"required Ollama models missing: {missing} (installed: {names})")
    return version, names


def _require_live_ollama_sync() -> tuple[str, list[str]]:
    return asyncio.run(_live_ollama())


@pytest.fixture()
async def live_ollama() -> tuple[str, list[str]]:
    return await _live_ollama()


@pytest.fixture()
def real_settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing at the real daemon and a temp home."""
    return load_settings(
        home=tmp_path / "home",
        ollama_url=OLLAMA_URL,
        chat_model=CHAT_MODEL,
        embed_model=EMBED_MODEL,
        request_timeout=TIMEOUT,
        extraction_enabled=False,
        summarization_enabled=False,
        memory_enabled=True,
        inject_context=True,
    )


@pytest.fixture()
async def real_service(real_settings: Settings, live_ollama: tuple[str, list[str]]):
    """A fully started Amnesia service backed by the real Ollama daemon."""
    service = Amnesia(real_settings)
    await service.start()
    try:
        yield service
    finally:
        await service.close()


@pytest.fixture()
async def real_upstream() -> httpx.AsyncClient:
    """Explicit upstream client so the proxy cannot reuse a stale connection."""
    async with httpx.AsyncClient(
        base_url=OLLAMA_URL,
        timeout=httpx.Timeout(connect=10.0, read=TIMEOUT, write=TIMEOUT, pool=10.0),
        trust_env=False,
    ) as client:
        yield client


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://amnesia.test",
        timeout=httpx.Timeout(TIMEOUT),
    )


@pytest.fixture()
def real_cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Env for driving the real CLI entry point against the real daemon."""
    for name in list(os.environ):
        if name.startswith("AMNESIA_"):
            monkeypatch.delenv(name, raising=False)
    home = tmp_path / "cli-home"
    monkeypatch.setenv("AMNESIA_HOME", str(home))
    monkeypatch.setenv("AMNESIA_OLLAMA_URL", OLLAMA_URL)
    monkeypatch.setenv("AMNESIA_MODEL", CHAT_MODEL)
    monkeypatch.setenv("AMNESIA_EMBED_MODEL", EMBED_MODEL)
    monkeypatch.setenv("AMNESIA_EXTRACTION", "0")
    monkeypatch.setenv("AMNESIA_LOG_LEVEL", "ERROR")
    # Loopback must never go through a corporate proxy.
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    return home


# ---------------------------------------------------------------------------
# Daemon reachability & real service health
# ---------------------------------------------------------------------------


async def test_real_ollama_reachable_and_models_installed(
    live_ollama: tuple[str, list[str]],
) -> None:
    version, names = live_ollama
    assert isinstance(version, str) and version
    assert any(name.startswith("dolphin3") for name in names), names
    assert any("nomic-embed-text" in name for name in names), names

    # resolve_chat_model must pick the requested model when it is installed.
    client = OllamaClient(OLLAMA_URL, timeout=15.0)
    try:
        assert await client.is_available() is True
        assert await client.resolve_chat_model(CHAT_MODEL) == CHAT_MODEL
    finally:
        await client.close()


async def test_real_service_health_and_embedding_dimension(real_service: Amnesia) -> None:
    health = real_service.health()
    assert health["ollama"]["available"] is True
    assert health["db_integrity"] == "ok"
    assert health["fts_enabled"] is True
    assert health["embeddings"]["enabled"] is True
    assert str(health["embeddings"]["model"]).startswith("ollama:")
    assert health["embeddings"]["dim"] == EMBED_DIM
    assert health["counts"]["memories"] == 0

    vector = await real_service.embed("the quick brown fox jumps over the lazy dog")
    assert len(vector) == EMBED_DIM
    assert all(isinstance(value, float) for value in vector[:8])


# ---------------------------------------------------------------------------
# Real proxy chat through the real daemon
# ---------------------------------------------------------------------------


async def test_real_chat_non_stream_through_proxy(
    real_service: Amnesia,
    real_upstream: httpx.AsyncClient,
) -> None:
    await real_service.remember(
        "User's favourite editor is VS Code", kind="preference", importance=0.9
    )
    app = create_app(real_service, upstream_client=real_upstream)

    async with _client(app) as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "user", "content": "Reply with exactly: OK"}
                ],
                "stream": False,
                "options": TINY_OPTIONS,
            },
            headers={"X-Amnesia-Session": "real-e2e-chat"},
        )
        assert response.status_code == 200, response.text[:500]
        data = response.json()
        content = data["message"]["content"]
        assert isinstance(content, str) and content.strip(), data

        # Memory injection actually happened in the prompt sent upstream.
        assert response.headers["X-Amnesia-Memory-Injected"] == "1"
        assert int(response.headers["X-Amnesia-Memory-Chars"]) > 0
        assert response.headers["X-Amnesia-Session"] == "real-e2e-chat"

        detail = await client.get("/amnesia/sessions/real-e2e-chat")
        assert detail.status_code == 200
        messages = detail.json()["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Reply with exactly: OK"
    assert messages[1]["content"] == content
    assert messages[0]["metadata"]["route"] == "ollama"


async def test_real_streaming_chat_persists_full_transcript(
    real_service: Amnesia,
    real_upstream: httpx.AsyncClient,
) -> None:
    app = create_app(real_service, upstream_client=real_upstream)
    async with _client(app) as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "user", "content": "Count from one to three."}
                ],
                "stream": True,
                "options": TINY_OPTIONS,
            },
            headers={"X-Amnesia-Session": "real-e2e-stream"},
        )
        assert response.status_code == 200, response.text[:500]
        assert "application/x-ndjson" in response.headers["content-type"]
        chunks = [line for line in response.text.splitlines() if line.strip()]
        assert len(chunks) >= 2, chunks
        import json

        decoded = [json.loads(line) for line in chunks]
        assert decoded[-1]["done"] is True
        streamed = "".join(
            str((chunk.get("message") or {}).get("content") or "")
            for chunk in decoded
        )
        assert streamed.strip()
        detail = await client.get("/amnesia/sessions/real-e2e-stream")
        assert detail.status_code == 200

    messages = detail.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == streamed


async def test_real_openai_chat_completions_non_stream(
    real_service: Amnesia,
    real_upstream: httpx.AsyncClient,
) -> None:
    app = create_app(real_service, upstream_client=real_upstream)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "user", "content": "Reply with exactly: OK"}
                ],
                "stream": False,
                "max_tokens": 16,
                "temperature": 0,
            },
            headers={"X-Amnesia-Session": "real-e2e-openai"},
        )
    assert response.status_code == 200, response.text[:500]
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"].strip()
    assert choice["finish_reason"] in {"stop", "length"}
    usage = data["usage"]
    assert usage["total_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ---------------------------------------------------------------------------
# Real embeddings through the OpenAI-compatible surface
# ---------------------------------------------------------------------------


async def test_real_v1_embeddings_returns_768_dimension(
    real_service: Amnesia,
    real_upstream: httpx.AsyncClient,
) -> None:
    app = create_app(real_service, upstream_client=real_upstream)
    async with _client(app) as client:
        response = await client.post(
            "/v1/embeddings",
            json={"model": EMBED_MODEL, "input": ["hello amnesia", "second text"]},
        )
    assert response.status_code == 200, response.text[:500]
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    for item in data["data"]:
        vector = item["embedding"]
        assert len(vector) == EMBED_DIM
        assert all(isinstance(value, (int, float)) for value in vector[:8])
    assert data["usage"]["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Real CLI doctor (exit code 0 against the live daemon)
# ---------------------------------------------------------------------------


def test_real_cli_doctor_exits_zero(
    real_cli_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)  # never read a .env from the repo root
    _require_live_ollama_sync()

    result = CliRunner().invoke(cli_module.app, ["doctor"])
    output = result.output.replace("\n", " ")
    assert result.exit_code == 0, f"doctor failed:\n{output}"
    assert "Amnesia doctor" in output
    assert "Stored data" in output
    assert "unreachable" not in output.lower()
    assert "hard failure" not in output.lower()
    assert (real_cli_env / "amnesia.db").exists()
