"""Shared pytest fixtures for Amnesia.

Design rules:
* No network, no real Ollama daemon, no slow sleeps.
* Every test gets an isolated temp home + SQLite file.
* ``fake_ollama`` implements the subset of the Ollama HTTP API that Amnesia uses
  (version/tags/embed/chat/generate) with httpx.MockTransport, records all
  requests, and lets tests queue canned replies/streams.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# Make ``import amnesia`` work regardless of how pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amnesia.config import Settings, load_settings  # noqa: E402
from amnesia.embeddings import HashEmbedder  # noqa: E402
from amnesia.ollama import OllamaClient  # noqa: E402
from amnesia.service import Amnesia  # noqa: E402
from amnesia.store import MemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Ollama daemon
# ---------------------------------------------------------------------------


def _fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dim):
        byte = digest[index % len(digest)]
        values.append((byte / 127.5) - 1.0)
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


class FakeOllama:
    """In-process fake of the Ollama HTTP API."""

    def __init__(self, *, base_url: str = "http://fake-ollama:11434") -> None:
        self.base_url = base_url
        self.version = "0.40.11"
        self.models: list[str] = [
            "dolphin3:latest",
            "R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m",
            "nomic-embed-text:latest",
        ]
        self.default_reply = "This is a fake local model reply."
        self.replies: list[str] = []
        self.embed_dim = 16
        self.requests: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []
        self.fail_status: int | None = None
        self.fail_connection: bool = False
        self.stream_split_words: bool = False
        self._upstream = httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(self._handle),
        )

    # -- test helpers ----------------------------------------------------
    @property
    def upstream_client(self) -> httpx.AsyncClient:
        """Async client the proxy should use as its upstream in tests."""
        return self._upstream

    @property
    def ollama_client(self) -> OllamaClient:
        """OllamaClient wired to this fake (for Amnesia / extractor tests)."""
        return OllamaClient(self.base_url, client=self._upstream)

    def queue_reply(self, content: str) -> None:
        self.replies.append(content)

    def queue_json(self, payload: dict[str, Any]) -> None:
        self.replies.append(json.dumps(payload))

    def chat_payloads(self) -> list[dict[str, Any]]:
        return [request["json"] for request in self.requests if request["path"] == "/api/chat"]

    def last_chat_payload(self) -> dict[str, Any]:
        payloads = self.chat_payloads()
        assert payloads, "no /api/chat request was recorded"
        return payloads[-1]

    def reset(self) -> None:
        self.requests.clear()
        self.embed_calls.clear()
        self.replies.clear()
        self.fail_status = None
        self.fail_connection = False

    async def aclose(self) -> None:
        await self._upstream.aclose()

    # -- transport handler ----------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        try:
            payload = json.loads(request.content) if request.content else None
        except ValueError:
            payload = None
        self.requests.append(
            {
                "method": request.method,
                "path": path,
                "json": payload,
                "headers": dict(request.headers),
            }
        )
        if self.fail_connection:
            raise httpx.ConnectError("fake Ollama is unreachable", request=request)
        if self.fail_status is not None:
            return httpx.Response(self.fail_status, json={"error": "fake upstream failure"})

        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})

        if path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": name,
                            "model": name,
                            "modified_at": "2026-10-01T00:00:00Z",
                            "size": 123,
                            "details": {"family": "llama", "parameter_size": "8B"},
                        }
                        for name in self.models
                    ]
                },
            )

        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})

        if path == "/api/show":
            model = (payload or {}).get("model", "dolphin3:latest")
            return httpx.Response(200, json={"model": model, "details": {}, "capabilities": ["completion"]})

        if path == "/api/embed":
            model = (payload or {}).get("model")
            inputs = (payload or {}).get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            self.embed_calls.append(list(inputs))
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "embeddings": [_fake_vector(str(text), self.embed_dim) for text in inputs],
                },
            )

        if path == "/api/chat":
            model = (payload or {}).get("model", "dolphin3:latest")
            content = self.replies.pop(0) if self.replies else self.default_reply
            if (payload or {}).get("stream", False):
                return httpx.Response(
                    200,
                    content=self._ndjson_stream(model, content),
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "created_at": "2026-10-01T00:00:00Z",
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 10,
                    "prompt_eval_count": 5,
                },
            )

        if path == "/api/generate":
            model = (payload or {}).get("model", "dolphin3:latest")
            content = self.replies.pop(0) if self.replies else self.default_reply
            if (payload or {}).get("stream", False):
                chunks = [
                    {"model": model, "response": content, "done": False},
                    {"model": model, "response": "", "done": True},
                ]
                return httpx.Response(
                    200,
                    content="\n".join(json.dumps(chunk) for chunk in chunks) + "\n",
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(200, json={"model": model, "response": content, "done": True})

        return httpx.Response(404, json={"error": f"fake ollama has no route {path}"})

    def _ndjson_stream(self, model: str, content: str) -> bytes:
        if self.stream_split_words:
            pieces = [word + " " for word in content.split(" ")]
        else:
            pieces = [content]
        chunks = [
            {"model": model, "message": {"role": "assistant", "content": piece}, "done": False}
            for piece in pieces
            if piece
        ]
        chunks.append(
            {
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "eval_count": 10,
            }
        )
        return ("\n".join(json.dumps(chunk) for chunk in chunks) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Settings / store / service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture()
def settings(tmp_home: Path) -> Settings:
    return load_settings(
        home=tmp_home,
        chat_model="dolphin3:latest",
        embed_model="nomic-embed-text",
        extraction_enabled=False,
        summarization_enabled=False,
        memory_enabled=True,
        inject_context=True,
        proxy_host="127.0.0.1",
        proxy_port=0,
        request_timeout=10.0,
    )


@pytest.fixture()
def store(settings: Settings) -> MemoryStore:
    memory_store = MemoryStore(settings.database).initialize()
    try:
        yield memory_store
    finally:
        memory_store.close()


@pytest.fixture()
def fake_ollama() -> FakeOllama:
    # MockTransport holds no sockets/threads, so no teardown is required; the
    # fixture deliberately stays sync so it can be used by both sync and async
    # tests without event-loop juggling.
    return FakeOllama()


@pytest.fixture()
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture()
async def service(
    settings: Settings,
    store: MemoryStore,
    hash_embedder: HashEmbedder,
    fake_ollama: FakeOllama,
) -> Amnesia:
    app_service = Amnesia(
        settings,
        store=store,
        embedder=hash_embedder,
        ollama=fake_ollama.ollama_client,
    )
    await app_service.start()
    try:
        yield app_service
    finally:
        await app_service.close()


@pytest.fixture()
def make_service(settings: Settings, store: MemoryStore, fake_ollama: FakeOllama):
    """Factory for services with custom settings (same store/fake Ollama)."""

    async def _factory(**overrides: Any) -> Amnesia:
        custom = settings.clone(**overrides) if overrides else settings
        app_service = Amnesia(
            custom,
            store=store,
            embedder=HashEmbedder(),
            ollama=fake_ollama.ollama_client,
        )
        await app_service.start()
        return app_service

    return _factory


# ---------------------------------------------------------------------------
# Proxy client fixtures (only available once amnesia/proxy.py exists)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def proxy_client(service: Amnesia, fake_ollama: FakeOllama):
    """httpx.AsyncClient speaking ASGI to the Amnesia proxy app."""
    proxy = pytest.importorskip("amnesia.proxy")

    app = proxy.create_app(service, upstream_client=fake_ollama.upstream_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://amnesia.test") as client:
        yield client
