"""Async client for the Ollama HTTP API.

Only the endpoints Amnesia needs are implemented, with careful handling of
NDJSON streaming, long generation timeouts and friendly errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterable, Sequence

import httpx

from .config import Settings, is_embedding_model_name, normalize_ollama_url

LOG = logging.getLogger("amnesia.ollama")

DEFAULT_TIMEOUT = 600.0


class OllamaError(RuntimeError):
    """Ollama request failed (network error or non-2xx response)."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fields Ollama understands, preserving tool calls/images."""
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": str(message.get("role", "user")),
            "content": message.get("content", "") or "",
        }
        for key in ("images", "tool_calls", "tool_call_id", "name", "thinking"):
            if key in message and message[key] is not None:
                item[key] = message[key]
        cleaned.append(item)
    return cleaned


class OllamaClient:
    """Thin, dependency-free async wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = normalize_ollama_url(base_url)
        self.timeout = float(timeout)
        self._client = client
        self._owns_client = client is None
        self._headers = dict(headers or {})

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaClient":
        return cls(settings.ollama_url, timeout=settings.request_timeout)

    # -- plumbing --------------------------------------------------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.timeout,
                    write=self.timeout,
                    pool=10.0,
                ),
                headers=self._headers,
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _raise_for_response(response: httpx.Response, action: str) -> None:
        if response.status_code < 400:
            return
        body = ""
        try:
            body = response.text[:500]
        except Exception:  # noqa: BLE001 - best effort error body
            body = ""
        raise OllamaError(
            f"Ollama {action} failed with HTTP {response.status_code}: {body}",
            status_code=response.status_code,
            body=body,
        )

    # -- discovery -------------------------------------------------------
    async def version(self) -> str:
        try:
            response = await self._get_client().get("/api/version")
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise OllamaError(f"Ollama version check failed: HTTP {response.status_code}")
        return str(response.json().get("version", "unknown"))

    async def tags(self) -> list[dict[str, Any]]:
        try:
            response = await self._get_client().get("/api/tags")
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        self._raise_for_response(response, "tags")
        models = response.json().get("models", [])
        return [dict(model) for model in models if isinstance(model, dict)]

    async def model_names(self) -> list[str]:
        names = []
        for model in await self.tags():
            name = model.get("name") or model.get("model")
            if name:
                names.append(str(name))
        return names

    async def is_available(self) -> bool:
        try:
            await self.version()
            return True
        except OllamaError:
            return False

    async def resolve_chat_model(self, preferred: str | None = None) -> str:
        """Pick a usable chat model: preferred when installed, else best guess."""
        names = await self.model_names()
        if preferred and preferred in names:
            return preferred
        if preferred:
            LOG.warning("Chat model %r is not installed; falling back", preferred)
        candidates = [name for name in names if not is_embedding_model_name(name)]
        if not candidates:
            raise OllamaError(
                "No chat models are installed in Ollama. Run `ollama pull <model>` first."
            )
        return candidates[0]

    # -- embeddings ------------------------------------------------------
    async def embeddings(self, model: str, input_: str | Sequence[str]) -> list[list[float]]:
        payload_input: str | list[str] = input_ if isinstance(input_, str) else list(input_)
        try:
            response = await self._get_client().post(
                "/api/embed",
                json={"model": model, "input": payload_input, "truncate": True},
            )
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama embeddings request failed: {exc}") from exc
        self._raise_for_response(response, "embeddings")
        return [list(map(float, vector)) for vector in response.json().get("embeddings", [])]

    # -- chat ------------------------------------------------------------
    def _chat_payload(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        stream: bool,
        options: dict[str, Any] | None,
        format_: str | dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        keep_alive: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": normalize_messages(messages),
            "stream": stream,
        }
        if options:
            payload["options"] = dict(options)
        if format_ is not None:
            payload["format"] = format_
        if tools:
            payload["tools"] = tools
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return payload

    async def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        format_: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion; returns Ollama's response object."""
        payload = self._chat_payload(
            model,
            messages,
            stream=False,
            options=options,
            format_=format_,
            tools=tools,
            keep_alive=keep_alive,
        )
        try:
            response = await self._get_client().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat request failed: {exc}") from exc
        self._raise_for_response(response, "chat")
        return dict(response.json())

    async def chat_stream(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        format_: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion yielding decoded NDJSON chunks."""
        payload = self._chat_payload(
            model,
            messages,
            stream=True,
            options=options,
            format_=format_,
            tools=tools,
            keep_alive=keep_alive,
        )
        client = self._get_client()
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:500]
                    raise OllamaError(
                        f"Ollama chat failed with HTTP {response.status_code}: {body}",
                        status_code=response.status_code,
                        body=body,
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        LOG.debug("Skipping malformed Ollama stream line: %r", line[:120])
                        continue
                    yield chunk
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat stream failed: {exc}") from exc

    async def chat_text(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str:
        data = await self.chat(model, messages, options=options, keep_alive=keep_alive)
        message = data.get("message") or {}
        return str(message.get("content", "") or "")

    async def chat_json(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        """Ask for a JSON object; the caller validates the shape."""
        data = await self.chat(
            model,
            messages,
            options=options,
            format_="json",
            keep_alive=keep_alive,
        )
        content = str((data.get("message") or {}).get("content", "") or "")
        from .utils import extract_json_object

        parsed = extract_json_object(content)
        if parsed is None:
            raise OllamaError(f"Model did not return valid JSON: {content[:300]}")
        return parsed

    # -- generate (passthrough support) ----------------------------------
    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            response = await self._get_client().post("/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama generate request failed: {exc}") from exc
        self._raise_for_response(response, "generate")
        return dict(response.json())
