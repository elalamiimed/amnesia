"""FastAPI HTTP proxy and memory dashboard for Amnesia.

The proxy sits between Ollama-speaking clients (Open WebUI, LibreChat,
Continue, ``ollama run``, plain ``curl``, the OpenAI SDK, ...) and the local
Ollama daemon.  Every chat request is transparently enriched with a compact
``<amnesia_memory>`` block built from the local memory store, and the resulting
exchange is persisted as a transcript (and queued for LLM extraction when that
workstream is enabled).

Three surfaces are exposed:

* **Ollama-native** - ``/api/chat`` and ``/api/generate`` with memory injection,
  plus transparent passthrough for ``/api/tags``, ``/api/version``, ``/api/ps``,
  ``/api/show``, ``/api/pull``, ``/api/delete`` and ``/api/embed``.
* **OpenAI-compatible** - ``/v1/chat/completions`` (SSE streaming),
  ``/v1/models`` and ``/v1/embeddings`` so hosted-style tools can talk to the
  local model with memory.
* **Memory API** - the ``/amnesia/*`` JSON endpoints consumed by the bundled
  dashboard (``/amnesia/ui``), the CLI, MCP clients and tests.

Run it with::

    python -m uvicorn --factory amnesia.proxy:get_app --host 127.0.0.1 --port 11435

or simply ``amnesia serve``.  The dashboard lives at
``http://127.0.0.1:11435/amnesia/ui``.
"""

from __future__ import annotations

import asyncio
import codecs
import hmac
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings, normalize_ollama_url
from .models import (
    ALL_EVENT_KINDS,
    ALL_MEMORY_KINDS,
    Session,
    coerce_tags,
)
from .ollama import normalize_messages
from .service import Amnesia
from .utils import estimate_tokens

LOG = logging.getLogger("amnesia.proxy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
DASHBOARD_FILE = STATIC_DIR / "dashboard.html"

MEMORY_TAG_OPEN = "<amnesia_memory>"
MEMORY_TAG_CLOSE = "</amnesia_memory>"
MEMORY_BLOCK_RE = re.compile(
    r"<amnesia_memory>.*?(?:</amnesia_memory>|\Z)",
    re.DOTALL | re.IGNORECASE,
)

SESSION_HEADERS: tuple[str, ...] = ("x-amnesia-session", "x-conversation-id")
MEMORY_HEADER = "x-amnesia-memory"
MEMORY_OFF_VALUES = frozenset({"off", "0", "false", "no", "none", "disabled"})
EXTRACTION_MODEL_PREFIX = "amnesia-extract"

VALID_MEMORY_KINDS: tuple[str, ...] = tuple(ALL_MEMORY_KINDS)
VALID_EVENT_KINDS: tuple[str, ...] = tuple(ALL_EVENT_KINDS)

#: Routes under ``/amnesia`` that stay reachable without an API token.
PUBLIC_AMNESIA_PATHS = frozenset({"/amnesia/health", "/amnesia/ui"})
PUBLIC_AMNESIA_PREFIXES: tuple[str, ...] = ("/amnesia/static/",)

UPSTREAM_CONNECT_TIMEOUT = 10.0
UPSTREAM_WRITE_TIMEOUT = 60.0

#: Response headers that must not be echoed back from the upstream daemon.
_SKIP_UPSTREAM_HEADERS = frozenset(
    {
        "content-length",
        "content-type",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "upgrade",
    }
)

_OLLAMA_PASSTHROUGH_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", "/api/tags"),
    ("GET", "/api/version"),
    ("GET", "/api/ps"),
    ("POST", "/api/show"),
    ("POST", "/api/pull"),
    ("POST", "/api/delete"),
    ("DELETE", "/api/delete"),
    ("POST", "/api/embed"),
)


# ---------------------------------------------------------------------------
# Request models for the memory API
# ---------------------------------------------------------------------------


class RememberRequest(BaseModel):
    """Body of ``POST /amnesia/remember``."""

    content: str = Field(description="Durable statement to remember")
    kind: str = "fact"
    key: str | None = None
    subject: str = "user"
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: Any = None
    source: str = "manual"
    session_id: str | None = None


class EventRequest(BaseModel):
    """Body of ``POST /amnesia/events``."""

    title: str
    kind: str = "note"
    details: str | None = None
    project: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    occurred_at: str | None = None
    tags: Any = None
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Prepared turn bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreparedTurn:
    """Everything the proxy learned about one chat request before forwarding."""

    route: str
    model: str
    client: str
    user_text: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    session: Session | None = None
    session_header: str | None = None
    injected: bool = False
    memory_chars: int = 0
    system: str | None = None
    system_changed: bool = False


# ---------------------------------------------------------------------------
# Generic HTTP helpers
# ---------------------------------------------------------------------------


def _split_content_type(value: str | None, default: str = "application/json") -> str:
    """Return the bare media type from a ``Content-Type`` header value."""
    if not value:
        return default
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type or default


def _forward_headers(response: httpx.Response, *, decoded: bool = False) -> dict[str, str]:
    """Copy safe headers from an upstream response (never content-length etc.).

    When the body has been decoded by httpx (``response.content``) the original
    ``Content-Encoding`` header would be wrong, so it is dropped as well.
    """
    headers: dict[str, str] = {}
    for key, value in response.headers.items():
        lowered = key.lower()
        if lowered in _SKIP_UPSTREAM_HEADERS:
            continue
        if decoded and lowered == "content-encoding":
            continue
        headers[key] = value
    return headers


def _passthrough_response(response: httpx.Response) -> Response:
    """Return a buffered upstream response, preserving status and headers."""
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=_split_content_type(response.headers.get("content-type")),
        headers=_forward_headers(response, decoded=True),
    )


async def _read_and_close(response: httpx.Response) -> bytes:
    """Read an entire upstream response body and close the response."""
    try:
        if getattr(response, "is_stream_consumed", False):
            return response.content
        return await response.aread()
    finally:
        await response.aclose()


async def _upstream_failure(
    app: FastAPI,
    response: httpx.Response,
    *,
    openai: bool = False,
) -> Response:
    """Translate a non-2xx upstream response.

    Client errors (4xx) are passed through unchanged for Ollama-native routes
    so clients keep Ollama's precise diagnostics (e.g. "model not found").
    Server errors (5xx) become an HTTP 502 gateway error with a readable JSON
    message, satisfying the proxy contract "upstream failure -> 502".
    """
    content = await _read_and_close(response)
    status = response.status_code
    snippet = content.decode("utf-8", "replace")[:600].strip()
    if status < 500 and not openai:
        return Response(
            content=content,
            status_code=status,
            media_type=_split_content_type(response.headers.get("content-type")),
            headers=_forward_headers(response, decoded=True),
        )
    if status >= 500:
        LOG.warning("Ollama upstream failed with HTTP %s: %s", status, snippet[:300])
        message = (
            f"Ollama upstream failed with HTTP {status}. "
            f"Is the local model/daemon healthy? Upstream said: {snippet or '(empty body)'}"
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": message,
                    "type": "upstream_error",
                    "upstream_status": status,
                }
            },
        )
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": f"Ollama rejected the request with HTTP {status}: {snippet}",
                "type": "upstream_error",
                "upstream_status": status,
            }
        },
    )


async def _iter_upstream_bytes(response: httpx.Response) -> AsyncIterator[bytes]:
    """Iterate an upstream response body without buffering it.

    Real HTTP responses are streamed chunk by chunk with ``aiter_raw``.  A few
    transports (notably ``httpx.MockTransport`` used by the tests) return
    responses that are already loaded; for those the body is replayed from
    memory so the proxy behaves identically without buffering real streams.
    """
    if getattr(response, "is_stream_consumed", False):
        content = response.content
        if content:
            yield content
        return
    async for chunk in response.aiter_raw():
        yield chunk


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _json_body(request: Request) -> dict[str, Any]:
    """Parse a JSON object request body or raise a friendly 400."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload


def _response_json(response: httpx.Response) -> dict[str, Any] | None:
    """Best-effort JSON object decoding of an upstream response."""
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError):
        LOG.debug("Upstream returned a non-JSON body: %r", response.text[:200])
        return None
    return data if isinstance(data, dict) else None


def _upstream_unreachable(app: FastAPI, exc: Exception) -> JSONResponse:
    """HTTP 502 payload for an unreachable Ollama daemon."""
    url = getattr(app.state.settings, "ollama_url", "the configured Ollama server")
    LOG.warning("Cannot reach Ollama upstream at %s: %s", url, exc)
    message = (
        f"Amnesia could not reach the Ollama daemon at {url}: {exc}. "
        "Make sure `ollama serve` is running and that AMNESIA_OLLAMA_URL points "
        "at the daemon (the proxy default is http://127.0.0.1:11434)."
    )
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": message,
                "type": "upstream_unreachable",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        },
    )


# ---------------------------------------------------------------------------
# App state helpers: upstream client, service, extraction scheduler
# ---------------------------------------------------------------------------


def _make_upstream_client(settings: Settings) -> httpx.AsyncClient:
    """Build the httpx client used to talk to the Ollama daemon."""
    timeout = httpx.Timeout(
        connect=UPSTREAM_CONNECT_TIMEOUT,
        read=max(5.0, float(getattr(settings, "request_timeout", 600.0) or 600.0)),
        write=UPSTREAM_WRITE_TIMEOUT,
        pool=UPSTREAM_CONNECT_TIMEOUT,
    )
    base_url = normalize_ollama_url(getattr(settings, "ollama_url", None))
    return httpx.AsyncClient(base_url=base_url, timeout=timeout)


def _upstream(app: FastAPI) -> httpx.AsyncClient:
    """Return (creating lazily if needed) the upstream httpx client."""
    client = getattr(app.state, "upstream_client", None)
    if client is None:
        client = _make_upstream_client(app.state.settings)
        app.state.upstream_client = client
        app.state.owns_upstream = True
        LOG.debug("Created upstream Ollama client for %s", app.state.settings.ollama_url)
    return client


def _upstream_url(app: FastAPI, path: str) -> str:
    """Resolve ``path`` against the upstream client's base URL.

    Tests and embedders may pass a client without ``base_url``; in that case
    fall back to an absolute URL built from settings.
    """
    client = _upstream(app)
    base = str(getattr(client, "base_url", "") or "")
    if base and base != "/":
        return path
    return normalize_ollama_url(app.state.settings.ollama_url).rstrip("/") + path


def _attach_scheduler(service: Amnesia, scheduler: Any) -> None:
    """Store the scheduler on the service so other layers can find it."""
    try:
        setattr(service, "extraction_scheduler", scheduler)
    except Exception:  # noqa: BLE001 - attachment is best effort
        LOG.debug("Could not attach extraction scheduler to the service", exc_info=True)


def _build_extraction_scheduler(service: Amnesia) -> Any | None:
    """Build the optional LLM extraction pipeline, degrading gracefully.

    Workstream A (``amnesia/extraction.py``) may not be available yet; the proxy
    must keep working without it.
    """
    try:
        from .extraction import ExtractionScheduler, MemoryExtractor
    except ImportError as exc:
        LOG.warning(
            "Automatic memory extraction is unavailable (%s); "
            "the proxy keeps working without it.",
            exc,
        )
        return None
    try:
        extractor = MemoryExtractor(service)
        scheduler = ExtractionScheduler(extractor)
    except Exception as exc:  # noqa: BLE001 - never break the proxy on startup
        LOG.warning("Could not initialise the extraction pipeline: %s", exc)
        return None
    LOG.info(
        "Automatic memory extraction enabled (model=%s)",
        service.extraction_model or "(chat model)",
    )
    return scheduler


async def _install_extraction(app: FastAPI, service: Amnesia) -> None:
    """Create/attach the extraction scheduler once, if configured."""
    if getattr(app.state, "extraction_checked", False):
        scheduler = getattr(app.state, "extraction_scheduler", None)
        if scheduler is not None:
            _attach_scheduler(service, scheduler)
        return
    app.state.extraction_checked = True

    scheduler = getattr(app.state, "extraction_scheduler", None)
    if scheduler is None and getattr(app.state.settings, "extraction_enabled", False):
        scheduler = _build_extraction_scheduler(service)
        if scheduler is not None:
            app.state.extraction_scheduler = scheduler
            app.state.owns_scheduler = True
    if scheduler is not None:
        _attach_scheduler(service, scheduler)


async def _create_owned_service(app: FastAPI) -> Amnesia:
    """Build, start and install a service owned by the app lifespan."""
    service = Amnesia(app.state.settings)
    await service.start()
    app.state.service = service
    app.state.owns_service = True
    await _install_extraction(app, service)
    return service


async def _ensure_service(app: FastAPI) -> Amnesia:
    """Return the app's service, creating it lazily when no lifespan ran."""
    service = getattr(app.state, "service", None)
    if service is not None:
        return service
    lock: asyncio.Lock = app.state.init_lock
    async with lock:
        service = getattr(app.state, "service", None)
        if service is None:
            service = await _create_owned_service(app)
        return service


async def _service_dependency(request: Request) -> Amnesia:
    """FastAPI dependency yielding the running :class:`Amnesia` service."""
    return await _ensure_service(request.app)


async def _schedule_extraction(app: FastAPI, session_id: str | None) -> None:
    """Fire-and-forget extraction scheduling; failures never surface."""
    if not session_id:
        return
    scheduler = getattr(app.state, "extraction_scheduler", None)
    if scheduler is None:
        service = getattr(app.state, "service", None)
        scheduler = getattr(service, "extraction_scheduler", None)
    if scheduler is None:
        return
    try:
        await _maybe_await(scheduler.schedule(session_id))
    except Exception:  # noqa: BLE001 - extraction is strictly best-effort
        LOG.warning("Could not schedule memory extraction for session %s", session_id, exc_info=True)


# ---------------------------------------------------------------------------
# Memory injection helpers
# ---------------------------------------------------------------------------


def _header_value(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _requested_session_id(request: Request) -> str | None:
    """Return the session id supplied by a client header, if any."""
    for name in SESSION_HEADERS:
        value = _header_value(request, name)
        if value:
            return value
    return None


def _memory_disabled_by_client(request: Request, payload: dict[str, Any]) -> bool:
    """Honour explicit client-side opt-outs from memory injection."""
    header = _header_value(request, MEMORY_HEADER)
    if header is not None and header.lower() in MEMORY_OFF_VALUES:
        return True
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        flag = metadata.get("amnesia_memory")
        if flag is False:
            return True
        if isinstance(flag, str) and flag.strip().lower() in MEMORY_OFF_VALUES:
            return True
    return False


def _is_extraction_model(model: str | None) -> bool:
    """True for internal extraction traffic (never injected or captured)."""
    return bool(model) and str(model).strip().lower().startswith(EXTRACTION_MODEL_PREFIX)


def _injection_allowed(
    app: FastAPI,
    request: Request,
    payload: dict[str, Any],
    model: str,
) -> bool:
    """Decide whether this request may receive a memory block."""
    settings = app.state.settings
    if not getattr(settings, "inject_context", True):
        return False
    if not getattr(settings, "memory_enabled", True):
        return False
    if _is_extraction_model(model):
        return False
    return not _memory_disabled_by_client(request, payload)


def _content_to_text(content: Any) -> str:
    """Flatten Ollama/OpenAI message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(part.get("content"), str):
                    parts.append(str(part["content"]))
        return " ".join(parts)
    return str(content)


def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    """Return the text of the newest user message (the memory query)."""
    for message in reversed(list(messages)):
        if str(message.get("role", "")).lower() != "user":
            continue
        text = _content_to_text(message.get("content")).strip()
        if text:
            return text
    return ""


def _strip_memory_block(text: str) -> str:
    """Remove a previously injected ``<amnesia_memory>`` block from ``text``."""
    if not text or "<amnesia_memory" not in text.lower():
        return text
    return MEMORY_BLOCK_RE.sub("", text).strip()


def _strip_memory_blocks(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove stale memory blocks echoed back by a client on re-entry.

    Handles both plain-string system content (Ollama/OpenAI) and OpenAI-style
    list content parts, so a stale block cannot be forwarded twice.
    """
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if str(item.get("role", "")).lower() in {"system", "developer"}:
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = _strip_memory_block(content)
            elif isinstance(content, list):
                parts: list[Any] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(_strip_memory_block(part))
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        new_part = dict(part)
                        new_part["text"] = _strip_memory_block(part["text"])
                        parts.append(new_part)
                    else:
                        parts.append(part)
                item["content"] = parts
        cleaned.append(item)
    return cleaned


def _flatten_system_contents(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn list-style OpenAI system content into a plain string."""
    flattened: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if str(item.get("role", "")).lower() in {"system", "developer"}:
            if isinstance(item.get("content"), list):
                item["content"] = _content_to_text(item["content"])
        flattened.append(item)
    return flattened


async def _memory_block(
    service: Amnesia,
    user_text: str,
    session: Session | None,
) -> str:
    """Build the memory block for a query; returns "" when there is none."""
    if not user_text.strip():
        return ""
    try:
        context = await service.context_for(
            user_text,
            session_id=session.id if session is not None else None,
        )
    except Exception:  # noqa: BLE001 - memory failures must never break chat
        LOG.warning("Building memory context failed; continuing without it", exc_info=True)
        return ""
    if context is None:
        return ""
    try:
        if getattr(context, "is_empty", False):
            return ""
        text = str(getattr(context, "text", "") or "").strip()
    except Exception:  # noqa: BLE001 - defensive: context objects are simple
        return ""
    return text


def _model_name(service: Amnesia, payload: dict[str, Any]) -> str:
    """Resolve the model for a request; fall back to the configured chat model."""
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return str(service.chat_model or "")


_CLIENT_LABELS: tuple[tuple[str, str], ...] = (
    ("open webui", "openwebui"),
    ("openwebui", "openwebui"),
    ("librechat", "librechat"),
    ("continue", "continue"),
    ("msty", "msty"),
    ("ollama", "ollama"),
)


def _client_label(request: Request, *, route: str) -> str:
    """Short, stable client label used in session/transcript metadata."""
    user_agent = (request.headers.get("user-agent") or "").lower()
    for needle, label in _CLIENT_LABELS:
        if needle in user_agent:
            return label
    return "openai" if route == "openai" else "ollama"


async def _resolve_session(
    service: Amnesia,
    request: Request,
    *,
    model: str,
    client: str,
) -> Session | None:
    """Resolve the transcript session for this request (never raises)."""
    requested = _requested_session_id(request)
    try:
        if requested:
            existing = service.get_session(requested)
            if existing is not None:
                return existing
            try:
                return service.start_session(
                    session_id=requested,
                    model=model or service.chat_model,
                    client=client,
                    metadata={"source": "proxy", "requested_id": True},
                )
            except Exception:  # noqa: BLE001 - duplicate/odd ids fall back below
                LOG.debug("Could not create requested session %r", requested, exc_info=True)
        return await service.ensure_session(
            model=model or service.chat_model,
            client=client,
            gap_minutes=service.settings.session_gap_minutes,
            metadata={"source": "proxy"},
        )
    except Exception:  # noqa: BLE001 - memory persistence is best-effort
        LOG.warning("Could not resolve a chat session for this request", exc_info=True)
        return None


async def _safe_finish_turn(
    app: FastAPI,
    turn: PreparedTurn,
    assistant_text: str,
    *,
    tool_calls: Sequence[Any] | None = None,
) -> None:
    """Persist a completed turn and schedule extraction - never raises."""
    if _is_extraction_model(turn.model):
        LOG.debug("Not persisting a turn generated by the extraction model")
        return
    session = turn.session
    assistant_text = assistant_text or ""
    if session is None:
        return
    if not turn.user_text and not assistant_text:
        return

    service = getattr(app.state, "service", None)
    if service is None:
        return

    metadata: dict[str, Any] = {
        "client": turn.client,
        "route": turn.route,
        "model": turn.model or None,
    }
    if turn.session_header:
        metadata["session_header"] = turn.session_header
    if tool_calls:
        metadata["tool_calls"] = len(tool_calls)
    try:
        await service.add_turn(
            session.id,
            turn.user_text or "",
            assistant_text,
            model=turn.model or None,
            metadata=metadata,
        )
    except Exception:  # noqa: BLE001 - never fail a chat because of persistence
        LOG.warning("Could not persist the chat transcript; continuing", exc_info=True)
        return
    await _schedule_extraction(app, session.id)


# ---------------------------------------------------------------------------
# Turn preparation
# ---------------------------------------------------------------------------


async def _prepare_messages_turn(
    app: FastAPI,
    request: Request,
    payload: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    *,
    route: str,
) -> PreparedTurn:
    """Resolve session + inject memory into a messages-style chat request."""
    service = await _ensure_service(app)
    model = _model_name(service, payload)
    client = _client_label(request, route=route)
    raw_messages = [dict(message) for message in messages if isinstance(message, dict)]
    turn = PreparedTurn(
        route=route,
        model=model,
        client=client,
        user_text=_last_user_text(raw_messages),
        messages=raw_messages,
        session_header=_requested_session_id(request),
    )
    if not _is_extraction_model(model):
        turn.session = await _resolve_session(service, request, model=model, client=client)

    if not _injection_allowed(app, request, payload, model):
        turn.messages = raw_messages
        return turn

    # Drop a stale block echoed by the client so the prompt cannot grow forever.
    cleaned = _strip_memory_blocks(raw_messages)
    block = await _memory_block(service, turn.user_text, turn.session)
    if block:
        cleaned = _flatten_system_contents(cleaned)
        cleaned = service.augment_messages(cleaned, block)
        turn.injected = True
        turn.memory_chars = len(block)
    turn.messages = cleaned
    return turn


async def _prepare_generate_turn(
    app: FastAPI,
    request: Request,
    payload: dict[str, Any],
) -> PreparedTurn:
    """Resolve session + inject memory into an Ollama ``/api/generate`` request."""
    service = await _ensure_service(app)
    model = _model_name(service, payload)
    client = _client_label(request, route="ollama")
    prompt_value = payload.get("prompt")
    prompt = prompt_value if isinstance(prompt_value, str) else _content_to_text(prompt_value)
    system_value = payload.get("system")
    system_text = system_value if isinstance(system_value, str) else _content_to_text(system_value)

    turn = PreparedTurn(
        route="ollama",
        model=model,
        client=client,
        user_text=prompt.strip(),
        session_header=_requested_session_id(request),
        system=system_text,
    )
    if not _is_extraction_model(model):
        turn.session = await _resolve_session(service, request, model=model, client=client)

    if not _injection_allowed(app, request, payload, model):
        return turn

    cleaned_system = _strip_memory_block(system_text)
    block = await _memory_block(service, turn.user_text, turn.session)
    if block:
        turn.system = f"{cleaned_system}\n\n{block}".strip() if cleaned_system else block
        turn.injected = True
        turn.memory_chars = len(block)
    else:
        turn.system = cleaned_system
    turn.system_changed = turn.system != system_text
    return turn


# ---------------------------------------------------------------------------
# NDJSON / SSE streaming utilities
# ---------------------------------------------------------------------------


class _NdjsonDecoder:
    """Incremental UTF-8 + newline decoder for Ollama's NDJSON streams."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""

    def feed(self, raw: bytes) -> list[dict[str, Any]]:
        """Decode a raw chunk into any complete NDJSON objects it contains."""
        self._buffer += self._decoder.decode(raw)
        return self._drain()

    def flush(self) -> list[dict[str, Any]]:
        """Flush the decoder at end-of-stream and return the final objects."""
        self._buffer += self._decoder.decode(b"", final=True)
        return self._drain()

    def _drain(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        while True:
            index = self._buffer.find("\n")
            if index < 0:
                break
            line, self._buffer = self._buffer[:index], self._buffer[index + 1 :]
            chunk = _parse_ndjson_line(line)
            if chunk is not None:
                chunks.append(chunk)
        return chunks


def _parse_ndjson_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        LOG.debug("Skipping malformed upstream stream line: %r", text[:120])
        return None
    return data if isinstance(data, dict) else None


def _ollama_chunk_text(chunk: dict[str, Any], field_name: str) -> str:
    """Extract incremental assistant text from an Ollama stream chunk."""
    if field_name == "response":
        value = chunk.get("response")
        return value if isinstance(value, str) else ""
    message = chunk.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _ollama_chunk_tool_calls(chunk: dict[str, Any]) -> list[Any]:
    message = chunk.get("message")
    if isinstance(message, dict):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            return calls
    return []


def _ollama_response_tool_calls(data: dict[str, Any] | None) -> list[Any]:
    if not isinstance(data, dict):
        return []
    message = data.get("message")
    if isinstance(message, dict):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            return calls
    return []


def _sse(payload: dict[str, Any]) -> bytes:
    """Serialize one OpenAI SSE ``data:`` event."""
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def _openai_tool_calls(raw_calls: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """Convert Ollama tool calls into the OpenAI shape."""
    if not raw_calls:
        return None
    converted: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            function = call
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments = "{}"
        converted.append(
            {
                "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {"name": str(name or ""), "arguments": arguments},
                "index": index,
            }
        )
    return converted or None


def _openai_finish_reason(chunk: dict[str, Any], *, has_tool_calls: bool = False) -> str:
    """Map an Ollama ``done_reason`` onto the OpenAI finish-reason vocabulary."""
    if has_tool_calls:
        return "tool_calls"
    reason = str(chunk.get("done_reason") or "").lower()
    if reason in {"tool_calls", "tool_call"}:
        return "tool_calls"
    if reason in {"length", "max_tokens", "limit"}:
        return "length"
    return "stop"


def _openai_usage(chunk: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
    completion_tokens = int(chunk.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def openai_response_from_ollama(
    data: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Convert a non-streaming Ollama ``/api/chat`` response to OpenAI shape."""
    message = data.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    tool_calls = _openai_tool_calls(message.get("tool_calls"))
    message_out: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message_out["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(data.get("model") or model or ""),
        "choices": [
            {
                "index": 0,
                "message": message_out,
                "finish_reason": _openai_finish_reason(data, has_tool_calls=bool(tool_calls)),
            }
        ],
        "usage": _openai_usage(data),
    }


class _OpenAISseStream:
    """Incremental converter from Ollama NDJSON chunks to OpenAI SSE events."""

    def __init__(self, *, model: str) -> None:
        self.completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = model
        self.role_sent = False
        self.finished = False
        self.usage: dict[str, int] | None = None

    def _chunk(
        self,
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return _sse(payload)

    def start(self) -> bytes:
        self.role_sent = True
        return self._chunk({"role": "assistant", "content": ""})

    def feed(self, chunk: dict[str, Any]) -> Iterator[bytes]:
        """Translate one Ollama chunk into zero or more SSE events."""
        if chunk.get("error"):
            yield _sse(
                {
                    "error": {
                        "message": str(chunk.get("error")),
                        "type": "upstream_error",
                    }
                }
            )
            return
        message = chunk.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        calls = message.get("tool_calls")
        tool_calls = calls if isinstance(calls, list) else []
        delta: dict[str, Any] = {}
        if content:
            delta["content"] = content
        if tool_calls:
            converted = _openai_tool_calls(tool_calls)
            if converted:
                delta["tool_calls"] = converted
        if chunk.get("done"):
            self.finished = True
            self.usage = _openai_usage(chunk)
            yield self._chunk(
                delta,
                finish_reason=_openai_finish_reason(chunk, has_tool_calls=bool(tool_calls)),
                usage=self.usage,
            )
            return
        if delta:
            yield self._chunk(delta)

    def finish(self) -> bytes:
        """Emit a terminal chunk when the upstream stream ended without ``done``."""
        if self.finished:
            return b""
        self.finished = True
        return self._chunk({}, finish_reason="stop", usage=self.usage)


# ---------------------------------------------------------------------------
# Upstream streaming / buffered helpers
# ---------------------------------------------------------------------------


def _decorate_response(response: Response, turn: PreparedTurn) -> Response:
    """Add Amnesia diagnostic/session headers to a proxy response."""
    try:
        if turn.session is not None:
            response.headers["X-Amnesia-Session"] = turn.session.id
        response.headers["X-Amnesia-Memory-Injected"] = "1" if turn.injected else "0"
        if turn.memory_chars:
            response.headers["X-Amnesia-Memory-Chars"] = str(turn.memory_chars)
    except Exception:  # noqa: BLE001 - headers are cosmetic
        LOG.debug("Could not decorate the proxy response", exc_info=True)
    return response


async def _stream_upstream(
    app: FastAPI,
    *,
    path: str,
    payload: dict[str, Any],
    turn: PreparedTurn,
    text_field: str,
    media_type: str,
) -> Response:
    """Forward a streaming Ollama request byte-for-byte, persisting the turn."""
    client = _upstream(app)
    try:
        upstream_request = client.build_request("POST", _upstream_url(app, path), json=payload)
        response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(app, exc)

    if response.status_code >= 400:
        return await _upstream_failure(app, response)

    headers = _forward_headers(response)
    content_type = _split_content_type(response.headers.get("content-type"), media_type)

    async def body() -> AsyncIterator[bytes]:
        decoder = _NdjsonDecoder()
        text_parts: list[str] = []
        tool_calls: list[Any] = []

        def consume(chunks: list[dict[str, Any]]) -> None:
            for chunk in chunks:
                if chunk.get("error"):
                    LOG.warning("Ollama stream reported an error: %s", chunk.get("error"))
                piece = _ollama_chunk_text(chunk, text_field)
                if piece:
                    text_parts.append(piece)
                tool_calls.extend(_ollama_chunk_tool_calls(chunk))

        try:
            async for raw in _iter_upstream_bytes(response):
                consume(decoder.feed(raw))
                yield raw
            consume(decoder.flush())
        finally:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001 - closing is best effort
                LOG.debug("Closing upstream stream failed", exc_info=True)
            await _safe_finish_turn(app, turn, "".join(text_parts), tool_calls=tool_calls)

    return StreamingResponse(
        body(),
        status_code=response.status_code,
        media_type=content_type,
        headers=headers,
    )


async def _json_upstream(
    app: FastAPI,
    *,
    path: str,
    payload: dict[str, Any],
    turn: PreparedTurn,
    text_extractor: Callable[[dict[str, Any]], str],
) -> Response:
    """Forward a non-streaming Ollama request, persisting the turn."""
    client = _upstream(app)
    try:
        response = await client.post(_upstream_url(app, path), json=payload)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(app, exc)

    if response.status_code >= 400:
        return await _upstream_failure(app, response)

    data = _response_json(response)
    if data is not None:
        assistant_text = str(text_extractor(data) or "")
        await _safe_finish_turn(app, turn, assistant_text, tool_calls=_ollama_response_tool_calls(data))
    return _passthrough_response(response)


def _generate_response_text(data: dict[str, Any]) -> str:
    value = data.get("response")
    return value if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# OpenAI request conversion
# ---------------------------------------------------------------------------


_OLLAMA_OPTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("seed", "seed"),
    ("stop", "stop"),
    ("max_tokens", "num_predict"),
    ("num_predict", "num_predict"),
    ("frequency_penalty", "frequency_penalty"),
    ("presence_penalty", "presence_penalty"),
    ("repeat_penalty", "repeat_penalty"),
)


def _openai_to_ollama_payload(
    payload: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    *,
    model: str,
    stream: bool,
) -> dict[str, Any]:
    """Translate an OpenAI chat-completions body into an Ollama chat payload."""
    ollama_payload: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(messages),
        "stream": bool(stream),
    }
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        ollama_payload["tools"] = tools
    options: dict[str, Any] = {}
    for source, target in _OLLAMA_OPTION_FIELDS:
        value = payload.get(source)
        if value is not None:
            options[target] = value
    if options:
        ollama_payload["options"] = options
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        format_type = str(response_format.get("type") or "").lower()
        if format_type == "json_object":
            ollama_payload["format"] = "json"
        elif format_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict) and isinstance(json_schema.get("schema"), dict):
                ollama_payload["format"] = json_schema["schema"]
    keep_alive = payload.get("keep_alive")
    if keep_alive is not None:
        ollama_payload["keep_alive"] = keep_alive
    return ollama_payload


async def _openai_json_response(
    app: FastAPI,
    *,
    payload: dict[str, Any],
    turn: PreparedTurn,
) -> Response:
    """Non-streaming ``/v1/chat/completions`` implementation."""
    client = _upstream(app)
    try:
        response = await client.post(_upstream_url(app, "/api/chat"), json=payload)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(app, exc)
    if response.status_code >= 400:
        return await _upstream_failure(app, response, openai=True)

    data = _response_json(response)
    if data is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Ollama returned a non-JSON response for /api/chat",
                    "type": "upstream_error",
                }
            },
        )
    message = data.get("message")
    assistant_text = ""
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        assistant_text = message["content"]
    await _safe_finish_turn(
        app,
        turn,
        assistant_text,
        tool_calls=_ollama_response_tool_calls(data),
    )
    return JSONResponse(content=openai_response_from_ollama(data, model=turn.model), status_code=200)


async def _openai_stream_response(
    app: FastAPI,
    *,
    payload: dict[str, Any],
    turn: PreparedTurn,
) -> Response:
    """Streaming ``/v1/chat/completions`` implementation (SSE)."""
    client = _upstream(app)
    try:
        upstream_request = client.build_request(
            "POST", _upstream_url(app, "/api/chat"), json=payload
        )
        response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(app, exc)
    if response.status_code >= 400:
        return await _upstream_failure(app, response, openai=True)

    headers = _forward_headers(response)
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"
    converter = _OpenAISseStream(model=turn.model)

    async def body() -> AsyncIterator[bytes]:
        decoder = _NdjsonDecoder()
        text_parts: list[str] = []
        tool_calls: list[Any] = []
        try:
            yield converter.start()
            async for raw in _iter_upstream_bytes(response):
                for chunk in decoder.feed(raw):
                    message = chunk.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            text_parts.append(content)
                        calls = message.get("tool_calls")
                        if isinstance(calls, list):
                            tool_calls.extend(calls)
                    for event in converter.feed(chunk):
                        yield event
            for chunk in decoder.flush():
                for event in converter.feed(chunk):
                    yield event
            trailing = converter.finish()
            if trailing:
                yield trailing
        except httpx.HTTPError as exc:  # pragma: no cover - mid-stream socket error
            LOG.warning("OpenAI stream aborted by an upstream error: %s", exc)
            yield _sse({"error": {"message": str(exc), "type": "upstream_unreachable"}})
        finally:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001 - closing is best effort
                LOG.debug("Closing upstream OpenAI stream failed", exc_info=True)
            await _safe_finish_turn(app, turn, "".join(text_parts), tool_calls=tool_calls)
        yield SSE_DONE

    return StreamingResponse(
        body(),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def _is_public_amnesia_path(path: str) -> bool:
    if path in PUBLIC_AMNESIA_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_AMNESIA_PREFIXES)


def _supplied_token(request: Request) -> str | None:
    """Extract a caller token from headers or a ``?token=`` query parameter."""
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()
    header_token = request.headers.get("x-amnesia-token")
    if header_token and header_token.strip():
        return header_token.strip()
    query_token = request.query_params.get("token")
    if query_token and query_token.strip():
        return query_token.strip()
    return None


def _tokens_match(supplied: str, expected: str) -> bool:
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except Exception:  # noqa: BLE001 - never fail auth because of encoding
        return False


def _install_auth_middleware(app: FastAPI) -> None:
    """Require a bearer token on ``/amnesia/*`` when ``settings.api_token`` is set."""

    @app.middleware("http")
    async def amnesia_auth(request: Request, call_next):  # type: ignore[no-untyped-def]
        settings = getattr(request.app.state, "settings", None)
        token = getattr(settings, "api_token", None)
        path = request.url.path
        if token and path.startswith("/amnesia") and not _is_public_amnesia_path(path):
            supplied = _supplied_token(request)
            if not supplied or not _tokens_match(supplied, str(token)):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": (
                                "Unauthorized. Pass `Authorization: Bearer <token>` "
                                "or `X-Amnesia-Token: <token>` (or ?token=<token>)."
                            ),
                            "type": "unauthorized",
                        }
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Ollama-native passthrough
# ---------------------------------------------------------------------------


async def _passthrough_request(request: Request) -> Response:
    """Transparently proxy an Ollama endpoint (streaming-safe)."""
    app = request.app
    client = _upstream(app)
    body: bytes | None = None
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        body = await request.body()
    headers: dict[str, str] | None = None
    if body:
        headers = {"content-type": request.headers.get("content-type", "application/json")}
    params = dict(request.query_params) or None
    try:
        upstream_request = client.build_request(
            request.method,
            _upstream_url(app, request.url.path),
            params=params,
            content=body,
            headers=headers,
        )
        response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(app, exc)

    if response.status_code >= 400:
        return await _upstream_failure(app, response)

    content_type = _split_content_type(response.headers.get("content-type"), "application/json")
    response_headers = _forward_headers(response)

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in _iter_upstream_bytes(response):
                yield chunk
        finally:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001 - closing is best effort
                LOG.debug("Closing passthrough stream failed", exc_info=True)

    return StreamingResponse(
        stream_body(),
        status_code=response.status_code,
        media_type=content_type,
        headers=response_headers,
    )


# ---------------------------------------------------------------------------
# Route registration: Ollama + OpenAI surfaces
# ---------------------------------------------------------------------------


def _register_proxy_routes(app: FastAPI) -> None:
    """Register the Ollama-native and OpenAI-compatible routes."""

    @app.post("/api/chat")
    async def api_chat(
        request: Request,
        service: Amnesia = Depends(_service_dependency),
    ) -> Response:
        """Ollama-native chat completion with memory injection and capture."""
        payload = await _json_body(request)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise HTTPException(
                status_code=400,
                detail="Request body must contain a non-empty 'messages' array",
            )
        messages = [dict(item) for item in raw_messages if isinstance(item, dict)]
        turn = await _prepare_messages_turn(
            request.app, request, payload, messages, route="ollama"
        )
        forwarded = {**payload, "messages": turn.messages}
        stream = bool(payload.get("stream", True))
        if stream:
            response = await _stream_upstream(
                request.app,
                path="/api/chat",
                payload=forwarded,
                turn=turn,
                text_field="message",
                media_type="application/x-ndjson",
            )
        else:
            response = await _json_upstream(
                request.app,
                path="/api/chat",
                payload=forwarded,
                turn=turn,
                text_extractor=_chat_message_text,
            )
        return _decorate_response(response, turn)

    @app.post("/api/generate")
    async def api_generate(
        request: Request,
        service: Amnesia = Depends(_service_dependency),
    ) -> Response:
        """Ollama-native text completion with memory injection and capture."""
        payload = await _json_body(request)
        turn = await _prepare_generate_turn(request.app, request, payload)
        forwarded = dict(payload)
        if turn.system_changed:
            forwarded["system"] = turn.system or ""
        stream = bool(payload.get("stream", True))
        if stream:
            response = await _stream_upstream(
                request.app,
                path="/api/generate",
                payload=forwarded,
                turn=turn,
                text_field="response",
                media_type="application/x-ndjson",
            )
        else:
            response = await _json_upstream(
                request.app,
                path="/api/generate",
                payload=forwarded,
                turn=turn,
                text_extractor=_generate_response_text,
            )
        return _decorate_response(response, turn)

    for method, path in _OLLAMA_PASSTHROUGH_ROUTES:
        app.add_api_route(
            path,
            _passthrough_request,
            methods=[method],
            include_in_schema=False,
            name=f"ollama-passthrough-{method.lower()}-{path.strip('/').replace('/', '-')}",
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        request: Request,
        service: Amnesia = Depends(_service_dependency),
    ) -> Response:
        """OpenAI-compatible chat completions (SSE streaming and non-streaming)."""
        payload = await _json_body(request)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise HTTPException(
                status_code=400,
                detail="Request body must contain a non-empty 'messages' array",
            )
        messages = [dict(item) for item in raw_messages if isinstance(item, dict)]
        turn = await _prepare_messages_turn(
            request.app, request, payload, messages, route="openai"
        )
        stream = bool(payload.get("stream", False))
        ollama_payload = _openai_to_ollama_payload(
            payload,
            turn.messages,
            model=turn.model or str(service.chat_model or ""),
            stream=stream,
        )
        if stream:
            response = await _openai_stream_response(
                request.app, payload=ollama_payload, turn=turn
            )
        else:
            response = await _openai_json_response(
                request.app, payload=ollama_payload, turn=turn
            )
        return _decorate_response(response, turn)

    @app.get("/v1/models")
    async def openai_models(request: Request) -> Response:
        """List local Ollama models in the OpenAI ``/v1/models`` shape."""
        app_ = request.app
        client = _upstream(app_)
        try:
            response = await client.get(_upstream_url(app_, "/api/tags"))
        except httpx.HTTPError as exc:
            return _upstream_unreachable(app_, exc)
        if response.status_code >= 400:
            return await _upstream_failure(app_, response, openai=True)
        data = _response_json(response) or {}
        created = int(time.time())
        models: list[dict[str, Any]] = []
        raw_models = data.get("models")
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("model")
                if not name:
                    continue
                models.append(
                    {"id": str(name), "object": "model", "created": created, "owned_by": "local"}
                )
        return JSONResponse({"object": "list", "data": models})

    @app.post("/v1/embeddings")
    async def openai_embeddings(
        request: Request,
        service: Amnesia = Depends(_service_dependency),
    ) -> Response:
        """OpenAI-compatible embeddings via Ollama ``/api/embed``."""
        payload = await _json_body(request)
        model = payload.get("model") or str(getattr(service.settings, "embed_model", "") or "")
        raw_input = payload.get("input")
        if raw_input is None:
            raise HTTPException(status_code=400, detail="Missing required field 'input'")
        if isinstance(raw_input, str):
            inputs: list[str] = [raw_input]
            send_value: str | list[str] = raw_input
        elif isinstance(raw_input, list):
            if not all(isinstance(item, str) for item in raw_input):
                raise HTTPException(
                    status_code=400,
                    detail="Only string inputs are supported for embeddings",
                )
            inputs = [str(item) for item in raw_input]
            send_value = inputs
        else:
            raise HTTPException(
                status_code=400,
                detail="'input' must be a string or a list of strings",
            )
        if not inputs:
            raise HTTPException(status_code=400, detail="'input' must not be empty")

        app_ = request.app
        client = _upstream(app_)
        try:
            response = await client.post(
                _upstream_url(app_, "/api/embed"),
                json={"model": model, "input": send_value, "truncate": True},
            )
        except httpx.HTTPError as exc:
            return _upstream_unreachable(app_, exc)
        if response.status_code >= 400:
            return await _upstream_failure(app_, response, openai=True)
        data = _response_json(response)
        if data is None:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "Ollama returned a non-JSON embeddings response",
                        "type": "upstream_error",
                    }
                },
            )
        raw_vectors = data.get("embeddings")
        vectors = raw_vectors if isinstance(raw_vectors, list) else []
        prompt_tokens = sum(estimate_tokens(text) for text in inputs)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "model": str(data.get("model") or model),
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
        )


def _chat_message_text(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


# ---------------------------------------------------------------------------
# Route registration: memory API + dashboard
# ---------------------------------------------------------------------------


def _clamp_limit(value: int, *, default: int = 50, maximum: int = 500) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, number))


def _validate_kind(value: str, allowed: Sequence[str], label: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label} {value!r}; expected one of: {', '.join(allowed)}",
        )
    return normalized


def _register_memory_routes(app: FastAPI) -> None:
    """Register the ``/amnesia/*`` JSON API and the dashboard page."""

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/amnesia/ui", status_code=307)

    @app.head("/", include_in_schema=False)
    async def root_head() -> Response:
        """Heartbeat used by the Ollama CLI before it starts chatting.

        ``ollama run`` probes ``HEAD /`` and aborts with "something went wrong"
        unless the server answers 2xx, so pointing OLLAMA_HOST at the proxy
        depends on this route.
        """
        return Response(status_code=200)

    @app.get("/amnesia", include_in_schema=False)
    async def amnesia_redirect() -> RedirectResponse:
        return RedirectResponse(url="/amnesia/ui", status_code=307)

    @app.get("/amnesia/health")
    async def amnesia_health(service: Amnesia = Depends(_service_dependency)) -> JSONResponse:
        """Canonical health payload (mirrors ``service.health()``)."""
        return JSONResponse(content=service.health())

    @app.post("/amnesia/shutdown")
    async def amnesia_shutdown(request: Request) -> JSONResponse:
        """Ask the running serve loop to shut down gracefully (loopback only).

        The CLI ``amnesia serve`` loop polls ``app.state.shutdown_requested``
        and flips uvicorn's ``server.should_exit`` so the server drains and
        stops without killing in-flight work. Shutdown is restricted to
        loopback callers: exposing it on a non-loopback interface would let
        remote clients stop the daemon.
        """
        client = request.client
        host = (client.host if client else "") or ""
        if host not in {"127.0.0.1", "::1", "localhost"} and not host.startswith("127."):
            LOG.warning("Rejected shutdown request from non-loopback client %r", host)
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": "Shutdown is only allowed from loopback clients",
                        "type": "forbidden",
                    }
                },
            )
        if not getattr(request.app.state, "shutdown_requested", False):
            LOG.info("Graceful shutdown requested via HTTP API (client=%s)", host)
        request.app.state.shutdown_requested = True
        return JSONResponse({"status": "shutting_down"})

    @app.get("/amnesia/profile")
    async def amnesia_profile(
        limit: int = Query(default=50, ge=1, le=500, description="Max identity facts"),
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Identity memory (who the user is)."""
        facts = service.profile(limit=_clamp_limit(limit))
        return JSONResponse(
            {
                "facts": [memory.to_dict() for memory in facts],
                "count": len(facts),
                "text": "\n".join(memory.display_line() for memory in facts),
            }
        )

    @app.get("/amnesia/recall")
    async def amnesia_recall(
        q: str = Query(default="", description="Search query"),
        limit: int = Query(default=10, ge=1, le=100),
        include_events: bool = Query(default=True),
        include_messages: bool = Query(default=False),
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Hybrid keyword + semantic recall across memories and events."""
        query = (q or "").strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Query parameter 'q' is required and cannot be empty"
            )
        hits = await service.recall(
            query,
            limit=_clamp_limit(limit, default=10, maximum=100),
            include_events=include_events,
            include_messages=include_messages,
        )
        return JSONResponse(
            {
                "query": query,
                "hits": [hit.to_dict() for hit in hits],
                "count": len(hits),
            }
        )

    @app.post("/amnesia/remember")
    async def amnesia_remember(
        body: RememberRequest,
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Store a durable memory (identity fact, preference or note)."""
        content = (body.content or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="'content' must not be empty")
        kind = _validate_kind(body.kind, VALID_MEMORY_KINDS, "memory kind")
        try:
            memory = await service.remember(
                content,
                kind=kind,
                key=body.key,
                subject=body.subject or "user",
                importance=body.importance,
                confidence=body.confidence,
                tags=coerce_tags(body.tags),
                source=body.source or "manual",
                session_id=body.session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "memory": memory.to_dict()})

    @app.delete("/amnesia/memories/{memory_id}")
    async def amnesia_forget(
        memory_id: int,
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Deactivate (soft-forget) one memory."""
        existing = service.store.get_memory(memory_id)
        if existing is None or not existing.active:
            raise HTTPException(status_code=404, detail=f"Memory #{memory_id} not found")
        deleted = await service.forget(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory #{memory_id} not found")
        return JSONResponse({"ok": True, "id": memory_id, "deleted": True})

    @app.get("/amnesia/events")
    async def amnesia_events(
        days: int = Query(default=7, ge=0, le=3650),
        limit: int = Query(default=50, ge=1, le=500),
        project: str | None = Query(default=None),
        kind: str | None = Query(default=None),
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Activity timeline events, newest first (``days=0`` means all time)."""
        events = service.timeline(
            days=days or None,
            limit=_clamp_limit(limit),
            project=project or None,
            kind=kind or None,
        )
        return JSONResponse(
            {"events": [event.to_dict() for event in events], "count": len(events), "days": days}
        )

    @app.post("/amnesia/events")
    async def amnesia_log_event(
        body: EventRequest,
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Record one activity event."""
        title = (body.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="'title' must not be empty")
        kind = _validate_kind(body.kind, VALID_EVENT_KINDS, "event kind")
        try:
            event = await service.log_event(
                title,
                kind=kind,
                details=body.details,
                project=body.project,
                tags=coerce_tags(body.tags),
                importance=body.importance,
                occurred_at=body.occurred_at,
                session_id=body.session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "event": event.to_dict()})

    @app.get("/amnesia/sessions")
    async def amnesia_sessions(
        limit: int = Query(default=20, ge=1, le=200),
        model: str | None = Query(default=None),
        client: str | None = Query(default=None),
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """Recent conversation sessions."""
        sessions = service.store.list_sessions(
            limit=_clamp_limit(limit, default=20, maximum=200),
            model=model or None,
            client=client or None,
        )
        return JSONResponse(
            {"sessions": [session.to_dict() for session in sessions], "count": len(sessions)}
        )

    @app.get("/amnesia/sessions/{session_id}")
    async def amnesia_session_detail(
        session_id: str,
        service: Amnesia = Depends(_service_dependency),
    ) -> JSONResponse:
        """One session plus its transcript."""
        session = service.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
        messages = service.store.session_messages(session_id)
        return JSONResponse(
            {
                "session": session.to_dict(),
                "messages": [message.to_dict() for message in messages],
                "count": len(messages),
            }
        )

    @app.get("/amnesia/stats")
    async def amnesia_stats(service: Amnesia = Depends(_service_dependency)) -> JSONResponse:
        """Aggregate store statistics for the dashboard cards."""
        return JSONResponse(content=service.stats().to_dict())

    @app.get("/amnesia/export")
    async def amnesia_export(service: Amnesia = Depends(_service_dependency)) -> Response:
        """Download a full JSON export (sessions, messages, memories, events)."""
        payload = service.store.export_data()
        filename = f"amnesia-export-{time.strftime('%Y%m%d-%H%M%S')}.json"
        body = json.dumps(payload, indent=2, ensure_ascii=False)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/amnesia/ui", response_class=HTMLResponse)
    async def amnesia_ui() -> Response:
        """Serve the self-contained web dashboard."""
        if DASHBOARD_FILE.is_file():
            return FileResponse(
                DASHBOARD_FILE,
                media_type="text/html; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        LOG.error("Dashboard asset is missing: %s", DASHBOARD_FILE)
        return HTMLResponse(
            content=(
                "<!doctype html><html><head><title>Amnesia</title></head><body>"
                "<h1>Amnesia dashboard unavailable</h1>"
                f"<p>Missing dashboard asset at <code>{DASHBOARD_FILE}</code>.</p>"
                "<p>The JSON memory API is still available under <code>/amnesia/</code>.</p>"
                "</body></html>"
            ),
            status_code=503,
        )

    if STATIC_DIR.is_dir():
        app.mount(
            "/amnesia/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="amnesia-static",
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the service, extraction scheduler and upstream client."""
    service = getattr(app.state, "service", None)
    if service is None:
        await _ensure_service(app)
    else:
        if not getattr(service, "started", False):
            await service.start()
        await _install_extraction(app, service)
    LOG.info(
        "Amnesia proxy ready: Ollama=%s, service=%s, extraction=%s",
        app.state.settings.ollama_url,
        "ready" if getattr(service, "started", False) else "external",
        "on" if getattr(app.state, "extraction_scheduler", None) else "off",
    )
    try:
        yield
    finally:
        scheduler = getattr(app.state, "extraction_scheduler", None)
        if getattr(app.state, "owns_scheduler", False) and scheduler is not None:
            try:
                await _maybe_await(scheduler.close())
                LOG.info("Extraction scheduler closed")
            except Exception:  # noqa: BLE001 - shutdown is best effort
                LOG.warning("Closing the extraction scheduler failed", exc_info=True)
        upstream = getattr(app.state, "upstream_client", None)
        if getattr(app.state, "owns_upstream", False) and upstream is not None:
            try:
                await upstream.aclose()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                LOG.debug("Closing the upstream client failed", exc_info=True)
        service = getattr(app.state, "service", None)
        if getattr(app.state, "owns_service", False) and service is not None:
            try:
                await service.close()
                LOG.info("Amnesia service closed")
            except Exception:  # noqa: BLE001 - shutdown is best effort
                LOG.warning("Closing the Amnesia service failed", exc_info=True)


def create_app(
    service: Amnesia | None = None,
    *,
    settings: Settings | None = None,
    upstream_client: httpx.AsyncClient | None = None,
    extraction_scheduler: Any = None,
) -> FastAPI:
    """Build the Amnesia HTTP proxy application.

    Parameters
    ----------
    service:
        Optional already-built :class:`Amnesia` facade.  Tests pass one backed
        by a fake Ollama client; in production the app builds its own during
        startup.
    settings:
        Optional settings override.  Defaults to ``service.settings`` when a
        service is supplied, else :func:`amnesia.config.load_settings`.
    upstream_client:
        Optional ``httpx.AsyncClient`` used for every Ollama request.  Tests
        pass a ``MockTransport``-backed client; production builds its own
        client from ``settings.ollama_url``.
    extraction_scheduler:
        Optional extraction scheduler.  When omitted and
        ``settings.extraction_enabled`` is true, the app tries to build one
        from :mod:`amnesia.extraction` during startup, degrading gracefully if
        that module is unavailable.
    """
    resolved_settings = settings or (
        service.settings if service is not None else load_settings()
    )

    application = FastAPI(
        title="Amnesia Proxy",
        description=(
            "Local-first long-term memory proxy for Ollama. Ollama-native and "
            "OpenAI-compatible chat endpoints with memory injection."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    application.state.settings = resolved_settings
    application.state.service = service
    application.state.owns_service = False
    application.state.upstream_client = upstream_client
    application.state.owns_upstream = False
    application.state.extraction_scheduler = extraction_scheduler
    application.state.owns_scheduler = False
    application.state.extraction_checked = extraction_scheduler is not None
    application.state.shutdown_requested = False
    application.state.init_lock = asyncio.Lock()

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        LOG.exception("Unhandled error while serving %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Internal Amnesia proxy error: {exc}",
                    "type": "internal_error",
                }
            },
        )

    _install_auth_middleware(application)
    _register_proxy_routes(application)
    _register_memory_routes(application)
    return application


def get_app() -> FastAPI:
    """Module-level factory usable as ``uvicorn ... --factory amnesia.proxy:get_app``."""
    return create_app(settings=load_settings())


def main() -> None:
    """Run the proxy with uvicorn (``python -m amnesia.proxy``)."""
    import uvicorn

    settings = load_settings()
    host = settings.proxy_host or "127.0.0.1"
    port = int(settings.proxy_port or 11435)
    LOG.info("Starting Amnesia proxy on http://%s:%s (dashboard: /amnesia/ui)", host, port)
    uvicorn.run(create_app(settings=settings), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()


__all__ = [
    "create_app",
    "get_app",
    "main",
    "openai_response_from_ollama",
]
