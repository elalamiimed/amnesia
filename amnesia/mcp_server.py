"""MCP (Model Context Protocol) stdio server exposing Amnesia memory.

MCP-capable clients (Claude Desktop, Cursor, Continue, ...) launch this process
and call the tools below to search and write the user's local long-term memory.
Everything stays on the machine: the tools talk to the same SQLite store the
CLI and the HTTP proxy use.

The protocol owns ``stdout``, therefore *nothing* but JSON-RPC may be written
there. All diagnostics go to ``stderr``.

Run it with::

    amnesia mcp

The tool implementations are plain async functions (``memory_search``,
``memory_remember``, ``memory_profile``, ``memory_log_event``,
``memory_timeline``, ``memory_stats``) so they can be imported and tested
directly; :func:`build_server` only wraps them in a ``FastMCP`` instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from .config import Settings, load_settings
from .models import ALL_MEMORY_KINDS
from .service import Amnesia
from .utils import run_sync

LOG = logging.getLogger("amnesia.mcp")

try:  # The MCP SDK is an optional dependency (``pip install amnesia[mcp]``).
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FastMCP = None  # type: ignore[assignment]
    MCP_AVAILABLE = False

# Tools exposed over MCP; also used by tests and docs.
TOOL_NAMES: tuple[str, ...] = (
    "memory_search",
    "memory_remember",
    "memory_profile",
    "memory_log_event",
    "memory_timeline",
    "memory_stats",
)

MAX_LIMIT = 100

# The service a running server uses. Tools called without an explicit
# ``service=`` argument fall back to it, which keeps the MCP wiring simple.
_default_service: Amnesia | None = None


def set_service(service: Amnesia | None) -> None:
    """Set (or clear) the process-wide service used by the MCP tools."""
    global _default_service
    _default_service = service


def get_service() -> Amnesia | None:
    """Return the process-wide service, if a server is running."""
    return _default_service


def _refresh_logging_handlers() -> None:
    """Detach ``amnesia`` log handlers bound to a closed stream.

    ``configure_logging`` installs a process-global handler once; when the
    server is launched repeatedly in the same interpreter (tests), that handler
    may point at a closed stream and emit noisy logging errors.
    """
    root = logging.getLogger("amnesia")
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is not None and getattr(stream, "closed", False):
            root.removeHandler(handler)
            with suppress(Exception):
                handler.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_limit(value: int, *, default: int = 20, maximum: int = MAX_LIMIT) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, number))


def _normalise_memory_kind(kind: str | None) -> str:
    value = (kind or "fact").strip().lower()
    if value == "truth":  # accepted alias for "fact"
        value = "fact"
    if value not in ALL_MEMORY_KINDS:
        raise ValueError(
            f"Unknown memory kind {kind!r}; expected one of {', '.join(ALL_MEMORY_KINDS)}."
        )
    return value


@asynccontextmanager
async def _resolve_service(service: Amnesia | None = None) -> AsyncIterator[Amnesia]:
    """Yield a usable service, starting (and closing) a private one if needed."""
    target = service or _default_service
    if target is not None:
        yield target
        return
    created = Amnesia(load_settings())
    await created.start()
    try:
        yield created
    finally:
        with suppress(Exception):
            await created.close()


def _stats_payload(stats: Any) -> dict[str, Any]:
    payload = stats.to_dict()
    payload["db_size_human"] = payload.get("db_size_human", "")
    return payload


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def memory_search(
    query: str,
    limit: int = 8,
    include_events: bool = True,
    *,
    service: Amnesia | None = None,
) -> dict[str, Any]:
    """Hybrid keyword + semantic search over stored memories and events."""
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string")
    capped = _clamp_limit(limit, default=8)
    async with _resolve_service(service) as svc:
        hits = await svc.recall(cleaned, limit=capped, include_events=bool(include_events))
    return {
        "query": cleaned,
        "count": len(hits),
        "hits": [hit.to_dict() for hit in hits],
    }


async def memory_remember(
    content: str,
    kind: str = "fact",
    key: str | None = None,
    importance: float = 0.6,
    *,
    service: Amnesia | None = None,
) -> dict[str, Any]:
    """Store a durable fact / preference / note about the user."""
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("content must be a non-empty string")
    resolved_kind = _normalise_memory_kind(kind)
    try:
        resolved_importance = max(0.0, min(1.0, float(importance)))
    except (TypeError, ValueError):
        resolved_importance = 0.6
    async with _resolve_service(service) as svc:
        memory = await svc.remember(
            cleaned,
            kind=resolved_kind,
            key=(key or "").strip() or None,
            importance=resolved_importance,
            source="mcp",
        )
    return {"memory": memory.to_dict()}


async def memory_profile(
    limit: int = 30,
    *,
    service: Amnesia | None = None,
) -> dict[str, Any]:
    """Return the identity facts Amnesia knows about the user."""
    capped = _clamp_limit(limit, default=30)
    async with _resolve_service(service) as svc:
        facts = await asyncio.to_thread(svc.profile, limit=capped)
    return {"count": len(facts), "facts": [fact.to_dict() for fact in facts]}


async def memory_log_event(
    title: str,
    kind: str = "note",
    details: str | None = None,
    project: str | None = None,
    *,
    service: Amnesia | None = None,
) -> dict[str, Any]:
    """Record an activity event (note, task, decision, learning, ...) on the timeline."""
    cleaned = (title or "").strip()
    if not cleaned:
        raise ValueError("title must be a non-empty string")
    async with _resolve_service(service) as svc:
        event = await svc.log_event(
            cleaned,
            kind=kind or "note",
            details=(details or "").strip() or None,
            project=(project or "").strip() or None,
            source="mcp",
        )
    return {"event": event.to_dict()}


async def memory_timeline(
    days: int = 7,
    limit: int = 20,
    *,
    service: Amnesia | None = None,
) -> dict[str, Any]:
    """Return recent activity events, newest first."""
    try:
        resolved_days = max(1, min(3650, int(days)))
    except (TypeError, ValueError):
        resolved_days = 7
    capped = _clamp_limit(limit, default=20)
    async with _resolve_service(service) as svc:
        events = await asyncio.to_thread(
            svc.timeline, days=resolved_days, limit=capped
        )
    return {
        "days": resolved_days,
        "count": len(events),
        "events": [event.to_dict() for event in events],
    }


async def memory_stats(*, service: Amnesia | None = None) -> dict[str, Any]:
    """Return counts, storage size and model/embedding status."""
    async with _resolve_service(service) as svc:
        stats = await asyncio.to_thread(svc.stats)
        payload = _stats_payload(stats)
        payload["memory_enabled"] = bool(svc.settings.memory_enabled)
        payload["ollama_available"] = svc.ollama_available
    return payload


# ---------------------------------------------------------------------------
# FastMCP wiring
# ---------------------------------------------------------------------------


def build_server(service: Amnesia | None = None) -> Any:
    """Create a :class:`FastMCP` server with all Amnesia tools registered.

    ``service`` is captured by the tools so the server never opens a second
    connection pool; when omitted, tools start a short-lived service per call.
    """
    if not MCP_AVAILABLE or FastMCP is None:  # pragma: no cover - guard only
        raise RuntimeError(
            "The 'mcp' package is not installed. Install Amnesia with the MCP "
            'extra: pip install "amnesia[mcp]"'
        )

    server = FastMCP(
        "amnesia",
        instructions=(
            "Amnesia is the user's local, private long-term memory. Use "
            "memory_search / memory_profile / memory_timeline to recall who the "
            "user is and what they have been doing, and memory_remember / "
            "memory_log_event to save durable facts and activity worth keeping. "
            "Never store secrets or credentials."
        ),
    )

    @server.tool(
        name="memory_search",
        description="Search the user's long-term memory (facts, notes, events).",
    )
    async def search_tool(
        query: str,
        limit: int = 8,
        include_events: bool = True,
    ) -> dict[str, Any]:
        return await memory_search(
            query, limit=limit, include_events=include_events, service=service
        )

    @server.tool(
        name="memory_remember",
        description=(
            "Save a durable fact, identity detail, preference or note about the "
            "user (kind = fact | identity | preference | note | episode | summary)."
        ),
    )
    async def remember_tool(
        content: str,
        kind: str = "fact",
        key: str | None = None,
        importance: float = 0.6,
    ) -> dict[str, Any]:
        return await memory_remember(
            content, kind=kind, key=key, importance=importance, service=service
        )

    @server.tool(
        name="memory_profile",
        description="Return the identity facts Amnesia knows about the user.",
    )
    async def profile_tool(limit: int = 30) -> dict[str, Any]:
        return await memory_profile(limit=limit, service=service)

    @server.tool(
        name="memory_log_event",
        description=(
            "Record an activity event on the user's timeline "
            "(kind = note | task | project | decision | learning | file | other)."
        ),
    )
    async def log_event_tool(
        title: str,
        kind: str = "note",
        details: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        return await memory_log_event(
            title, kind=kind, details=details, project=project, service=service
        )

    @server.tool(
        name="memory_timeline",
        description="Return recent activity events from the user's timeline.",
    )
    async def timeline_tool(days: int = 7, limit: int = 20) -> dict[str, Any]:
        return await memory_timeline(days=days, limit=limit, service=service)

    @server.tool(
        name="memory_stats",
        description="Return Amnesia memory counts, storage size and model status.",
    )
    async def stats_tool() -> dict[str, Any]:
        return await memory_stats(service=service)

    return server


# Backwards/alternative name kept for convenience in tests and integrations.
create_server = build_server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, verbose: bool = False) -> int:
    """Run the stdio MCP server. Returns a process exit code.

    Logging is quiet (WARNING) by default so MCP clients see a clean stderr;
    pass ``verbose=True`` for INFO logs, or set ``AMNESIA_LOG_LEVEL``.
    """
    del argv  # stdio transport takes no arguments; accepted for future use
    if not MCP_AVAILABLE:  # pragma: no cover - depends on the environment
        print(
            "The MCP server needs the 'mcp' package.\n"
            'Install it with:  pip install "amnesia[mcp]"\n'
            "or:               pip install mcp",
            file=sys.stderr,
        )
        return 1

    if verbose:
        settings: Settings = load_settings(log_level="INFO")
    elif os.environ.get("AMNESIA_LOG_LEVEL", "").strip():
        settings = load_settings()
    else:
        settings = load_settings(log_level="WARNING")
    _refresh_logging_handlers()

    async def _serve() -> None:
        service = Amnesia(settings)
        await service.start()
        set_service(service)
        try:
            server = build_server(service)
            LOG.info("Amnesia MCP server ready (stdio); %s tools", len(TOOL_NAMES))
            await server.run_stdio_async()
        finally:
            set_service(None)
            scheduler = getattr(service, "extraction_scheduler", None)
            if scheduler is not None:
                for method in ("drain", "close"):
                    with suppress(Exception):
                        result = getattr(scheduler, method, None)
                        if result is not None:
                            maybe = result()
                            if asyncio.iscoroutine(maybe):
                                await maybe
            with suppress(Exception):
                await service.close()

    try:
        run_sync(_serve())
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        return 0
    except Exception as exc:  # noqa: BLE001 - stdio server must fail cleanly
        LOG.exception("Amnesia MCP server failed")
        print(f"amnesia MCP server failed: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "MCP_AVAILABLE",
    "TOOL_NAMES",
    "build_server",
    "create_server",
    "get_service",
    "main",
    "memory_log_event",
    "memory_profile",
    "memory_remember",
    "memory_search",
    "memory_stats",
    "memory_timeline",
    "set_service",
]
