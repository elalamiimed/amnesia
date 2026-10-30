"""Typer command-line interface for Amnesia.

``amnesia`` is the primary way users talk to their local models with memory:

    amnesia init --interactive     # create ~/.amnesia + onboard your profile
    amnesia chat                   # streaming REPL with memory injection
    amnesia ask "what was I doing yesterday?"
    amnesia remember "I prefer concise answers"
    amnesia recall "python vs rust"
    amnesia doctor                 # environment / database checks

Everything is local: the CLI only talks to the SQLite store in
``~/.amnesia/amnesia.db`` and to the Ollama daemon on the configured URL.

Design notes
------------
* :func:`make_service` is the single construction point for an
  :class:`amnesia.service.Amnesia`; tests monkeypatch it so no test ever talks
  to a real Ollama daemon.
* Every command starts and closes its own service (the store is cheap to open);
  long-running commands (``serve``) own the service for their lifetime.
* Expected failures surface as friendly rich messages plus a non-zero exit
  code - never a raw traceback.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, NoReturn, TypeVar

import typer
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import Settings, load_settings
from .models import (
    ALL_EVENT_KINDS,
    ALL_MEMORY_KINDS,
    Event,
    Memory,
    Message,
    RecallHit,
    Session,
    Stats,
)
from .ollama import OllamaError
from .service import Amnesia
from .utils import human_time, run_sync, truncate

LOG = logging.getLogger("amnesia.cli")

T = TypeVar("T")

# Console bound lazily to the current stdout/stderr (rich resolves the stream
# on every write, which keeps CliRunner tests honest).
console = Console(highlight=False, soft_wrap=True)
err_console = Console(stderr=True, highlight=False, soft_wrap=True)


def _harden_streams() -> None:
    """Never crash on a legacy Windows code page (GBK, cp1252, ...).

    Python's redirected stdio uses the locale encoding; rich renders tables and
    panels safely, but plain unicode decoration would raise ``UnicodeEncodeError``
    on e.g. a cp936 console. Replacing unencodable characters keeps the CLI
    usable everywhere.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - best effort, never block the command
            continue


def _stream_encoding() -> str:
    try:
        return str(getattr(console.file, "encoding", None) or "utf-8")
    except Exception:  # noqa: BLE001 - exotic stream objects
        return "utf-8"


def _glyph(unicode_text: str, ascii_fallback: str) -> str:
    """Use a pretty glyph when the terminal encoding supports it, else ASCII."""
    try:
        unicode_text.encode(_stream_encoding())
    except (LookupError, UnicodeEncodeError):
        return ascii_fallback
    return unicode_text

MEMORY_KIND_HELP = "fact | identity | preference | note | episode | summary"
EVENT_KIND_HELP = "note | task | project | decision | learning | file | conversation | other"

CHAT_HELP: tuple[tuple[str, str], ...] = (
    ("/help", "Show this help"),
    ("/remember TEXT", "Save a durable fact"),
    ("/recall QUERY", "Search your memory"),
    ("/profile", "Show what Amnesia knows about you"),
    ("/timeline [DAYS]", "Recent activity"),
    ("/events [DAYS]", "Activity with details"),
    ("/forget ID", "Deactivate a memory by id"),
    ("/new", "Start a new session"),
    ("/session", "Show the current session"),
    ("/model [NAME]", "Show or switch the model"),
    ("/memory on|off", "Toggle memory injection"),
    ("/stats", "Show memory statistics"),
    ("/export PATH", "Export everything to JSON"),
    ("/quit", "Leave chat"),
)

ONBOARDING_QUESTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "name",
        "What should I call you?",
        "identity",
        "User's name is {value}",
    ),
    (
        "role",
        "What do you do? (role, job or field)",
        "identity",
        "User's role is {value}",
    ),
    (
        "location",
        "Where are you based? (city, country, timezone)",
        "fact",
        "User is based in {value}",
    ),
    (
        "languages",
        "Which languages do you work in?",
        "fact",
        "User works in these languages: {value}",
    ),
    (
        "projects",
        "What are you currently working on?",
        "fact",
        "User is currently working on: {value}",
    ),
    (
        "answer_style",
        "How should your assistant answer? (e.g. concise, detailed, code-first)",
        "preference",
        "Prefers answers that are {value}",
    ),
)


# ---------------------------------------------------------------------------
# Service construction & lifecycle
# ---------------------------------------------------------------------------

# Verbosity of the current invocation: set by the global ``--verbose/-v`` flag.
# 0 keeps the CLI quiet (WARNING), 1 shows INFO, 2+ shows DEBUG.
_VERBOSE_COUNT = 0

DEFAULT_LOG_LEVEL = "WARNING"


def _cli_log_level(default: str = DEFAULT_LOG_LEVEL) -> str | None:
    """Resolve the log level for this invocation.

    Explicit ``-v`` flags win; otherwise an ``AMNESIA_LOG_LEVEL`` set by the
    user is respected, and the quiet CLI default applies last. Returns ``None``
    when no override is needed (let :func:`amnesia.config.load_settings` decide).
    """
    if _VERBOSE_COUNT >= 2:
        return "DEBUG"
    if _VERBOSE_COUNT == 1:
        return "INFO"
    if os.environ.get("AMNESIA_LOG_LEVEL", "").strip():
        return None
    return default


def _load_settings(*, default_log_level: str = DEFAULT_LOG_LEVEL) -> Settings:
    """Load settings, applying the CLI's quiet-by-default log level."""
    level = _cli_log_level(default_log_level)
    if level is None:
        return load_settings()
    return load_settings(log_level=level)


def make_service(settings: Settings | None = None) -> Amnesia:
    """Build the :class:`Amnesia` service used by every command.

    Tests monkeypatch this function to inject a temp store, a
    :class:`~amnesia.embeddings.HashEmbedder` and a fake Ollama client.
    """
    return Amnesia(settings or _load_settings())


def _refresh_logging_handlers() -> None:
    """Drop ``amnesia`` log handlers bound to a stale (closed) stream.

    ``configure_logging`` only installs a handler once per process. When the
    CLI is driven repeatedly *in-process* (tests, notebooks, embedding Amnesia
    in another tool) the first handler can point at a stdout/stderr that has
    since been closed, which produces noisy "--- Logging error ---" blocks.
    Removing stale handlers lets the next ``service.start()`` install a fresh
    handler bound to the current streams.
    """
    root = logging.getLogger("amnesia")
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is not None and getattr(stream, "closed", False):
            root.removeHandler(handler)
            with suppress(Exception):
                handler.close()


async def _shutdown(service: Amnesia) -> None:
    """Drain the extraction scheduler (if any) and close the service."""
    scheduler = getattr(service, "extraction_scheduler", None)
    if scheduler is not None:
        for method_name in ("drain", "close"):
            method = getattr(scheduler, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - shutdown must never mask the command
                LOG.debug("Extraction scheduler %s failed", method_name, exc_info=True)
    with suppress(Exception):
        await service.close()


def _run(
    operation: Callable[[Amnesia], Awaitable[T]],
    settings: Settings | None = None,
) -> T:
    """Start a service, run ``operation`` inside one event loop, close it."""
    _refresh_logging_handlers()

    async def _main() -> T:
        service = make_service(settings or _load_settings())
        try:
            await service.start()
            return await operation(service)
        finally:
            await _shutdown(service)

    return run_sync(_main())


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _fail(message: str, *, code: int = 1) -> NoReturn:
    """Print a friendly error and exit non-zero."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)


def _note(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def _success(message: str) -> None:
    console.print(f"[green]{_glyph('✓', '+')}[/green] {message}")


def _warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def _print_json(payload: Any) -> None:
    """Emit indented JSON on stdout (no wrapping, no ANSI, machine readable)."""
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _human_bytes(size: Any) -> str:
    try:
        value = float(max(0, int(size)))
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _short_id(identifier: str, width: int = 8) -> str:
    return identifier if len(identifier) <= width else identifier[:width]


def _truncate(text: str, width: int = 100) -> str:
    return truncate(str(text or "").replace("\n", " "), width, suffix="…")


def _text(value: Any, style: str | None = None) -> Text:
    """Build a rich Text so arbitrary user/model content is never parsed as markup."""
    return Text(str(value), style=style)


def _styled(value: Any, style: str) -> Text:
    return Text(str(value), style=style)


def _new_table(*, title: Text | str | None = None, title_style: str = "bold") -> Table:
    return Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        title=title,
        title_style=title_style,
        title_justify="left",
        pad_edge=False,
    )


def _status_icon(status: str) -> Text:
    if status == "ok":
        return Text(_glyph("✓", "OK"), style="bold green")
    if status == "warn":
        return Text("!", style="bold yellow")
    return Text(_glyph("✗", "X"), style="bold red")


def _memory_table(memories: Sequence[Memory], *, title: Text | str = "Memory") -> Table:
    table = _new_table(title=title)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Kind", style="magenta", no_wrap=True)
    table.add_column("Imp", justify="right", no_wrap=True)
    table.add_column("Memory", overflow="fold")
    for memory in memories:
        content = Text()
        if memory.key:
            content.append(f"{memory.key}: ", style="cyan")
        content.append(memory.content)
        table.add_row(
            Text(str(memory.id), style="dim"),
            _styled(memory.kind, "magenta"),
            Text(f"{memory.importance:.2f}"),
            content,
        )
    return table


def _event_table(
    events: Sequence[Event],
    *,
    title: Text | str = "Timeline",
    show_details: bool = False,
) -> Table:
    table = _new_table(title=title)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("When", style="cyan", no_wrap=True)
    table.add_column("Kind", style="magenta", no_wrap=True)
    table.add_column("Project", style="blue", no_wrap=True)
    table.add_column("Event", overflow="fold")
    if show_details:
        table.add_column("Details", overflow="fold", style="dim")
    for event in events:
        row = [
            Text(str(event.id), style="dim"),
            Text(human_time(event.occurred_at)),
            _styled(event.kind, "magenta"),
            _styled(event.project or "—", "blue" if event.project else "dim"),
            _text(event.title),
        ]
        if show_details:
            row.append(_text(_truncate(event.details or "", 120)))
        table.add_row(*row)
    return table


def _recall_table(hits: Sequence[RecallHit], *, query: str) -> Table:
    table = _new_table(title=Text(f'Recall "{_truncate(query, 60)}" · {len(hits)} hit(s)'))
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Kind", style="magenta", no_wrap=True)
    table.add_column("When", style="cyan", no_wrap=True)
    table.add_column("Content", overflow="fold")
    for hit in hits:
        label = hit.memory_kind or hit.event_kind or hit.kind
        if hit.project:
            label = f"{label} [{hit.project}]"
        table.add_row(
            Text(f"{hit.score:.3f}", style="green"),
            _styled(label, "magenta"),
            Text(human_time(hit.created_at)),
            _text(_truncate(hit.content, 140)),
        )
    return table


def _sessions_table(sessions: Sequence[Session], *, latest_id: str | None = None) -> Table:
    table = _new_table(title="Sessions")
    table.add_column("Id", style="cyan", no_wrap=True)
    table.add_column("Started", style="dim", no_wrap=True)
    table.add_column("Client", no_wrap=True)
    table.add_column("Model", overflow="ellipsis", no_wrap=True)
    table.add_column("Msgs", justify="right", no_wrap=True)
    table.add_column("Title", overflow="fold")
    for session in sessions:
        marker = " [green]•[/green]" if session.id == latest_id else ""
        table.add_row(
            _styled(_short_id(session.id), "cyan"),
            Text(human_time(session.started_at)),
            _text(session.client),
            _text(_truncate(session.model or "—", 28)),
            Text(str(session.message_count)),
            _text(session.title or "(untitled)") + Text.from_markup(marker),
        )
    return table


def _messages_table(messages: Sequence[Message]) -> Table:
    table = _new_table(title="Messages")
    table.add_column("Role", no_wrap=True)
    table.add_column("When", style="dim", no_wrap=True)
    table.add_column("Content", overflow="fold")
    for message in messages:
        style = "green" if message.role == "assistant" else "cyan" if message.role == "user" else "dim"
        table.add_row(
            _styled(message.role, style),
            Text(human_time(message.created_at)),
            _text(_truncate(message.content, 240)),
        )
    return table


def _stats_table(stats: Stats) -> Table:
    table = _new_table(title="Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Identity memories", Text(str(stats.identity_memories)))
    table.add_row("Episode memories", Text(str(stats.episode_memories)))
    table.add_row("Total memories", Text(str(stats.memories)))
    table.add_row("Events", Text(str(stats.events)))
    table.add_row("Sessions", Text(str(stats.sessions)))
    table.add_row("Messages", Text(str(stats.messages)))
    table.add_row("Embedded memories", Text(str(stats.embedded_memories)))
    table.add_row("Embedded events", Text(str(stats.embedded_events)))
    table.add_row("Database size", Text(_human_bytes(stats.db_size_bytes)))
    table.add_row("Embedding model", Text(stats.embed_model or "—"))
    table.add_row("Chat model", Text(stats.chat_model or "—"))
    table.add_row("Last activity", Text(human_time(stats.last_activity) if stats.last_activity else "—"))
    return table


def _memory_saved_panel(memory: Memory, *, title: str = "Memory saved") -> Panel:
    body = Text()
    body.append(f"#{memory.id} ", style="dim")
    body.append(memory.kind, style="magenta")
    if memory.key:
        body.append(f" · {memory.key}", style="cyan")
    if memory.source:
        body.append(f" · via {memory.source}", style="dim")
    body.append("\n")
    body.append(memory.content)
    return Panel(body, title=f"[green]{title}[/green]", border_style="green", padding=(0, 1))


def _event_saved_panel(event: Event, *, title: str = "Note saved") -> Panel:
    body = Text()
    body.append(f"#{event.id} ", style="dim")
    body.append(event.kind, style="magenta")
    if event.project:
        body.append(f" · {event.project}", style="blue")
    body.append("\n")
    body.append(event.title)
    if event.details:
        body.append("\n")
        body.append(event.details, style="dim")
    return Panel(body, title=f"[green]{title}[/green]", border_style="green", padding=(0, 1))


def _banner() -> Panel:
    body = Group(
        Text("Your local models forgot. Amnesia remembers.", style="bold"),
        Text(
            "Local-first long-term memory for Ollama - identity, activity and "
            "recall, all in one SQLite file on this machine.",
            style="dim",
        ),
    )
    return Panel(body, title="[bold cyan]amnesia[/bold cyan]", border_style="cyan", padding=(0, 2))


# ---------------------------------------------------------------------------
# Validation & lookup helpers
# ---------------------------------------------------------------------------


def _coerce_memory_kind(kind: str | None, *, default: str = "fact") -> str:
    value = (kind or default).strip().lower()
    if value == "truth":  # accepted alias for "fact"
        value = "fact"
    if value not in ALL_MEMORY_KINDS:
        _fail(
            f"Unknown memory kind {kind!r}. Choose one of: "
            f"{', '.join(ALL_MEMORY_KINDS)}."
        )
    return value


def _coerce_event_kind(kind: str | None, *, default: str = "note") -> str:
    value = (kind or default).strip().lower()
    if value not in ALL_EVENT_KINDS:
        _fail(
            f"Unknown event kind {kind!r}. Choose one of: "
            f"{', '.join(ALL_EVENT_KINDS)}."
        )
    return value


def _find_session(service: Amnesia, token: str | None) -> Session | None:
    """Look up a session by full id or unique prefix."""
    if not token:
        return None
    exact = service.get_session(token)
    if exact is not None:
        return exact
    if len(token) < 4:
        return None
    matches = [
        session
        for session in service.store.list_sessions(limit=1000)
        if session.id.startswith(token)
    ]
    if len(matches) > 1:
        _fail(f"Session prefix {token!r} is ambiguous ({len(matches)} sessions match).")
    return matches[0] if matches else None


def _require_ollama(service: Amnesia) -> None:
    if not service.ollama_available:
        _fail(
            f"Ollama is not reachable at {service.settings.ollama_url}. "
            "Start it with `ollama serve` and try again."
        )


def _resolve_model(service: Amnesia, override: str | None) -> str:
    if override:
        return override
    model = service.chat_model
    if not model:
        _fail(
            "No chat model configured. Pass a model name, set AMNESIA_MODEL, "
            "or install one with `ollama pull <model>`."
        )
    return model


def _model_installed(model: str | None, names: Sequence[str]) -> bool:
    if not model:
        return False
    target = model.split(":")[0]
    return any(name == model or name.split(":")[0] == target for name in names)


# ---------------------------------------------------------------------------
# Extraction pipeline (guarded: workstream A may land independently)
# ---------------------------------------------------------------------------


def _build_extraction_pipeline(service: Amnesia) -> Any:
    """Build and attach a MemoryExtractor/ExtractionScheduler if available."""
    if not service.settings.extraction_enabled:
        return None
    try:
        from .extraction import ExtractionScheduler, MemoryExtractor
    except Exception as exc:  # noqa: BLE001 - optional component
        LOG.debug("Extraction pipeline unavailable: %s", exc)
        return None
    try:
        scheduler = ExtractionScheduler(MemoryExtractor(service))
    except Exception:  # noqa: BLE001 - never block the CLI on distillation
        LOG.warning("Could not initialise the extraction pipeline", exc_info=True)
        return None
    try:
        service.extraction_scheduler = scheduler  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - service attribute is optional by design
        LOG.debug("Could not attach scheduler to service", exc_info=True)
    return scheduler


async def _schedule_extraction(service: Amnesia, session_id: str | None) -> None:
    """Fire-and-forget distillation after a turn, when the pipeline exists."""
    if not session_id or not service.settings.extraction_enabled:
        return
    scheduler = getattr(service, "extraction_scheduler", None)
    if scheduler is None:
        scheduler = _build_extraction_pipeline(service)
    if scheduler is None:
        return
    try:
        scheduler.schedule(session_id)
    except Exception:  # noqa: BLE001 - extraction is best-effort
        LOG.debug("Could not schedule extraction", exc_info=True)


def _build_extractor(service: Amnesia) -> Any:
    try:
        from .extraction import MemoryExtractor
    except Exception as exc:  # noqa: BLE001 - optional component
        LOG.debug("Extractor unavailable: %s", exc)
        return None
    try:
        return MemoryExtractor(service)
    except Exception:  # noqa: BLE001 - report a friendly error upstream
        LOG.warning("Could not initialise the extractor", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Error handling decorator
# ---------------------------------------------------------------------------

_FRIENDLY_ERRORS: tuple[type[BaseException], ...] = (
    OllamaError,
    ValueError,
    FileNotFoundError,
    IsADirectoryError,
    PermissionError,
    json.JSONDecodeError,
)


def _friendly(func: Callable[..., T]) -> Callable[..., T]:
    """Convert expected failures into a clean message + non-zero exit code."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except KeyboardInterrupt:
            err_console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(code=130) from None
        except _FRIENDLY_ERRORS as exc:
            err_console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from None
        except Exception as exc:  # noqa: BLE001 - last resort, never a traceback
            LOG.exception("Command %s failed", getattr(func, "__name__", "?"))
            err_console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
            err_console.print("[dim]Set AMNESIA_LOG_LEVEL=DEBUG for details.[/dim]")
            raise typer.Exit(code=1) from None

    return wrapper


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="amnesia",
    help=(
        "Amnesia - local-first long-term memory for Ollama models.\n\n"
        "Memory commands are available without Ollama; `chat`, `ask`, `serve` "
        "and `distill` need the daemon running."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

session_app = typer.Typer(
    help="Inspect recorded chat sessions.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
config_app = typer.Typer(
    help="Inspect Amnesia configuration.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
app.add_typer(session_app, name="session")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]amnesia[/bold] {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the Amnesia version and exit.",
            is_eager=True,
            callback=_version_callback,
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Show INFO logs (-v) or DEBUG logs (-vv). Default is quiet.",
        ),
    ] = 0,
) -> None:
    """Give your local models a long-term memory."""
    del version  # handled by the eager callback before any command runs
    global _VERBOSE_COUNT
    _VERBOSE_COUNT = int(verbose)
    _harden_streams()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

async def _detect_environment(service: Amnesia) -> dict[str, Any]:
    info: dict[str, Any] = {
        "ollama_url": service.settings.ollama_url,
        "ollama_available": bool(service.ollama_available),
        "version": None,
        "models": [],
        "chat_model": service.chat_model,
        "embed_model": service.settings.embed_model,
        "embedding_id": service.embedding_model_id,
    }
    if service.ollama_available:
        with suppress(OllamaError):
            info["version"] = await service.ollama.version()
        with suppress(OllamaError):
            info["models"] = await service.ollama.model_names()
    return info


async def _store_onboarding(
    service: Amnesia, answers: Sequence[tuple[str, str, str]]
) -> int:
    stored = 0
    for key, kind, content in answers:
        try:
            await service.remember(
                content,
                kind=kind,
                key=key,
                importance=0.85,
                source="onboarding",
            )
            stored += 1
        except Exception as exc:  # noqa: BLE001 - keep onboarding going
            LOG.warning("Could not store onboarding answer %r: %s", key, exc)
    return stored


def _collect_onboarding() -> list[tuple[str, str, str]]:
    console.print(
        Panel(
            "Answer a few questions so your local models know who you are.\n"
            "[dim]Press Enter to skip a question, Ctrl+C to stop.[/dim]",
            title="[bold cyan]Onboarding[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    answers: list[tuple[str, str, str]] = []
    for key, question, kind, template in ONBOARDING_QUESTIONS:
        try:
            raw = console.input(f"[bold cyan]?[/bold cyan] {question} ")
        except (EOFError, KeyboardInterrupt, typer.Abort):
            console.print()
            break
        value = raw.strip()
        if not value:
            continue
        answers.append((key, kind, template.format(value=value)))
    return answers


def _render_init_report(
    settings: Settings,
    info: dict[str, Any],
    *,
    stored: int,
    already: bool,
) -> None:
    console.print(_banner())
    table = _new_table(title="Environment")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Home", _styled(str(settings.home), ""))
    table.add_row("Database", _styled(str(settings.database), ""))
    table.add_row("Config", _styled(str(settings.config_file) + (" (exists)" if settings.config_file.exists() else " (new)"), "dim"))
    table.add_row("Ollama", _styled(info["ollama_url"], ""))
    if info["ollama_available"]:
        table.add_row("Ollama version", _styled(str(info["version"] or "unknown"), "green"))
        models = info["models"]
        listing = ", ".join(models[:6]) + (" …" if len(models) > 6 else "") if models else "none installed"
        table.add_row("Installed models", _text(f"{len(models)}: {listing}" if models else listing))
        table.add_row("Chat model", _styled(str(info["chat_model"] or "not configured"), "green" if info["chat_model"] else "yellow"))
        table.add_row(
            "Embedding model",
            _styled(str(info["embedding_id"] or info["embed_model"]), ""),
        )
    else:
        table.add_row("Ollama version", _styled("unreachable", "yellow"))
        table.add_row(
            "Next step",
            _styled("Start the daemon with `ollama serve` to chat.", "yellow"),
        )
    console.print(table)

    embed_model = str(settings.embed_model or "")
    if info["ollama_available"] and embed_model and not _model_installed(embed_model, info["models"]):
        _warn(
            f"Embedding model {embed_model!r} is not installed; memories still work "
            f"with keyword/hash embeddings. Run: ollama pull {embed_model}"
        )

    if stored:
        _success(f"Stored {stored} onboarding memory item(s) - your profile is ready.")

    if already:
        console.print(
            Panel(
                "Amnesia is already initialised here.\n"
                "Use [bold]amnesia init --force[/bold] to re-run detection and onboarding.",
                border_style="yellow",
                padding=(0, 1),
            )
        )
    else:
        _success(f"Initialised {settings.database}")

    next_steps = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    next_steps.add_column("Command", style="cyan", no_wrap=True)
    next_steps.add_column("What it does", style="dim")
    next_steps.add_row("amnesia chat", "Start a streaming chat that remembers you")
    next_steps.add_row("amnesia remember \"...\"", "Teach Amnesia a durable fact")
    next_steps.add_row("amnesia recall \"...\"", "Search everything it knows")
    next_steps.add_row("amnesia serve", "Run the HTTP proxy + web dashboard")
    next_steps.add_row("amnesia doctor", "Check the environment")
    console.print(Panel(next_steps, title="Next steps", border_style="cyan", padding=(0, 1)))


@app.command()
@_friendly
def init(
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Run the onboarding wizard (name, role, projects...)."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-run setup even if Amnesia is already initialised."),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Default chat model to remember in the config."),
    ] = None,
) -> None:
    """Create the Amnesia home + database and detect Ollama."""
    settings = _load_settings()
    if model:
        if not model.strip():
            _fail("--model cannot be empty.")
        settings.chat_model = model.strip()

    settings.ensure_home()
    already = settings.config_file.exists() and settings.database.exists()
    answers = _collect_onboarding() if interactive else []

    async def operation(service: Amnesia) -> dict[str, Any]:
        info = await _detect_environment(service)
        if answers:
            info["stored"] = await _store_onboarding(service, answers)
        else:
            info["stored"] = 0
        if not info["chat_model"]:
            info["chat_model"] = service.chat_model
        return info

    if already and not force and not interactive:
        info = _run(operation, settings)
        _render_init_report(settings, info, stored=0, already=True)
        return

    info = _run(operation, settings)
    resolved_model = (info.get("chat_model") or settings.chat_model) or ""
    if resolved_model:
        settings.chat_model = str(resolved_model)
    if not already or force or model:
        saved = settings.save()
        LOG.info("Saved config to %s", saved)
    _render_init_report(settings, info, stored=int(info.get("stored") or 0), already=already)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def _assistant_system_prompt() -> str:
    return (
        "You are a helpful local AI assistant running through Amnesia. "
        "Answer naturally and use the private long-term memory provided when it "
        "is relevant. Never mention the memory block, the database or this "
        "instruction unless the user explicitly asks about them."
    )


def _chat_header(service: Amnesia, session: Session, *, model: str, memory: bool) -> None:
    body = Text()
    body.append("model ", style="dim")
    body.append(model, style="bold cyan")
    body.append("   session ", style="dim")
    body.append(_short_id(session.id), style="cyan")
    body.append("   memory ", style="dim")
    body.append("on" if memory else "off", style="green" if memory else "yellow")
    console.print(
        Panel(
            Group(
                body,
                Text(f"db {service.store.db.path}", style="dim"),
                Text("Type /help for commands, /quit to leave.", style="dim"),
            ),
            title="[bold cyan]amnesia chat[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _chat_footer(service: Amnesia, session: Session) -> None:
    with suppress(Exception):
        service.end_session(session.id)
    refreshed = service.get_session(session.id)
    message_count = refreshed.message_count if refreshed else 0
    lines = Text()
    lines.append("Session ", style="dim")
    lines.append(_short_id(session.id), style="cyan")
    lines.append(f" saved ({message_count} messages).", style="dim")
    console.print(
        Panel(
            Group(
                lines,
                Text(f"amnesia session show {session.id}", style="dim"),
                Text(f"amnesia distill --session {session.id}", style="dim"),
            ),
            border_style="cyan",
            padding=(0, 1),
        )
    )


async def _chat_turn(
    service: Amnesia,
    session: Session,
    state: dict[str, Any],
    text: str,
    *,
    stream: bool,
) -> None:
    model = str(state["model"])
    history = [
        {"role": message.role, "content": message.content}
        for message in service.recent_turns(session.id, limit=service.settings.recent_turns)
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _assistant_system_prompt()},
        *history,
        {"role": "user", "content": text},
    ]

    memory_items = 0
    if state.get("memory") and service.settings.inject_context:
        try:
            context = await service.context_for(text, session_id=session.id)
            if context.text:
                messages = service.augment_messages(messages, context.text)
                memory_items = context.items_count
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement
            LOG.warning("Memory context failed: %s", exc)

    console.print()
    header = Text()
    header.append("assistant ", style="bold green")
    header.append("›", style="green")
    if memory_items:
        header.append(f"  ({memory_items} memory item(s))", style="dim")
    console.print(header)

    reply_parts: list[str] = []
    try:
        if stream:
            async for chunk in service.ollama.chat_stream(model, messages):
                piece = str((chunk.get("message") or {}).get("content") or "")
                if piece:
                    reply_parts.append(piece)
                    console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
        else:
            data = await service.ollama.chat(model, messages)
            reply = str((data.get("message") or {}).get("content") or "")
            if reply:
                reply_parts.append(reply)
                console.print(reply, markup=False, highlight=False, soft_wrap=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted.[/yellow]")
        return
    except OllamaError as exc:
        console.print()
        err_console.print(
            f"[bold red]Error:[/bold red] {exc}\n"
            "[dim]Is Ollama running? Try `ollama serve`.[/dim]"
        )
        return

    console.print()
    reply_text = "".join(reply_parts).strip()
    if not reply_text:
        _note("(the model returned an empty reply)")
        return
    try:
        await service.add_turn(session.id, text, reply_text, model=model)
    except Exception as exc:  # noqa: BLE001 - never lose the reply to a db hiccup
        LOG.warning("Could not persist the conversation turn: %s", exc)
    await _schedule_extraction(service, session.id)


def _help_table() -> Table:
    table = _new_table(title="Chat commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for command, description in CHAT_HELP:
        table.add_row(command, description)
    return table


async def _slash_remember(service: Amnesia, session: Session, argument: str) -> None:
    if not argument:
        _note("Usage: /remember TEXT")
        return
    memory = await service.remember(
        argument, kind="fact", source="chat", session_id=session.id
    )
    console.print(_memory_saved_panel(memory))


async def _slash_recall(service: Amnesia, argument: str) -> None:
    if not argument:
        _note("Usage: /recall QUERY")
        return
    hits = await service.recall(argument, limit=8)
    if not hits:
        _note("Nothing relevant found in memory yet.")
        return
    console.print(_recall_table(hits, query=argument))


def _slash_profile(service: Amnesia) -> None:
    facts = service.profile(limit=50)
    if not facts:
        _note("No profile yet - use /remember TEXT or `amnesia init --interactive`.")
        return
    console.print(_memory_table(facts, title="Who Amnesia thinks you are"))


def _slash_timeline(service: Amnesia, argument: str, *, show_details: bool = False) -> None:
    days = 7
    if argument:
        try:
            days = max(1, int(argument))
        except ValueError:
            _note("Usage: /timeline [DAYS]  (days must be an integer)")
            return
    events = service.timeline(days=days, limit=20)
    if not events:
        _note(f"No activity recorded in the last {days} day(s).")
        return
    console.print(
        _event_table(
            events,
            title=f"Activity · last {days} day(s)",
            show_details=show_details,
        )
    )


async def _slash_forget(service: Amnesia, argument: str) -> None:
    if not argument:
        _note("Usage: /forget ID")
        return
    try:
        memory_id = int(argument)
    except ValueError:
        _note("Usage: /forget ID  (ID must be an integer)")
        return
    forgotten = await service.forget(memory_id=memory_id)
    if forgotten:
        _success(f"Memory #{memory_id} deactivated.")
    else:
        _warn(f"No active memory with id #{memory_id}.")


def _slash_model(service: Amnesia, state: dict[str, Any], argument: str) -> None:
    if not argument:
        _note(f"Current model: {state['model']}")
        return
    state["model"] = argument
    _success(f"Model set to {argument} for this session.")


def _slash_memory(state: dict[str, Any], argument: str) -> None:
    value = argument.strip().lower()
    if value in {"on", "off"}:
        state["memory"] = value == "on"
    elif value:
        _note("Usage: /memory on|off")
        return
    _success(f"Memory injection is {'on' if state.get('memory') else 'off'}.")


async def _slash_export(service: Amnesia, argument: str) -> None:
    if not argument:
        _note("Usage: /export PATH")
        return
    destination = service.export(argument)
    _success(f"Exported memory to {destination}")


async def _handle_slash(
    service: Amnesia,
    session: Session,
    state: dict[str, Any],
    raw: str,
) -> tuple[str, Session | None]:
    """Handle one slash command; returns (action, replacement_session)."""
    command, _, argument = raw.partition(" ")
    command = command.strip().lower()
    argument = argument.strip()

    if command in {"/quit", "/exit", "/q"}:
        return "quit", None
    if command in {"/help", "/?"}:
        console.print(_help_table())
        return "continue", None
    if command == "/remember":
        await _slash_remember(service, session, argument)
        return "continue", None
    if command == "/recall":
        await _slash_recall(service, argument)
        return "continue", None
    if command == "/profile":
        _slash_profile(service)
        return "continue", None
    if command == "/timeline":
        _slash_timeline(service, argument)
        return "continue", None
    if command == "/events":
        _slash_timeline(service, argument, show_details=True)
        return "continue", None
    if command == "/forget":
        await _slash_forget(service, argument)
        return "continue", None
    if command == "/new":
        with suppress(Exception):
            service.end_session(session.id)
        new_session = service.start_session(model=str(state["model"]), client="cli")
        _success(f"Started session {new_session.id}")
        return "continue", new_session
    if command == "/session":
        body = Text()
        body.append("id ", style="dim")
        body.append(session.id, style="cyan")
        body.append("\nmodel ", style="dim")
        body.append(str(session.model or state["model"]))
        body.append("\nclient ", style="dim")
        body.append(session.client)
        body.append("\nstarted ", style="dim")
        body.append(human_time(session.started_at))
        refreshed = service.get_session(session.id)
        body.append("\nmessages ", style="dim")
        body.append(str(refreshed.message_count if refreshed else 0))
        console.print(Panel(body, title="Current session", border_style="cyan", padding=(0, 1)))
        return "continue", None
    if command == "/model":
        _slash_model(service, state, argument)
        return "continue", None
    if command == "/memory":
        _slash_memory(state, argument)
        return "continue", None
    if command == "/stats":
        console.print(_stats_table(service.stats()))
        return "continue", None
    if command == "/export":
        await _slash_export(service, argument)
        return "continue", None
    _warn(f"Unknown command {command!r} - type /help for the list.")
    return "continue", None


async def _chat_loop(
    service: Amnesia,
    *,
    model_override: str | None,
    session_id: str | None,
    memory: bool,
    stream: bool,
) -> None:
    _require_ollama(service)
    model = _resolve_model(service, model_override)

    if session_id:
        session = _find_session(service, session_id)
        if session is None:
            session = service.start_session(
                session_id=session_id, model=model, client="cli"
            )
            _note(f"Created session {session.id}")
    else:
        session = await service.ensure_session(model=model, client="cli")

    state: dict[str, Any] = {"model": model, "memory": memory}
    _chat_header(service, session, model=model, memory=memory)

    while True:
        try:
            raw = console.input("[bold cyan]you ›[/] ")
        except (EOFError, KeyboardInterrupt, typer.Abort):
            console.print()
            break
        text = raw.strip()
        if not text:
            continue
        if text.startswith("/"):
            try:
                action, replacement = await _handle_slash(service, session, state, text)
            except (OllamaError, ValueError) as exc:
                err_console.print(f"[bold red]Error:[/bold red] {exc}")
                continue
            if action == "quit":
                break
            if replacement is not None:
                session = replacement
            continue
        try:
            await _chat_turn(service, session, state, text, stream=stream)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except OllamaError as exc:
            err_console.print(f"[bold red]Error:[/bold red] {exc}")
        except Exception:  # noqa: BLE001 - keep the REPL alive
            LOG.exception("Chat turn failed")
            err_console.print("[bold red]Error:[/bold red] the turn failed; see logs.")

    _chat_footer(service, session)


@app.command()
@_friendly
def chat(
    model: Annotated[
        str | None,
        typer.Argument(help="Model to chat with (default: configured model)."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Resume or create this session id."),
    ] = None,
    no_memory: Annotated[
        bool,
        typer.Option("--no-memory", help="Disable memory injection for this chat."),
    ] = False,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Wait for the full reply instead of streaming."),
    ] = False,
) -> None:
    """Chat with a local Ollama model that remembers you (streaming REPL)."""
    settings = _load_settings()
    _run(
        lambda service: _chat_loop(
            service,
            model_override=model,
            session_id=session,
            memory=not no_memory,
            stream=not no_stream,
        ),
        settings,
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


async def _ask_once(
    service: Amnesia,
    text: str,
    *,
    model_override: str | None,
    memory: bool,
    stream: bool,
) -> None:
    _require_ollama(service)
    model = _resolve_model(service, model_override)
    session = await service.ensure_session(model=model, client="cli")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _assistant_system_prompt()},
    ]
    memory_items = 0
    if memory and service.settings.inject_context:
        try:
            context = await service.context_for(text, session_id=session.id)
            if context.text:
                messages = service.augment_messages(messages, context.text)
                memory_items = context.items_count
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement
            LOG.warning("Memory context failed: %s", exc)
    messages.append({"role": "user", "content": text})

    parts: list[str] = []
    if stream:
        async for chunk in service.ollama.chat_stream(model, messages):
            piece = str((chunk.get("message") or {}).get("content") or "")
            if piece:
                parts.append(piece)
                console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
        console.print()
    else:
        data = await service.ollama.chat(model, messages)
        reply = str((data.get("message") or {}).get("content") or "")
        parts.append(reply)
        console.print(Panel(_text(reply), border_style="cyan", padding=(0, 1)))

    answer = "".join(parts).strip()
    footer = Text()
    footer.append(f"model {model}", style="dim")
    if memory:
        footer.append(f" · {memory_items} memory item(s)", style="dim")
    else:
        footer.append(" · memory off", style="dim")
    footer.append(f" · session {_short_id(session.id)}", style="dim")
    console.print(footer)

    if answer:
        await service.add_turn(session.id, text, answer, model=model)
        await _schedule_extraction(service, session.id)


@app.command()
@_friendly
def ask(
    text: Annotated[str, typer.Argument(help="The question to ask.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use for this answer."),
    ] = None,
    no_memory: Annotated[
        bool,
        typer.Option("--no-memory", help="Ask without injecting memories."),
    ] = False,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Wait for the full answer instead of streaming."),
    ] = False,
) -> None:
    """Ask a one-shot question with memory injected into the prompt."""
    settings = _load_settings()
    _run(
        lambda service: _ask_once(
            service,
            text,
            model_override=model,
            memory=not no_memory,
            stream=not no_stream,
        ),
        settings,
    )


# ---------------------------------------------------------------------------
# remember / forget
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def remember(
    text: Annotated[str, typer.Argument(help="The fact, preference or note to store.")],
    kind: Annotated[
        str,
        typer.Option("--kind", "-k", help=f"Memory kind: {MEMORY_KIND_HELP}."),
    ] = "fact",
    key: Annotated[
        str | None,
        typer.Option("--key", help="Stable key: re-remembering the key updates it in place."),
    ] = None,
    importance: Annotated[
        float,
        typer.Option("--importance", help="0..1, how important this is for retrieval."),
    ] = 0.6,
) -> None:
    """Store a durable fact about you in long-term memory."""
    resolved_kind = _coerce_memory_kind(kind)
    if not text.strip():
        _fail("Cannot remember empty text.")

    async def operation(service: Amnesia) -> Memory:
        return await service.remember(
            text,
            kind=resolved_kind,
            key=(key or "").strip() or None,
            importance=max(0.0, min(1.0, importance)),
            source="cli",
        )

    memory = _run(operation, _load_settings())
    console.print(_memory_saved_panel(memory))


@app.command()
@_friendly
def forget(
    memory_id: Annotated[
        int | None,
        typer.Argument(help="Memory id (see `amnesia profile`)."),
    ] = None,
    key: Annotated[
        str | None,
        typer.Option("--key", help="Forget the memory stored under this key instead."),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="Kind used together with --key (default: search every memory kind).",
        ),
    ] = None,
    hard: Annotated[
        bool,
        typer.Option("--hard", help="Delete permanently instead of deactivating."),
    ] = False,
) -> None:
    """Forget a memory by id or by key."""
    if memory_id is None and not (key or "").strip():
        _fail("Provide a memory id or --key KEY. Example: `amnesia forget 12`.")
    target_key = (key or "").strip() or None
    explicit_kind = _coerce_memory_kind(kind) if (target_key and kind) else None

    async def operation(service: Amnesia) -> bool:
        if memory_id is not None:
            return await service.forget(memory_id=memory_id, hard=hard)
        if target_key is None:  # pragma: no cover - guarded above
            return False
        kinds_to_try = [explicit_kind] if explicit_kind else list(ALL_MEMORY_KINDS)
        for candidate in kinds_to_try:
            if await service.forget(key=target_key, kind=candidate, hard=hard):
                return True
        return False

    forgotten = _run(operation, _load_settings())
    if not forgotten:
        target = f"#{memory_id}" if memory_id is not None else f"key {key!r}"
        _fail(f"No active memory found for {target}.")
    action = "permanently deleted" if hard else "deactivated"
    _success(f"Memory {f'#{memory_id}' if memory_id is not None else repr(key)} {action}.")


# ---------------------------------------------------------------------------
# recall / profile / timeline
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def recall(
    query: Annotated[str, typer.Argument(help="What to search for.")],
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of hits (1-100)."),
    ] = 10,
    events: Annotated[
        bool,
        typer.Option("--events/--no-events", help="Include activity events in the results."),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Search memories + activity with hybrid keyword/semantic recall."""
    capped = max(1, min(100, int(limit)))
    if not query.strip():
        _fail("Query cannot be empty.")

    async def operation(service: Amnesia) -> list[RecallHit]:
        return await service.recall(query, limit=capped, include_events=events)

    hits = _run(operation, _load_settings())
    if json_output:
        _print_json({"query": query, "count": len(hits), "hits": [hit.to_dict() for hit in hits]})
        return
    if not hits:
        console.print(
            Panel(
                f'Nothing relevant found for "{query}".\n'
                "Try `amnesia remember TEXT` or a broader query.",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return
    console.print(_recall_table(hits, query=query))


@app.command()
@_friendly
def profile(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of identity memories."),
    ] = 50,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show the durable identity facts Amnesia knows about you."""
    capped = max(1, min(500, int(limit)))

    async def operation(service: Amnesia) -> list[Memory]:
        return service.profile(limit=capped)

    memories = _run(operation, _load_settings())
    if json_output:
        _print_json({"count": len(memories), "facts": [memory.to_dict() for memory in memories]})
        return
    if not memories:
        console.print(
            Panel(
                "No profile yet.\n"
                "Add facts with `amnesia remember TEXT` or run "
                "`amnesia init --interactive`.",
                title="Profile",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return
    console.print(_memory_table(memories, title=f"Profile · {len(memories)} item(s)"))


@app.command()
@_friendly
def timeline(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="How many days back to look."),
    ] = 7,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of events."),
    ] = 50,
    project: Annotated[
        str | None,
        typer.Option("--project", "-p", help="Only show events for this project."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show your recent activity timeline."""
    resolved_days = max(1, min(3650, int(days)))
    capped = max(1, min(500, int(limit)))

    async def operation(service: Amnesia) -> list[Event]:
        return service.timeline(
            days=resolved_days, limit=capped, project=(project or "").strip() or None
        )

    events = _run(operation, _load_settings())
    if json_output:
        _print_json(
            {
                "days": resolved_days,
                "project": project,
                "count": len(events),
                "events": [event.to_dict() for event in events],
            }
        )
        return
    if not events:
        console.print(
            Panel(
                f"No activity recorded in the last {resolved_days} day(s)"
                + (f" for project {project!r}" if project else "")
                + ".\nLog something with `amnesia note TEXT`.",
                title="Timeline",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return
    scope = f" · {project}" if project else ""
    console.print(_event_table(events, title=f"Timeline · last {resolved_days} day(s){scope}"))


# ---------------------------------------------------------------------------
# note / sessions / session show
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def note(
    text: Annotated[str, typer.Argument(help="What happened (a task, decision, learning...).")],
    kind: Annotated[
        str,
        typer.Option("--kind", "-k", help=f"Event kind: {EVENT_KIND_HELP}."),
    ] = "note",
    project: Annotated[
        str | None,
        typer.Option("--project", "-p", help="Project this activity belongs to."),
    ] = None,
    importance: Annotated[
        float,
        typer.Option("--importance", help="0..1, how important this is for retrieval."),
    ] = 0.5,
) -> None:
    """Record an activity event on your timeline."""
    resolved_kind = _coerce_event_kind(kind)
    if not text.strip():
        _fail("Note text cannot be empty.")

    async def operation(service: Amnesia) -> Event:
        return await service.log_event(
            text,
            kind=resolved_kind,
            project=(project or "").strip() or None,
            importance=max(0.0, min(1.0, importance)),
            source="cli",
        )

    event = _run(operation, _load_settings())
    console.print(_event_saved_panel(event))


@app.command()
@_friendly
def sessions(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of sessions."),
    ] = 20,
) -> None:
    """List your chat sessions."""
    capped = max(1, min(500, int(limit)))

    async def operation(service: Amnesia) -> list[Session]:
        return service.store.list_sessions(limit=capped)

    found = _run(operation, _load_settings())
    if not found:
        console.print(
            Panel(
                "No sessions yet.\nStart one with `amnesia chat` or `amnesia ask ...`.",
                title="Sessions",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return
    console.print(_sessions_table(found, latest_id=found[0].id))
    _note(f"Show one with: amnesia session show {_short_id(found[0].id)}")


@session_app.command("show")
@_friendly
def session_show(
    session_id: Annotated[str, typer.Argument(help="Full session id or unique prefix.")],
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum messages to display."),
    ] = 40,
) -> None:
    """Show one session with its transcript."""
    capped = max(1, min(1000, int(limit)))

    async def operation(service: Amnesia) -> tuple[Session | None, list[Message]]:
        found = _find_session(service, session_id)
        if found is None:
            return None, []
        return found, service.store.session_messages(found.id, limit=capped)

    found, messages = _run(operation, _load_settings())
    if found is None:
        _fail(f"Session {session_id!r} not found. Run `amnesia sessions` to list them.")

    meta = Text()
    meta.append("id ", style="dim")
    meta.append(found.id, style="cyan")
    meta.append("\nmodel ", style="dim")
    meta.append(found.model or "—")
    meta.append("\nclient ", style="dim")
    meta.append(found.client)
    meta.append("\nstarted ", style="dim")
    meta.append(human_time(found.started_at))
    if found.ended_at:
        meta.append("\nended ", style="dim")
        meta.append(human_time(found.ended_at))
    meta.append("\nmessages ", style="dim")
    meta.append(str(found.message_count))
    if found.summary:
        meta.append("\nsummary ", style="dim")
        meta.append(found.summary)
    console.print(
        Panel(
            meta,
            title=Text(found.title or "Session"),
            border_style="cyan",
            padding=(0, 1),
        )
    )
    if messages:
        console.print(_messages_table(messages))
    else:
        _note("This session has no messages.")


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def distill(
    session: Annotated[
        str | None,
        typer.Option("--session", help="Session id to distill."),
    ] = None,
    all_sessions: Annotated[
        bool,
        typer.Option("--all", help="Distill every recorded session."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum sessions to process with --all."),
    ] = 20,
) -> None:
    """Extract facts and events from past conversations (LLM distillation)."""
    capped = max(1, min(200, int(limit)))

    async def operation(service: Amnesia) -> list[tuple[str, dict[str, Any]]]:
        extractor = _build_extractor(service)
        if extractor is None:
            _fail(
                "Automatic distillation is unavailable in this build "
                "(amnesia.extraction could not be imported).\n"
                "Chat, recall and manual memory still work."
            )
        targets: list[str] = []
        if all_sessions:
            targets = [item.id for item in service.store.list_sessions(limit=capped)]
        elif session:
            found = _find_session(service, session)
            if found is None:
                _fail(f"Session {session!r} not found. Run `amnesia sessions` to list them.")
            targets = [found.id]
        else:
            latest = service.store.latest_session()
            if latest is None:
                _fail("No sessions to distill yet. Start one with `amnesia chat`.")
            targets = [latest.id]
        results: list[tuple[str, dict[str, Any]]] = []
        for session_id in targets:
            try:
                result = await extractor.distill_session(session_id)
            except Exception as exc:  # noqa: BLE001 - report per session, keep going
                LOG.warning("Distillation failed for %s: %s", session_id, exc)
                result = {"facts": 0, "events": 0, "summary": None, "skipped": False, "error": str(exc)}
            results.append((session_id, dict(result)))
        return results

    results = _run(operation, _load_settings())

    table = _new_table(title="Distillation")
    table.add_column("Session", style="cyan", no_wrap=True)
    table.add_column("Facts", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Summary", overflow="fold")
    table.add_column("Status", style="dim")
    total_facts = total_events = 0
    for session_id, result in results:
        facts = int(result.get("facts") or 0)
        events = int(result.get("events") or 0)
        total_facts += facts
        total_events += events
        summary = result.get("summary")
        status = "skipped" if result.get("skipped") else ("error" if result.get("error") else "distilled")
        table.add_row(
            _styled(_short_id(session_id), "cyan"),
            Text(str(facts)),
            Text(str(events)),
            _text(_truncate(summary or "—", 80)),
            _styled(status, "yellow" if status != "distilled" else "green"),
        )
    console.print(table)
    _success(
        f"Distilled {len(results)} session(s): {total_facts} fact(s), {total_events} event(s)."
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _serve_urls(host: str, port: int) -> tuple[str, str]:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    base = f"http://{display_host}:{port}"
    return base, f"{base}/amnesia/ui"


def _load_proxy_module() -> Any:
    """Import the optional HTTP proxy module.

    Kept as a small module-level seam so tests can simulate a build without
    workstream B (and so a broken/missing proxy becomes a friendly CLI error
    instead of an import traceback).
    """
    from . import proxy  # noqa: PLC0415 - optional component

    return proxy


def _load_takeover_module() -> Any:
    """Import the optional Ollama takeover module (``serve --takeover``)."""
    from . import ollama_control  # noqa: PLC0415 - optional component

    return ollama_control


def _make_takeover(**kwargs: Any) -> Any:
    """Build the takeover controller; a seam for tests."""
    module = _load_takeover_module()
    return module.OllamaTakeover(**kwargs)


def _make_proxy_app(
    create_app: Callable[..., Any],
    service: Amnesia,
    settings: Settings,
    scheduler: Any,
) -> Any:
    """Call proxy.create_app with only the keyword arguments it supports."""
    try:
        parameters = inspect.signature(create_app).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        parameters = {}
    kwargs: dict[str, Any] = {}
    if "settings" in parameters:
        kwargs["settings"] = settings
    if "extraction_scheduler" in parameters:
        kwargs["extraction_scheduler"] = scheduler
    try:
        return create_app(service, **kwargs)
    except TypeError:
        LOG.debug("create_app rejected optional kwargs; retrying with service only")
        return create_app(service)


def _factory_supports_no_args(factory: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return False
    for parameter in parameters.values():
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ) and parameter.default is parameter.empty:
            return False
    return True


def _print_serve_banner(
    settings: Settings,
    host: str,
    port: int,
    *,
    extraction: bool,
    takeover: Any = None,
) -> None:
    base, dashboard = _serve_urls(host, port)
    if takeover is None:
        body = Group(
            Text(f"Ollama proxy      {base}", style="bold"),
            Text(f"Dashboard         {dashboard}", style="bold"),
            Text(f"OpenAI-compatible {base}/v1", style="dim"),
            Text(f"Ollama-compatible {base}  (point OLLAMA_HOST here)", style="dim"),
            Text(
                f"memory {'on' if settings.memory_enabled else 'off'} · "
                f"extraction {'on' if extraction else 'off'} ·"
                f"db {settings.database}",
                style="dim",
            ),
            Text("Press Ctrl+C to stop.", style="dim"),
        )
    else:
        upstream = str(getattr(takeover, "upstream_url", "?"))
        body = Group(
            Text(f"Ollama clients    {base}  (plain `ollama run`, no OLLAMA_HOST needed)", style="bold"),
            Text(f"Ollama daemon     {upstream}  (private to Amnesia)", style="dim"),
            Text(f"Dashboard         {dashboard}", style="bold"),
            Text(f"OpenAI-compatible {base}/v1", style="dim"),
            Text(
                f"takeover on · memory {'on' if settings.memory_enabled else 'off'} · "
                f"extraction {'on' if extraction else 'off'} ·"
                f"db {settings.database}",
                style="dim",
            ),
            Text("Press Ctrl+C to stop (Ollama is restored afterwards).", style="dim"),
        )
    console.print(
        Panel(
            body,
            title="[bold cyan]amnesia serve[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _warn_if_ollama_host_bypasses(port: int) -> None:
    """Warn when the inherited ``OLLAMA_HOST`` would make clients bypass Amnesia."""
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return
    try:
        from .config import normalize_ollama_url  # noqa: PLC0415

        normalized = normalize_ollama_url(raw)
        env_port = int(normalized.rsplit(":", 1)[-1])
    except (ImportError, TypeError, ValueError):
        env_port = -1
    if env_port != int(port):
        _warn(
            f"OLLAMA_HOST={raw} points Ollama clients away from Amnesia; unset it "
            f"or set it to http://127.0.0.1:{port}."
        )


# ---------------------------------------------------------------------------
# Background memory mode (serve --background + stop)
# ---------------------------------------------------------------------------

BACKGROUND_START_TIMEOUT = 180.0
STOP_TIMEOUT = 60.0
_STATE_POLL_INTERVAL = 0.4


def _serve_logs_dir(settings: Settings) -> Path:
    return Path(settings.home) / "logs"


def _serve_log_path(settings: Settings) -> Path:
    return _serve_logs_dir(settings) / "serve.log"


def _serve_state_path(settings: Settings) -> Path:
    return _serve_logs_dir(settings) / "serve.pid"


def _read_serve_state(settings: Settings) -> dict[str, Any] | None:
    try:
        payload = json.loads(_serve_state_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not str(payload.get("url") or ""):
        return None
    return payload


def _write_serve_state(settings: Settings, payload: dict[str, Any]) -> None:
    path = _serve_state_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:  # pragma: no cover - bookkeeping only
        LOG.debug("Could not write serve state file %s", path, exc_info=True)


def _clear_serve_state(settings: Settings) -> None:
    try:
        _serve_state_path(settings).unlink()
    except OSError:
        pass


def _api_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _probe_amnesia_health(
    base_url: str, token: str | None = None, *, timeout: float = 2.0
) -> dict[str, Any] | None:
    """Return the health payload of a running Amnesia server, or ``None``."""
    try:
        import httpx  # noqa: PLC0415 - keeps CLI start-up light
    except ImportError:  # pragma: no cover - declared dependency
        return None
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/amnesia/health",
            headers=_api_headers(token),
            timeout=timeout,
            trust_env=False,
        )
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _request_shutdown(base_url: str, token: str | None = None) -> int:
    """Ask a running Amnesia server to shut down gracefully; return HTTP status."""
    import httpx  # noqa: PLC0415

    response = httpx.post(
        f"{base_url.rstrip('/')}/amnesia/shutdown",
        headers=_api_headers(token),
        timeout=5.0,
        trust_env=False,
    )
    return int(response.status_code)


def _background_command(
    host: str,
    port: int,
    *,
    takeover: bool,
    upstream_port: int,
    keep_ollama: bool,
    no_restart_ollama: bool,
    no_extraction: bool,
) -> list[str]:
    """Build the command line for the detached serve process."""
    command = [
        sys.executable,
        "-m",
        "amnesia",
        "serve",
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    if takeover:
        command += ["--takeover", "--upstream-port", str(int(upstream_port))]
    if no_extraction:
        command.append("--no-extraction")
    if takeover and keep_ollama:
        command.append("--keep-ollama")
    if takeover and no_restart_ollama:
        command.append("--no-restart-ollama")
    return command


def _spawn_background_server(command: list[str], *, log_path: Path, cwd: Path) -> Any:
    """Start the detached serve process, appending its output to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    start_new_session = False
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:  # pragma: no cover - POSIX
        start_new_session = True
    header = f"\n=== amnesia serve started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    with open(log_path, "ab", buffering=0) as log_file:
        log_file.write(header.encode("utf-8", "replace"))
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=os.environ.copy(),
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    return process


def _wait_for_background(
    process: Any,
    base_url: str,
    token: str | None,
    *,
    timeout: float = BACKGROUND_START_TIMEOUT,
) -> dict[str, Any] | None:
    """Poll the detached server until its health endpoint answers."""
    deadline = time.monotonic() + max(5.0, float(timeout))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        health = _probe_amnesia_health(base_url, token)
        if health is not None:
            return health
        time.sleep(_STATE_POLL_INTERVAL)
    return None


def _tail_log(path: Path, lines: int = 10) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _open_url(url: str) -> None:
    try:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser must never fail a command
        LOG.debug("Could not open %s in a browser", url, exc_info=True)


def _print_takeover_background_summary(
    settings: Settings, dashboard: str, log_path: Path
) -> None:
    _success("Memory mode is running in the background.")
    _note(f"Dashboard / profile: {dashboard}#profile")
    _note(f"Database: {settings.database}")
    _note(f"Log: {log_path}")
    _note("Stop it with: amnesia stop")


@app.command()
@_friendly
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind address (default: from settings)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Bind port (default: from settings)."),
    ] = None,
    no_extraction: Annotated[
        bool,
        typer.Option("--no-extraction", help="Disable automatic distillation while serving."),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Restart the proxy when its code changes (development)."),
    ] = False,
    takeover: Annotated[
        bool,
        typer.Option(
            "--takeover",
            help=(
                "Bind Ollama's standard port 11434 and move the real daemon to a "
                "private port, so plain `ollama run` clients get memory with no "
                "OLLAMA_HOST configuration."
            ),
        ),
    ] = False,
    upstream_port: Annotated[
        int,
        typer.Option(
            "--upstream-port",
            help="Private port for the Ollama daemon that --takeover manages (default 11435).",
        ),
    ] = 11435,
    keep_ollama: Annotated[
        bool,
        typer.Option(
            "--keep-ollama",
            help="In takeover mode, fail instead of stopping an Ollama app/daemon already on 11434.",
        ),
    ] = False,
    no_restart_ollama: Annotated[
        bool,
        typer.Option(
            "--no-restart-ollama",
            help="Do not restart the Ollama desktop app after takeover mode ends.",
        ),
    ] = False,
    background: Annotated[
        bool,
        typer.Option(
            "--background",
            help=(
                "Start the server detached, wait until it is healthy, then return. "
                "Stop it later with `amnesia stop`."
            ),
        ),
    ] = False,
    open_dashboard: Annotated[
        bool,
        typer.Option(
            "--open",
            help="With --background, open the dashboard (profile view) in the browser.",
        ),
    ] = False,
) -> None:
    """Run the HTTP proxy (Ollama + OpenAI compatible) and dashboard.

    With ``--takeover`` Amnesia binds Ollama's standard port instead, so the
    plain ``ollama run`` CLI (and other clients that default to 11434) get
    memory injection and conversation capture without exporting OLLAMA_HOST.
    """
    # Operators expect startup/request info from a server; other commands are quiet.
    settings = _load_settings(default_log_level="INFO")
    if background and reload:
        _fail("--background cannot be combined with --reload.")
    takeover_module: Any = None
    if takeover:
        if reload:
            _fail("--takeover cannot be combined with --reload.")
        try:
            takeover_module = _load_takeover_module()
        except Exception as exc:  # noqa: BLE001 - friendly missing-module error
            _fail(f"Takeover mode is unavailable in this build ({exc}).")
    default_bind_port = (
        int(takeover_module.DEFAULT_OLLAMA_PORT) if takeover else int(settings.proxy_port)
    )
    bind_host = (host or settings.proxy_host or "127.0.0.1").strip()
    bind_port = int(port if port is not None else default_bind_port)
    if not (1 <= bind_port <= 65535):
        _fail(f"Invalid port {bind_port}; expected 1-65535.")
    if takeover:
        if not (1 <= int(upstream_port) <= 65535):
            _fail(f"Invalid --upstream-port {upstream_port}; expected 1-65535.")
        if int(upstream_port) == bind_port:
            _fail("--upstream-port must differ from the port Amnesia binds.")
        if bind_port != default_bind_port:
            _warn(
                f"Plain Ollama clients default to port {default_bind_port}, so "
                f"--port {bind_port} will not capture them."
            )

    if background:
        if reload:
            _fail("--background cannot be combined with --reload.")
        base, dashboard = _serve_urls(bind_host, bind_port)
        existing = _probe_amnesia_health(base, settings.api_token)
        if existing is not None:
            _success(f"Amnesia is already running at {base}")
            _note(f"Dashboard / profile: {dashboard}#profile")
            if open_dashboard:
                _open_url(f"{dashboard}#profile")
            return
        command = _background_command(
            bind_host,
            bind_port,
            takeover=takeover,
            upstream_port=int(upstream_port),
            keep_ollama=keep_ollama,
            no_restart_ollama=no_restart_ollama,
            no_extraction=no_extraction,
        )
        log_path = _serve_log_path(settings)
        _note(f"Starting memory mode in the background (log: {log_path}) ...")
        process = _spawn_background_server(command, log_path=log_path, cwd=Path.cwd())
        health = _wait_for_background(process, base, settings.api_token)
        if health is None:
            if process.poll() is None:
                with suppress(Exception):
                    process.terminate()
            tail = _tail_log(log_path)
            detail = f"\n{tail}" if tail else ""
            _fail(f"The background server did not become healthy.{detail}")
        _write_serve_state(
            settings,
            {
                "pid": int(process.pid),
                "url": base,
                "dashboard": dashboard,
                "mode": "takeover" if takeover else "proxy",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "log": str(log_path),
            },
        )
        _print_takeover_background_summary(settings, dashboard, log_path)
        if open_dashboard:
            _open_url(f"{dashboard}#profile")
        return

    if open_dashboard:
        _warn("--open only works together with --background.")

    try:
        proxy = _load_proxy_module()
    except Exception as exc:  # noqa: BLE001 - friendly missing-module error
        _fail(f"The HTTP proxy is unavailable in this build ({exc}).")

    create_app = getattr(proxy, "create_app", None)
    if not callable(create_app):
        _fail("amnesia.proxy does not expose create_app(); cannot start the server.")

    extraction_enabled = bool(settings.extraction_enabled and not no_extraction)

    takeover_ctl: Any = None
    if takeover and takeover_module is not None:
        listen_base, _ = _serve_urls(bind_host, bind_port)
        takeover_error = getattr(takeover_module, "TakeoverError", RuntimeError)
        _note(f"Takeover: moving Ollama off {listen_base} and binding Amnesia there ...")
        try:
            takeover_ctl = _make_takeover(
                host=bind_host,
                port=bind_port,
                upstream_port=int(upstream_port),
                stop_existing=not keep_ollama,
                restart_desktop=not no_restart_ollama,
                log_dir=Path(settings.home) / "logs",
            )
            takeover_ctl.start()
        except takeover_error as exc:
            _fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - keep the CLI friendly
            _fail(f"Could not start takeover mode: {exc}")
        settings = settings.clone(ollama_url=takeover_ctl.upstream_url)
        _success(
            f"Ollama {getattr(takeover_ctl, 'version', '?')} is running privately at "
            f"{takeover_ctl.upstream_url}"
        )
        _note(f"Plain `ollama run` clients now reach Amnesia on {listen_base}.")
        _warn_if_ollama_host_bypasses(bind_port)

    if reload and _factory_supports_no_args(create_app):
        try:
            import uvicorn  # noqa: PLC0415 - only needed for this command
        except ImportError as exc:  # pragma: no cover - declared dependency
            _fail(f"uvicorn is required to serve the proxy ({exc}).")
        if no_extraction:
            # The reload factory builds its own settings in a subprocess, so the
            # flag must be passed through the environment.
            os.environ["AMNESIA_EXTRACTION"] = "false"
        _print_serve_banner(settings, bind_host, bind_port, extraction=extraction_enabled)
        try:
            uvicorn.run(
                "amnesia.proxy:create_app",
                factory=True,
                host=bind_host,
                port=bind_port,
                reload=True,
                log_level=str(settings.log_level).lower(),
            )
        except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
            pass
        console.print("[dim]Server stopped.[/dim]")
        return
    if reload:
        _warn(
            "This proxy build cannot be reloaded in-process; "
            "starting without hot reload."
        )

    async def operation(service: Amnesia) -> None:
        scheduler = None
        if extraction_enabled:
            scheduler = _build_extraction_pipeline(service)
            if scheduler is None:
                _warn("Automatic extraction is enabled but the pipeline is unavailable.")
        # The proxy owns the extraction lifecycle: give it the effective
        # settings so --no-extraction really means no extraction, even though
        # the daemon's own settings still allow it.
        effective_settings = service.settings.clone(extraction_enabled=extraction_enabled)
        app_instance = _make_proxy_app(create_app, service, effective_settings, scheduler)

        try:
            import uvicorn  # noqa: PLC0415 - only needed for this command
        except ImportError as exc:  # pragma: no cover - declared dependency
            _fail(f"uvicorn is required to serve the proxy ({exc}).")

        _print_serve_banner(
            service.settings,
            bind_host,
            bind_port,
            extraction=scheduler is not None,
            takeover=takeover_ctl,
        )
        if not service.ollama_available:
            _warn(
                f"Ollama is not reachable at {service.settings.ollama_url}; "
                "the dashboard and memory API still work and chats recover "
                "as soon as the daemon is up."
            )
        server = uvicorn.Server(
            uvicorn.Config(
                app_instance,
                host=bind_host,
                port=bind_port,
                log_level=str(service.settings.log_level).lower(),
            )
        )
        app_state = getattr(app_instance, "state", None)
        if app_state is not None:
            app_state.shutdown_requested = False

        async def _watch_shutdown_request() -> None:
            """Honour `POST /amnesia/shutdown` by asking uvicorn to stop."""
            while not getattr(app_state, "shutdown_requested", False):
                await asyncio.sleep(_STATE_POLL_INTERVAL)
            LOG.info("Graceful shutdown requested via the memory API")
            server.should_exit = True

        watcher = asyncio.create_task(_watch_shutdown_request())
        try:
            await server.serve()
        except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
            pass
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            console.print("[dim]Server stopped.[/dim]")

    try:
        _run(operation, settings)
    finally:
        if takeover_ctl is not None:
            takeover_ctl.stop()
            if getattr(takeover_ctl, "desktop_restarted", False):
                _note("Restarted the Ollama desktop app.")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def stop(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Seconds to wait for a graceful stop."),
    ] = STOP_TIMEOUT,
) -> None:
    """Gracefully stop a running Amnesia server and restore Ollama.

    Works for background instances (``amnesia serve --background``) and for
    foreground ones: the server shuts down, terminates the private Ollama
    daemon and restarts the Ollama desktop app if takeover had stopped it.
    """
    settings = _load_settings()
    state = _read_serve_state(settings)
    if state is not None:
        # Trust the recorded instance only: if it is gone, never fall through
        # to another server (which may belong to a different AMNESIA_HOME).
        candidates = [str(state["url"])]
    else:
        candidates = []
        for candidate in (
            "http://127.0.0.1:11434",
            _serve_urls(settings.proxy_host or "127.0.0.1", int(settings.proxy_port))[0],
        ):
            if candidate not in candidates:
                candidates.append(candidate)

    target: str | None = None
    for candidate in candidates:
        if _probe_amnesia_health(candidate, settings.api_token) is not None:
            target = candidate
            break
    if target is None:
        _clear_serve_state(settings)
        if json_output:
            _print_json({"running": False, "stopped": False})
        else:
            _note("Amnesia is not running.")
            _note(f"Memory is stored in {settings.database}")
        return

    try:
        status = _request_shutdown(target, settings.api_token)
    except Exception as exc:  # noqa: BLE001 - network errors are user-facing
        _fail(f"Could not ask the Amnesia server at {target} to stop: {exc}")
    if status == 404:
        _fail(
            "This Amnesia instance was started by an older build without a graceful "
            "stop endpoint. Press Ctrl+C in its terminal once, then start it again "
            "with: amnesia serve --takeover --background"
        )
    if status not in (200, 202):
        _fail(f"The Amnesia server at {target} refused to stop (HTTP {status}).")

    deadline = time.monotonic() + max(5.0, float(timeout))
    while time.monotonic() < deadline:
        if _probe_amnesia_health(target, settings.api_token) is None:
            break
        time.sleep(_STATE_POLL_INTERVAL)
    else:
        _fail(
            f"Amnesia is still running after {timeout:.0f}s. Check the server log at "
            f"{_serve_log_path(settings)}."
        )

    state = _read_serve_state(settings)
    if state is not None and str(state.get("url", "")) == target:
        _clear_serve_state(settings)
    if json_output:
        _print_json(
            {
                "running": True,
                "stopped": True,
                "url": target,
                "database": str(settings.database),
            }
        )
        return
    _success("Memory mode stopped; Ollama is back to normal.")
    _note(f"Memory stayed in {settings.database}")


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def mcp() -> None:
    """Run the stdio MCP server for MCP clients (Claude Desktop, Cursor, ...)."""
    try:
        from .mcp_server import main as mcp_main
    except Exception as exc:  # noqa: BLE001 - optional dependency
        err_console.print(
            "[bold red]Error:[/bold red] could not load the MCP server "
            f"({exc}).\nInstall the extra with: pip install \"amnesia[mcp]\""
        )
        raise typer.Exit(code=1) from None
    code = mcp_main(verbose=bool(_VERBOSE_COUNT))
    if code:
        raise typer.Exit(code=int(code))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


async def _doctor_check(service: Amnesia) -> int:
    """Render the health report; returns the number of hard failures."""
    health = service.health()
    rows: list[tuple[str, str, Text | str]] = []

    db_path = Path(str(health.get("db_path") or service.settings.database))
    db_exists = db_path.exists()
    integrity = str(health.get("db_integrity") or "unknown")
    journal = "unknown"
    with suppress(Exception):
        journal = str(service.store.db.scalar("PRAGMA journal_mode", default="unknown"))
    db_size = _human_bytes(health.get("db_size_bytes"))
    rows.append(
        (
            "ok" if db_exists else "fail",
            "Database",
            f"{db_path} ({db_size}, journal={journal})",
        )
    )
    rows.append(
        (
            "ok" if integrity == "ok" else "fail",
            "Integrity check",
            integrity if integrity == "ok" else f"{integrity} - back up and recreate the database",
        )
    )
    rows.append(
        (
            "ok" if health.get("fts_enabled") else "warn",
            "Full-text search",
            "SQLite FTS5 active"
            if health.get("fts_enabled")
            else "FTS5 unavailable, keyword search falls back to LIKE",
        )
    )

    ollama = health.get("ollama") or {}
    available = bool(ollama.get("available"))
    version: str | None = None
    models: list[str] = []
    if available:
        with suppress(OllamaError):
            version = await service.ollama.version()
        with suppress(OllamaError):
            models = await service.ollama.model_names()
    rows.append(
        (
            "ok" if available else "fail",
            "Ollama",
            f"{service.settings.ollama_url} (version {version or 'unknown'})"
            if available
            else f"{service.settings.ollama_url} unreachable - run `ollama serve`",
        )
    )
    if models:
        listing = ", ".join(models[:5]) + (" …" if len(models) > 5 else "")
        rows.append(("ok", "Installed models", f"{len(models)}: {listing}"))
    else:
        rows.append(
            (
                "warn" if available else "fail",
                "Installed models",
                "none found - run `ollama pull <model>`",
            )
        )

    chat_model = ollama.get("chat_model") or service.settings.chat_model
    if chat_model:
        installed = _model_installed(str(chat_model), models)
        rows.append(
            (
                "ok" if installed or not models else "warn",
                "Chat model",
                str(chat_model)
                if installed or not models
                else f"{chat_model} (not installed - run `ollama pull {chat_model}`)",
            )
        )
    else:
        rows.append(("warn", "Chat model", "not configured - pass --model or set AMNESIA_MODEL"))

    embed_setting = str(service.settings.embed_model or "")
    embedding_id = service.embedding_model_id
    if embedding_id and embedding_id.startswith("ollama:"):
        rows.append(("ok", "Embeddings", f"{embedding_id} (semantic search active)"))
    else:
        installed = _model_installed(embed_setting, models)
        rows.append(
            (
                "warn",
                "Embeddings",
                f"{embed_setting or 'disabled'} not active - keyword hash fallback in use"
                + (
                    ""
                    if installed
                    else f"; run `ollama pull {embed_setting}` for semantic recall"
                    if embed_setting
                    else ""
                ),
            )
        )

    extraction_model = ollama.get("extraction_model") or service.settings.extraction_model
    if extraction_model:
        rows.append(("ok", "Extraction model", str(extraction_model)))
    else:
        rows.append(
            (
                "warn" if service.settings.extraction_enabled else "ok",
                "Extraction",
                "disabled"
                if not service.settings.extraction_enabled
                else "no extraction model configured",
            )
        )

    table = _new_table(title="Amnesia doctor")
    table.add_column("", width=2, no_wrap=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for status, check, detail in rows:
        table.add_row(_status_icon(status), _text(check), _text(detail))
    console.print(table)

    counts = health.get("counts") or {}
    counts_table = _new_table(title="Stored data")
    counts_table.add_column("Memories", justify="right")
    counts_table.add_column("Events", justify="right")
    counts_table.add_column("Sessions", justify="right")
    counts_table.add_column("Messages", justify="right")
    counts_table.add_column("Embedded", justify="right")
    embedded = int(counts.get("embedded_memories") or 0) + int(counts.get("embedded_events") or 0)
    counts_table.add_row(
        Text(str(counts.get("memories", 0))),
        Text(str(counts.get("events", 0))),
        Text(str(counts.get("sessions", 0))),
        Text(str(counts.get("messages", 0))),
        Text(str(embedded)),
    )
    console.print(counts_table)

    warnings = list(health.get("warnings") or [])
    failures = [row for row in rows if row[0] == "fail"]
    warning_rows = [row for row in rows if row[0] == "warn"]
    for warning in warnings:
        console.print(f"[yellow]![/yellow] [dim]{warning}[/dim]")
    if failures:
        console.print(
            Panel(
                f"{len(failures)} hard failure(s) and {len(warning_rows)} warning(s) found.\n"
                "Fix the rows marked ✗ above and run `amnesia doctor` again.",
                border_style="red",
                padding=(0, 1),
            )
        )
    elif warning_rows:
        console.print(
            Panel(
                f"Amnesia is usable with {len(warning_rows)} warning(s).\n"
                "Warnings only degrade optional features (e.g. semantic recall).",
                border_style="yellow",
                padding=(0, 1),
            )
        )
    else:
        console.print(
            Panel(
                "Everything checks out - Amnesia is healthy.",
                border_style="green",
                padding=(0, 1),
            )
        )
    return len(failures)


@app.command()
@_friendly
def doctor() -> None:
    """Check the database, Ollama and models; exit 1 on hard failures."""
    settings = _load_settings()
    failures = _run(_doctor_check, settings)
    if failures:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def stats(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show memory, activity and storage statistics."""

    async def operation(service: Amnesia) -> dict[str, Any]:
        payload = service.stats().to_dict()
        payload["health"] = {
            "ollama_available": service.ollama_available,
            "embeddings_enabled": service.embeddings_enabled,
            "memory_enabled": service.settings.memory_enabled,
        }
        return payload

    payload = _run(operation, _load_settings())
    if json_output:
        _print_json(payload)
        return
    table = _new_table(title="Amnesia statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = [
        ("Identity memories", payload.get("identity_memories", 0)),
        ("Episode memories", payload.get("episode_memories", 0)),
        ("Total memories", payload.get("memories", 0)),
        ("Events", payload.get("events", 0)),
        ("Sessions", payload.get("sessions", 0)),
        ("Messages", payload.get("messages", 0)),
        ("Embedded memories", payload.get("embedded_memories", 0)),
        ("Embedded events", payload.get("embedded_events", 0)),
        ("Database size", payload.get("db_size_human", "?")),
        ("Database path", payload.get("db_path", "?")),
        ("Embedding model", payload.get("embed_model") or "—"),
        ("Chat model", payload.get("chat_model") or "—"),
        ("Last activity", payload.get("last_activity_human") or "—"),
    ]
    for label, value in rows:
        table.add_row(_text(label), _text(value))
    console.print(table)


# ---------------------------------------------------------------------------
# export / import / backup / purge
# ---------------------------------------------------------------------------


@app.command()
@_friendly
def export(
    destination: Annotated[str, typer.Argument(help="Destination JSON file path.")],
) -> None:
    """Export all memories, events and sessions to a JSON file."""

    async def operation(service: Amnesia) -> tuple[Path, Stats]:
        path = service.export(destination)
        return path, service.stats()

    path, summary = _run(operation, _load_settings())
    size = _human_bytes(path.stat().st_size) if path.exists() else "?"
    console.print(
        Panel(
            Group(
                Text(f"Exported to {path}"),
                Text(
                    f"{summary.memories} memories · {summary.events} events ·"
                    f"{summary.sessions} sessions · {size}",
                    style="dim",
                ),
            ),
            title="[green]Export complete[/green]",
            border_style="green",
            padding=(0, 1),
        )
    )


@app.command("import")
@_friendly
def import_data(
    source: Annotated[str, typer.Argument(help="Exported JSON file to import.")],
    no_merge: Annotated[
        bool,
        typer.Option("--no-merge", help="Do not merge keyed memories; add copies instead."),
    ] = False,
) -> None:
    """Import memories, events and sessions from an export file."""
    path = Path(source).expanduser()
    if not path.exists():
        _fail(f"Import file not found: {path}")
    if not path.is_file():
        _fail(f"Import path is not a file: {path}")

    async def operation(service: Amnesia) -> dict[str, int]:
        return service.import_(path, merge=not no_merge)

    counts = _run(operation, _load_settings())
    table = _new_table(title="Import complete")
    table.add_column("Kind", style="cyan")
    table.add_column("Imported", justify="right")
    total = 0
    for label, key in (
        ("Memories", "memories"),
        ("Events", "events"),
        ("Sessions", "sessions"),
        ("Messages", "messages"),
    ):
        value = int(counts.get(key) or 0)
        total += value
        table.add_row(_text(label), Text(str(value)))
    console.print(table)
    _success(f"Imported {total} item(s) from {path}.")


@app.command()
@_friendly
def backup(
    destination: Annotated[str, typer.Argument(help="Destination path for the SQLite backup.")],
) -> None:
    """Back up the database file (safe copy, WAL friendly)."""

    async def operation(service: Amnesia) -> Path:
        return service.backup(destination)

    path = _run(operation, _load_settings())
    size = _human_bytes(path.stat().st_size) if path.exists() else "?"
    _success(f"Database backed up to {path} ({size})")


@app.command()
@_friendly
def purge(
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Required confirmation: actually delete the data."),
    ] = False,
    events_older_than: Annotated[
        int | None,
        typer.Option(
            "--events-older-than",
            help="Only delete events older than this many days (implies no full wipe).",
        ),
    ] = None,
) -> None:
    """Delete memories/events/sessions (requires --yes)."""
    if not yes:
        console.print(
            Panel(
                "This permanently deletes local memory data.\n"
                "Re-run with [bold]--yes[/bold] to confirm:\n\n"
                "  amnesia purge --yes\n"
                "  amnesia purge --yes --events-older-than 90",
                title="[red]Refusing to purge[/red]",
                border_style="red",
                padding=(0, 1),
            )
        )
        raise typer.Exit(code=1)

    if events_older_than is not None and int(events_older_than) < 0:
        _fail("--events-older-than must be >= 0 days.")

    async def operation(service: Amnesia) -> tuple[str, int]:
        if events_older_than is not None:
            removed = service.store.purge_old_events(days=int(events_older_than))
            return "events", int(removed)
        service.store.purge_all()
        return "all", 0

    mode, removed = _run(operation, _load_settings())
    if mode == "events":
        _success(
            f"Deleted {removed} event(s) older than {events_older_than} day(s)."
        )
    else:
        _success("All memories, events, sessions and messages deleted.")


# ---------------------------------------------------------------------------
# config show
# ---------------------------------------------------------------------------


def _format_setting(value: Any) -> Text:
    if isinstance(value, bool):
        return _styled("on" if value else "off", "green" if value else "dim")
    if value is None or value == "":
        return _styled("—", "dim")
    if isinstance(value, dict):
        return _styled(json.dumps(value, ensure_ascii=False), "dim")
    return _text(value)


@config_app.command("show")
@_friendly
def config_show(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show the resolved configuration (file + environment)."""
    settings = _load_settings()
    payload = settings.to_dict(redact_token=True)
    payload["config_file"] = str(settings.config_file)
    payload["config_file_exists"] = settings.config_file.exists()
    payload["database_exists"] = settings.database.exists()
    payload["effective_chat_model"] = settings.effective_chat_model
    payload["effective_extraction_model"] = settings.effective_extraction_model
    if json_output:
        _print_json(payload)
        return

    groups: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = (
        (
            "Paths",
            (
                ("home", payload["home"]),
                ("database", payload["db_path"]),
                ("config file", payload["config_file"]),
                ("config exists", payload["config_file_exists"]),
            ),
        ),
        (
            "Models",
            (
                ("chat model", payload["effective_chat_model"]),
                ("embedding model", payload["embed_model"]),
                ("extraction model", payload["effective_extraction_model"]),
                ("context window", payload["context_window"]),
            ),
        ),
        (
            "Memory",
            (
                ("memory enabled", payload["memory_enabled"]),
                ("inject context", payload["inject_context"]),
                ("extraction enabled", payload["extraction_enabled"]),
                ("extraction every (turns)", payload["extraction_every_n_turns"]),
                ("extraction min chars", payload["extraction_min_chars"]),
                ("summarization", payload["summarization_enabled"]),
                ("redact secrets", payload["redact_secrets"]),
                ("memory token budget", payload["memory_token_budget"]),
                ("profile token budget", payload["profile_token_budget"]),
                ("recent turns", payload["recent_turns"]),
            ),
        ),
        (
            "Ollama & proxy",
            (
                ("ollama url", payload["ollama_url"]),
                ("request timeout (s)", payload["request_timeout"]),
                ("proxy host", payload["proxy_host"]),
                ("proxy port", payload["proxy_port"]),
                ("api token", payload["api_token"]),
            ),
        ),
        (
            "Sessions & logging",
            (
                ("session gap (min)", payload["session_gap_minutes"]),
                ("log level", payload["log_level"]),
                ("timezone", payload["timezone"]),
            ),
        ),
    )

    console.print(
        Panel(
            f"home [cyan]{settings.home}[/cyan]\n"
            f"config [cyan]{settings.config_file}[/cyan] "
            f"({'exists' if settings.config_file.exists() else 'missing'})",
            title="[bold cyan]Amnesia configuration[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    for group_name, entries in groups:
        table = _new_table(title=group_name)
        table.add_column("Setting", style="dim", no_wrap=True)
        table.add_column("Value", overflow="fold")
        for label, value in entries:
            table.add_row(_text(label), _format_setting(value))
        console.print(table)
    _note("Environment variables (AMNESIA_*) override the config file.")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m`
    app()


__all__ = ["app", "make_service"]
