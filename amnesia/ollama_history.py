"""Import past chats from the Ollama desktop app's private SQLite database.

The Ollama desktop app keeps its own transcript cache in
``%LOCALAPPDATA%\\Ollama\\db.sqlite`` (tables ``chats`` and ``messages``).
Unlike the HTTP API - which is stateless and stores nothing - those transcripts
can be read *after the fact*.  Amnesia can turn them into its own sessions and
messages so older desktop conversations count towards your memory.

This module is deliberately conservative:

* it opens Ollama's database **read-only** (and falls back to ``immutable=1``
  when a live WAL prevents a normal read-only connection),
* it tolerates schema additions / missing optional columns,
* it never writes to the Ollama database and never calls the network,
* imported sessions get deterministic ids (``ollama-desktop-<chat id>``) so
  re-running the importer skips what is already present.

Distilling the imported transcripts into durable facts/events is the CLI's job
(``amnesia import-ollama-history`` does it by default); this module only copies
the conversations.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .store import MemoryStore
from .utils import utc_now_iso

LOG = logging.getLogger("amnesia.ollama_history")

#: Only these roles are imported; tool/thinking rows are not conversations.
IMPORTED_ROLES = frozenset({"user", "assistant", "system"})
#: Prefix of the deterministic Amnesia session ids created by the importer.
SESSION_PREFIX = "ollama-desktop-"
#: Maximum length of a derived session title.
TITLE_MAX_CHARS = 80

_DEFAULT_PLATFORM_PATHS = {
    "darwin": "Library/Application Support/Ollama/db.sqlite",
    "linux": ".local/share/Ollama/db.sqlite",
}


class OllamaHistoryError(RuntimeError):
    """Raised when Ollama's desktop history cannot be read (user-facing)."""


@dataclass(frozen=True)
class HistoryMessage:
    """One message row from Ollama's desktop database."""

    role: str
    content: str
    created_at: str | None = None
    message_id: str | int | None = None
    model: str | None = None


@dataclass(frozen=True)
class HistoryChat:
    """One desktop chat with its messages, oldest-first."""

    chat_id: str
    title: str = ""
    created_at: str | None = None
    messages: tuple[HistoryMessage, ...] = ()
    model: str | None = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def first_message_at(self) -> str | None:
        for message in self.messages:
            if message.created_at:
                return message.created_at
        return None


@dataclass
class ImportResult:
    """Counters returned by :func:`import_history`."""

    chats_found: int = 0
    chats_imported: int = 0
    chats_skipped: int = 0
    messages_imported: int = 0
    session_ids: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(self.chats_imported)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------


def default_history_path() -> Path:
    """Return the default Ollama desktop history path for this platform.

    ``AMNESIA_OLLAMA_HISTORY_DB`` overrides the detected location.
    """
    override = os.environ.get("AMNESIA_OLLAMA_HISTORY_DB", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "Ollama" / "db.sqlite"
    return Path.home() / _DEFAULT_PLATFORM_PATHS.get(sys.platform, _DEFAULT_PLATFORM_PATHS["linux"])


# ---------------------------------------------------------------------------
# Timestamp / title helpers
# ---------------------------------------------------------------------------


def _iso_utc(value: Any) -> str | None:
    """Convert an Ollama timestamp (text with offset, or unix seconds) to UTC ISO."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if abs(seconds) > 1e11:  # milliseconds
            seconds /= 1000.0
        try:
            moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return moment.isoformat(timespec="seconds").replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _iso_utc(int(text))
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _first_line(text: str, width: int = TITLE_MAX_CHARS) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: max(1, width - 1)].rstrip() + "…"


def _derive_title(chat: HistoryChat) -> str:
    if chat.title.strip():
        return _first_line(chat.title)
    for message in chat.messages:
        if message.role == "user" and message.content.strip():
            return _first_line(message.content)
    for message in chat.messages:
        if message.content.strip():
            return _first_line(message.content)
    return "(empty chat)"


def session_id_for(chat_id: str | int) -> str:
    """Deterministic Amnesia session id for an Ollama desktop chat."""
    return f"{SESSION_PREFIX}{chat_id}"


# ---------------------------------------------------------------------------
# Reading Ollama's database
# ---------------------------------------------------------------------------


def _open_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite file without ever writing to it (WAL-safe when possible)."""
    base = path.expanduser().resolve()
    attempts = [
        f"file:{base.as_posix()}?mode=ro",
        # ``immutable=1`` ignores the WAL; used only when a live -wal/-shm
        # prevents the normal read-only connection.
        f"file:{base.as_posix()}?mode=ro&immutable=1",
    ]
    last_error: Exception | None = None
    for uri in attempts:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=2.0)
            connection.row_factory = sqlite3.Row
            connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            return connection
        except sqlite3.Error as exc:
            last_error = exc
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
    raise OllamaHistoryError(
        f"Could not open the Ollama desktop database at {base}: {last_error}. "
        "Close the Ollama desktop app and try again."
    )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _select_expression(columns: set[str], name: str, fallback: str = "NULL") -> str:
    return f'"{name}"' if name in columns else f"{fallback} AS {name}"


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def read_ollama_history(
    db_path: str | Path,
    *,
    limit: int | None = None,
) -> list[HistoryChat]:
    """Read Ollama desktop chats, oldest first, without modifying the database."""
    path = Path(db_path).expanduser()
    if not path.exists():
        raise OllamaHistoryError(f"Ollama desktop database not found: {path}")
    if not path.is_file():
        raise OllamaHistoryError(f"Ollama desktop database path is not a file: {path}")

    connection = _open_read_only(path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "chats" not in tables or "messages" not in tables:
            raise OllamaHistoryError(
                f"{path} does not look like an Ollama desktop database "
                "(missing the 'chats' or 'messages' table)."
            )

        chat_columns = _columns(connection, "chats")
        message_columns = _columns(connection, "messages")
        if "id" not in chat_columns or "id" not in message_columns:
            raise OllamaHistoryError(f"Unsupported Ollama desktop schema in {path}.")

        chat_rows = connection.execute(
            "SELECT "
            + ", ".join(
                [
                    _select_expression(chat_columns, "id"),
                    _select_expression(chat_columns, "title", "''"),
                    _select_expression(chat_columns, "created_at"),
                ]
            )
            + " FROM chats"
        ).fetchall()

        message_rows = connection.execute(
            "SELECT "
            + ", ".join(
                [
                    _select_expression(message_columns, "id"),
                    _select_expression(message_columns, "chat_id"),
                    _select_expression(message_columns, "role", "'user'"),
                    _select_expression(message_columns, "content", "''"),
                    _select_expression(message_columns, "created_at"),
                    _select_expression(message_columns, "model_name"),
                ]
            )
            + " FROM messages ORDER BY id ASC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise OllamaHistoryError(f"Could not read the Ollama desktop database: {exc}") from exc
    finally:
        with suppress(sqlite3.Error):
            connection.close()

    messages_by_chat: dict[str, list[HistoryMessage]] = {}
    for row in message_rows:
        chat_id = _safe_str(row["chat_id"]).strip()
        role = _safe_str(row["role"]).strip().lower() or "user"
        content = _safe_str(row["content"])
        if not chat_id or role not in IMPORTED_ROLES or not content.strip():
            continue
        messages_by_chat.setdefault(chat_id, []).append(
            HistoryMessage(
                role=role,
                content=content,
                created_at=_iso_utc(row["created_at"]),
                message_id=row["id"],
                model=_safe_str(row["model_name"]).strip() or None,
            )
        )

    chats: list[HistoryChat] = []
    for row in chat_rows:
        chat_id = _safe_str(row["id"]).strip()
        if not chat_id:
            continue
        messages = tuple(messages_by_chat.get(chat_id, ()))
        if not messages:
            continue
        model = next((item.model for item in messages if item.model), None)
        chats.append(
            HistoryChat(
                chat_id=chat_id,
                title=_safe_str(row["title"]),
                created_at=_iso_utc(row["created_at"]),
                messages=messages,
                model=model,
            )
        )

    chats.sort(key=lambda item: (item.created_at or "", item.chat_id))
    if limit is not None and int(limit) > 0:
        chats = chats[-int(limit) :]
    return chats


# ---------------------------------------------------------------------------
# Importing into Amnesia
# ---------------------------------------------------------------------------


def import_history(
    store: MemoryStore | None,
    chats: Sequence[HistoryChat],
    *,
    dry_run: bool = False,
) -> ImportResult:
    """Copy Ollama desktop chats into Amnesia sessions/messages.

    Idempotent: a chat already imported under its deterministic session id is
    skipped, so re-running the importer never duplicates sessions or messages.
    A ``None`` store is allowed for a pure dry run (no duplicate detection).
    """
    result = ImportResult(chats_found=len(chats), dry_run=bool(dry_run))
    for chat in chats:
        session_id = session_id_for(chat.chat_id)
        if store is not None and store.get_session(session_id) is not None:
            result.chats_skipped += 1
            continue
        if dry_run or store is None:
            result.chats_imported += 1
            result.messages_imported += chat.message_count
            result.session_ids.append(session_id)
            continue

        started_at = chat.created_at or chat.first_message_at
        store.create_session(
            session_id=session_id,
            model=chat.model,
            client="ollama-desktop",
            title=_derive_title(chat),
            metadata={
                "source": "ollama-desktop",
                "ollama_chat_id": chat.chat_id,
                "imported_at": utc_now_iso(),
            },
            started_at=started_at,
        )
        for message in chat.messages:
            store.add_message(
                session_id,
                message.role,
                message.content,
                created_at=message.created_at,
                metadata={
                    "source": "ollama-desktop",
                    "ollama_message_id": (
                        None if message.message_id is None else str(message.message_id)
                    ),
                },
            )
        result.chats_imported += 1
        result.messages_imported += chat.message_count
        result.session_ids.append(session_id)
    return result


__all__ = [
    "IMPORTED_ROLES",
    "SESSION_PREFIX",
    "HistoryChat",
    "HistoryMessage",
    "ImportResult",
    "OllamaHistoryError",
    "default_history_path",
    "import_history",
    "read_ollama_history",
    "session_id_for",
]
