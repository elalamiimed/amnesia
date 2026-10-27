"""SQLite schema, connection management and FTS5 handling.

One small SQLite file is the entire memory substrate. Design goals:

* WAL mode: the proxy can read while the extractor writes.
* FTS5 external-content indexes kept in sync by triggers.
* Explicit, additive migrations (``PRAGMA user_version``) so a memory store
  survives upgrades - losing memories would be a tragic kind of irony.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .utils import json_dumps, json_loads, utc_now_iso

LOG = logging.getLogger("amnesia.db")

SCHEMA_VERSION = 1
DEFAULT_DB_NAME = "amnesia.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    model      TEXT,
    client     TEXT NOT NULL DEFAULT 'cli',
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    metadata   TEXT NOT NULL DEFAULT '{}',
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    token_count INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT NOT NULL,
    subject          TEXT NOT NULL DEFAULT 'user',
    key              TEXT,
    content          TEXT NOT NULL,
    importance       REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'manual',
    session_id       TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    message_id       INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    tags             TEXT NOT NULL DEFAULT '[]',
    active           INTEGER NOT NULL DEFAULT 1,
    embedding        BLOB,
    embedding_model  TEXT,
    embedding_dim    INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    title           TEXT NOT NULL,
    details         TEXT,
    project         TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    importance      REAL NOT NULL DEFAULT 0.5,
    occurred_at     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    session_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    message_id      INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    source          TEXT NOT NULL DEFAULT 'manual',
    embedding       BLOB,
    embedding_model TEXT,
    embedding_dim   INTEGER
);

CREATE TABLE IF NOT EXISTS extractions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    message_id INTEGER,
    kind       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'done',
    items      INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind, active);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject, active);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_occurred ON events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_extractions_session ON extractions(session_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS ux_memories_kind_subject_key
    ON memories(kind, subject, key)
    WHERE key IS NOT NULL AND active = 1;
"""

# FTS5 is created separately so a build without FTS5 still works (keyword
# search then falls back to LIKE queries).
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    title, details, project, tags,
    content='events',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
        VALUES ('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
        VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, title, details, project, tags)
        VALUES (new.id, new.title, new.details, new.project, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, title, details, project, tags)
        VALUES ('delete', old.id, old.title, old.details, old.project, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, title, details, project, tags)
        VALUES ('delete', old.id, old.title, old.details, old.project, old.tags);
    INSERT INTO events_fts(rowid, title, details, project, tags)
        VALUES (new.id, new.title, new.details, new.project, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class Database:
    """Thread-safe SQLite wrapper used by the synchronous store layer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.fts_enabled = False

    # -- connection ------------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized; call initialize() first")
        return self._conn

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        self._conn = conn
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- lifecycle -------------------------------------------------------
    def initialize(self) -> "Database":
        with self._lock:
            conn = self.connect()
            conn.executescript(SCHEMA_SQL)
            try:
                conn.executescript(FTS_SQL)
                self.fts_enabled = True
            except sqlite3.OperationalError as exc:  # pragma: no cover - depends on build
                LOG.warning("SQLite FTS5 unavailable (%s); keyword search degrades to LIKE", exc)
                self.fts_enabled = False
            current = self._user_version(conn)
            if current == 0:
                self._set_meta(conn, "created_at", utc_now_iso())
                self._set_meta(conn, "schema_version", str(SCHEMA_VERSION))
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif current < SCHEMA_VERSION:
                self._migrate(conn, current)
            elif current > SCHEMA_VERSION:
                LOG.warning(
                    "Database schema v%s is newer than this Amnesia build (v%s)",
                    current,
                    SCHEMA_VERSION,
                )
            conn.commit()
            self._rebuild_fts_if_needed(conn)
        return self

    @staticmethod
    def _user_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply additive migrations. v1 is the initial schema."""
        LOG.info("Migrating Amnesia database from schema v%s to v%s", from_version, SCHEMA_VERSION)
        # Future migrations: if from_version < 2: conn.executescript(MIGRATION_V2)
        self._set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _rebuild_fts_if_needed(self, conn: sqlite3.Connection) -> None:
        """One-time FTS backfill for stores created before FTS was enabled."""
        if not self.fts_enabled:
            return
        try:
            counted = conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
            expected = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
            if counted != expected:
                conn.executescript(
                    "INSERT INTO memories_fts(memories_fts) VALUES('rebuild');"
                    "INSERT INTO events_fts(events_fts) VALUES('rebuild');"
                    "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
                )
                conn.commit()
        except sqlite3.OperationalError:  # pragma: no cover
            LOG.debug("FTS rebuild skipped", exc_info=True)

    # -- access primitives ----------------------------------------------
    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return str(row["value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._set_meta(self.connection, key, value)
            self.connection.commit()

    def get_json_meta(self, key: str, default: Any = None) -> Any:
        raw = self.get_meta(key)
        return json_loads(raw, default=default) if raw is not None else default

    def set_json_meta(self, key: str, value: Any) -> None:
        self.set_meta(key, json_dumps(value))

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Serialize writes; commits on success and rolls back on error."""
        with self._lock:
            conn = self.connection
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.write() as conn:
            cursor = conn.execute(sql, tuple(params))
            return int(cursor.lastrowid or 0)

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        with self.write() as conn:
            conn.executemany(sql, [tuple(row) for row in rows])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = 0) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    # -- health ----------------------------------------------------------
    def stats_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, sql in {
            "memories": "SELECT count(*) FROM memories WHERE active = 1",
            "identity_memories": (
                "SELECT count(*) FROM memories WHERE active = 1 "
                "AND kind IN ('fact','identity','preference')"
            ),
            "episode_memories": (
                "SELECT count(*) FROM memories WHERE active = 1 "
                "AND kind IN ('episode','summary','note')"
            ),
            "events": "SELECT count(*) FROM events",
            "sessions": "SELECT count(*) FROM sessions",
            "messages": "SELECT count(*) FROM messages",
            "embedded_memories": "SELECT count(*) FROM memories WHERE embedding IS NOT NULL",
            "embedded_events": "SELECT count(*) FROM events WHERE embedding IS NOT NULL",
        }.items():
            try:
                counts[name] = int(self.scalar(sql))
            except sqlite3.OperationalError:
                counts[name] = 0
        return counts

    def integrity_check(self) -> str:
        row = self.query_one("PRAGMA integrity_check")
        return str(row[0]) if row else "unknown"

    def size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                if candidate.exists():
                    total += candidate.stat().st_size
            except OSError:
                continue
        return total


def fts_match_query(text: str, *, max_terms: int = 12) -> str:
    """Build a safe FTS5 MATCH expression from free text.

    Uses OR-joined quoted terms so natural language questions still match
    documents that only share some of the words. Returns '' when nothing
    useful can be extracted.
    """
    import re

    terms = re.findall(r"[A-Za-z0-9_]{2,}", text or "")
    seen: list[str] = []
    for term in terms:
        lowered = term.lower()
        if lowered not in seen:
            seen.append(lowered)
        if len(seen) >= max_terms:
            break
    if not seen:
        return ""
    return " OR ".join(f'"{term}"' for term in seen)


def like_pattern(text: str) -> str:
    """Escape a string for a LIKE query and wrap it in wildcards."""
    escaped = (text or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
