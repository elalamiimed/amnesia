"""Synchronous persistence layer for memories, events, sessions and messages.

The store is deliberately synchronous: SQLite is fast locally and keeping the
write path simple makes the proxy, CLI and extractor trivially safe to share.
Async orchestration (embeddings, LLM calls) lives in :mod:`amnesia.service`.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from .db import (
    DEFAULT_DB_NAME,
    Database,
    fts_match_query,
    like_pattern,
)
from .embeddings import pack_vector, unpack_vector
from .models import (
    ALL_EVENT_KINDS,
    ALL_MEMORY_KINDS,
    IDENTITY_KINDS,
    EPISODE_KINDS,
    Event,
    Message,
    Memory,
    Session,
    Stats,
    coerce_tags,
)
from .utils import estimate_tokens, json_dumps, json_loads, utc_now_iso

LOG = logging.getLogger("amnesia.store")

MEMORY_COLUMNS = (
    "kinds",
    "kind",
    "subject",
    "key",
    "content",
    "importance",
    "confidence",
    "source",
    "session_id",
    "message_id",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "access_count",
    "tags",
    "active",
    "embedding",
    "embedding_model",
    "embedding_dim",
)

_IDENTITY_ORDER = (
    "CASE kind WHEN 'identity' THEN 0 WHEN 'fact' THEN 1 WHEN 'preference' THEN 2 ELSE 3 END"
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _limit(value: int, default: int = 50, maximum: int = 1000) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, number))


class MemoryStore:
    """Typed access to the Amnesia SQLite database."""

    def __init__(self, path: str | Path | Database) -> None:
        if isinstance(path, Database):
            self.db = path
        else:
            self.db = Database(path)
        self._initialized = False

    # -- lifecycle -------------------------------------------------------
    def initialize(self) -> "MemoryStore":
        self.db.initialize()
        self._initialized = True
        return self

    @classmethod
    def open(cls, home: str | Path) -> "MemoryStore":
        home_path = Path(home)
        store = cls(home_path / DEFAULT_DB_NAME)
        return store.initialize()

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def create_session(
        self,
        *,
        model: str | None = None,
        client: str = "cli",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        started_at: str | None = None,
    ) -> Session:
        session = Session(
            id=session_id or str(uuid.uuid4()),
            started_at=started_at or utc_now_iso(),
            client=client,
            model=model,
            title=title,
            metadata=dict(metadata or {}),
        )
        self.db.execute(
            "INSERT INTO sessions(id, title, model, client, started_at, metadata) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.title,
                session.model,
                session.client,
                session.started_at,
                json_dumps(session.metadata),
            ),
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        row = self.db.query_one(
            "SELECT s.*, (SELECT count(*) FROM messages m WHERE m.session_id = s.id) "
            "AS message_count FROM sessions s WHERE s.id = ?",
            (session_id,),
        )
        return Session.from_row(row) if row else None

    def end_session(self, session_id: str, *, summary: str | None = None) -> None:
        self.db.execute(
            "UPDATE sessions SET ended_at = ?, summary = COALESCE(?, summary) WHERE id = ?",
            (utc_now_iso(), summary, session_id),
        )

    def set_session_title(self, session_id: str, title: str) -> None:
        self.db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    def set_session_summary(self, session_id: str, summary: str) -> None:
        self.db.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))

    def list_sessions(
        self,
        *,
        limit: int = 20,
        model: str | None = None,
        client: str | None = None,
    ) -> list[Session]:
        sql = (
            "SELECT s.*, (SELECT count(*) FROM messages m WHERE m.session_id = s.id) "
            "AS message_count FROM sessions s"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("s.model = ?")
            params.append(model)
        if client:
            clauses.append("s.client = ?")
            params.append(client)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY s.started_at DESC LIMIT ?"
        params.append(_limit(limit))
        return [Session.from_row(row) for row in self.db.query(sql, params)]

    def latest_session(
        self,
        *,
        model: str | None = None,
        client: str | None = None,
    ) -> Session | None:
        sessions = self.list_sessions(limit=1, model=model, client=client)
        return sessions[0] if sessions else None

    def delete_session(self, session_id: str) -> bool:
        existing = self.get_session(session_id)
        if not existing:
            return False
        self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        token_count: int | None = None,
    ) -> Message:
        clean_content = content or ""
        message = Message(
            id=0,
            session_id=session_id,
            role=role,
            content=clean_content,
            created_at=created_at or utc_now_iso(),
            token_count=token_count if token_count is not None else estimate_tokens(clean_content),
            metadata=dict(metadata or {}),
        )
        message_id = self.db.execute(
            "INSERT INTO messages(session_id, role, content, created_at, token_count, metadata) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                message.session_id,
                message.role,
                message.content,
                message.created_at,
                message.token_count,
                json_dumps(message.metadata),
            ),
        )
        message.id = message_id
        return message

    def add_messages(self, session_id: str, messages: Sequence[dict[str, Any]]) -> list[Message]:
        added: list[Message] = []
        for item in messages:
            added.append(
                self.add_message(
                    session_id,
                    str(item.get("role", "user")),
                    str(item.get("content", "")),
                    metadata=item.get("metadata"),
                    created_at=item.get("created_at"),
                )
            )
        return added

    def get_message(self, message_id: int) -> Message | None:
        row = self.db.query_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return Message.from_row(row) if row else None

    def session_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        ascending: bool = True,
    ) -> list[Message]:
        order = "ASC" if ascending else "DESC"
        if limit:
            rows = self.db.query(
                f"SELECT * FROM messages WHERE session_id = ? ORDER BY id {order} LIMIT ?",
                (session_id, _limit(limit)),
            )
            messages = [Message.from_row(row) for row in rows]
            return messages if ascending else list(reversed(messages))
        rows = self.db.query(
            f"SELECT * FROM messages WHERE session_id = ? ORDER BY id {order}",
            (session_id,),
        )
        return [Message.from_row(row) for row in rows]

    def recent_messages(self, session_id: str | None, limit: int = 6) -> list[Message]:
        """Most recent messages for a session, oldest-first (prompt order)."""
        if not session_id:
            return []
        return self.session_messages(session_id, limit=limit, ascending=False)

    def count_messages(self, session_id: str | None = None) -> int:
        if session_id:
            return int(self.db.scalar("SELECT count(*) FROM messages WHERE session_id = ?", (session_id,)))
        return int(self.db.scalar("SELECT count(*) FROM messages"))

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------
    def add_memory(
        self,
        *,
        kind: str,
        content: str,
        subject: str = "user",
        key: str | None = None,
        importance: float = 0.5,
        confidence: float = 0.7,
        source: str = "manual",
        session_id: str | None = None,
        message_id: int | None = None,
        tags: Iterable[str] | str | None = None,
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
        created_at: str | None = None,
    ) -> Memory:
        now = created_at or utc_now_iso()
        tag_list = coerce_tags(tags)
        normalized_kind = str(kind)
        if normalized_kind not in ALL_MEMORY_KINDS:
            LOG.warning("Unknown memory kind %r; storing as note", kind)
            normalized_kind = "note"

        blob: bytes | None = None
        dim: int | None = None
        if embedding is not None and len(embedding) > 0:
            blob = pack_vector(embedding)
            dim = len(embedding)

        memory_id = self.db.execute(
            "INSERT INTO memories(kind, subject, key, content, importance, confidence, source, "
            "session_id, message_id, created_at, updated_at, tags, active, embedding, "
            "embedding_model, embedding_dim) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            (
                normalized_kind,
                subject,
                key,
                content.strip(),
                _clamp(importance),
                _clamp(confidence),
                source,
                session_id,
                message_id,
                now,
                now,
                json_dumps(tag_list),
                blob,
                embedding_model,
                dim,
            ),
        )
        created = self.get_memory(memory_id)
        assert created is not None
        return created

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self.db.query_one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return Memory.from_row(row) if row else None

    def find_memory_by_key(
        self,
        kind: str,
        key: str,
        *,
        subject: str = "user",
    ) -> Memory | None:
        row = self.db.query_one(
            "SELECT * FROM memories WHERE kind = ? AND subject = ? AND key = ? AND active = 1 "
            "ORDER BY updated_at DESC LIMIT 1",
            (kind, subject, key),
        )
        return Memory.from_row(row) if row else None

    def upsert_memory_by_key(
        self,
        *,
        kind: str,
        key: str,
        content: str,
        subject: str = "user",
        importance: float = 0.6,
        confidence: float = 0.7,
        source: str = "manual",
        session_id: str | None = None,
        message_id: int | None = None,
        tags: Iterable[str] | str | None = None,
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
    ) -> Memory:
        """Create or refresh a keyed fact (one canonical statement per key)."""
        existing = self.find_memory_by_key(kind, key, subject=subject)
        if existing is None:
            return self.add_memory(
                kind=kind,
                content=content,
                subject=subject,
                key=key,
                importance=importance,
                confidence=confidence,
                source=source,
                session_id=session_id,
                message_id=message_id,
                tags=tags,
                embedding=embedding,
                embedding_model=embedding_model,
            )
        merged_tags = sorted({*existing.tags, *coerce_tags(tags)})
        updates: dict[str, Any] = {
            "content": content.strip(),
            "importance": max(existing.importance, _clamp(importance)),
            "confidence": max(existing.confidence, _clamp(confidence)),
            "source": source,
            "tags": merged_tags,
            "updated_at": utc_now_iso(),
        }
        if embedding is not None and len(embedding) > 0:
            updates["embedding"] = embedding
            updates["embedding_model"] = embedding_model
        self.update_memory(existing.id, **updates)
        refreshed = self.get_memory(existing.id)
        assert refreshed is not None
        return refreshed

    def update_memory(self, memory_id: int, **fields: Any) -> bool:
        allowed = {
            "kind",
            "subject",
            "key",
            "content",
            "importance",
            "confidence",
            "source",
            "session_id",
            "message_id",
            "tags",
            "active",
            "embedding",
            "embedding_model",
        }
        updates: dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        if "tags" in updates:
            updates["tags"] = json_dumps(coerce_tags(updates["tags"]))
        if "importance" in updates:
            updates["importance"] = _clamp(updates["importance"])
        if "confidence" in updates:
            updates["confidence"] = _clamp(updates["confidence"])
        if "active" in updates:
            updates["active"] = 1 if updates["active"] else 0
        if "embedding" in updates:
            vector = updates.pop("embedding")
            if vector is None:
                updates["embedding"] = None
                updates["embedding_dim"] = None
            else:
                updates["embedding"] = pack_vector(vector)
                updates["embedding_dim"] = len(vector)
        updates["updated_at"] = utc_now_iso()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        params = [*updates.values(), memory_id]
        self.db.execute(f"UPDATE memories SET {assignments} WHERE id = ?", params)
        return True

    def set_memory_embedding(
        self,
        memory_id: int,
        embedding: Sequence[float],
        *,
        embedding_model: str | None,
    ) -> None:
        self.update_memory(
            memory_id,
            embedding=embedding,
            embedding_model=embedding_model,
        )

    def deactivate_memory(self, memory_id: int) -> bool:
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        self.db.execute(
            "UPDATE memories SET active = 0, updated_at = ?, key = NULL WHERE id = ?",
            (utc_now_iso(), memory_id),
        )
        return True

    def delete_memory(self, memory_id: int) -> bool:
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return True

    def list_identity_memories(self, *, limit: int = 30, subject: str = "user") -> list[Memory]:
        rows = self.db.query(
            "SELECT * FROM memories WHERE active = 1 AND subject = ? "
            "AND kind IN ('fact','identity','preference') "
            f"ORDER BY {_IDENTITY_ORDER}, importance DESC, updated_at DESC LIMIT ?",
            (subject, _limit(limit, default=30)),
        )
        return [Memory.from_row(row) for row in rows]

    def list_episode_memories(self, *, limit: int = 30) -> list[Memory]:
        rows = self.db.query(
            "SELECT * FROM memories WHERE active = 1 AND kind IN ('episode','summary','note') "
            "ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (_limit(limit, default=30),),
        )
        return [Memory.from_row(row) for row in rows]

    def list_memories(
        self,
        *,
        kind: str | None = None,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories"
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active = 1")
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(_limit(limit, default=100))
        return [Memory.from_row(row) for row in self.db.query(sql, params)]

    def count_memories(self, *, active_only: bool = True) -> int:
        sql = "SELECT count(*) FROM memories" + (" WHERE active = 1" if active_only else "")
        return int(self.db.scalar(sql))

    def touch_memories(self, memory_ids: Iterable[int]) -> None:
        ids = [int(memory_id) for memory_id in memory_ids]
        if not ids:
            return
        now = utc_now_iso()
        placeholders = ",".join("?" for _ in ids)
        self.db.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now, *ids],
        )

    def find_similar_memories(
        self,
        embedding: Sequence[float],
        *,
        threshold: float = 0.92,
        kinds: Sequence[str] | None = None,
        subject: str | None = None,
        exclude_ids: Iterable[int] = (),
        limit: int = 5,
        embedding_model: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Nearest-neighbour scan used for dedupe/merge decisions."""
        from .embeddings import cosine_similarity

        excluded = {int(item) for item in exclude_ids}
        candidates = self._vector_candidates("memories", kinds=kinds, subject=subject)
        scored: list[tuple[Memory, float]] = []
        for memory, vector, model in candidates:
            if memory.id in excluded:
                continue
            if embedding_model and model and model != embedding_model:
                continue
            similarity = cosine_similarity(embedding, vector)
            if similarity >= threshold:
                memory.similarity = similarity
                scored.append((memory, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:_limit(limit, default=5)]

    # ------------------------------------------------------------------
    # Events (activity timeline)
    # ------------------------------------------------------------------
    def add_event(
        self,
        *,
        title: str,
        kind: str = "note",
        details: str | None = None,
        project: str | None = None,
        tags: Iterable[str] | str | None = None,
        importance: float = 0.5,
        occurred_at: str | None = None,
        session_id: str | None = None,
        message_id: int | None = None,
        source: str = "manual",
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
    ) -> Event:
        now = utc_now_iso()
        normalized_kind = str(kind)
        if normalized_kind not in ALL_EVENT_KINDS:
            normalized_kind = "other"
        blob: bytes | None = None
        dim: int | None = None
        if embedding is not None and len(embedding) > 0:
            blob = pack_vector(embedding)
            dim = len(embedding)
        event_id = self.db.execute(
            "INSERT INTO events(kind, title, details, project, tags, importance, occurred_at, "
            "created_at, session_id, message_id, source, embedding, embedding_model, embedding_dim) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                normalized_kind,
                title.strip(),
                details,
                project,
                json_dumps(coerce_tags(tags)),
                _clamp(importance),
                occurred_at or now,
                now,
                session_id,
                message_id,
                source,
                blob,
                embedding_model,
                dim,
            ),
        )
        event = self.get_event(event_id)
        assert event is not None
        return event

    def get_event(self, event_id: int) -> Event | None:
        row = self.db.query_one("SELECT * FROM events WHERE id = ?", (event_id,))
        return Event.from_row(row) if row else None

    def update_event(self, event_id: int, **fields: Any) -> bool:
        allowed = {
            "kind",
            "title",
            "details",
            "project",
            "tags",
            "importance",
            "occurred_at",
            "embedding",
            "embedding_model",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        if "tags" in updates:
            updates["tags"] = json_dumps(coerce_tags(updates["tags"]))
        if "importance" in updates:
            updates["importance"] = _clamp(updates["importance"])
        if "embedding" in updates:
            vector = updates.pop("embedding")
            if vector is None:
                updates["embedding"] = None
                updates["embedding_dim"] = None
            else:
                updates["embedding"] = pack_vector(vector)
                updates["embedding_dim"] = len(vector)
        assignments = ", ".join(f"{column} = ?" for column in updates)
        self.db.execute(
            f"UPDATE events SET {assignments} WHERE id = ?",
            [*updates.values(), event_id],
        )
        return True

    def delete_event(self, event_id: int) -> bool:
        if not self.get_event(event_id):
            return False
        self.db.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return True

    def list_events(
        self,
        *,
        limit: int = 50,
        days: int | None = None,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[Event]:
        sql = "SELECT * FROM events"
        clauses: list[str] = []
        params: list[Any] = []
        if days is not None and days > 0:
            from .utils import days_ago_iso

            clauses.append("occurred_at >= ?")
            params.append(days_ago_iso(days))
        if project:
            clauses.append("project = ?")
            params.append(project)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurred_at DESC, id DESC LIMIT ?"
        params.append(_limit(limit))
        return [Event.from_row(row) for row in self.db.query(sql, params)]

    def count_events(self) -> int:
        return int(self.db.scalar("SELECT count(*) FROM events"))

    def event_projects(self, *, limit: int = 20) -> list[tuple[str, int]]:
        rows = self.db.query(
            "SELECT project, count(*) AS total FROM events WHERE project IS NOT NULL "
            "AND project != '' GROUP BY project ORDER BY total DESC LIMIT ?",
            (_limit(limit),),
        )
        return [(str(row["project"]), int(row["total"])) for row in rows]

    def events_by_day(self, *, days: int = 30) -> list[tuple[str, int]]:
        from .utils import days_ago_iso

        rows = self.db.query(
            "SELECT substr(occurred_at, 1, 10) AS day, count(*) AS total FROM events "
            "WHERE occurred_at >= ? GROUP BY day ORDER BY day",
            (days_ago_iso(days),),
        )
        return [(str(row["day"]), int(row["total"])) for row in rows]

    # ------------------------------------------------------------------
    # Keyword search (FTS5 with LIKE fallback)
    # ------------------------------------------------------------------
    def _keyword_rows(
        self,
        table: str,
        fts_table: str,
        columns: str,
        query: str,
        *,
        limit: int,
        where: str = "",
        params: Sequence[Any] = (),
        order_extra: str = "",
    ) -> list[sqlite3.Row]:
        match = fts_match_query(query)
        if not match:
            return []
        placeholders = ""
        extra_params: list[Any] = []
        if where:
            placeholders = f" AND ({where})"
            extra_params = list(params)
        if self.db.fts_enabled:
            sql = (
                f"SELECT t.*, bm25({fts_table}) AS _rank FROM {fts_table} "
                f"JOIN {table} t ON t.id = {fts_table}.rowid "
                f"WHERE {fts_table} MATCH ?{placeholders} "
                f"ORDER BY bm25({fts_table}){order_extra} LIMIT ?"
            )
            try:
                return list(
                    self.db.query(sql, [match, *extra_params, _limit(limit)])
                )
            except sqlite3.OperationalError as exc:
                LOG.debug("FTS query failed (%s); falling back to LIKE", exc)
        # LIKE fallback: match the whole query and each significant token.
        import re

        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9_]{2,}", query)
        ][:8] or [query.strip()]
        columns_list = columns.split(",")
        like_groups: list[str] = []
        base_params: list[Any] = []
        for token in tokens:
            like = like_pattern(token)
            like_groups.append(
                "(" + " OR ".join(f"{column} LIKE ? ESCAPE '\\'" for column in columns_list) + ")"
            )
            base_params.extend([like] * len(columns_list))
        sql = f"SELECT t.*, 0.0 AS _rank FROM {table} t WHERE " + " OR ".join(like_groups)
        if where:
            sql += f" AND ({where})"
        sql += " ORDER BY t.rowid DESC LIMIT ?"
        try:
            return list(
                self.db.query(
                    sql,
                    [*base_params, *extra_params, _limit(limit)],
                )
            )
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _fts_score(row: sqlite3.Row) -> float:
        """Convert bm25 rank (lower is better) into a 0..1 relevance score."""
        try:
            rank = float(row["_rank"])
        except (IndexError, KeyError, TypeError, ValueError):
            return 0.5
        if rank == 0.0:
            return 0.5
        return 1.0 / (1.0 + abs(rank))

    def search_memories_keyword(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Sequence[str] | None = None,
        subject: str | None = None,
    ) -> list[tuple[Memory, float]]:
        where_parts = ["t.active = 1"]
        params: list[Any] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            where_parts.append(f"t.kind IN ({placeholders})")
            params.extend(kinds)
        if subject:
            where_parts.append("t.subject = ?")
            params.append(subject)
        rows = self._keyword_rows(
            "memories",
            "memories_fts",
            "content,tags",
            query,
            limit=limit,
            where=" AND ".join(where_parts),
            params=params,
        )
        return [(Memory.from_row(row), self._fts_score(row)) for row in rows]

    def search_events_keyword(
        self,
        query: str,
        *,
        limit: int = 20,
        projects: Sequence[str] | None = None,
    ) -> list[tuple[Event, float]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if projects:
            placeholders = ",".join("?" for _ in projects)
            where_parts.append(f"t.project IN ({placeholders})")
            params.extend(projects)
        if not where_parts:
            where_parts.append("1 = 1")
        rows = self._keyword_rows(
            "events",
            "events_fts",
            "title,details,project,tags",
            query,
            limit=limit,
            where=" AND ".join(where_parts),
            params=params,
        )
        return [(Event.from_row(row), self._fts_score(row)) for row in rows]

    def search_messages_keyword(
        self,
        query: str,
        *,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[Message, float]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if session_id:
            where_parts.append("t.session_id = ?")
            params.append(session_id)
        if not where_parts:
            where_parts.append("1 = 1")
        rows = self._keyword_rows(
            "messages",
            "messages_fts",
            "content",
            query,
            limit=limit,
            where=" AND ".join(where_parts),
            params=params,
        )
        return [(Message.from_row(row), self._fts_score(row)) for row in rows]

    # ------------------------------------------------------------------
    # Vector search (brute-force cosine over float32 BLOBs)
    # ------------------------------------------------------------------
    def _vector_candidates(
        self,
        table: str,
        *,
        kinds: Sequence[str] | None = None,
        subject: str | None = None,
        model: str | None = None,
    ) -> list[tuple[Any, list[float], str | None]]:
        sql = f"SELECT * FROM {table} WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if table == "memories":
            sql += " AND active = 1"
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            sql += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        rows = self.db.query(sql, params)
        candidates: list[tuple[Any, list[float], str | None]] = []
        for row in rows:
            try:
                vector = unpack_vector(row["embedding"])
            except (TypeError, ValueError):
                continue
            if not vector:
                continue
            if model and row["embedding_model"] and row["embedding_model"] != model:
                continue
            model_name = row["embedding_model"] if "embedding_model" in row.keys() else None
            obj = Memory.from_row(row) if table == "memories" else Event.from_row(row)
            candidates.append((obj, vector, model_name))
        return candidates

    def search_memories_vector(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 20,
        kinds: Sequence[str] | None = None,
        subject: str | None = None,
        min_similarity: float = 0.0,
        embedding_model: str | None = None,
    ) -> list[tuple[Memory, float]]:
        from .embeddings import cosine_similarity

        if not embedding:
            return []
        scored: list[tuple[Memory, float]] = []
        for memory, vector, _model in self._vector_candidates(
            "memories", kinds=kinds, subject=subject, model=embedding_model
        ):
            similarity = cosine_similarity(embedding, vector)
            if similarity >= min_similarity:
                memory.similarity = similarity
                scored.append((memory, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: _limit(limit)]

    def search_events_vector(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 20,
        min_similarity: float = 0.0,
        embedding_model: str | None = None,
    ) -> list[tuple[Event, float]]:
        from .embeddings import cosine_similarity

        if not embedding:
            return []
        scored: list[tuple[Event, float]] = []
        for event, vector, _model in self._vector_candidates(
            "events", model=embedding_model
        ):
            similarity = cosine_similarity(embedding, vector)
            if similarity >= min_similarity:
                event.similarity = similarity
                scored.append((event, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: _limit(limit)]

    # ------------------------------------------------------------------
    # Embedding backlog
    # ------------------------------------------------------------------
    def memories_missing_embeddings(self, *, limit: int = 50) -> list[Memory]:
        rows = self.db.query(
            "SELECT * FROM memories WHERE active = 1 AND (embedding IS NULL OR embedding_dim IS NULL) "
            "ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (_limit(limit),),
        )
        return [Memory.from_row(row) for row in rows]

    def events_missing_embeddings(self, *, limit: int = 50) -> list[Event]:
        rows = self.db.query(
            "SELECT * FROM events WHERE embedding IS NULL OR embedding_dim IS NULL "
            "ORDER BY occurred_at DESC LIMIT ?",
            (_limit(limit),),
        )
        return [Event.from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Extraction audit trail
    # ------------------------------------------------------------------
    def record_extraction(
        self,
        *,
        kind: str,
        session_id: str | None = None,
        message_id: int | None = None,
        status: str = "done",
        items: int = 0,
        error: str | None = None,
    ) -> int:
        return self.db.execute(
            "INSERT INTO extractions(session_id, message_id, kind, status, items, error, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (session_id, message_id, kind, status, int(items), error, utc_now_iso()),
        )

    def last_extraction(self, session_id: str) -> sqlite3.Row | None:
        return self.db.query_one(
            "SELECT * FROM extractions WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        )

    def count_turns_since_extraction(self, session_id: str) -> int:
        """User messages added after the last successful fact extraction."""
        row = self.db.query_one(
            "SELECT max(id) AS last_id FROM extractions WHERE session_id = ? AND status = 'done'",
            (session_id,),
        )
        last_id = int(row["last_id"]) if row and row["last_id"] else 0
        return int(
            self.db.scalar(
                "SELECT count(*) FROM messages WHERE session_id = ? AND role = 'user' AND id > ?",
                (session_id, last_id),
            )
        )

    # ------------------------------------------------------------------
    # Stats / export / maintenance
    # ------------------------------------------------------------------
    def stats(
        self,
        *,
        embed_model: str = "",
        chat_model: str | None = None,
    ) -> Stats:
        counts = self.db.stats_counts()
        last_activity = None
        for sql in (
            "SELECT max(created_at) FROM messages",
            "SELECT max(occurred_at) FROM events",
            "SELECT max(updated_at) FROM memories",
        ):
            value = self.db.scalar(sql, default=None)
            if value and (last_activity is None or str(value) > last_activity):
                last_activity = str(value)
        return Stats(
            memories=counts.get("memories", 0),
            identity_memories=counts.get("identity_memories", 0),
            episode_memories=counts.get("episode_memories", 0),
            events=counts.get("events", 0),
            sessions=counts.get("sessions", 0),
            messages=counts.get("messages", 0),
            embedded_memories=counts.get("embedded_memories", 0),
            embedded_events=counts.get("embedded_events", 0),
            db_size_bytes=self.db.size_bytes(),
            db_path=str(self.db.path),
            embed_model=embed_model,
            chat_model=chat_model,
            last_activity=last_activity,
        )

    def export_data(self) -> dict[str, Any]:
        sessions = [Session.from_row(row) for row in self.db.query("SELECT * FROM sessions ORDER BY started_at")]
        messages = [Message.from_row(row) for row in self.db.query("SELECT * FROM messages ORDER BY id")]
        memories = [
            Memory.from_row(row)
            for row in self.db.query("SELECT * FROM memories ORDER BY id")
        ]
        events = [
            Event.from_row(row) for row in self.db.query("SELECT * FROM events ORDER BY id")
        ]
        return {
            "format": "amnesia-export",
            "version": 1,
            "exported_at": utc_now_iso(),
            "sessions": [session.to_dict() for session in sessions],
            "messages": [message.to_dict() for message in messages],
            "memories": [memory.to_dict() for memory in memories],
            "events": [event.to_dict() for event in events],
        }

    def import_data(self, payload: dict[str, Any], *, merge: bool = True) -> dict[str, int]:
        """Import an export payload. Sessions are re-created with fresh ids."""
        if not isinstance(payload, dict):
            raise ValueError("Import payload must be a JSON object")
        counts = {"sessions": 0, "messages": 0, "memories": 0, "events": 0}
        id_map: dict[str, str] = {}
        for session_data in payload.get("sessions", []):
            original_id = str(session_data.get("id", ""))
            session = self.create_session(
                model=session_data.get("model"),
                client=str(session_data.get("client", "import")),
                title=session_data.get("title"),
                metadata={"imported": True, "original_id": original_id},
                started_at=session_data.get("started_at"),
            )
            id_map[original_id] = session.id
            if session_data.get("ended_at"):
                self.end_session(session.id, summary=session_data.get("summary"))
            counts["sessions"] += 1
        for message_data in payload.get("messages", []):
            session_id = id_map.get(str(message_data.get("session_id", "")))
            if not session_id:
                continue
            self.add_message(
                session_id,
                str(message_data.get("role", "user")),
                str(message_data.get("content", "")),
                created_at=message_data.get("created_at"),
                metadata={"imported": True},
            )
            counts["messages"] += 1
        for memory_data in payload.get("memories", []):
            kind = str(memory_data.get("kind", "fact"))
            key = memory_data.get("key")
            content = str(memory_data.get("content", "")).strip()
            if not content:
                continue
            target = self.find_memory_by_key(kind, str(key)) if key else None
            if target and merge:
                self.update_memory(
                    target.id,
                    content=content,
                    importance=float(memory_data.get("importance", 0.5)),
                    confidence=float(memory_data.get("confidence", 0.7)),
                    tags=memory_data.get("tags") or [],
                )
            else:
                session_id = id_map.get(str(memory_data.get("session_id") or ""))
                self.add_memory(
                    kind=kind,
                    content=content,
                    subject=str(memory_data.get("subject", "user")),
                    key=str(key) if key else None,
                    importance=float(memory_data.get("importance", 0.5)),
                    confidence=float(memory_data.get("confidence", 0.7)),
                    source=str(memory_data.get("source", "import")),
                    session_id=session_id,
                    tags=memory_data.get("tags") or [],
                )
            counts["memories"] += 1
        for event_data in payload.get("events", []):
            title = str(event_data.get("title", "")).strip()
            if not title:
                continue
            self.add_event(
                title=title,
                kind=str(event_data.get("kind", "note")),
                details=event_data.get("details"),
                project=event_data.get("project"),
                tags=event_data.get("tags") or [],
                importance=float(event_data.get("importance", 0.5)),
                occurred_at=event_data.get("occurred_at"),
                source=str(event_data.get("source", "import")),
            )
            counts["events"] += 1
        return counts

    def backup_to(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            with target:
                self.db.connection.backup(target)
        finally:
            target.close()
        return destination

    def purge_all(self) -> None:
        """Danger: remove every memory, event, session and message."""
        with self.db.write() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM extractions")

    def purge_old_events(self, *, days: int) -> int:
        from .utils import days_ago_iso

        before = self.count_events()
        self.db.execute("DELETE FROM events WHERE occurred_at < ?", (days_ago_iso(days),))
        return before - self.count_events()

    def last_activity_iso(self) -> str | None:
        values = [
            self.db.scalar("SELECT max(created_at) FROM messages", default=None),
            self.db.scalar("SELECT max(occurred_at) FROM events", default=None),
        ]
        present = [str(value) for value in values if value]
        return max(present) if present else None
