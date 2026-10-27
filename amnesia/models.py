"""Typed views over the Amnesia memory tables.

These dataclasses are the shared currency between the store, retrieval,
service, CLI, proxy and MCP layers. They intentionally mirror the SQLite
schema and add small conveniences (serialization + display helpers).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .utils import estimate_tokens, human_time, json_loads, relative_time


# ---------------------------------------------------------------------------
# Enumerations (string-valued so they round-trip through SQLite/JSON)
# ---------------------------------------------------------------------------


class Role(str):
    """Message roles, kept as plain strings for maximum client compatibility."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryKind(str):
    """Kinds of durable memory.

    Identity-ish kinds (fact/identity/preference) make up the "who the user is"
    profile. Episode-ish kinds (summary/episode/note) are recallable moments or
    explicit notes.
    """

    FACT = "fact"
    IDENTITY = "identity"
    PREFERENCE = "preference"
    EPISODE = "episode"
    SUMMARY = "summary"
    NOTE = "note"


IDENTITY_KINDS: tuple[str, ...] = (
    MemoryKind.FACT,
    MemoryKind.IDENTITY,
    MemoryKind.PREFERENCE,
)
EPISODE_KINDS: tuple[str, ...] = (
    MemoryKind.EPISODE,
    MemoryKind.SUMMARY,
    MemoryKind.NOTE,
)
ALL_MEMORY_KINDS: tuple[str, ...] = IDENTITY_KINDS + EPISODE_KINDS


class EventKind(str):
    """Kinds of activity events tracked on the timeline."""

    CONVERSATION = "conversation"
    PROJECT = "project"
    TASK = "task"
    DECISION = "decision"
    FILE = "file"
    LEARNING = "learning"
    NOTE = "note"
    OTHER = "other"


ALL_EVENT_KINDS: tuple[str, ...] = (
    EventKind.CONVERSATION,
    EventKind.PROJECT,
    EventKind.TASK,
    EventKind.DECISION,
    EventKind.FILE,
    EventKind.LEARNING,
    EventKind.NOTE,
    EventKind.OTHER,
)


def _tags_from_row(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json_loads(str(value), default=[])
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


# ---------------------------------------------------------------------------
# Sessions & messages
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Session:
    id: str
    started_at: str
    client: str = "cli"
    model: str | None = None
    title: str | None = None
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    message_count: int = 0

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Session":
        return cls(
            id=str(row["id"]),
            started_at=str(row["started_at"]),
            client=str(_row_get(row, "client", "cli") or "cli"),
            model=_row_get(row, "model"),
            title=_row_get(row, "title"),
            ended_at=_row_get(row, "ended_at"),
            metadata=json_loads(_row_get(row, "metadata", "{}"), default={}) or {},
            summary=_row_get(row, "summary"),
            message_count=int(_row_get(row, "message_count", 0) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_human"] = human_time(self.started_at)
        data["ended_human"] = human_time(self.ended_at) if self.ended_at else None
        return data


@dataclass(slots=True)
class Message:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Message":
        content = str(row["content"] or "")
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            content=content,
            created_at=str(row["created_at"]),
            token_count=_row_get(row, "token_count") or estimate_tokens(content),
            metadata=json_loads(_row_get(row, "metadata", "{}"), default={}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Memories & events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Memory:
    id: int
    kind: str
    content: str
    created_at: str
    updated_at: str
    subject: str = "user"
    key: str | None = None
    importance: float = 0.5
    confidence: float = 0.7
    source: str = "manual"
    session_id: str | None = None
    message_id: int | None = None
    last_accessed_at: str | None = None
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    active: bool = True
    embedding: list[float] | None = field(default=None, repr=False, compare=False)
    similarity: float | None = None

    @property
    def is_identity(self) -> bool:
        return self.kind in IDENTITY_KINDS

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Memory":
        return cls(
            id=int(row["id"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            subject=str(_row_get(row, "subject", "user") or "user"),
            key=_row_get(row, "key"),
            importance=float(_row_get(row, "importance", 0.5) or 0.5),
            confidence=float(_row_get(row, "confidence", 0.7) or 0.7),
            source=str(_row_get(row, "source", "manual") or "manual"),
            session_id=_row_get(row, "session_id"),
            message_id=_row_get(row, "message_id"),
            last_accessed_at=_row_get(row, "last_accessed_at"),
            access_count=int(_row_get(row, "access_count", 0) or 0),
            tags=_tags_from_row(_row_get(row, "tags", "[]")),
            active=bool(_row_get(row, "active", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "subject": self.subject,
            "key": self.key,
            "content": self.content,
            "importance": round(self.importance, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "tags": self.tags,
            "active": self.active,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "created_human": human_time(self.created_at),
            "relative": relative_time(self.created_at),
        }

    def display_line(self, *, show_id: bool = False) -> str:
        prefix = f"#{self.id} " if show_id else ""
        key = f"{self.key}: " if self.key else ""
        return f"{prefix}{key}{self.content}"


@dataclass(slots=True)
class Event:
    id: int
    kind: str
    title: str
    occurred_at: str
    created_at: str
    details: str | None = None
    project: str | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    session_id: str | None = None
    message_id: int | None = None
    source: str = "manual"
    embedding: list[float] | None = field(default=None, repr=False, compare=False)
    similarity: float | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Event":
        return cls(
            id=int(row["id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            occurred_at=str(row["occurred_at"]),
            created_at=str(row["created_at"]),
            details=_row_get(row, "details"),
            project=_row_get(row, "project"),
            tags=_tags_from_row(_row_get(row, "tags", "[]")),
            importance=float(_row_get(row, "importance", 0.5) or 0.5),
            session_id=_row_get(row, "session_id"),
            message_id=_row_get(row, "message_id"),
            source=str(_row_get(row, "source", "manual") or "manual"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "details": self.details,
            "project": self.project,
            "tags": self.tags,
            "importance": round(self.importance, 3),
            "occurred_at": self.occurred_at,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "source": self.source,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "occurred_human": human_time(self.occurred_at),
            "relative": relative_time(self.occurred_at),
        }

    def display_line(self, *, show_id: bool = False) -> str:
        prefix = f"#{self.id} " if show_id else ""
        project = f"[{self.project}] " if self.project else ""
        when = human_time(self.occurred_at)
        return f"{prefix}{when} {project}{self.title}"


# ---------------------------------------------------------------------------
# Retrieval / context results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RecallHit:
    """One ranked thing retrieved from memory for a query."""

    kind: str  # "memory" | "event" | "message"
    id: int
    content: str
    score: float
    created_at: str | None = None
    memory_kind: str | None = None
    event_kind: str | None = None
    project: str | None = None
    importance: float | None = None
    similarity: float | None = None
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "content": self.content,
            "score": round(self.score, 4),
            "created_at": self.created_at,
            "memory_kind": self.memory_kind,
            "event_kind": self.event_kind,
            "project": self.project,
            "importance": self.importance,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "when": human_time(self.created_at),
        }

    def display_line(self) -> str:
        label = self.memory_kind or self.event_kind or self.kind
        project = f" [{self.project}]" if self.project else ""
        return f"[{label}{project}] {self.content}"


@dataclass(slots=True)
class MemoryContext:
    """The formatted memory block injected into a model's prompt."""

    query: str = ""
    profile: list[Memory] = field(default_factory=list)
    facts: list[Memory] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    episodes: list[Memory] = field(default_factory=list)
    text: str = ""
    tokens_estimate: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.profile or self.facts or self.events or self.episodes)

    @property
    def items_count(self) -> int:
        return len(self.profile) + len(self.facts) + len(self.events) + len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "text": self.text,
            "tokens_estimate": self.tokens_estimate,
            "profile": [m.to_dict() for m in self.profile],
            "facts": [m.to_dict() for m in self.facts],
            "events": [e.to_dict() for e in self.events],
            "episodes": [m.to_dict() for m in self.episodes],
        }


@dataclass(slots=True)
class Stats:
    memories: int = 0
    identity_memories: int = 0
    episode_memories: int = 0
    events: int = 0
    sessions: int = 0
    messages: int = 0
    embedded_memories: int = 0
    embedded_events: int = 0
    db_size_bytes: int = 0
    db_path: str = ""
    embed_model: str = ""
    chat_model: str | None = None
    last_activity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["db_size_human"] = _human_bytes(self.db_size_bytes)
        data["last_activity_human"] = (
            human_time(self.last_activity) if self.last_activity else None
        )
        return data


def _human_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


@dataclass(slots=True)
class Exchange:
    """A user/assistant pair distilled from one conversation turn."""

    user: str
    assistant: str
    model: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        parts = [self.user.strip()]
        if self.assistant.strip():
            parts.append(self.assistant.strip())
        return "\n\n".join(parts)


def is_identity_kind(kind: str) -> bool:
    return kind in IDENTITY_KINDS


def coerce_tags(tags: Iterable[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return [str(tag).strip() for tag in tags if str(tag).strip()]
