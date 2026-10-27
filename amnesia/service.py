"""Async facade over the memory store.

Everything user-facing (CLI, HTTP proxy, MCP server) talks to :class:`Amnesia`:
it owns the store, the embedder and the Ollama client, and implements the
memory-level operations (remember / recall / profile / timeline / context).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Settings, configure_logging, load_settings
from .embeddings import Embedder, create_embedder
from .models import (
    IDENTITY_KINDS,
    Event,
    Memory,
    MemoryContext,
    Message,
    RecallHit,
    Session,
    Stats,
)
from .ollama import OllamaClient, OllamaError
from .retrieval import ContextBuilder, recall as recall_memories
from .store import MemoryStore
from .utils import (
    estimate_tokens,
    redact_secrets,
    truncate,
    utc_now_iso,
)

LOG = logging.getLogger("amnesia.service")


class Amnesia:
    """High-level, async memory service."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        ollama: OllamaClient | None = None,
        builder: ContextBuilder | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or MemoryStore(self.settings.database)
        self._embedder = embedder
        self._ollama = ollama
        self._builder = builder
        self._started = False
        self._embedding_failures = 0
        self.ollama_available: bool | None = None
        self.resolved_chat_model: str | None = self.settings.chat_model
        self.resolved_extraction_model: str | None = self.settings.extraction_model
        self.warnings: list[str] = []

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> "Amnesia":
        configure_logging(self.settings)
        self.settings.ensure_home()
        self.store.initialize()
        self._ollama = self._ollama or OllamaClient.from_settings(self.settings)

        if self._embedder is None:
            try:
                self._embedder = await create_embedder(self.settings)
            except Exception as exc:  # noqa: BLE001 - never block startup
                LOG.warning("Embedding backend unavailable: %s", exc)
                from .embeddings import HashEmbedder

                self._embedder = HashEmbedder()

        try:
            await self._ollama.version()
            self.ollama_available = True
        except OllamaError as exc:
            self.ollama_available = False
            self.warnings.append(str(exc))
            LOG.warning("Ollama is not reachable yet: %s", exc)

        if self.ollama_available:
            try:
                self.resolved_chat_model = await self._ollama.resolve_chat_model(
                    self.settings.chat_model
                )
            except OllamaError as exc:
                self.warnings.append(str(exc))
                LOG.warning("Could not resolve a chat model: %s", exc)
            self.resolved_extraction_model = (
                self.settings.extraction_model or self.resolved_chat_model
            )

        if not self.settings.chat_model and self.resolved_chat_model:
            self.settings.chat_model = self.resolved_chat_model

        self._started = True
        LOG.info(
            "Amnesia ready: db=%s, embeddings=%s, model=%s",
            self.store.db.path,
            self.embedder.name if self.embedder else "none",
            self.resolved_chat_model or "(not resolved)",
        )
        return self

    async def close(self) -> None:
        if self._embedder is not None:
            try:
                await self._embedder.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                LOG.debug("Embedder close failed", exc_info=True)
        if self._ollama is not None:
            await self._ollama.close()
        self.store.close()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    # -- collaborators ---------------------------------------------------
    @property
    def ollama(self) -> OllamaClient:
        if self._ollama is None:
            self._ollama = OllamaClient.from_settings(self.settings)
        return self._ollama

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    @property
    def builder(self) -> ContextBuilder:
        if self._builder is None:
            self._builder = ContextBuilder(
                self.store,
                self.settings,
                default_budget=self.settings.memory_token_budget,
                profile_budget=min(
                    self.settings.profile_token_budget,
                    max(80, self.settings.memory_token_budget // 3),
                ),
            )
        return self._builder

    @property
    def embedding_model_id(self) -> str | None:
        if self._embedder is None or not self._embedder.dim:
            return None
        return self._embedder.name

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self._embedder and self._embedder.dim)

    @property
    def chat_model(self) -> str | None:
        return self.resolved_chat_model or self.settings.chat_model

    @property
    def extraction_model(self) -> str | None:
        return (
            self.resolved_extraction_model
            or self.settings.effective_extraction_model
            or self.chat_model
        )

    # -- embeddings ------------------------------------------------------
    async def embed(self, text: str) -> list[float]:
        if not self.embeddings_enabled or not text.strip():
            return []
        assert self._embedder is not None
        try:
            return await self._embedder.embed_one(text)
        except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
            self._embedding_failures += 1
            LOG.warning("Embedding failed (%s); storing without vector", exc)
            return []

    def _embedding_kwargs(self, vector: Sequence[float] | None) -> dict[str, Any]:
        if not vector:
            return {}
        return {"embedding": list(vector), "embedding_model": self.embedding_model_id}

    # -- identity memory -------------------------------------------------
    async def remember(
        self,
        content: str,
        *,
        kind: str = "fact",
        key: str | None = None,
        subject: str = "user",
        importance: float = 0.6,
        confidence: float = 0.8,
        tags: Iterable[str] | str | None = None,
        source: str = "manual",
        session_id: str | None = None,
        message_id: int | None = None,
        dedupe: bool = True,
    ) -> Memory:
        text = content.strip()
        if not text:
            raise ValueError("Cannot remember empty content")
        if self.settings.redact_secrets:
            text = redact_secrets(text)
        vector = await self.embed(text)

        if key:
            return self.store.upsert_memory_by_key(
                kind=kind,
                key=key,
                content=text,
                subject=subject,
                importance=importance,
                confidence=confidence,
                source=source,
                session_id=session_id,
                message_id=message_id,
                tags=tags,
                **self._embedding_kwargs(vector),
            )

        if dedupe and vector:
            similar = self.store.find_similar_memories(
                vector,
                threshold=0.93,
                kinds=[kind],
                subject=subject,
                limit=1,
                embedding_model=self.embedding_model_id,
            )
            if similar:
                existing, similarity = similar[0]
                self.store.update_memory(
                    existing.id,
                    content=text,
                    importance=max(existing.importance, importance),
                    confidence=max(existing.confidence, confidence),
                    tags=sorted({*existing.tags, *(tags or [])}),
                    **self._embedding_kwargs(vector),
                )
                LOG.debug(
                    "Merged near-duplicate memory #%s (similarity %.2f)",
                    existing.id,
                    similarity,
                )
                refreshed = self.store.get_memory(existing.id)
                if refreshed is not None:
                    return refreshed

        return self.store.add_memory(
            kind=kind,
            content=text,
            subject=subject,
            importance=importance,
            confidence=confidence,
            source=source,
            session_id=session_id,
            message_id=message_id,
            tags=tags,
            **self._embedding_kwargs(vector),
        )

    async def forget(
        self,
        memory_id: int | None = None,
        *,
        key: str | None = None,
        kind: str | None = None,
        subject: str = "user",
        hard: bool = False,
    ) -> bool:
        target: Memory | None = None
        if memory_id is not None:
            target = self.store.get_memory(int(memory_id))
        elif key is not None:
            target = self.store.find_memory_by_key(kind or "fact", key, subject=subject)
        if target is None:
            return False
        return (
            self.store.delete_memory(target.id)
            if hard
            else self.store.deactivate_memory(target.id)
        )

    def profile(self, *, limit: int = 50) -> list[Memory]:
        return self.store.list_identity_memories(limit=limit)

    async def profile_text(self, *, limit: int = 50) -> str:
        memories = await asyncio.to_thread(self.profile, limit=limit)
        if not memories:
            return ""
        return "\n".join(memory.display_line() for memory in memories)

    # -- activity memory -------------------------------------------------
    async def log_event(
        self,
        title: str,
        *,
        kind: str = "note",
        details: str | None = None,
        project: str | None = None,
        tags: Iterable[str] | str | None = None,
        importance: float = 0.5,
        occurred_at: str | None = None,
        session_id: str | None = None,
        message_id: int | None = None,
        source: str = "manual",
    ) -> Event:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Event title cannot be empty")
        clean_details = details.strip() if details else None
        if self.settings.redact_secrets:
            clean_title = redact_secrets(clean_title)
            clean_details = redact_secrets(clean_details) if clean_details else None
        vector = await self.embed(
            f"{clean_title}\n{clean_details or ''}\n{project or ''}".strip()
        )
        return self.store.add_event(
            title=clean_title,
            kind=kind,
            details=clean_details,
            project=project,
            tags=tags,
            importance=importance,
            occurred_at=occurred_at,
            session_id=session_id,
            message_id=message_id,
            source=source,
            **self._embedding_kwargs(vector),
        )

    def timeline(
        self,
        *,
        days: int | None = 7,
        limit: int = 50,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[Event]:
        return self.store.list_events(days=days, limit=limit, project=project, kind=kind)

    def timeline_text(self, *, days: int | None = 7, limit: int = 50) -> str:
        events = self.timeline(days=days, limit=limit)
        return "\n".join(event.display_line() for event in events)

    # -- recall ----------------------------------------------------------
    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        include_events: bool = True,
        include_messages: bool = False,
    ) -> list[RecallHit]:
        vector = await self.embed(query)
        return recall_memories(
            self.store,
            query,
            vector or None,
            limit=limit,
            include_events=include_events,
            include_messages=include_messages,
            embedding_model=self.embedding_model_id,
        )

    # -- context for prompts --------------------------------------------
    async def context_for(
        self,
        user_message: str,
        *,
        budget: int | None = None,
        session_id: str | None = None,
        include_events: bool = True,
        include_episodes: bool = True,
    ) -> MemoryContext:
        if not self.settings.memory_enabled or not user_message.strip():
            return MemoryContext(query=user_message)
        vector = await self.embed(user_message)
        session_summary = None
        if session_id and self.settings.summarization_enabled:
            session = self.store.get_session(session_id)
            if session and session.summary:
                session_summary = truncate(session.summary, 400)
        effective_budget = budget or min(
            self.settings.memory_token_budget,
            max(200, self.settings.context_window // 4),
        )
        return self.builder.build(
            user_message,
            vector or None,
            budget=effective_budget,
            profile_budget=min(
                self.settings.profile_token_budget,
                max(80, effective_budget // 3),
            ),
            embedding_model=self.embedding_model_id,
            session_summary=session_summary,
            include_events=include_events,
            include_episodes=include_episodes,
        )

    @staticmethod
    def augment_messages(
        messages: Sequence[dict[str, Any]],
        memory_block: str,
    ) -> list[dict[str, Any]]:
        """Return a copy of ``messages`` with the memory block added to the system prompt."""
        copied = [dict(message) for message in messages]
        if not memory_block.strip():
            return copied
        if copied and str(copied[0].get("role")) == "system":
            original = str(copied[0].get("content") or "").rstrip()
            separator = "\n\n" if original else ""
            copied[0]["content"] = f"{original}{separator}{memory_block}"
            return copied
        return [{"role": "system", "content": memory_block}, *copied]

    @staticmethod
    def system_message(memory_block: str) -> dict[str, Any]:
        return {"role": "system", "content": memory_block}

    # -- sessions & transcripts -----------------------------------------
    def start_session(
        self,
        *,
        model: str | None = None,
        client: str = "cli",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        return self.store.create_session(
            model=model or self.chat_model,
            client=client,
            title=title,
            metadata=metadata,
            session_id=session_id,
        )

    def get_session(self, session_id: str) -> Session | None:
        return self.store.get_session(session_id)

    async def ensure_session(
        self,
        *,
        session_id: str | None = None,
        model: str | None = None,
        client: str = "cli",
        gap_minutes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Resume the latest session or start a new one when idle too long."""
        if session_id:
            existing = self.store.get_session(session_id)
            if existing:
                return existing
        gap = gap_minutes if gap_minutes is not None else self.settings.session_gap_minutes
        latest = self.store.latest_session(model=model or self.chat_model, client=client)
        if latest is not None:
            last_activity = self.store.last_activity_iso() or latest.started_at
            from .utils import parse_iso, utc_now

            last_dt = parse_iso(last_activity)
            if last_dt is not None:
                idle_minutes = (utc_now() - last_dt).total_seconds() / 60.0
                if idle_minutes <= gap:
                    return latest
        return self.start_session(model=model or self.chat_model, client=client, metadata=metadata)

    async def add_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Message, Message]:
        user_message = self.store.add_message(
            session_id,
            "user",
            user_text,
            metadata={"model": model, **(metadata or {})},
        )
        assistant_message = self.store.add_message(
            session_id,
            "assistant",
            assistant_text,
            metadata={"model": model, **(metadata or {})},
        )
        session = self.store.get_session(session_id)
        if session is not None and not session.title:
            title = truncate(user_text.strip().replace("\n", " "), 80, suffix="")
            if title:
                self.store.set_session_title(session_id, title)
        return user_message, assistant_message

    async def ingest_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Message, Message]:
        return await self.add_turn(
            session_id, user_text, assistant_text, model=model, metadata=metadata
        )

    def recent_turns(
        self,
        session_id: str | None,
        *,
        limit: int | None = None,
    ) -> list[Message]:
        return self.store.recent_messages(
            session_id, limit=limit or self.settings.recent_turns
        )

    def end_session(self, session_id: str, *, summary: str | None = None) -> None:
        self.store.end_session(session_id, summary=summary)

    # -- embeddings backfill --------------------------------------------
    async def backfill_embeddings(self, *, limit: int = 100) -> dict[str, int]:
        if not self.embeddings_enabled:
            return {"memories": 0, "events": 0}
        updated = {"memories": 0, "events": 0}
        for memory in self.store.memories_missing_embeddings(limit=limit):
            vector = await self.embed(memory.content)
            if vector:
                self.store.set_memory_embedding(
                    memory.id, vector, embedding_model=self.embedding_model_id
                )
                updated["memories"] += 1
        for event in self.store.events_missing_embeddings(limit=limit):
            vector = await self.embed(
                f"{event.title}\n{event.details or ''}\n{event.project or ''}"
            )
            if vector:
                self.store.update_event(
                    event.id,
                    embedding=vector,
                    embedding_model=self.embedding_model_id,
                )
                updated["events"] += 1
        return updated

    # -- stats / data management ---------------------------------------
    def stats(self) -> Stats:
        return self.store.stats(
            embed_model=self.embedder.name if self.embedder else "",
            chat_model=self.chat_model,
        )

    def export(self, destination: str | Path) -> Path:
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.store.export_data()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def import_(self, source: str | Path, *, merge: bool = True) -> dict[str, int]:
        path = Path(source).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.store.import_data(payload, merge=merge)

    def backup(self, destination: str | Path) -> Path:
        return self.store.backup_to(destination)

    def health(self) -> dict[str, Any]:
        session = self.store.latest_session()
        return {
            "started": self.started,
            "db_path": str(self.store.db.path),
            "db_size_bytes": self.store.db.size_bytes(),
            "db_integrity": self.store.db.integrity_check(),
            "fts_enabled": self.store.db.fts_enabled,
            "embeddings": {
                "enabled": self.embeddings_enabled,
                "model": self.embedding_model_id,
                "dim": self.embedder.dim if self.embedder else None,
                "failures": self._embedding_failures,
            },
            "ollama": {
                "url": self.settings.ollama_url,
                "available": self.ollama_available,
                "chat_model": self.chat_model,
                "extraction_model": self.extraction_model,
            },
            "settings": {
                "memory_enabled": self.settings.memory_enabled,
                "inject_context": self.settings.inject_context,
                "extraction_enabled": self.settings.extraction_enabled,
                "memory_token_budget": self.settings.memory_token_budget,
            },
            "counts": self.store.db.stats_counts(),
            "latest_session": session.to_dict() if session else None,
            "warnings": self.warnings,
            "timestamp": utc_now_iso(),
        }

    # -- small conveniences used by the CLI ------------------------------
    async def quick_context_preview(self, query: str, *, budget: int = 400) -> str:
        context = await self.context_for(query, budget=budget)
        return context.text or "(no relevant memory yet)"

    def token_estimate(self, text: str) -> int:
        return estimate_tokens(text)


__all__ = ["Amnesia"]
