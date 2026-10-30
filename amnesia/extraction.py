"""LLM-powered extraction and distillation for Amnesia.

The local chat model is the only source of understanding in Amnesia, so this
module owns the "intelligence" pipeline: it asks the local model to distil a
conversation exchange (or a whole session) into durable profile facts, concrete
activity events and a short session summary.

Pipeline::

    exchange / session transcript
        -> prompt (strict JSON contract + known profile to avoid duplicates)
        -> ollama.chat(..., format_="json")
        -> tolerant JSON parsing + validation/clipping (never trust the model)
        -> ExtractionResult
        -> service.remember(...) / service.log_event(...)
           (dedupe, secret redaction and embeddings are applied there)
        -> extraction audit row (store.record_extraction)

Design rules
------------
* Never raise because of bad model output: log it, record an ``error`` audit
  row and return an empty result.
* Every write goes through the :class:`~amnesia.service.Amnesia` facade so
  redaction, dedupe and embedding stay in one place.
* Background work goes through :class:`ExtractionScheduler`, which is safe to
  call from a request handler and is always drained/cancelled on shutdown.
* Prompts are deliberately small: CPU-only Ollama models are slow and the
  extraction call should never dominate a chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .config import Settings, load_settings
from .models import ALL_EVENT_KINDS, IDENTITY_KINDS, Message, Session
from .utils import (
    UTC,
    collapse_whitespace,
    extract_json_object,
    parse_iso,
    redact_secrets,
    to_iso,
    truncate,
    utc_now,
    utc_now_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from .ollama import OllamaClient
    from .service import Amnesia
    from .store import MemoryStore

LOG = logging.getLogger("amnesia.extraction")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 12
MAX_TURNS_HARD_LIMIT = 50

MAX_FACTS = 8
MAX_EVENTS = 10
MAX_CONTENT_CHARS = 400
MAX_TITLE_CHARS = 160
MAX_DETAILS_CHARS = 500
MAX_SUMMARY_CHARS = 600
MAX_MESSAGE_CHARS = 1500
MAX_TRANSCRIPT_CHARS = 7000
MAX_PROFILE_CHARS = 1200
MAX_AUDIT_ERROR_CHARS = 300

#: Options sent to Ollama for the extraction call (small, deterministic output).
EXTRACTION_OPTIONS: dict[str, Any] = {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_predict": 700,
    "num_ctx": 4096,
}

#: Options for the summary-only call (shorter answer).
SUMMARY_OPTIONS: dict[str, Any] = {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_predict": 256,
    "num_ctx": 4096,
}

AUDIT_KIND_EXCHANGE = "exchange"
AUDIT_KIND_DISTILL = "distill"
AUDIT_KIND_SUMMARY = "summary"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are Amnesia's memory distiller. Read a chat between a user and their local
AI assistant and extract durable long-term memory from it.

Return ONE JSON object and nothing else:
{
  "facts": [
    {"kind": "identity|fact|preference",
     "key": "short slot name or null",
     "content": "canonical short statement",
     "importance": 0.0,
     "confidence": 0.0}
  ],
  "events": [
    {"kind": "project|task|decision|file|learning|note|other",
     "title": "short action title",
     "details": "one optional sentence or null",
     "project": "project name or null",
     "importance": 0.0,
     "occurred_at": "ISO-8601 timestamp or null"}
  ],
  "summary": "one to three sentences, or null"
}

Rules:
- Facts are DURABLE information about the user: identity (name, location, job,
  education, languages), stable preferences (communication style, tools,
  routines, likes/dislikes), ongoing projects and constraints.
- Ignore small talk, one-off questions, temporary states and anything that is
  only about the assistant.
- Write facts as canonical third-person statements, never as quotes:
  "Prefers concise answers", not "The user said they prefer concise answers".
- Use a short lowercase key only for slots that can change later: name,
  location, job, education, language, editor, os, timezone. Otherwise null.
- Events are concrete things the user did, decided, built, fixed or learned.
  Capture them even when mentioned in passing ("today I finished X"). Skip
  greetings, small talk and hypotheticals.
- occurred_at is an ISO-8601 UTC timestamp, only when the text clearly states a
  date (including "yesterday" relative to the given date); otherwise null.
- importance and confidence are decimals between 0.0 and 1.0: use ~0.9 for core
  identity facts, ~0.6-0.8 for stable preferences/projects, ~0.4 for minor
  details. Never leave them at 0.
- summary summarizes the exchange/session in one to three sentences, or null.
- Never include secrets: no passwords, API keys, tokens or credentials.
- Output strict JSON only: no markdown fences, no comments, no prose."""

SUMMARY_SYSTEM_PROMPT = """\
You summarize one conversation session for a long-term memory system.

Return ONE JSON object and nothing else:
{"summary": "one to three sentences describing what the user wanted, what was
decided or built, and what is still open"}

Rules:
- Focus on the durable outcome, not a turn-by-turn narration.
- If a previous summary is provided, refine it instead of starting over.
- Never include secrets or credentials.
- If nothing is worth remembering, return {"summary": null}.
- Strict JSON only: no markdown fences, no comments, no prose."""

#: Closing instruction placed right before generation: local models follow this
#: much more reliably than the same rule buried in the system prompt.
FINAL_DIRECTIVE = (
    "Return the JSON object now. Always fill BOTH lists: include an entry in "
    "events for every concrete piece of work the user reports doing "
    "(finished, built, fixed, decided, learned)."
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExtractionResult:
    """Validated model output for one extraction call.

    ``facts``/``events`` only contain entries that passed validation and have
    been clipped to sane lengths, so persistence code can trust them.
    ``raw`` is the parsed model payload (``None`` when the model did not return
    usable JSON).
    """

    facts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def item_count(self) -> int:
        """Number of facts plus events in the result."""
        return len(self.facts) + len(self.events)

    @property
    def is_empty(self) -> bool:
        """True when the model found nothing worth remembering."""
        return not self.facts and not self.events and not self.summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [dict(fact) for fact in self.facts],
            "events": [dict(event) for event in self.events],
            "summary": self.summary,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------

_NONE_STRINGS = frozenset(
    {
        "",
        "-",
        "n/a",
        "na",
        "nil",
        "none",
        "not specified",
        "not stated",
        "null",
        "unknown",
        "unspecified",
    }
)

_MEMORY_KIND_ALIASES: dict[str, str] = {
    "personal": "identity",
    "profile": "identity",
    "who": "identity",
    "knowledge": "fact",
    "facts": "fact",
    "info": "fact",
    "information": "fact",
    "pref": "preference",
    "preferences": "preference",
    "like": "preference",
    "likes": "preference",
    "style": "preference",
}

_EVENT_KIND_ALIASES: dict[str, str] = {
    "action": "task",
    "activity": "task",
    "code": "file",
    "coding": "file",
    "doc": "file",
    "document": "file",
    "feature": "project",
    "fix": "task",
    "learned": "learning",
    "learning": "learning",
    "note": "note",
    "other": "other",
    "plan": "task",
    "repo": "project",
    "work": "project",
}

_QUALITATIVE_SCORES: dict[str, float] = {
    "critical": 0.95,
    "high": 0.85,
    "low": 0.35,
    "medium": 0.6,
    "moderate": 0.55,
    "very high": 0.95,
    "very low": 0.2,
}


def _is_noneish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_STRINGS
    return False


def _as_list(value: Any) -> list[Any]:
    """Coerce a model field into a list without raising."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return [value]


def _clamp_score(value: Any, default: float) -> float:
    """Turn a score-ish model value into a float in ``(0, 1]``.

    Small local models routinely emit ``importance: 0.0`` / ``confidence: 0.0``
    as a "no opinion" placeholder; a zero score carries no information and would
    rank fresh memories last, so it falls back to the caller's default.
    """
    if isinstance(value, bool):
        return 1.0 if value else default
    number: float | None = None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().lower().rstrip("%").strip()
        if text in _QUALITATIVE_SCORES:
            return _QUALITATIVE_SCORES[text]
        try:
            number = float(text)
        except ValueError:
            return default
    if number is None or not math.isfinite(number) or number <= 0.0:
        return default
    if 1.0 < number <= 100.0:
        number /= 100.0
    return min(1.0, number)


def _first_text(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty string-ish value for one of ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip() and not _is_noneish(value):
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return ""


def _clean_optional(value: Any, *, limit: int) -> str | None:
    """Collapse a model-provided optional text field; ``None`` when empty."""
    if _is_noneish(value):
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if not _is_noneish(item)]
        text = "; ".join(part for part in parts if part)
    else:
        text = collapse_whitespace(str(value))
    if not text:
        return None
    return truncate(text, limit)


def _clean_key(value: Any) -> str | None:
    """Normalize an upsertable slot name (``"Name"`` -> ``"name"``)."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text or text in _NONE_STRINGS:
        return None
    text = re.sub(r"[^a-z0-9_.\-]+", "_", text.replace(" ", "_"))
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text[:48] or None


def _normalize_memory_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in IDENTITY_KINDS:
        return text
    return _MEMORY_KIND_ALIASES.get(text, "fact")


def _normalize_event_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ALL_EVENT_KINDS:
        return text
    return _EVENT_KIND_ALIASES.get(text, "note")


def _normalize_occurred_at(value: Any) -> str | None:
    """Validate a model-provided date; return a UTC ISO string when sane."""
    if _is_noneish(value):
        return None
    if isinstance(value, datetime):
        parsed: datetime | None = value
    elif isinstance(value, str):
        parsed = parse_iso(value.strip())
    else:
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    now = utc_now()
    if parsed > now + timedelta(days=1):
        return None
    if parsed.year < 1990:
        return None
    return to_iso(parsed)


def _coerce_fact(item: Any) -> dict[str, Any] | None:
    """Validate one model fact into ``{kind, key, content, importance, confidence}``."""
    if isinstance(item, str):
        mapping: dict[str, Any] = {"content": item}
    elif isinstance(item, Mapping):
        mapping = dict(item)
    else:
        return None

    content = _first_text(
        mapping, ("content", "statement", "fact", "text", "value", "description")
    )
    content = truncate(collapse_whitespace(content), MAX_CONTENT_CHARS)
    if not content:
        return None

    kind = _normalize_memory_kind(mapping.get("kind") or mapping.get("type"))
    key = _clean_key(
        mapping.get("key") or mapping.get("slot") or mapping.get("name_or_slot")
    )
    return {
        "kind": kind,
        "key": key,
        "content": content,
        "importance": _clamp_score(mapping.get("importance"), 0.6),
        "confidence": _clamp_score(mapping.get("confidence"), 0.75),
    }


def _coerce_event(item: Any) -> dict[str, Any] | None:
    """Validate one model event into the event dict consumed by ``log_event``."""
    if isinstance(item, str):
        mapping: dict[str, Any] = {"title": item}
    elif isinstance(item, Mapping):
        mapping = dict(item)
    else:
        return None

    title = _first_text(
        mapping, ("title", "event", "name", "summary", "what", "action")
    )
    title = truncate(collapse_whitespace(title), MAX_TITLE_CHARS)
    if not title:
        return None

    details = _clean_optional(
        mapping.get("details") or mapping.get("description") or mapping.get("notes"),
        limit=MAX_DETAILS_CHARS,
    )
    project = _clean_optional(
        mapping.get("project") or mapping.get("repo") or mapping.get("repository"),
        limit=80,
    )
    return {
        "kind": _normalize_event_kind(mapping.get("kind") or mapping.get("type")),
        "title": title,
        "details": details,
        "project": project,
        "importance": _clamp_score(mapping.get("importance"), 0.5),
        "occurred_at": _normalize_occurred_at(
            mapping.get("occurred_at")
            or mapping.get("when")
            or mapping.get("date")
            or mapping.get("timestamp")
        ),
    }


def _coerce_summary(value: Any) -> str | None:
    """Normalize a model summary (string, list of sentences or dict)."""
    if _is_noneish(value):
        return None
    if isinstance(value, Mapping):
        return _coerce_summary(value.get("summary") or value.get("text"))
    if isinstance(value, (list, tuple)):
        value = " ".join(str(part).strip() for part in value if not _is_noneish(part))
    text = collapse_whitespace(str(value)).strip().strip('"').strip("'")
    if not text or _is_noneish(text):
        return None
    text = re.sub(r"(?i)^(session\s+)?summary\s*[:\-]\s*", "", text).strip()
    if not text:
        return None
    return truncate(text, MAX_SUMMARY_CHARS)


# ---------------------------------------------------------------------------
# Tolerant JSON parsing
# ---------------------------------------------------------------------------


def _balanced_slice(text: str, start: int, opener: str, closer: str) -> str | None:
    """Return the balanced ``opener..closer`` block starting at ``start``.

    When the block never closes (truncated model output) the remainder of the
    text is returned so callers can still salvage leading items.
    """
    if start < 0 or start >= len(text) or text[start] != opener:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _objects_in_array(array_text: str) -> list[dict[str, Any]]:
    """Extract every valid JSON object from a (possibly truncated) array text."""
    objects: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = array_text.find("{", cursor)
        if start == -1:
            break
        segment = _balanced_slice(array_text, start, "{", "}")
        if not segment:
            break
        try:
            parsed = json.loads(segment)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            objects.append(parsed)
        cursor = start + max(1, len(segment))
    return objects


def _salvage_array(text: str, key: str) -> list[Any] | None:
    """Best-effort recovery of one array field from truncated JSON."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if match is None:
        return None
    start = text.find("[", match.start())
    if start == -1:
        return None
    array_text = _balanced_slice(text, start, "[", "]")
    if not array_text:
        return None
    try:
        parsed = json.loads(array_text)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    objects = _objects_in_array(array_text)
    return objects or None


def _salvage_string(text: str, key: str) -> str | None:
    """Best-effort recovery of one string field from truncated JSON."""
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL
    )
    if match is None:
        return None
    try:
        return str(json.loads(f'"{match.group(1)}"'))
    except ValueError:
        return match.group(1)


def _salvage_payload(text: str) -> dict[str, Any] | None:
    """Rebuild a payload dict from malformed/truncated JSON fragments."""
    salvaged: dict[str, Any] = {}
    facts = _salvage_array(text, "facts")
    if facts is not None:
        salvaged["facts"] = facts
    events = _salvage_array(text, "events")
    if events is not None:
        salvaged["events"] = events
    summary = _salvage_string(text, "summary")
    if summary is not None:
        salvaged["summary"] = summary
    return salvaged or None


def _repair_json(text: str) -> str:
    """Cheap repairs for the JSON mistakes local models make constantly."""
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
    repaired = (
        repaired.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return repaired


def _looks_like_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in ("facts", "events", "summary"))


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    """Extract an extraction payload from raw model text, tolerating junk."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text.strip()

    parsed = extract_json_object(candidate)
    if _looks_like_payload(parsed):
        return parsed  # type: ignore[return-value]

    repaired = _repair_json(candidate)
    repaired_parsed = extract_json_object(repaired)
    if _looks_like_payload(repaired_parsed):
        return repaired_parsed  # type: ignore[return-value]

    salvaged = _salvage_payload(repaired)
    if salvaged:
        LOG.debug("Salvaged partial extraction JSON from malformed model output")
        return salvaged

    # A syntactically valid but schema-less object is still "JSON returned";
    # the caller stores nothing but should not count it as a hard failure.
    if isinstance(parsed, dict):
        return parsed
    if isinstance(repaired_parsed, dict):
        return repaired_parsed
    return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """Turns conversation text into stored facts, events and session summaries.

    Parameters
    ----------
    service:
        A started :class:`~amnesia.service.Amnesia` instance (the facade that
        owns the store and Ollama client).
    model:
        Optional model override for extraction; defaults to
        ``service.extraction_model`` and then ``service.chat_model``.
    ollama:
        Optional client override (tests / advanced wiring).
    settings:
        Optional settings override; defaults to ``service.settings``.
    """

    def __init__(
        self,
        service: "Amnesia",
        *,
        model: str | None = None,
        ollama: "OllamaClient | None" = None,
        settings: Settings | None = None,
    ) -> None:
        if service is None:
            raise ValueError("MemoryExtractor requires an Amnesia service")
        self.service = service
        self._model = (model or "").strip() or None
        self._ollama = ollama
        self.settings: Settings = (
            settings or getattr(service, "settings", None) or load_settings()
        )
        try:
            self._store: "MemoryStore | None" = getattr(service, "store", None)
        except Exception:  # noqa: BLE001 - a broken service must not break init
            LOG.debug("Could not access service.store", exc_info=True)
            self._store = None

    # -- collaborators ---------------------------------------------------
    @property
    def store(self) -> "MemoryStore | None":
        return self._store

    @property
    def default_model(self) -> str | None:
        """Resolve the extraction model: explicit -> service -> settings."""
        candidates: list[Any] = [
            self._model,
            getattr(self.service, "extraction_model", None),
            getattr(self.service, "chat_model", None),
        ]
        settings = self.settings
        candidates.extend(
            [
                getattr(settings, "extraction_model", None),
                getattr(settings, "chat_model", None),
            ]
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @property
    def extraction_enabled(self) -> bool:
        return bool(getattr(self.settings, "extraction_enabled", False))

    def _get_ollama(self) -> "OllamaClient":
        if self._ollama is not None:
            return self._ollama
        client = getattr(self.service, "ollama", None)
        if client is None:
            raise RuntimeError("No Ollama client available for extraction")
        return client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def extract_exchange(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str | None = None,
    ) -> ExtractionResult:
        """Ask the model for facts/events/summary about one user/assistant turn.

        This method is intentionally side-effect free apart from the extraction
        audit row: persistence is the job of :meth:`distill_session`. It never
        raises because of model or transport problems.
        """
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not user and not assistant:
            return ExtractionResult()

        model = self.default_model
        if not model:
            LOG.warning("No extraction model configured; skipping exchange extraction")
            self._audit(
                kind=AUDIT_KIND_EXCHANGE,
                session_id=session_id,
                status="error",
                error="no extraction model available",
            )
            return ExtractionResult()

        try:
            messages = self._exchange_messages(user, assistant)
            raw_text = await self._chat_json(messages, options=EXTRACTION_OPTIONS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - extraction is best effort
            LOG.warning("Exchange extraction failed: %s", exc)
            self._audit(
                kind=AUDIT_KIND_EXCHANGE,
                session_id=session_id,
                status="error",
                error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
            )
            return ExtractionResult()

        result = self._parse_result(raw_text)
        if result.raw is None:
            LOG.warning(
                "Model returned unparseable extraction JSON: %s",
                truncate(raw_text, 200),
            )
            self._audit(
                kind=AUDIT_KIND_EXCHANGE,
                session_id=session_id,
                status="error",
                error="invalid JSON from model",
            )
            return result

        self._audit(
            kind=AUDIT_KIND_EXCHANGE,
            session_id=session_id,
            status="done",
            items=result.item_count + (1 if result.summary else 0),
        )
        return result

    async def extract_session_summary(self, session_id: str) -> str | None:
        """Summarize a session, persist the summary and return it.

        The summary is written to ``sessions.summary`` and mirrored as a
        ``kind="summary"`` memory (keyed per session so refreshes update instead
        of piling up). Explicit calls run even when automatic summarization is
        disabled; only the automatic pipeline honours that setting.
        """
        try:
            if not session_id or self.store is None:
                return None
            session = self.store.get_session(session_id)
            if session is None:
                LOG.warning("Cannot summarize unknown session %s", session_id)
                return None
            messages = self._recent_messages(session_id, DEFAULT_MAX_TURNS)
            transcript = self._format_transcript(messages)
            if not transcript.strip():
                LOG.debug("Session %s has no messages to summarize", session_id)
                return None
            if not self.default_model:
                LOG.warning("No extraction model configured; skipping session summary")
                self._audit(
                    kind=AUDIT_KIND_SUMMARY,
                    session_id=session_id,
                    status="error",
                    error="no extraction model available",
                )
                return None

            prompt = self._summary_messages(
                transcript, session=session, previous_summary=session.summary
            )
            raw_text = await self._chat_json(prompt, options=SUMMARY_OPTIONS)
            summary = self._parse_summary(raw_text)
            if not summary:
                LOG.warning(
                    "Model returned no usable summary for session %s: %s",
                    session_id,
                    truncate(raw_text, 200),
                )
                self._audit(
                    kind=AUDIT_KIND_SUMMARY,
                    session_id=session_id,
                    status="error",
                    error="invalid summary from model",
                )
                return None

            if not await self._persist_summary(session_id, summary):
                self._audit(
                    kind=AUDIT_KIND_SUMMARY,
                    session_id=session_id,
                    status="error",
                    error="could not persist summary",
                )
                return None
            self._audit(
                kind=AUDIT_KIND_SUMMARY,
                session_id=session_id,
                status="done",
                items=1,
            )
            return _coerce_summary(summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - extraction is best effort
            LOG.warning("Summary extraction failed for session %s: %s", session_id, exc)
            self._audit(
                kind=AUDIT_KIND_SUMMARY,
                session_id=session_id,
                status="error",
                error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
            )
            return None

    async def distill_session(
        self,
        session_id: str,
        *,
        force: bool = False,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict[str, Any]:
        """Extract and persist facts/events/summary for one session.

        Returns ``{"facts": n, "events": n, "summary": str | None,
        "skipped": bool}`` (plus an ``error`` key when something went wrong).
        ``force=True`` bypasses the cadence/min-length/summarization gates and
        is what the explicit ``amnesia distill`` command should use.
        """
        try:
            if not session_id or self.store is None:
                return self._result(facts=0, events=0, summary=None, skipped=True)
            session = self.store.get_session(session_id)
            if session is None:
                LOG.warning("Cannot distill unknown session %s", session_id)
                return self._result(
                    facts=0,
                    events=0,
                    summary=None,
                    skipped=True,
                    error=f"unknown session {session_id}",
                )

            turns = self._normalize_turns(max_turns)
            messages = self._recent_messages(session_id, turns)
            if not messages:
                LOG.debug("Session %s has no messages; nothing to distill", session_id)
                return self._result(facts=0, events=0, summary=None, skipped=True)

            if not force:
                every = self._every_n_turns()
                if self._turns_since(session_id) < every:
                    LOG.debug("Session %s: extraction cadence not reached", session_id)
                    return self._result(
                        facts=0,
                        events=0,
                        summary=None,
                        skipped=True,
                        error="cadence",
                    )
                min_chars = self._min_chars()
                if self._combined_length(messages) < min_chars:
                    LOG.debug(
                        "Session %s: transcript below extraction_min_chars (%s)",
                        session_id,
                        min_chars,
                    )
                    return self._result(
                        facts=0,
                        events=0,
                        summary=None,
                        skipped=True,
                        error="min_chars",
                    )

            if not self.default_model:
                LOG.warning("No extraction model configured; skipping distillation")
                self._audit(
                    kind=AUDIT_KIND_DISTILL,
                    session_id=session_id,
                    status="error",
                    error="no extraction model available",
                )
                return self._result(
                    facts=0,
                    events=0,
                    summary=None,
                    skipped=False,
                    error="no extraction model available",
                )

            prompt = self._transcript_messages(messages, session=session)
            try:
                raw_text = await self._chat_json(prompt, options=EXTRACTION_OPTIONS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - extraction is best effort
                LOG.warning("Distillation failed for session %s: %s", session_id, exc)
                self._audit(
                    kind=AUDIT_KIND_DISTILL,
                    session_id=session_id,
                    status="error",
                    error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
                )
                return self._result(
                    facts=0,
                    events=0,
                    summary=None,
                    skipped=False,
                    error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
                )

            result = self._parse_result(raw_text)
            if result.raw is None:
                LOG.warning(
                    "Model returned unparseable extraction JSON for session %s: %s",
                    session_id,
                    truncate(raw_text, 200),
                )
                self._audit(
                    kind=AUDIT_KIND_DISTILL,
                    session_id=session_id,
                    status="error",
                    error="invalid JSON from model",
                )
                return self._result(
                    facts=0,
                    events=0,
                    summary=None,
                    skipped=False,
                    error="invalid JSON from model",
                )

            facts_stored, fact_failures = await self._store_facts(
                result.facts, session_id=session_id
            )
            events_stored, event_failures = await self._store_events(
                result.events, session_id=session_id
            )

            summary: str | None = None
            should_summarize = force or bool(
                getattr(self.settings, "summarization_enabled", True)
            )
            if result.summary and should_summarize:
                if await self._persist_summary(session_id, result.summary):
                    summary = _coerce_summary(result.summary)
                    if summary and getattr(self.settings, "redact_secrets", True):
                        # Keep the returned preview consistent with what was
                        # actually persisted (and with the CLI echo).
                        summary = redact_secrets(summary)

            failures = fact_failures + event_failures
            items = facts_stored + events_stored + (1 if summary else 0)
            error = f"{failures} item(s) failed to persist" if failures else None
            status = "done" if failures == 0 else ("partial" if items else "error")
            self._audit(
                kind=AUDIT_KIND_DISTILL,
                session_id=session_id,
                status=status,
                items=items,
                error=error,
            )
            return self._result(
                facts=facts_stored,
                events=events_stored,
                summary=summary,
                skipped=False,
                error=error,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - extraction is best effort
            LOG.warning(
                "Distillation crashed for session %s: %s", session_id, exc, exc_info=True
            )
            self._audit(
                kind=AUDIT_KIND_DISTILL,
                session_id=session_id,
                status="error",
                error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
            )
            return self._result(
                facts=0,
                events=0,
                summary=None,
                skipped=False,
                error=truncate(str(exc), MAX_AUDIT_ERROR_CHARS),
            )

    async def maybe_extract(self, session_id: str) -> None:
        """Extract only when the configured cadence/thresholds say it is time.

        Safe to call after every turn: it is a cheap no-op unless extraction is
        enabled, the session is due (``extraction_every_n_turns``) and enough
        text has accumulated (``extraction_min_chars``). Never raises.
        """
        try:
            if not session_id or self.store is None:
                return
            if not self.extraction_enabled:
                LOG.debug("Extraction disabled; skipping session %s", session_id)
                return
            session = self.store.get_session(session_id)
            if session is None:
                LOG.debug("maybe_extract: unknown session %s", session_id)
                return
            every = self._every_n_turns()
            turns = self._turns_since(session_id)
            if turns < every:
                LOG.debug(
                    "Session %s: %s turn(s) since last extraction (need %s)",
                    session_id,
                    turns,
                    every,
                )
                return
            messages = self._recent_messages(session_id, DEFAULT_MAX_TURNS)
            min_chars = self._min_chars()
            if self._combined_length(messages) < min_chars:
                LOG.debug(
                    "Session %s: transcript below extraction_min_chars (%s)",
                    session_id,
                    min_chars,
                )
                return
            await self.distill_session(session_id, force=False)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - must never bubble into callers
            LOG.warning(
                "maybe_extract failed for session %s", session_id, exc_info=True
            )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _exchange_messages(self, user_text: str, assistant_text: str) -> list[dict[str, Any]]:
        parts = [f"Today is {utc_now_iso()[:10]} (UTC)."]
        profile = self._profile_block()
        if profile:
            parts.append(
                "Known profile (do not duplicate these facts; only report changes):\n"
                + profile
            )
        exchange = [f"User: {truncate(user_text, MAX_MESSAGE_CHARS)}"]
        if assistant_text:
            exchange.append(f"Assistant: {truncate(assistant_text, MAX_MESSAGE_CHARS)}")
        parts.append("Conversation exchange:\n" + "\n".join(exchange))
        parts.append(FINAL_DIRECTIVE)
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _transcript_messages(
        self,
        messages: Sequence[Message],
        *,
        session: Session | None = None,
    ) -> list[dict[str, Any]]:
        parts = [f"Today is {utc_now_iso()[:10]} (UTC)."]
        if session is not None and session.title:
            parts.append(f"Session title: {truncate(session.title, 120)}")
        profile = self._profile_block()
        if profile:
            parts.append(
                "Known profile (do not duplicate these facts; only report changes):\n"
                + profile
            )
        parts.append(
            "Conversation transcript (most recent turns, oldest first):\n"
            + self._format_transcript(messages)
        )
        parts.append(FINAL_DIRECTIVE)
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _summary_messages(
        self,
        transcript: str,
        *,
        session: Session | None = None,
        previous_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        parts = [f"Today is {utc_now_iso()[:10]} (UTC)."]
        if session is not None and session.title:
            parts.append(f"Session title: {truncate(session.title, 120)}")
        if previous_summary:
            parts.append(
                "Previous summary (refine or extend it):\n"
                + truncate(collapse_whitespace(previous_summary), MAX_SUMMARY_CHARS)
            )
        parts.append("Conversation transcript:\n" + transcript)
        parts.append("Return the JSON object now.")
        return [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _profile_block(self) -> str:
        """Compact list of known profile facts, used to avoid duplicates."""
        if self.store is None:
            return ""
        try:
            memories = self.store.list_identity_memories(limit=25)
        except Exception:  # noqa: BLE001 - prompt enrichment is best effort
            LOG.debug("Could not load profile facts for the prompt", exc_info=True)
            return ""
        lines: list[str] = []
        used = 0
        for memory in memories:
            content = collapse_whitespace(memory.content)
            if not content:
                continue
            line = f"- [{memory.key}] {content}" if memory.key else f"- {content}"
            line = truncate(line, 160)
            if used + len(line) > MAX_PROFILE_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call & parsing
    # ------------------------------------------------------------------
    async def _chat_json(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        options: Mapping[str, Any],
    ) -> str:
        """Call Ollama in JSON mode and return the raw content string."""
        model = self.default_model
        if not model:
            raise RuntimeError("no extraction model available")
        client = self._get_ollama()
        response = await client.chat(
            model,
            list(messages),
            format_="json",
            options=dict(options),
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("Ollama returned a non-object chat response")
        message = response.get("message") or {}
        if not isinstance(message, Mapping):
            raise RuntimeError("Ollama returned a malformed chat message")
        return str(message.get("content") or "")

    def _parse_result(self, raw_text: str) -> ExtractionResult:
        """Validate and clip a raw model answer into an ExtractionResult."""
        payload = _parse_json_payload(raw_text)
        if payload is None:
            return ExtractionResult()
        facts = [
            fact
            for fact in (_coerce_fact(item) for item in _as_list(payload.get("facts")))
            if fact is not None
        ][:MAX_FACTS]
        events = [
            event
            for event in (
                _coerce_event(item) for item in _as_list(payload.get("events"))
            )
            if event is not None
        ][:MAX_EVENTS]
        return ExtractionResult(
            facts=facts,
            events=events,
            summary=_coerce_summary(payload.get("summary")),
            raw=payload,
        )

    def _parse_summary(self, raw_text: str) -> str | None:
        """Parse a summary answer, tolerating a plain-prose response."""
        payload = _parse_json_payload(raw_text)
        if payload is not None:
            return _coerce_summary(payload.get("summary"))
        prose = collapse_whitespace(raw_text or "").strip().strip('"').strip("'")
        if not prose or re.match(r"(?i)^(sorry|i (cannot|can't|am unable))", prose):
            return None
        if 15 <= len(prose) <= 1500:
            LOG.debug("Using plain-prose model answer as the session summary")
            return _coerce_summary(prose)
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def _store_facts(
        self, facts: Sequence[Mapping[str, Any]], *, session_id: str
    ) -> tuple[int, int]:
        """Persist validated facts; returns ``(stored, failed)``."""
        remember = getattr(self.service, "remember", None)
        if remember is None or not facts:
            return 0, 0
        stored = 0
        failed = 0
        for fact in facts:
            try:
                await remember(
                    str(fact.get("content") or ""),
                    kind=str(fact.get("kind") or "fact"),
                    key=fact.get("key") or None,
                    importance=float(fact.get("importance", 0.6)),
                    confidence=float(fact.get("confidence", 0.75)),
                    source="extraction",
                    session_id=session_id,
                )
                stored += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad fact must not stop the run
                failed += 1
                LOG.warning("Could not store extracted fact: %s", exc)
        return stored, failed

    async def _store_events(
        self, events: Sequence[Mapping[str, Any]], *, session_id: str
    ) -> tuple[int, int]:
        """Persist validated events; returns ``(stored, failed)``."""
        log_event = getattr(self.service, "log_event", None)
        if log_event is None or not events:
            return 0, 0
        stored = 0
        failed = 0
        for event in events:
            try:
                await log_event(
                    str(event.get("title") or ""),
                    kind=str(event.get("kind") or "note"),
                    details=event.get("details"),
                    project=event.get("project"),
                    importance=float(event.get("importance", 0.5)),
                    occurred_at=event.get("occurred_at"),
                    session_id=session_id,
                    source="extraction",
                )
                stored += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad event must not stop the run
                failed += 1
                LOG.warning("Could not store extracted event: %s", exc)
        return stored, failed

    async def _persist_summary(self, session_id: str, summary: str) -> bool:
        """Write the summary to the session row and the summary memory."""
        clean = _coerce_summary(summary)
        if not clean:
            return False
        if getattr(self.settings, "redact_secrets", True):
            # Summaries bypass service.remember for the session row, so apply
            # the same redaction guarantee here before persisting.
            clean = redact_secrets(clean)
        persisted = False
        if self.store is not None:
            try:
                self.store.set_session_summary(session_id, clean)
                persisted = True
            except Exception as exc:  # noqa: BLE001 - memory write is best effort
                LOG.warning("Could not set session summary for %s: %s", session_id, exc)
        remember = getattr(self.service, "remember", None)
        if remember is not None:
            try:
                await remember(
                    clean,
                    kind="summary",
                    key=f"session-summary:{session_id}",
                    importance=0.65,
                    confidence=0.8,
                    source="extraction",
                    session_id=session_id,
                )
                persisted = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - memory write is best effort
                LOG.warning("Could not store summary memory for %s: %s", session_id, exc)
        return persisted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _result(
        *,
        facts: int,
        events: int,
        summary: str | None,
        skipped: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "facts": facts,
            "events": events,
            "summary": summary,
            "skipped": skipped,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def _normalize_turns(max_turns: int) -> int:
        try:
            turns = int(max_turns)
        except (TypeError, ValueError):
            return DEFAULT_MAX_TURNS
        return max(1, min(turns, MAX_TURNS_HARD_LIMIT))

    def _every_n_turns(self) -> int:
        try:
            return max(1, int(getattr(self.settings, "extraction_every_n_turns", 1)))
        except (TypeError, ValueError):
            return 1

    def _min_chars(self) -> int:
        try:
            return max(0, int(getattr(self.settings, "extraction_min_chars", 0)))
        except (TypeError, ValueError):
            return 0

    def _turns_since(self, session_id: str) -> int:
        if self.store is None:
            return 0
        try:
            return int(self.store.count_turns_since_extraction(session_id))
        except Exception:  # noqa: BLE001 - fall back to "not due"
            LOG.debug("Could not count turns since extraction", exc_info=True)
            return 0

    def _recent_messages(
        self, session_id: str, max_turns: int
    ) -> list[Message]:
        """The most recent messages, oldest-first (prompt order)."""
        if self.store is None:
            return []
        limit = max(1, int(max_turns)) * 2
        try:
            return list(
                self.store.session_messages(session_id, limit=limit, ascending=False)
            )
        except Exception:  # noqa: BLE001 - treat as empty transcript
            LOG.debug("Could not load session messages", exc_info=True)
            return []

    @staticmethod
    def _combined_length(messages: Sequence[Message]) -> int:
        return sum(len((message.content or "").strip()) for message in messages)

    @staticmethod
    def _format_transcript(messages: Sequence[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            content = collapse_whitespace(message.content or "")
            if not content:
                continue
            role = {"user": "User", "assistant": "Assistant"}.get(
                message.role, message.role.title() or "Message"
            )
            lines.append(f"{role}: {truncate(content, MAX_MESSAGE_CHARS)}")
        transcript = "\n".join(lines)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = "[...earlier turns omitted...]\n" + transcript[
                -MAX_TRANSCRIPT_CHARS:
            ]
        return transcript

    def _audit(
        self,
        *,
        kind: str,
        session_id: str | None = None,
        status: str = "done",
        items: int = 0,
        error: str | None = None,
    ) -> None:
        """Record an extraction run; auditing must never break the pipeline."""
        if self.store is None:
            return
        try:
            self.store.record_extraction(
                kind=kind,
                session_id=session_id,
                status=status,
                items=int(items),
                error=truncate(error, MAX_AUDIT_ERROR_CHARS) if error else None,
            )
        except Exception:  # noqa: BLE001 - audit is best effort
            LOG.debug("Could not record extraction audit row", exc_info=True)


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------


class ExtractionScheduler:
    """Fire-and-forget background extraction with bounded concurrency.

    The proxy and CLI call :meth:`schedule` from synchronous request paths; the
    scheduler dedupes in-flight sessions, never raises from :meth:`schedule`,
    and guarantees that :meth:`close` awaits or cancels every task so shutdown
    is free of "Task was destroyed but it is pending" warnings.
    """

    def __init__(self, extractor: MemoryExtractor, *, max_concurrency: int = 1) -> None:
        self.extractor = extractor
        self.max_concurrency = self._normalize_concurrency(max_concurrency)
        self._semaphore: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._in_flight: set[str] = set()
        self._closed = False

    @staticmethod
    def _normalize_concurrency(value: Any) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    # -- introspection ---------------------------------------------------
    @property
    def pending(self) -> int:
        """Number of scheduled tasks that have not finished yet."""
        return len(self._tasks)

    @property
    def in_flight_sessions(self) -> set[str]:
        """Sessions currently queued or running (dedupe key)."""
        return set(self._in_flight)

    @property
    def closed(self) -> bool:
        return self._closed

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    # -- public API ------------------------------------------------------
    def schedule(self, session_id: str) -> None:
        """Queue background extraction for ``session_id``; never raises."""
        try:
            if not session_id or self._closed:
                return
            if session_id in self._in_flight:
                LOG.debug("Extraction already queued for session %s", session_id)
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Called from sync code with no loop: silently skip, the CLI
                # path should use asyncio.run(...) around drain() instead.
                LOG.debug("No running event loop; skipping scheduled extraction")
                return
            self._in_flight.add(session_id)
            task = loop.create_task(
                self._run(session_id), name=f"amnesia-extraction-{session_id[:8]}"
            )
            self._tasks.add(task)
            task.add_done_callback(self._on_task_done)
        except Exception:  # noqa: BLE001 - scheduling must never break a request
            LOG.warning("Could not schedule extraction for %s", session_id, exc_info=True)

    async def drain(self) -> None:
        """Wait until every currently scheduled task has finished."""
        while True:
            pending = [task for task in self._tasks if not task.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        """Cancel and await all outstanding work; idempotent."""
        self._closed = True
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._in_flight.clear()

    async def __aenter__(self) -> "ExtractionScheduler":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- internals -------------------------------------------------------
    async def _run(self, session_id: str) -> None:
        try:
            async with self._get_semaphore():
                await self.extractor.maybe_extract(session_id)
        except asyncio.CancelledError:
            LOG.debug("Extraction cancelled for session %s", session_id)
            raise
        except Exception:  # noqa: BLE001 - background work must not crash the loop
            LOG.warning(
                "Background extraction failed for session %s", session_id, exc_info=True
            )
        finally:
            self._in_flight.discard(session_id)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:  # pragma: no cover - defensive
            return
        if error is not None:  # pragma: no cover - _run already logs
            LOG.warning("Extraction task failed: %s", error)


__all__ = ["ExtractionResult", "MemoryExtractor", "ExtractionScheduler"]
