"""Small shared helpers: time, token estimates, redaction, JSON, async bridge.

Kept dependency-light on purpose - these helpers are used by every layer.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Iterable, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

UTC = timezone.utc


def utc_now() -> datetime:
    """Timezone-aware wall clock in UTC."""
    return datetime.now(tz=UTC)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second resolution (sortable as text)."""
    return utc_now().replace(microsecond=0).isoformat()


def to_iso(dt: datetime) -> str:
    """Serialize a datetime to an ISO string; naive values are assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.replace(microsecond=0).isoformat()


def parse_iso(value: str | datetime | None) -> datetime | None:
    """Parse ISO timestamps stored by Amnesia (handles trailing Z)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def local_now() -> datetime:
    return datetime.now().astimezone()


def human_time(value: str | datetime | None, *, with_time: bool = True) -> str:
    """Render a stored UTC timestamp in the machine's local timezone."""
    dt = parse_iso(value)
    if dt is None:
        return "unknown"
    local = dt.astimezone()
    if with_time:
        return local.strftime("%Y-%m-%d %H:%M")
    return local.strftime("%Y-%m-%d")


def relative_time(value: str | datetime | None) -> str:
    """Friendly '3 hours ago' style rendering for CLI output."""
    dt = parse_iso(value)
    if dt is None:
        return "unknown"
    delta = utc_now() - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


def days_ago_iso(days: int) -> str:
    return to_iso(utc_now() - timedelta(days=days))


# ---------------------------------------------------------------------------
# Tokens & text
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap token estimate; good enough for context budgeting.

    ~4 characters per token for English, considered a safe upper bound for
    budget planning on local tokenizers.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def truncate(text: str, limit: int, *, suffix: str = "...") -> str:
    """Truncate on a word boundary when possible."""
    if limit <= 0 or len(text) <= limit:
        return text[: max(0, limit)] if limit > 0 else ""
    cut = text[: max(0, limit - len(suffix))]
    pivot = cut.rfind(" ")
    if pivot > limit * 0.6:
        cut = cut[:pivot]
    return cut.rstrip() + suffix


def collapse_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


# ---------------------------------------------------------------------------
# Redaction - never persist obvious secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # key = value / key: value forms. The key may carry prefixes/separators
    # (api_token, access_token, client_secret, db_password, ...), and the
    # leading negative lookbehind keeps us from matching inside other words.
    (
        re.compile(
            r"(?i)(?<![a-z0-9])"
            r"((?:[a-z0-9]+[_-])*(?:api[_-]?key|secret|token|password|passwd|pwd|"
            r"access[_-]?key|private[_-]?key|auth[_-]?token|bearer))"
            r"(\s*[:=]\s*)([^\s,;\"']{6,})"
        ),
        r"\1\2[REDACTED]",
    ),
    # Authorization headers
    (
        re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([A-Za-z0-9._~+/=-]{12,})"),
        r"\1[REDACTED]",
    ),
    # Well-known token shapes
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})\b"), "[REDACTED]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,})\b"), "[REDACTED]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED]"),
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b"), "[REDACTED]"),
    (re.compile(r"\b([A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})\b"), "[REDACTED]"),
    # Connection strings with inline credentials, including empty usernames
    # such as redis://:password@host:6379/0.
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^/\s@]*:)([^@/\s]{1,})(@)"),
        r"\1[REDACTED]\3",
    ),
]


def redact_secrets(text: str) -> str:
    """Best-effort removal of secrets from text that is about to be stored."""
    if not text:
        return text
    result = text
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    # PEM private key blocks
    result = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED PRIVATE KEY]",
        result,
        flags=re.DOTALL,
    )
    return result


def looks_secret(text: str) -> bool:
    """True when text contains something that should probably not be stored raw."""
    return redact_secrets(text) != text


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM answer.

    Tolerates code fences, leading prose, and trailing commentary, which local
    models produce constantly.
    """
    if not text:
        return None
    candidate = text.strip()
    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        pass
    # Scan for a balanced JSON object.
    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(candidate)):
            char = candidate[idx]
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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(candidate[start : idx + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except ValueError:
                        break
        start = candidate.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Async bridge for CLI code paths
# ---------------------------------------------------------------------------


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync code, even inside a running loop.

    CLI commands are sync by nature; this lets them reuse the async service.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an event loop (e.g. notebooks / embedded use): run in a
    # dedicated thread with its own loop.
    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("async bridge returned no result")
    return result[0]


def chunked(items: Iterable[T], size: int) -> Iterable[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
