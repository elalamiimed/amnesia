"""Unit tests for the shared helpers in :mod:`amnesia.utils` and :mod:`amnesia.db`.

These helpers are used by every layer of Amnesia (store, retrieval, service,
extraction, proxy, CLI), so subtle regressions here would silently corrupt
memory or defeat secret redaction.  The tests are fully offline and
deterministic: time is frozen by monkeypatching ``amnesia.utils.utc_now`` and
nothing touches the network or a real database.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from amnesia import utils
from amnesia.db import fts_match_query, like_pattern
from amnesia.utils import (
    UTC,
    chunked,
    collapse_whitespace,
    days_ago_iso,
    estimate_tokens,
    extract_json_object,
    human_time,
    json_dumps,
    json_loads,
    looks_secret,
    parse_iso,
    redact_secrets,
    relative_time,
    run_sync,
    to_iso,
    truncate,
    utc_now,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("My api_key=abcdef123456 is here", "abcdef123456"),
        ("api-key: super-secret-value", "super-secret-value"),
        ("password: hunter2hunter2", "hunter2hunter2"),
        ("my secret = topsecretvalue", "topsecretvalue"),
        ("token=ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
        ("Authorization: Bearer abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
        ("AWS key AKIAIOSFODNN7EXAMPLE here", "AKIAIOSFODNN7EXAMPLE"),
        ("slack xoxb-1234567890-abcdef", "xoxb-1234567890-abcdef"),
        ("openai sk-abcdefghijklmnop123456", "sk-abcdefghijklmnop123456"),
    ],
)
def test_redact_secrets_removes_common_secret_shapes(raw: str, secret: str) -> None:
    redacted = redact_secrets(raw)
    assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_connection_string_credentials() -> None:
    dsn = "postgresql://alice:hunter2@db.internal:5432/app"
    redacted = redact_secrets(f"connecting to {dsn}")
    assert "hunter2" not in redacted
    assert "postgresql://alice:[REDACTED]@db.internal:5432/app" in redacted


def test_redact_secrets_pem_private_key_block() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234567890abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    redacted = redact_secrets(f"key follows\n{pem}\ndone")
    assert "MIIEow" not in redacted
    assert "[REDACTED PRIVATE KEY]" in redacted
    assert redacted.startswith("key follows")


def test_redact_secrets_jwt() -> None:
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    assert "[REDACTED]" in redact_secrets(f"Bearer {jwt}")
    assert jwt not in redact_secrets(jwt)


def test_redact_secrets_leaves_benign_text_untouched() -> None:
    text = "Prefers concise answers; uses VS Code on Windows 11."
    assert redact_secrets(text) == text
    assert looks_secret(text) is False


def test_redact_secrets_does_not_flag_word_tokenizer() -> None:
    assert redact_secrets("The tokenizer splits on whitespace") == (
        "The tokenizer splits on whitespace"
    )


def test_redact_secrets_empty_input() -> None:
    assert redact_secrets("") == ""
    assert looks_secret("") is False


def test_redact_secrets_empty_userinfo_connection_string() -> None:
    """Regression: redis://:pass@host style DSNs (empty username) are redacted."""
    redacted = redact_secrets("redis://:hunter2@localhost:6379/0")
    assert "hunter2" not in redacted
    assert redacted == "redis://:[REDACTED]@localhost:6379/0"


def test_redact_secrets_underscore_prefixed_secret_keys() -> None:
    """Regression: api_token=, access_token=, client_secret= are redacted."""
    for raw, secret in (
        ("api_token=tok_abcdef123456", "tok_abcdef123456"),
        ("access_token=ya29.abcdefghijklmnop", "ya29.abcdefghijklmnop"),
        ("client_secret=cs_abcdef123456", "cs_abcdef123456"),
        ("AMNESIA_API_TOKEN=envtoken123456", "envtoken123456"),
    ):
        redacted = redact_secrets(raw)
        assert secret not in redacted, raw
        assert "[REDACTED]" in redacted, raw


# ---------------------------------------------------------------------------
# Tolerant JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"facts": [], "summary": null}') == {
        "facts": [],
        "summary": None,
    }


def test_extract_json_object_markdown_fence() -> None:
    text = '```json\n{"facts": [{"content": "Prefers dark mode"}]}\n```'
    assert extract_json_object(text) == {"facts": [{"content": "Prefers dark mode"}]}


def test_extract_json_object_surrounded_by_prose() -> None:
    text = 'Sure! Here is the JSON:\n{"facts": []}\nHope that helps.'
    assert extract_json_object(text) == {"facts": []}


def test_extract_json_object_prefers_fenced_block() -> None:
    text = 'Noise {"ignored": true}\n```\n{"facts": [1]}\n```'
    assert extract_json_object(text) == {"facts": [1]}


def test_extract_json_object_handles_nested_braces_in_strings() -> None:
    text = 'prefix {"content": "uses {curly} braces", "n": 1} suffix'
    assert extract_json_object(text) == {"content": "uses {curly} braces", "n": 1}


def test_extract_json_object_handles_escaped_quotes() -> None:
    text = r'{"content": "he said \"hi\" }", "ok": true}'
    assert extract_json_object(text) == {"content": 'he said "hi" }', "ok": True}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here at all",
        "[1, 2, 3]",
        '"just a string"',
        "42",
        "null",
        '{"truncated": [1, 2',
    ],
)
def test_extract_json_object_returns_none_without_object(text: str) -> None:
    assert extract_json_object(text) is None


def test_extract_json_object_skips_malformed_object() -> None:
    text = '{not valid json} then {"valid": true}'
    assert extract_json_object(text) == {"valid": True}


# ---------------------------------------------------------------------------
# Token estimates & text helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("a" * 400, 100),
    ],
)
def test_estimate_tokens(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_truncate_short_and_exact_text_unchanged() -> None:
    assert truncate("short", 10) == "short"
    assert truncate("exactly-ten", 11) == "exactly-ten"


def test_truncate_prefers_word_boundary_and_adds_suffix() -> None:
    text = "hello world extra words here"
    result = truncate(text, 22)
    assert result.endswith("...")
    assert len(result) <= 22
    assert result.startswith("hello world extra")
    assert "wor" not in result.replace("world", "")


def test_truncate_custom_suffix_and_zero_limit() -> None:
    assert truncate("hello world", 8, suffix="~").endswith("~")
    assert truncate("hello", 0) == ""
    assert truncate("hello", -3) == ""


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  a \t  b\n c  ") == "a b\n c"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def test_utc_now_is_timezone_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_utc_now_iso_is_parseable_and_second_resolution() -> None:
    value = utc_now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", value)
    assert parse_iso(value) is not None


def test_to_iso_treats_naive_as_utc() -> None:
    assert to_iso(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05+00:00"
    aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    assert to_iso(aware) == "2026-01-02T03:04:05+02:00"


@pytest.mark.parametrize(
    ("value", "expected_iso"),
    [
        ("2026-01-02T03:04:05Z", "2026-01-02T03:04:05+00:00"),
        ("2026-01-02T03:04:05+00:00", "2026-01-02T03:04:05+00:00"),
        ("2026-01-02T03:04:05", "2026-01-02T03:04:05+00:00"),
        ("2026-01-02 03:04:05", "2026-01-02T03:04:05+00:00"),
    ],
)
def test_parse_iso_accepts_common_shapes(value: str, expected_iso: str) -> None:
    parsed = parse_iso(value)
    assert parsed is not None
    assert to_iso(parsed) == expected_iso


def test_parse_iso_passthrough_and_invalid_values() -> None:
    aware = datetime(2026, 1, 2, tzinfo=UTC)
    assert parse_iso(aware) is aware
    naive = datetime(2026, 1, 2, 3, 4, 5)
    assert parse_iso(naive) == naive.replace(tzinfo=UTC)
    assert parse_iso(None) is None
    assert parse_iso("   ") is None
    assert parse_iso("not a date") is None
    assert parse_iso("2026-13-45T99:99:99") is None


def test_human_time_renders_local_time() -> None:
    value = "2026-01-02T20:30:00+00:00"
    local = parse_iso(value).astimezone()
    assert human_time(value) == local.strftime("%Y-%m-%d %H:%M")
    assert human_time(value, with_time=False) == local.strftime("%Y-%m-%d")
    assert human_time(None) == "unknown"
    assert human_time("garbage") == "unknown"


def test_relative_time_boundaries_with_frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(utils, "utc_now", lambda: frozen)

    def ago(**kwargs: float) -> str:
        return relative_time(to_iso(frozen - timedelta(**kwargs)))

    assert relative_time("not a date") == "unknown"
    assert ago(seconds=30) == "just now"
    assert ago(minutes=5) == "5m ago"
    assert ago(hours=3) == "3h ago"
    assert ago(days=2) == "2d ago"
    assert ago(days=45) == "1mo ago"
    assert ago(days=400) == "1y ago"
    assert relative_time(to_iso(frozen + timedelta(minutes=5))) == "in the future"


def test_days_ago_iso_is_about_n_days_ago() -> None:
    parsed = parse_iso(days_ago_iso(3))
    assert parsed is not None
    delta = utc_now() - parsed
    assert timedelta(days=2, hours=23) < delta < timedelta(days=3, minutes=1)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def test_json_dumps_is_compact_and_unicode_friendly() -> None:
    payload = {"name": "Amélie", "n": 1}
    encoded = json_dumps(payload)
    assert " " not in encoded
    assert "Amélie" in encoded
    assert json.loads(encoded) == payload


def test_json_dumps_serializes_unknown_types_via_str() -> None:
    assert '"when"' in json_dumps({"when": datetime(2026, 1, 1)})


@pytest.mark.parametrize(("raw", "expected"), [("", None), (None, None), ("{bad", None)])
def test_json_loads_falls_back_to_default(raw: str | None, expected: object) -> None:
    assert json_loads(raw, default=expected) == expected


def test_json_loads_valid_payload() -> None:
    assert json_loads('{"a": [1, 2]}') == {"a": [1, 2]}
    assert json_loads("[]", default={}) == []


# ---------------------------------------------------------------------------
# FTS query building & LIKE escaping
# ---------------------------------------------------------------------------


def test_fts_match_query_quotes_and_ors_terms() -> None:
    match = fts_match_query("Tell me about Python and FTS5!")
    for term in ('"tell"', '"me"', '"about"', '"python"', '"and"', '"fts5"'):
        assert term in match
    assert " OR " in match


def test_fts_match_query_dedupes_case_insensitively() -> None:
    assert fts_match_query("Python python PYTHON") == '"python"'


def test_fts_match_query_drops_punctuation_and_short_tokens() -> None:
    assert fts_match_query("?!.,; a b") == ""
    assert fts_match_query("") == ""
    assert fts_match_query(None) == ""  # type: ignore[arg-type]


def test_fts_match_query_is_injection_safe() -> None:
    match = fts_match_query('hello" OR * DROP TABLE')
    assert match == '"hello" OR "or" OR "drop" OR "table"'
    assert "*" not in match
    assert match.count('"') == 2 * 4


def test_fts_match_query_caps_terms() -> None:
    words = " ".join(f"term{index}" for index in range(20))
    match = fts_match_query(words, max_terms=3)
    assert match.count(" OR ") == 2
    assert match.split(" OR ") == ['"term0"', '"term1"', '"term2"']


def test_like_pattern_escapes_wildcards_and_backslashes() -> None:
    assert like_pattern("100%") == "%100\\%%"
    assert like_pattern("a_b") == "%a\\_b%"
    assert like_pattern("c:\\temp") == "%c:\\\\temp%"


# ---------------------------------------------------------------------------
# Async bridge & batching
# ---------------------------------------------------------------------------


def test_run_sync_from_sync_context() -> None:
    async def answer() -> int:
        return 42

    assert run_sync(answer()) == 42


def test_run_sync_propagates_exceptions() -> None:
    async def boom() -> None:
        raise ValueError("exploded")

    with pytest.raises(ValueError, match="exploded"):
        run_sync(boom())


async def test_run_sync_inside_running_loop_uses_thread_bridge() -> None:
    """CLI-style sync helpers must work even when called inside a live loop."""

    async def answer() -> str:
        return "bridged"

    assert run_sync(answer()) == "bridged"


async def test_run_sync_inside_running_loop_propagates_exceptions() -> None:
    async def boom() -> None:
        raise RuntimeError("loop bridge failed")

    with pytest.raises(RuntimeError, match="loop bridge failed"):
        run_sync(boom())


def test_chunked_batches_until_exhausted() -> None:
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(chunked([], 3)) == []
    assert list(chunked(["a"], 3)) == [["a"]]
