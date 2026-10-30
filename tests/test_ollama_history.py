"""Tests for the read-only Ollama desktop history importer.

Everything here is hermetic: a tiny temporary SQLite file mimics the Ollama
desktop schema and a temporary Amnesia store receives the import.  The real
Ollama database is never touched, no network is used and no Ollama daemon is
required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import amnesia.ollama_history as history
from amnesia.ollama_history import (
    HistoryChat,
    HistoryMessage,
    OllamaHistoryError,
    import_history,
    read_ollama_history,
    session_id_for,
)
from amnesia.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_history_db(
    path: Path,
    *,
    titles: tuple[str, ...] = ("", "Second chat"),
    include_optional_columns: bool = True,
) -> Path:
    """Create a minimal Ollama-desktop-like database."""
    connection = sqlite3.connect(path)
    try:
        if include_optional_columns:
            connection.execute(
                "CREATE TABLE chats(id TEXT PRIMARY KEY, title TEXT, created_at TEXT)"
            )
            connection.execute(
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, "
                "content TEXT, created_at TEXT, model_name TEXT)"
            )
        else:
            connection.execute("CREATE TABLE chats(id TEXT PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE messages(id INTEGER PRIMARY KEY, chat_id TEXT, "
                "role TEXT, content TEXT)"
            )
        rows = [
            ("c1", titles[0], "2026-10-26 04:52:02.5831601+08:00"),
            ("c2", titles[1] if len(titles) > 1 else "", "2026-10-28 16:59:24.7296061+08:00"),
        ]
        if include_optional_columns:
            connection.executemany("INSERT INTO chats VALUES (?, ?, ?)", rows)
            messages = [
                (1, "c1", "user", "Hello there", "2026-10-26 04:52:02.5831601+08:00", None),
                (2, "c2", "user", "What is up?", "2026-10-28 16:59:24.7296061+08:00", None),
                (
                    3,
                    "c2",
                    "assistant",
                    "Not much!",
                    "2026-10-28 17:01:15.8817029+08:00",
                    "dolphin3",
                ),
                (4, "c2", "tool", "tool payload", "2026-10-28 17:01:16+08:00", None),
                (5, "c2", "user", "   ", "2026-10-28 17:01:17+08:00", None),
                (6, "ghost", "user", "orphan message", "2026-10-28 17:01:18+08:00", None),
            ]
            connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", messages)
        else:
            connection.executemany("INSERT INTO chats VALUES (?)", [("c1",), ("c2",)])
            connection.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                [
                    (1, "c1", "user", "Hello there"),
                    (2, "c2", "user", "What is up?"),
                ],
            )
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture()
def history_db(tmp_path: Path) -> Path:
    return _create_history_db(tmp_path / "db.sqlite")


@pytest.fixture()
def store(tmp_path: Path):
    store = MemoryStore(tmp_path / "amnesia.db").initialize()
    try:
        yield store
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Reading Ollama's database
# ---------------------------------------------------------------------------


def test_reads_chats_oldest_first_and_skips_unusable_rows(history_db: Path) -> None:
    chats = read_ollama_history(history_db)

    assert [chat.chat_id for chat in chats] == ["c1", "c2"]
    first, second = chats

    # Empty chat title stays empty here; the importer derives one from content.
    assert first.title == ""
    assert first.created_at == "2026-10-25T20:52:02Z"
    assert first.message_count == 1
    assert first.messages[0].role == "user"
    assert first.messages[0].content == "Hello there"
    assert first.messages[0].created_at == "2026-10-25T20:52:02Z"
    assert first.messages[0].message_id == 1

    # tool rows, blank content and messages for missing chats are dropped.
    assert second.title == "Second chat"
    assert [message.role for message in second.messages] == ["user", "assistant"]
    assert [message.content for message in second.messages] == ["What is up?", "Not much!"]
    assert second.model == "dolphin3"
    assert second.messages[1].created_at == "2026-10-28T09:01:15Z"


def test_read_limit_keeps_newest_chats(history_db: Path) -> None:
    chats = read_ollama_history(history_db, limit=1)
    assert [chat.chat_id for chat in chats] == ["c2"]


def test_read_tolerates_missing_optional_columns(tmp_path: Path) -> None:
    db = _create_history_db(tmp_path / "minimal.sqlite", include_optional_columns=False)

    chats = read_ollama_history(db)

    assert len(chats) == 2
    assert chats[0].title == ""
    assert chats[0].created_at is None
    assert chats[0].messages[0].model is None


def test_read_missing_file_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(OllamaHistoryError, match="not found"):
        read_ollama_history(tmp_path / "nope.sqlite")


def test_read_rejects_a_non_ollama_database(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE unrelated(id INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(OllamaHistoryError, match="does not look like"):
        read_ollama_history(path)


# ---------------------------------------------------------------------------
# Timestamp / title / id helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-26 04:52:02.5831601+08:00", "2026-10-25T20:52:02Z"),
        ("2026-10-26T04:52:02Z", "2026-10-26T04:52:02Z"),
        (1_790_000_000, "2026-09-21T14:13:20Z"),
        (None, None),
        ("", None),
        ("not-a-date", None),
    ],
)
def test_iso_utc_conversion(value: object, expected: str | None) -> None:
    assert history._iso_utc(value) == expected


def test_session_id_for_is_deterministic() -> None:
    assert session_id_for("abc-123") == "ollama-desktop-abc-123"
    assert session_id_for(42) == "ollama-desktop-42"


def test_derive_title_uses_first_user_message() -> None:
    chat = HistoryChat(
        chat_id="c1",
        title="",
        messages=(
            HistoryMessage(role="assistant", content="welcome"),
            HistoryMessage(role="user", content="  Explain   decorators  "),
        ),
    )
    assert history._derive_title(chat) == "Explain decorators"


# ---------------------------------------------------------------------------
# Importing into Amnesia
# ---------------------------------------------------------------------------


def test_import_creates_sessions_and_messages(history_db: Path, store: MemoryStore) -> None:
    chats = read_ollama_history(history_db)

    result = import_history(store, chats)

    assert result.chats_found == 2
    assert result.chats_imported == 2
    assert result.chats_skipped == 0
    assert result.messages_imported == 3
    assert result.session_ids == [
        "ollama-desktop-c1",
        "ollama-desktop-c2",
    ]

    first = store.get_session("ollama-desktop-c1")
    assert first is not None
    assert first.client == "ollama-desktop"
    assert first.metadata["source"] == "ollama-desktop"
    assert first.metadata["ollama_chat_id"] == "c1"
    assert first.started_at == "2026-10-25T20:52:02Z"
    first_messages = store.session_messages(first.id)
    assert [message.role for message in first_messages] == ["user"]
    assert first_messages[0].metadata["ollama_message_id"] == "1"
    assert first_messages[0].created_at == "2026-10-25T20:52:02Z"

    second = store.get_session("ollama-desktop-c2")
    assert second is not None
    assert second.title == "Second chat"
    assert second.model == "dolphin3"
    second_messages = store.session_messages(second.id)
    assert [message.role for message in second_messages] == ["user", "assistant"]
    assert [message.content for message in second_messages] == ["What is up?", "Not much!"]


def test_import_is_idempotent(history_db: Path, store: MemoryStore) -> None:
    chats = read_ollama_history(history_db)
    import_history(store, chats)

    again = import_history(store, chats)

    assert again.chats_found == 2
    assert again.chats_imported == 0
    assert again.chats_skipped == 2
    assert again.messages_imported == 0
    assert again.session_ids == []
    assert len(store.list_sessions(limit=10)) == 2


def test_dry_run_writes_nothing(history_db: Path, store: MemoryStore) -> None:
    chats = read_ollama_history(history_db)

    result = import_history(store, chats, dry_run=True)

    assert result.dry_run is True
    assert result.chats_imported == 2
    assert result.messages_imported == 3
    assert len(store.list_sessions(limit=10)) == 0
    assert len(store.session_messages("ollama-desktop-c1")) == 0


def test_dry_run_without_a_store(history_db: Path) -> None:
    chats = read_ollama_history(history_db)

    result = import_history(None, chats, dry_run=True)

    assert result.chats_imported == 2
    assert result.chats_skipped == 0
    assert result.messages_imported == 3


def test_import_skips_already_imported_and_adds_new_chats(history_db: Path, store: MemoryStore) -> None:
    chats = read_ollama_history(history_db)
    import_history(store, chats[:1])

    result = import_history(store, chats)

    assert result.chats_imported == 1
    assert result.chats_skipped == 1
    assert result.session_ids == ["ollama-desktop-c2"]
    assert len(store.list_sessions(limit=10)) == 2


def test_default_history_path_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.sqlite"
    monkeypatch.setenv("AMNESIA_OLLAMA_HISTORY_DB", str(custom))
    assert history.default_history_path() == custom
