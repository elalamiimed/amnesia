"""Tests for the Amnesia Typer CLI.

Every test drives the real CLI through ``typer.testing.CliRunner`` but the
service factory is monkeypatched to a temp SQLite store + HashEmbedder + fake
Ollama client, so no test touches the network, a real daemon or ``~/.amnesia``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

import amnesia.cli as cli
from amnesia.config import Settings
from amnesia.embeddings import HashEmbedder
from amnesia.service import Amnesia
from amnesia.store import MemoryStore

pytestmark = pytest.mark.usefixtures("run_cli")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_home: Path,
    settings: Settings,
    fake_ollama: Any,
) -> Any:
    """Invoke the CLI against an isolated temp home and a fake Ollama."""

    # Never let a developer's real AMNESIA_* environment leak into tests.
    for name in list(os.environ):
        if name.startswith("AMNESIA_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AMNESIA_HOME", str(tmp_home))

    def _factory(settings_arg: Settings | None = None) -> Amnesia:
        # The fixture settings are deliberately used for every invocation: they
        # keep extraction/summarisation disabled and point at the temp store, so
        # the CLI tests stay deterministic. Only the log level the CLI resolved
        # (quiet by default, -v for INFO) is propagated, because it is part of
        # the behaviour under test.
        effective = settings
        if settings_arg is not None and settings_arg.log_level:
            effective = settings.clone(log_level=settings_arg.log_level)
        return Amnesia(
            effective,
            embedder=HashEmbedder(),
            ollama=fake_ollama.ollama_client,
        )

    monkeypatch.setattr(cli, "make_service", _factory)
    runner = CliRunner()

    def _invoke(*args: str, input: str | None = None) -> Any:
        return runner.invoke(cli.app, list(args), input=input)

    return _invoke


@contextmanager
def _store(settings: Settings) -> Iterator[MemoryStore]:
    store = MemoryStore(settings.database).initialize()
    try:
        yield store
    finally:
        store.close()


def _json_stdout(result: Any) -> Any:
    """Parse the JSON a command wrote to stdout (logs go to stderr)."""
    return json.loads(result.stdout)


def _assert_ok(result: Any) -> None:
    assert result.exit_code == 0, f"unexpected failure:\n{result.output}"


def _flat(text: str) -> str:
    """Collapse rich table wrapping so content assertions stay robust."""
    return " ".join(text.split())


def _session_messages(settings: Settings) -> list[Any]:
    with _store(settings) as store:
        latest = store.latest_session()
        assert latest is not None
        return store.session_messages(latest.id)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_version_flag(run_cli: Any) -> None:
    result = run_cli("--version")
    _assert_ok(result)
    assert "amnesia" in result.output.lower()


def test_help_lists_contract_commands(run_cli: Any) -> None:
    result = run_cli("--help")
    _assert_ok(result)
    for command in (
        "init",
        "chat",
        "ask",
        "remember",
        "forget",
        "recall",
        "profile",
        "timeline",
        "note",
        "sessions",
        "distill",
        "serve",
        "mcp",
        "doctor",
        "stats",
        "export",
        "import",
        "backup",
        "purge",
        "session",
        "config",
    ):
        assert command in result.output


def test_no_args_shows_help(run_cli: Any) -> None:
    result = run_cli()
    assert result.exit_code != 0
    assert "Usage:" in result.output


COMMANDS_WITH_HELP = (
    "init",
    "chat",
    "ask",
    "remember",
    "forget",
    "recall",
    "profile",
    "timeline",
    "note",
    "sessions",
    "distill",
    "serve",
    "mcp",
    "doctor",
    "stats",
    "export",
    "import",
    "backup",
    "purge",
)


def test_every_command_has_a_working_help_page(run_cli: Any) -> None:
    """docs are written against these flags; a broken parameter breaks the CLI."""
    for command in COMMANDS_WITH_HELP:
        result = run_cli(command, "--help")
        assert result.exit_code == 0, f"`{command} --help` failed:\n{result.output}"
        assert "Usage:" in result.output
    for subcommand in (("session", "show"), ("config", "show")):
        result = run_cli(*subcommand, "--help")
        assert result.exit_code == 0, f"`{' '.join(subcommand)} --help` failed"
        assert "Usage:" in result.output


# ---------------------------------------------------------------------------
# Windows legacy code page safety (CP936/GBK, cp1252, ...)
# ---------------------------------------------------------------------------


def test_success_glyph_falls_back_on_non_utf8_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GBK stream cannot encode ✓; the CLI must fall back to ASCII."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk")
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=stream, highlight=False, soft_wrap=True, width=80),
    )
    cli._success("saved")  # must not raise UnicodeEncodeError
    monkeypatch.setattr(cli, "err_console", cli.console)
    stream.flush()
    rendered = buffer.getvalue().decode("gbk")
    assert "saved" in rendered
    assert "+" in rendered


def test_cli_survives_gbk_stdout_when_redirected(tmp_home: Path) -> None:
    """Regression: redirected CP936 consoles crashed on the ✓ decoration glyph.

    Runs the real entry point in a subprocess (stdout is a pipe, so Python picks
    the locale encoding - CP936 on this machine) exactly like `amnesia init |
    Out-String`.
    """
    env = os.environ.copy()
    env["AMNESIA_HOME"] = str(tmp_home)
    env["AMNESIA_OLLAMA_URL"] = "http://127.0.0.1:1"  # offline: connection refused
    env["AMNESIA_EMBED_MODEL"] = "none"  # skip the embedding-model probe
    env["AMNESIA_MEMORY_ENABLED"] = "0"
    env["AMNESIA_LOG_LEVEL"] = "ERROR"
    env["PYTHONPATH"] = str(REPO_ROOT)
    # Bypass any corporate HTTP proxy so the "offline" Ollama URL fails fast
    # instead of waiting on a proxy round-trip.
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env.pop("PYTHONIOENCODING", None)

    def _run(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "amnesia", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            timeout=60,
        )

    initialised = _run("init")
    assert initialised.returncode == 0, initialised.stderr.decode("utf-8", "replace")
    assert b"Initialised" in initialised.stdout
    assert b"codec can't encode" not in initialised.stderr
    assert b"Traceback" not in initialised.stderr

    backed_up = _run("backup", str(tmp_home / "backup.db"))
    assert backed_up.returncode == 0, backed_up.stderr.decode("utf-8", "replace")
    assert b"codec can't encode" not in backed_up.stderr

    # `doctor` exits 1 because Ollama is deliberately unreachable, but it must
    # finish and print its report instead of dying on the status glyphs.
    diagnosed = _run("doctor")
    assert diagnosed.returncode == 1
    assert b"hard failure" in diagnosed.stdout
    assert b"codec can't encode" not in diagnosed.stderr
    assert b"Traceback" not in diagnosed.stderr


# ---------------------------------------------------------------------------
# Logging verbosity (quiet by default, -v for INFO)
# ---------------------------------------------------------------------------


def test_cli_is_quiet_by_default(run_cli: Any) -> None:
    result = run_cli("remember", "Quiet by default", "--kind", "fact")
    _assert_ok(result)
    assert "Amnesia ready" not in result.stderr
    assert "amnesia.service" not in result.stderr
    assert "INFO" not in result.stderr


def test_verbose_flag_enables_info_logs(run_cli: Any) -> None:
    result = run_cli("-v", "remember", "Verbose mode", "--kind", "fact")
    _assert_ok(result)
    assert "Amnesia ready" in result.stderr
    assert "INFO" in result.stderr


def test_env_log_level_is_respected(run_cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMNESIA_LOG_LEVEL", "INFO")
    result = run_cli("remember", "Env log level", "--kind", "fact")
    _assert_ok(result)
    assert "Amnesia ready" in result.stderr


def test_verbose_fraction_maps_to_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_VERBOSE_COUNT", 0)
    assert cli._cli_log_level("WARNING") == "WARNING"
    monkeypatch.setattr(cli, "_VERBOSE_COUNT", 1)
    assert cli._cli_log_level("WARNING") == "INFO"
    monkeypatch.setattr(cli, "_VERBOSE_COUNT", 2)
    assert cli._cli_log_level("WARNING") == "DEBUG"


# ---------------------------------------------------------------------------
# remember / profile / recall round-trip
# ---------------------------------------------------------------------------


def test_remember_profile_recall_roundtrip(run_cli: Any) -> None:
    remembered = run_cli(
        "remember",
        "Prefers concise, code-first answers",
        "--kind",
        "preference",
        "--importance",
        "0.9",
    )
    _assert_ok(remembered)
    assert "Memory saved" in remembered.output

    profile = run_cli("profile")
    _assert_ok(profile)
    assert "concise, code-first answers" in _flat(profile.output)

    recalled = run_cli("recall", "concise answers")
    _assert_ok(recalled)
    assert "concise, code-first answers" in _flat(recalled.output)
    assert "hit(s)" in recalled.output


def test_profile_json_output(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Lives in Shenzhen", "--kind", "fact"))
    result = run_cli("profile", "--json")
    _assert_ok(result)
    payload = _json_stdout(result)
    assert payload["count"] == 1
    assert payload["facts"][0]["content"] == "Lives in Shenzhen"


def test_remember_with_key_upserts(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Uses VS Code", "--kind", "fact", "--key", "editor"))
    _assert_ok(run_cli("remember", "Uses Vim keybindings", "--kind", "fact", "--key", "editor"))
    payload = _json_stdout(run_cli("profile", "--json"))
    assert payload["count"] == 1
    assert payload["facts"][0]["content"] == "Uses Vim keybindings"


def test_recall_json_output(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Building the Amnesia CLI", "--kind", "fact"))
    result = run_cli("recall", "Amnesia CLI", "--json")
    _assert_ok(result)
    payload = _json_stdout(result)
    assert payload["query"] == "Amnesia CLI"
    assert payload["count"] >= 1
    assert any("Amnesia CLI" in hit["content"] for hit in payload["hits"])
    assert set(payload["hits"][0]) >= {"kind", "id", "content", "score"}


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


def test_forget_by_id(run_cli: Any, settings: Settings) -> None:
    _assert_ok(run_cli("remember", "Temporary fact", "--kind", "fact"))
    with _store(settings) as store:
        memories = store.list_identity_memories()
        memory_id = memories[0].id

    result = run_cli("forget", str(memory_id))
    _assert_ok(result)
    assert "deactivated" in result.output

    profile = _json_stdout(run_cli("profile", "--json"))
    assert profile["count"] == 0


def test_forget_unknown_id_is_friendly(run_cli: Any) -> None:
    result = run_cli("forget", "424242")
    assert result.exit_code == 1
    assert "Error" in result.output
    assert "No active memory" in result.output
    assert "Traceback" not in result.output


def test_forget_requires_target(run_cli: Any) -> None:
    result = run_cli("forget")
    assert result.exit_code == 1
    assert "Provide a memory id or --key" in result.output


def test_forget_by_key_searches_all_kinds(run_cli: Any) -> None:
    """`forget --key` without --kind must find preferences/identity keys too."""
    _assert_ok(
        run_cli("remember", "Prefers dark mode", "--kind", "preference", "--key", "theme")
    )
    _assert_ok(
        run_cli("remember", "Works as an engineer", "--kind", "identity", "--key", "job")
    )

    result = run_cli("forget", "--key", "theme")
    _assert_ok(result)
    assert "deactivated" in result.output
    assert _json_stdout(run_cli("profile", "--json"))["count"] == 1

    result = run_cli("forget", "--key", "job")
    _assert_ok(result)
    assert _json_stdout(run_cli("profile", "--json"))["count"] == 0


def test_forget_by_key_explicit_kind_still_works(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Uses Vim", "--kind", "fact", "--key", "editor"))
    _assert_ok(run_cli("forget", "--key", "editor", "--kind", "fact"))

    # An explicit kind that does not match must not silently delete a different kind.
    _assert_ok(run_cli("remember", "Likes tea", "--kind", "preference", "--key", "drink"))
    mismatch = run_cli("forget", "--key", "drink", "--kind", "fact")
    assert mismatch.exit_code == 1
    assert "No active memory" in mismatch.output
    assert _json_stdout(run_cli("profile", "--json"))["count"] == 1


def test_bad_kind_is_friendly(run_cli: Any) -> None:
    result = run_cli("remember", "something", "--kind", "banana")
    assert result.exit_code == 1
    assert "Unknown memory kind" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# notes / timeline
# ---------------------------------------------------------------------------


def test_note_and_timeline(run_cli: Any) -> None:
    result = run_cli(
        "note",
        "Shipped the CLI workstream",
        "--kind",
        "project",
        "--project",
        "amnesia",
        "--importance",
        "0.8",
    )
    _assert_ok(result)
    assert "Note saved" in result.output

    timeline = run_cli("timeline", "--days", "7")
    _assert_ok(timeline)
    assert "Shipped the CLI workstream" in _flat(timeline.output)
    assert "amnesia" in timeline.output

    payload = _json_stdout(run_cli("timeline", "--json", "--days", "7"))
    assert payload["days"] == 7
    assert payload["count"] == 1
    assert payload["events"][0]["kind"] == "project"
    assert payload["events"][0]["project"] == "amnesia"


def test_timeline_project_filter(run_cli: Any) -> None:
    _assert_ok(run_cli("note", "Alpha task", "--project", "alpha"))
    _assert_ok(run_cli("note", "Beta task", "--project", "beta"))
    payload = _json_stdout(run_cli("timeline", "--project", "alpha", "--json"))
    assert payload["count"] == 1
    assert payload["events"][0]["title"] == "Alpha task"


def test_note_invalid_kind(run_cli: Any) -> None:
    result = run_cli("note", "whatever", "--kind", "banana")
    assert result.exit_code == 1
    assert "Unknown event kind" in result.output


# ---------------------------------------------------------------------------
# stats / config
# ---------------------------------------------------------------------------


def test_stats_json_output(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Likes tests", "--kind", "preference"))
    _assert_ok(run_cli("note", "Wrote tests"))
    result = run_cli("stats", "--json")
    _assert_ok(result)
    payload = _json_stdout(result)
    assert payload["memories"] == 1
    assert payload["events"] == 1
    assert payload["db_size_human"]
    assert payload["health"]["ollama_available"] is True


def test_config_show_json(run_cli: Any, tmp_home: Path) -> None:
    result = run_cli("config", "show", "--json")
    _assert_ok(result)
    payload = _json_stdout(result)
    assert payload["home"] == str(tmp_home)
    assert payload["db_path"].endswith("amnesia.db")
    assert payload["memory_enabled"] is True
    assert payload["api_token"] is None


def test_config_show_text(run_cli: Any) -> None:
    result = run_cli("config", "show")
    _assert_ok(result)
    assert "Amnesia configuration" in result.output
    assert "memory enabled" in result.output


# ---------------------------------------------------------------------------
# export / import / backup / purge
# ---------------------------------------------------------------------------


def test_export_creates_json_file(run_cli: Any, tmp_path: Path) -> None:
    _assert_ok(run_cli("remember", "Export me", "--kind", "fact"))
    _assert_ok(run_cli("note", "Export note"))
    destination = tmp_path / "exports" / "backup.json"

    result = run_cli("export", str(destination))
    _assert_ok(result)
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["format"] == "amnesia-export"
    assert any(memory["content"] == "Export me" for memory in payload["memories"])
    assert len(payload["events"]) == 1


def test_backup_creates_file(run_cli: Any, tmp_path: Path) -> None:
    _assert_ok(run_cli("remember", "Backup me", "--kind", "fact"))
    destination = tmp_path / "backups" / "amnesia-backup.db"

    result = run_cli("backup", str(destination))
    _assert_ok(result)
    assert destination.exists() and destination.stat().st_size > 0
    assert "backed up" in result.output.lower()


def test_purge_requires_yes(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "Keep me", "--kind", "fact"))
    refused = run_cli("purge")
    assert refused.exit_code == 1
    assert "Refusing to purge" in refused.output
    # Nothing was deleted.
    assert _json_stdout(run_cli("profile", "--json"))["count"] == 1


def test_export_import_roundtrip(run_cli: Any, tmp_path: Path) -> None:
    _assert_ok(run_cli("remember", "Import me later", "--kind", "fact"))
    _assert_ok(run_cli("note", "Import this event"))
    destination = tmp_path / "roundtrip.json"
    _assert_ok(run_cli("export", str(destination)))

    purged = run_cli("purge", "--yes")
    _assert_ok(purged)
    assert _json_stdout(run_cli("profile", "--json"))["count"] == 0

    imported = run_cli("import", str(destination))
    _assert_ok(imported)
    assert "Imported" in imported.output
    payload = _json_stdout(run_cli("profile", "--json"))
    assert any(fact["content"] == "Import me later" for fact in payload["facts"])
    timeline = _json_stdout(run_cli("timeline", "--json", "--days", "2"))
    assert any(event["title"] == "Import this event" for event in timeline["events"])


def test_import_missing_file_is_friendly(run_cli: Any, tmp_path: Path) -> None:
    result = run_cli("import", str(tmp_path / "does-not-exist.json"))
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_purge_old_events(run_cli: Any, settings: Settings) -> None:
    with _store(settings) as store:
        store.add_event(title="Ancient event", occurred_at="2020-01-01T00:00:00+00:00")
    result = run_cli("purge", "--yes", "--events-older-than", "1")
    _assert_ok(result)
    assert "Deleted 1 event(s)" in result.output
    timeline = _json_stdout(run_cli("timeline", "--json"))
    assert timeline["count"] == 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_healthy_with_fake_ollama(run_cli: Any) -> None:
    _assert_ok(run_cli("remember", "A fact", "--kind", "fact"))
    result = run_cli("doctor")
    _assert_ok(result)
    assert "Amnesia doctor" in result.output
    assert "Ollama" in result.output
    assert "0.40.11" in result.output
    assert "Stored data" in result.output


def test_doctor_fails_when_ollama_down(run_cli: Any, fake_ollama: Any) -> None:
    fake_ollama.fail_connection = True
    result = run_cli("doctor")
    assert result.exit_code == 1
    assert "unreachable" in result.output
    assert "hard failure" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# chat (streaming REPL)
# ---------------------------------------------------------------------------


def test_chat_streams_reply_and_persists_turn(
    run_cli: Any, settings: Settings, fake_ollama: Any
) -> None:
    fake_ollama.queue_reply("Hello from the fake model.")
    fake_ollama.stream_split_words = True

    result = run_cli("chat", input="hi there\n/quit\n")
    _assert_ok(result)
    assert "Hello from the fake model." in result.output
    assert "amnesia chat" in result.output

    messages = _session_messages(settings)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "hi there"
    assert messages[1].content.strip() == "Hello from the fake model."


def test_chat_non_streaming(run_cli: Any, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("Non-stream reply")
    result = run_cli("chat", "--no-stream", input="hello\n/quit\n")
    _assert_ok(result)
    assert "Non-stream reply" in result.output


def test_chat_injects_memory(run_cli: Any, fake_ollama: Any) -> None:
    _assert_ok(run_cli("remember", "User loves tea", "--kind", "fact", "--importance", "0.9"))
    fake_ollama.queue_reply("Noted.")
    result = run_cli("chat", input="what do I like?\n/quit\n")
    _assert_ok(result)

    payload = fake_ollama.last_chat_payload()
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert "<amnesia_memory>" in system["content"]
    assert "User loves tea" in system["content"]


def test_chat_no_memory_disables_injection(run_cli: Any, fake_ollama: Any) -> None:
    _assert_ok(run_cli("remember", "User loves tea", "--kind", "fact"))
    fake_ollama.queue_reply("Noted.")
    result = run_cli("chat", "--no-memory", input="what do I like?\n/quit\n")
    _assert_ok(result)

    payload = fake_ollama.last_chat_payload()
    combined = " ".join(str(message.get("content") or "") for message in payload["messages"])
    assert "<amnesia_memory>" not in combined
    assert "User loves tea" not in combined


def test_chat_slash_commands(run_cli: Any, settings: Settings, tmp_path: Path) -> None:
    export_path = tmp_path / "chat-export.json"
    result = run_cli(
        "chat",
        input=(
            "/help\n"
            "/remember I prefer dark mode\n"
            "/profile\n"
            "/memory off\n"
            "/memory on\n"
            "/stats\n"
            f"/export {export_path}\n"
            "/new\n"
            "/session\n"
            "/quit\n"
        ),
    )
    _assert_ok(result)
    assert "Chat commands" in result.output
    assert "Memory saved" in result.output
    assert "I prefer dark mode" in result.output
    assert "Memory injection is off" in result.output
    assert "Started session" in result.output
    assert export_path.exists()

    with _store(settings) as store:
        assert any("dark mode" in memory.content for memory in store.list_identity_memories())


def test_chat_forget_slash_command(run_cli: Any, settings: Settings) -> None:
    _assert_ok(run_cli("remember", "Forget this soon", "--kind", "fact"))
    with _store(settings) as store:
        memory_id = store.list_identity_memories()[0].id
    result = run_cli("chat", input=f"/forget {memory_id}\n/quit\n")
    _assert_ok(result)
    assert "deactivated" in result.output


def test_chat_unknown_slash_command(run_cli: Any) -> None:
    result = run_cli("chat", input="/nope\n/quit\n")
    _assert_ok(result)
    assert "Unknown command" in result.output


def test_chat_requires_ollama(run_cli: Any, fake_ollama: Any) -> None:
    fake_ollama.fail_connection = True
    result = run_cli("chat", input="/quit\n")
    assert result.exit_code == 1
    assert "Ollama is not reachable" in result.output
    assert "Traceback" not in result.output


def test_chat_history_is_sent(run_cli: Any, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("first reply")
    fake_ollama.queue_reply("second reply")
    result = run_cli("chat", input="first question\nsecond question\n/quit\n")
    _assert_ok(result)

    payload = fake_ollama.last_chat_payload()
    roles = [message["role"] for message in payload["messages"]]
    assert roles[0] == "system"
    assert "user" in roles and "assistant" in roles
    assert any(
        message["content"] == "first reply" for message in payload["messages"]
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_prints_answer_and_persists(run_cli: Any, settings: Settings, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("A one-shot answer.")
    result = run_cli("ask", "What do you know?")
    _assert_ok(result)
    assert "A one-shot answer." in result.output

    messages = _session_messages(settings)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "What do you know?"


def test_ask_non_streaming_injects_memory(run_cli: Any, fake_ollama: Any) -> None:
    _assert_ok(run_cli("remember", "User is a Python developer", "--kind", "identity"))
    fake_ollama.queue_reply("You are a Python developer.")
    result = run_cli("ask", "what is my job?", "--no-stream")
    _assert_ok(result)
    assert "You are a Python developer." in result.output

    payload = fake_ollama.last_chat_payload()
    system = payload["messages"][0]
    assert "<amnesia_memory>" in system["content"]
    assert "User is a Python developer" in system["content"]


def test_ask_without_ollama_is_friendly(run_cli: Any, fake_ollama: Any) -> None:
    fake_ollama.fail_connection = True
    result = run_cli("ask", "hello")
    assert result.exit_code == 1
    assert "Ollama is not reachable" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


def test_sessions_and_session_show(run_cli: Any, settings: Settings, fake_ollama: Any) -> None:
    fake_ollama.queue_reply("Session answer")
    _assert_ok(run_cli("chat", input="session question\n/quit\n"))

    listing = run_cli("sessions")
    _assert_ok(listing)
    assert "Sessions" in listing.output
    assert "session question" in _flat(listing.output)

    with _store(settings) as store:
        session = store.latest_session()
        assert session is not None
        session_id = session.id

    shown = run_cli("session", "show", session_id[:8])
    _assert_ok(shown)
    assert "session question" in _flat(shown.output)
    assert "Session answer" in _flat(shown.output)


def test_session_show_missing(run_cli: Any) -> None:
    result = run_cli("session", "show", "does-not-exist")
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_non_interactive(run_cli: Any, tmp_home: Path) -> None:
    result = run_cli("init")
    _assert_ok(result)
    assert "Initialised" in result.output
    assert (tmp_home / "amnesia.db").exists()
    assert (tmp_home / "config.json").exists()


def test_init_interactive_stores_onboarding(run_cli: Any) -> None:
    answers = "\n".join(
        [
            "Bilbo Baggins",  # name
            "Software engineer",  # role
            "Shenzhen",  # location
            "English, Chinese",  # languages
            "Amnesia",  # projects
            "concise",  # answer style
        ]
    )
    result = run_cli("init", "--interactive", input=answers + "\n")
    _assert_ok(result)
    assert "Stored 6 onboarding memory item(s)" in result.output

    profile = _json_stdout(run_cli("profile", "--json"))
    contents = " ".join(fact["content"] for fact in profile["facts"])
    assert "Bilbo Baggins" in contents
    assert "Software engineer" in contents
    assert "Shenzhen" in contents
    assert all(fact["source"] == "onboarding" for fact in profile["facts"])


# ---------------------------------------------------------------------------
# serve / mcp / distill guardrails
# ---------------------------------------------------------------------------


def test_serve_help_documents_flags(run_cli: Any) -> None:
    result = run_cli("serve", "--help")
    _assert_ok(result)
    for flag in ("--host", "--port", "--no-extraction", "--reload"):
        assert flag in result.output


def test_serve_builds_proxy_app_and_calls_uvicorn(
    run_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` must build the real proxy app and hand it to uvicorn's Server.

    ``sys.modules["uvicorn"]`` is replaced with a fake, so this test can never
    bind a port or start a real server.
    """
    import sys
    import types

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            captured["app"] = app
            captured["config"] = kwargs

    class _FakeServer:
        def __init__(self, config: Any) -> None:
            captured["server_config"] = config

        async def serve(self) -> None:
            captured["served"] = True

    # Fail loudly if anything tries to bind a port / start uvicorn's run().
    def _forbidden(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - safety net
        raise AssertionError("serve must not start a real server in tests")

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(Config=_FakeConfig, Server=_FakeServer, run=_forbidden),
    )
    result = run_cli(
        "serve", "--host", "127.0.0.1", "--port", "12345", "--no-extraction"
    )
    _assert_ok(result)
    assert captured.get("served") is True
    assert captured["config"]["host"] == "127.0.0.1"
    assert captured["config"]["port"] == 12345
    # Operators see startup info from `serve` even though other commands are quiet.
    assert captured["config"]["log_level"] == "info"
    assert captured["app"] is not None
    flat = _flat(result.output)
    assert "Dashboard" in flat or "dashboard" in flat
    assert "Server stopped." in flat


def test_serve_without_proxy_module_is_friendly(
    run_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If workstream B is unavailable, `serve` fails cleanly instead of tracing."""

    def _missing_proxy() -> Any:
        raise ImportError("simulated missing proxy module")

    monkeypatch.setattr(cli, "_load_proxy_module", _missing_proxy)
    result = run_cli("serve", "--port", "12345")
    assert result.exit_code == 1
    assert "proxy" in result.output.lower()
    assert "Traceback" not in result.output


def test_distill_without_extraction_module_is_friendly(
    run_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken_build(service: Amnesia) -> Any:
        return None

    monkeypatch.setattr(cli, "_build_extractor", _broken_build)
    result = run_cli("distill")
    assert result.exit_code == 1
    assert "unavailable" in result.output.lower()
    assert "Traceback" not in result.output


class _FakeExtractor:
    """Stands in for MemoryExtractor so the distill command is tested in isolation."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def distill_session(
        self, session_id: str, *, force: bool = False, max_turns: int = 12
    ) -> dict[str, Any]:
        self.calls.append(session_id)
        return {
            "facts": 2,
            "events": 1,
            "summary": "A concise session summary.",
            "skipped": False,
        }


def test_distill_latest_session(
    run_cli: Any, fake_ollama: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ollama.queue_reply("distill me")
    _assert_ok(run_cli("chat", input="remember this chat\n/quit\n"))

    extractor = _FakeExtractor()
    monkeypatch.setattr(cli, "_build_extractor", lambda service: extractor)
    result = run_cli("distill")
    _assert_ok(result)
    assert len(extractor.calls) == 1
    flat = _flat(result.output)
    assert "A concise session summary." in flat
    assert "2 fact(s), 1 event(s)" in flat


def test_distill_all_sessions(
    run_cli: Any, fake_ollama: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ollama.queue_reply("first")
    _assert_ok(run_cli("chat", "--session", "session-one", input="hello\n/quit\n"))
    fake_ollama.queue_reply("second")
    _assert_ok(run_cli("chat", "--session", "session-two", input="hello again\n/quit\n"))

    extractor = _FakeExtractor()
    monkeypatch.setattr(cli, "_build_extractor", lambda service: extractor)
    result = run_cli("distill", "--all", "--limit", "5")
    _assert_ok(result)
    assert set(extractor.calls) == {"session-one", "session-two"}
    assert "Distilled 2 session(s)" in _flat(result.output)


def test_mcp_command_without_sdk_is_friendly(
    run_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import amnesia.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server, "MCP_AVAILABLE", False)
    result = run_cli("mcp")
    assert result.exit_code == 1
    assert "mcp" in result.output.lower()
