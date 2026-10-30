"""Unit tests for ``amnesia/ollama_control.py`` (zero-config takeover mode).

The module manages the user's real Ollama processes and sockets, so every test
here is fully hermetic: process discovery, port probes, HTTP version checks,
daemon spawning and desktop-app restart are replaced through the module's
private seams.  Nothing opens a socket, starts a subprocess, enumerates psutil,
writes outside ``tmp_path``, or sleeps for real.
"""

from __future__ import annotations

import time as real_time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import amnesia.ollama_control as oc
from amnesia.ollama_control import (
    DEFAULT_OLLAMA_PORT,
    DEFAULT_UPSTREAM_PORT,
    OllamaProcess,
    OllamaTakeover,
    StopReport,
    TakeoverError,
    stop_existing_ollama,
)

# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------


class FakePopen:
    """Minimal ``subprocess.Popen`` stand-in for the managed daemon."""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _fake_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap ``ollama_control.time`` for a clock that never sleeps."""
    monkeypatch.setattr(
        oc,
        "time",
        SimpleNamespace(monotonic=real_time.monotonic, sleep=lambda _seconds: None),
    )


def _forbidden(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - safety net
    raise AssertionError("unexpected process/socket side effect in a unit test")


def _ollama_processes() -> list[OllamaProcess]:
    return [
        OllamaProcess(pid=101, name="ollama app.exe"),
        OllamaProcess(
            pid=202,
            name="ollama.exe",
            cmdline=("C:/Program Files/Ollama/ollama.exe", "serve"),
        ),
        OllamaProcess(
            pid=303,
            name="ollama.exe",
            cmdline=("C:/Program Files/Ollama/ollama.exe", "run", "dolphin3:latest"),
        ),
    ]


# ---------------------------------------------------------------------------
# A1. Process classification
# ---------------------------------------------------------------------------


def test_ollama_process_classifies_desktop_app_by_name() -> None:
    process = OllamaProcess(pid=11, name="ollama app.exe")
    assert process.is_desktop_app is True
    assert process.is_server is False
    described = process.describe()
    assert "desktop app" in described
    assert "pid=11" in described


def test_ollama_process_classifies_desktop_app_by_exe_path() -> None:
    process = OllamaProcess(
        pid=12,
        name="Ollama",
        exe="C:/Users/user/AppData/Local/Programs/Ollama/ollama app.exe",
    )
    assert process.is_desktop_app is True


def test_ollama_process_classifies_macos_app_bundle() -> None:
    process = OllamaProcess(
        pid=13,
        name="Ollama",
        exe="/Applications/Ollama.app/Contents/MacOS/Ollama",
        cmdline=("/Applications/Ollama.app/Contents/MacOS/Ollama",),
    )
    assert process.is_desktop_app is True


def test_ollama_process_classifies_server_and_chat_client() -> None:
    daemon = OllamaProcess(
        pid=21,
        name="ollama.exe",
        cmdline=("C:/Program Files/Ollama/ollama.exe", "serve"),
    )
    assert daemon.is_server is True
    assert daemon.is_desktop_app is False
    assert daemon.describe().startswith("daemon")
    assert "serve" in daemon.command

    client = OllamaProcess(
        pid=22,
        name="ollama.exe",
        cmdline=("C:/Program Files/Ollama/ollama.exe", "run", "dolphin3:latest"),
    )
    assert client.is_server is False
    assert client.is_desktop_app is False
    assert "dolphin3" in client.command


def test_ollama_process_server_detection_is_case_insensitive() -> None:
    process = OllamaProcess(pid=23, name="ollama.exe", cmdline=("ollama.exe", "Serve"))
    assert process.is_server is True


def test_stop_report_empty_and_flagged_states() -> None:
    empty = StopReport()
    assert empty.stopped == []
    assert empty.desktop_stopped is False
    assert empty.summary == "nothing to stop"

    desktop = OllamaProcess(pid=31, name="ollama app.exe")
    daemon = OllamaProcess(pid=32, name="ollama.exe", cmdline=("ollama.exe", "serve"))
    report = StopReport(stopped=[desktop, daemon])
    assert report.desktop_stopped is True
    assert "desktop app" in report.summary
    assert "daemon pid=32" in report.summary


# ---------------------------------------------------------------------------
# A2. stop_existing_ollama
# ---------------------------------------------------------------------------


def test_stop_existing_stops_app_and_daemon_but_not_chat_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sleep(monkeypatch)
    port_state = {"open": True}
    terminated: list[int] = []
    probed_hosts: list[str] = []

    def fake_port_open(host: str, port: int, *, timeout: float = 0.35) -> bool:
        probed_hosts.append(host)
        return port_state["open"]

    def fake_terminate(process: OllamaProcess, *, timeout: float = 8.0) -> bool:
        terminated.append(process.pid)
        port_state["open"] = False  # a killed daemon releases the default port
        return True

    monkeypatch.setattr(oc, "_port_open", fake_port_open)
    monkeypatch.setattr(oc, "_iter_ollama_processes", lambda: _ollama_processes())
    monkeypatch.setattr(oc, "_terminate_process", fake_terminate)

    report = stop_existing_ollama(host="0.0.0.0", port=DEFAULT_OLLAMA_PORT)

    assert terminated == [101, 202]  # desktop app first, then the serve daemon
    assert [process.pid for process in report.stopped] == [101, 202]
    assert report.desktop_stopped is True
    assert "desktop app pid=101" in report.summary
    assert "daemon pid=202" in report.summary
    assert 303 not in terminated  # `ollama run` client is left alone
    assert probed_hosts and set(probed_hosts) == {"127.0.0.1"}  # wildcard normalised


def test_stop_existing_reports_only_successfully_terminated_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sleep(monkeypatch)
    port_state = {"open": True}

    def fake_port_open(host: str, port: int, *, timeout: float = 0.35) -> bool:
        return port_state["open"]

    def fake_terminate(process: OllamaProcess, *, timeout: float = 8.0) -> bool:
        port_state["open"] = False
        if process.pid == 202:  # daemon refused to die
            return False
        return True

    monkeypatch.setattr(oc, "_port_open", fake_port_open)
    monkeypatch.setattr(oc, "_iter_ollama_processes", lambda: _ollama_processes())
    monkeypatch.setattr(oc, "_terminate_process", fake_terminate)

    report = stop_existing_ollama(port=DEFAULT_OLLAMA_PORT)

    assert [process.pid for process in report.stopped] == [101]
    assert report.desktop_stopped is True
    assert "pid=101" in report.summary
    assert "pid=202" not in report.summary


# ---------------------------------------------------------------------------
# A3. Managed takeover happy path
# ---------------------------------------------------------------------------


def test_managed_takeover_starts_daemon_and_reports_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sleep(monkeypatch)
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: False)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "C:/fake/ollama.exe")
    process = FakePopen(pid=4321)
    spawned: list[dict[str, Any]] = []

    def fake_spawn(
        binary: str, *, host: str, port: int, log_path: Path
    ) -> FakePopen:
        spawned.append(
            {"binary": binary, "host": host, "port": port, "log_path": Path(log_path)}
        )
        return process

    def fake_wait(url: str, *, timeout: float, process: Any = None, log_path: Any = None):
        return "0.40.13"

    monkeypatch.setattr(oc, "_spawn_daemon", fake_spawn)
    monkeypatch.setattr(oc, "_wait_until_healthy", fake_wait)

    takeover = OllamaTakeover(
        host="127.0.0.1",
        port=DEFAULT_OLLAMA_PORT,
        upstream_port=DEFAULT_UPSTREAM_PORT,
        log_dir=tmp_path,
    )
    try:
        started = takeover.start()
        assert started is takeover
        assert takeover.version == "0.40.13"
        assert takeover.managed is True
        assert takeover.takeover_active is True
        assert takeover.upstream_url == "http://127.0.0.1:11435"
        assert takeover.listen_url == "http://127.0.0.1:11434"
        assert takeover.process is process
        assert takeover._atexit_registered is True

        assert len(spawned) == 1
        spawn_call = spawned[0]
        assert spawn_call["binary"] == "C:/fake/ollama.exe"
        assert spawn_call["host"] == "127.0.0.1"
        assert spawn_call["port"] == 11435
        assert spawn_call["log_path"] == tmp_path / "ollama-takeover.log"
        assert spawn_call["log_path"].parent == tmp_path
    finally:
        takeover.stop()

    assert process.terminated is True
    assert takeover.process is None
    assert takeover.takeover_active is False
    assert takeover._atexit_registered is False


# ---------------------------------------------------------------------------
# A4. Stopping the existing Ollama and restoring the desktop app
# ---------------------------------------------------------------------------


def test_takeover_stops_existing_ollama_and_restores_desktop_app_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sleep(monkeypatch)
    port_state = {"open": True}
    terminated: list[int] = []
    restarts: list[str] = []

    def fake_port_open(host: str, port: int, *, timeout: float = 0.35) -> bool:
        return port_state["open"]

    def fake_terminate_process(process: OllamaProcess, *, timeout: float = 8.0) -> bool:
        terminated.append(process.pid)
        port_state["open"] = False
        return True

    monkeypatch.setattr(oc, "_port_open", fake_port_open)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_version", lambda url, **kw: "0.40.13")
    monkeypatch.setattr(oc, "_port_owner_pid", lambda port: None)
    monkeypatch.setattr(oc, "_iter_ollama_processes", lambda: _ollama_processes()[:2])
    monkeypatch.setattr(oc, "_terminate_process", fake_terminate_process)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "C:/fake/ollama.exe")
    monkeypatch.setattr(oc, "_spawn_daemon", lambda *a, **kw: FakePopen(pid=5555))
    monkeypatch.setattr(oc, "_wait_until_healthy", lambda url, **kw: "0.40.13")
    monkeypatch.setattr(oc, "_restart_desktop_app", lambda: restarts.append("restart") or True)

    takeover = OllamaTakeover(log_dir=tmp_path)
    try:
        takeover.start()
        assert takeover.stopped is not None
        assert takeover.stopped.desktop_stopped is True
        assert terminated == [101, 202]
        assert takeover._desktop_restore_pending is True
        assert takeover.desktop_restarted is False
    finally:
        takeover.stop()

    assert restarts == ["restart"]
    assert takeover.desktop_restarted is True
    assert takeover._desktop_restore_pending is False
    assert takeover.process is None
    assert takeover._atexit_registered is False

    # Idempotent: a second stop must not restart the desktop app again.
    takeover.stop()
    assert restarts == ["restart"]
    assert takeover.desktop_restarted is True


# ---------------------------------------------------------------------------
# A5. Failure paths: actionable errors and cleanup
# ---------------------------------------------------------------------------


def test_fails_when_port_is_held_by_another_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: True)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_version", lambda url, **kw: None)
    monkeypatch.setattr(oc, "_port_owner_pid", lambda port: 777)
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    message = str(info.value)
    assert "another program" in message
    assert "pid 777" in message
    assert "--port" in message
    assert takeover.takeover_active is False
    assert takeover.process is None
    assert takeover._atexit_registered is False


def test_fails_when_another_amnesia_already_owns_the_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: True)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: True)
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    assert "already serving an Amnesia proxy" in str(info.value)


def test_fails_when_ollama_runs_and_keep_flag_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: True)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_version", lambda url, **kw: "0.40.13")
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(stop_existing=False, log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    message = str(info.value)
    assert "Quit it first" in message
    assert "--keep-ollama" in message
    assert takeover.stopped is None


def test_fails_when_upstream_port_is_already_in_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        oc,
        "_port_open",
        lambda host, port, **kw: int(port) == DEFAULT_UPSTREAM_PORT,
    )
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_binary", _forbidden)
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    message = str(info.value)
    assert "--upstream-port" in message
    assert "already in use" in message
    assert takeover.process is None


def test_health_timeout_terminates_the_spawned_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: False)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "C:/fake/ollama.exe")
    process = FakePopen(pid=6666)
    monkeypatch.setattr(oc, "_spawn_daemon", lambda *a, **kw: process)

    def fake_wait(*args: Any, **kwargs: Any) -> str:
        raise TakeoverError("Timed out after 60s waiting for Ollama at http://127.0.0.1:11435.")

    monkeypatch.setattr(oc, "_wait_until_healthy", fake_wait)

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    assert "Timed out" in str(info.value)
    assert process.terminated is True
    assert takeover.process is None
    assert takeover.takeover_active is False
    assert takeover._atexit_registered is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 0},
        {"port": 70000},
        {"port": -1},
        {"upstream_port": 0},
        {"upstream_port": 70000},
        {"port": 11434, "upstream_port": 11434},
    ],
)
def test_invalid_ports_are_rejected(
    tmp_path: Path, kwargs: dict[str, int]
) -> None:
    with pytest.raises(TakeoverError):
        OllamaTakeover(log_dir=tmp_path, **kwargs)


def test_failure_after_stopping_desktop_app_restores_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after the tray app was stopped must leave the user with Ollama."""
    _fake_sleep(monkeypatch)
    port_state = {"open": True}
    restarts: list[str] = []
    desktop = _ollama_processes()[0]

    def fake_port_open(host: str, port: int, *, timeout: float = 0.35) -> bool:
        return port_state["open"]

    def fake_terminate_process(process: OllamaProcess, *, timeout: float = 8.0) -> bool:
        port_state["open"] = False
        return True

    def failing_spawn(*args: Any, **kwargs: Any) -> FakePopen:
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(oc, "_port_open", fake_port_open)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_version", lambda url, **kw: "0.40.13")
    monkeypatch.setattr(oc, "_port_owner_pid", lambda port: None)
    monkeypatch.setattr(oc, "_iter_ollama_processes", lambda: [desktop])
    monkeypatch.setattr(oc, "_terminate_process", fake_terminate_process)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "C:/fake/ollama.exe")
    monkeypatch.setattr(oc, "_spawn_daemon", failing_spawn)
    monkeypatch.setattr(oc, "_restart_desktop_app", lambda: restarts.append("restart") or True)

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="spawn boom"):
        takeover.start()

    assert restarts == ["restart"]
    assert takeover._desktop_restore_pending is False
    assert takeover.takeover_active is False
    assert takeover.process is None
    assert takeover._atexit_registered is False


# ---------------------------------------------------------------------------
# A6. stop() before start()
# ---------------------------------------------------------------------------


def test_stop_before_start_is_a_harmless_noop(tmp_path: Path) -> None:
    takeover = OllamaTakeover(log_dir=tmp_path)
    takeover.stop()  # must not raise and must not touch anything

    assert takeover.process is None
    assert takeover.desktop_restarted is False
    assert takeover.takeover_active is False
    assert takeover._atexit_registered is False


# ---------------------------------------------------------------------------
# A7. External upstream mode
# ---------------------------------------------------------------------------


def test_external_upstream_skips_daemon_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_urls: list[str] = []
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: False)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)

    def fake_version(url: str, **kwargs: Any) -> str:
        checked_urls.append(url)
        return "0.40.13"

    monkeypatch.setattr(oc, "_ollama_version", fake_version)
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(
        upstream_url="http://127.0.0.1:9999", log_dir=tmp_path
    )
    try:
        takeover.start()
        assert takeover.managed is False
        assert takeover.version == "0.40.13"
        assert takeover.upstream_url == "http://127.0.0.1:9999"
        assert takeover.process is None
        assert checked_urls == ["http://127.0.0.1:9999"]
        # Equal ports are fine for an external upstream: nothing is spawned.
        external = OllamaTakeover(
            upstream_url="http://127.0.0.1:9999",
            port=DEFAULT_OLLAMA_PORT,
            upstream_port=DEFAULT_OLLAMA_PORT,
            log_dir=tmp_path,
        )
        assert external.managed is False
    finally:
        takeover.stop()

    assert takeover._atexit_registered is False


def test_external_upstream_unreachable_raises_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: False)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_version", lambda url, **kw: None)
    monkeypatch.setattr(oc, "_spawn_daemon", _forbidden)

    takeover = OllamaTakeover(upstream_url="http://127.0.0.1:9999", log_dir=tmp_path)
    with pytest.raises(TakeoverError) as info:
        takeover.start()

    message = str(info.value)
    assert "not answering" in message
    assert "http://127.0.0.1:9999" in message
    assert "Start it first" in message
    assert takeover.takeover_active is False


# ---------------------------------------------------------------------------
# A8. Loopback normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("::", "127.0.0.1"),
        ("[::]", "127.0.0.1"),
        ("*", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "localhost"),
        ("192.168.1.5", "192.168.1.5"),
    ],
)
def test_loopback_normalises_wildcard_binds(host: str, expected: str) -> None:
    assert oc._loopback(host) == expected


# ---------------------------------------------------------------------------
# A9. Managed-daemon pid bookkeeping (crash recovery)
# ---------------------------------------------------------------------------


def _install_managed_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Install the happy-path seams for a managed takeover; return spawn calls."""
    _fake_sleep(monkeypatch)
    spawned: list[dict[str, Any]] = []

    def fake_spawn(
        binary: str, *, host: str, port: int, log_path: Path
    ) -> FakePopen:
        spawned.append(
            {"binary": binary, "host": host, "port": port, "log_path": Path(log_path)}
        )
        return FakePopen(pid=4321)

    monkeypatch.setattr(oc, "_port_open", lambda host, port, **kw: False)
    monkeypatch.setattr(oc, "_amnesia_already_running", lambda host, port: False)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "C:/fake/ollama.exe")
    monkeypatch.setattr(oc, "_spawn_daemon", fake_spawn)
    monkeypatch.setattr(oc, "_wait_until_healthy", lambda url, **kw: "0.40.13")
    return spawned


def test_pid_file_tracks_the_managed_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_managed_seams(monkeypatch)
    pid_path = tmp_path / "ollama-takeover.pid"

    takeover = OllamaTakeover(log_dir=tmp_path)
    try:
        takeover.start()
        assert pid_path.read_text(encoding="ascii") == "4321"
    finally:
        takeover.stop()

    assert not pid_path.exists()


def test_stale_managed_daemon_is_recovered_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _install_managed_seams(monkeypatch)
    pid_path = tmp_path / "ollama-takeover.pid"
    pid_path.write_text("9999", encoding="ascii")
    recovered: list[tuple[int, int, float]] = []

    def fake_recover(pid: int, *, port: int, timeout: float) -> bool:
        recovered.append((pid, port, timeout))
        return True

    monkeypatch.setattr(oc, "_terminate_stale_managed_daemon", fake_recover)

    takeover = OllamaTakeover(log_dir=tmp_path)
    try:
        takeover.start()
        assert len(spawned) == 1
        # The stale pid was recovered before the new daemon was spawned, and
        # the pid file now tracks the new process.
        assert recovered and recovered[0][0] == 9999
        assert recovered[0][1] == DEFAULT_UPSTREAM_PORT
        assert pid_path.read_text(encoding="ascii") == "4321"
    finally:
        takeover.stop()


def test_pid_file_is_removed_when_start_fails_after_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_managed_seams(monkeypatch)

    def failing_wait(*args: Any, **kwargs: Any) -> str:
        raise TakeoverError("busy")

    monkeypatch.setattr(oc, "_wait_until_healthy", failing_wait)
    pid_path = tmp_path / "ollama-takeover.pid"

    takeover = OllamaTakeover(log_dir=tmp_path)
    with pytest.raises(TakeoverError, match="busy"):
        takeover.start()

    assert not pid_path.exists()
