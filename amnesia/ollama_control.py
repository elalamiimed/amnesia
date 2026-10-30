"""Takeover mode: put Amnesia on Ollama's default port.

Ollama clients -- the ``ollama`` CLI, the desktop app and most SDKs -- default
to ``http://127.0.0.1:11434``.  Amnesia can only add memory to requests that
pass through it, so this module lets ``amnesia serve --takeover`` bind that
standard port and move the real Ollama daemon to a private loopback port that
Amnesia manages as a child process.

The result is zero-configuration memory: plain ``ollama run <model>`` (or any
Ollama/OpenAI client that talks to the default port) receives the same memory
injection and conversation capture as ``amnesia chat``, without exporting
``OLLAMA_HOST`` in every shell.

Nothing here runs unless takeover mode is explicitly requested.  When it is,
a running Ollama desktop app or ``ollama serve`` daemon on port 11434 is
stopped first (the desktop app is restarted when Amnesia exits), the managed
daemon is spawned with ``OLLAMA_HOST=127.0.0.1:<upstream-port>`` and removed
when Amnesia stops.  The user's own environment is never modified.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("amnesia.ollama_control")

#: Default port Ollama clients talk to.
DEFAULT_OLLAMA_PORT = 11434
#: Default private port for the managed Ollama daemon.
DEFAULT_UPSTREAM_PORT = 11435
#: How long to wait for a spawned daemon to answer ``/api/version``.
DEFAULT_START_TIMEOUT = 60.0
#: How long to wait for the old daemon to release port 11434.
DEFAULT_STOP_TIMEOUT = 15.0
#: Health-poll interval while waiting for the managed daemon.
HEALTH_INTERVAL = 0.35


class TakeoverError(RuntimeError):
    """Raised when takeover mode cannot safely start (friendly for the CLI)."""


# ---------------------------------------------------------------------------
# Process discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaProcess:
    """One running ``ollama`` process discovered through psutil."""

    pid: int
    name: str = ""
    exe: str | None = None
    cmdline: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        return " ".join(self.cmdline) or (self.exe or self.name)

    @property
    def is_desktop_app(self) -> bool:
        lowered = f"{self.name} {self.exe or ''} {self.command}".lower()
        return "ollama app" in lowered or ".app/contents/macos" in lowered

    @property
    def is_server(self) -> bool:
        args = [part.lower() for part in self.cmdline[1:]]
        if "serve" in args:
            return True
        # ``ollama serve`` can also be started with a path as argv[0] only;
        # fall back to the process name for daemons we cannot introspect.
        return False

    def describe(self) -> str:
        kind = "desktop app" if self.is_desktop_app else "daemon"
        return f"{kind} pid={self.pid} ({self.name or '?'})"


@dataclass
class StopReport:
    """Outcome of :func:`stop_existing_ollama`."""

    stopped: list[OllamaProcess] = field(default_factory=list)

    @property
    def desktop_stopped(self) -> bool:
        return any(item.is_desktop_app for item in self.stopped)

    @property
    def summary(self) -> str:
        if not self.stopped:
            return "nothing to stop"
        return ", ".join(f"{item.describe()}" for item in self.stopped)


def _iter_ollama_processes() -> list[OllamaProcess]:
    """Return the running Ollama processes (tray app, daemons, clients)."""
    try:
        import psutil  # noqa: PLC0415 - optional dependency, Windows takeover only
    except ImportError as exc:  # pragma: no cover - declared in the windows extra
        raise TakeoverError(
            "Takeover mode needs the 'psutil' package to manage the Ollama "
            "process. Install it with: pip install psutil"
        ) from exc

    found: list[OllamaProcess] = []
    for process in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = process.info or {}
            pid = int(info.get("pid") or 0)
            name = str(info.get("name") or "")
            exe = info.get("exe")
            exe = str(exe) if exe else None
            cmdline = tuple(str(part) for part in (info.get("cmdline") or ()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
        if not pid or pid == os.getpid():
            continue
        haystack = f"{name} {exe or ''} {' '.join(cmdline)}".lower()
        if "ollama" not in haystack:
            continue
        found.append(OllamaProcess(pid=pid, name=name, exe=exe, cmdline=cmdline))
    return found


def _port_owner_pid(port: int) -> int | None:
    """Best-effort PID listening on ``port`` (used only in error messages)."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:  # pragma: no cover - path only reached with psutil present
        return None
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, OSError):  # pragma: no cover - needs admin on some boxes
        return None
    for connection in connections:
        if connection.laddr and connection.laddr.port == port and connection.pid:
            return int(connection.pid)
    return None


def _terminate_process(process: OllamaProcess, *, timeout: float = 8.0) -> bool:
    """Terminate one Ollama process; return True when it is gone."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise TakeoverError("psutil is required for takeover mode.") from exc
    try:
        handle = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied as exc:
        raise TakeoverError(
            f"Cannot stop {process.describe()}: access denied. Quit Ollama manually "
            "(tray icon > Quit Ollama) and run the command again."
        ) from exc
    try:
        handle.terminate()
        try:
            handle.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            handle.kill()
            handle.wait(timeout=5.0)
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied as exc:
        raise TakeoverError(
            f"Cannot stop {process.describe()}: access denied. Quit Ollama manually "
            "(tray icon > Quit Ollama) and run the command again."
        ) from exc
    except OSError as exc:  # pragma: no cover - unusual Windows failures
        LOG.debug("Terminating %s failed: %s", process.describe(), exc)
        return False
    return True


def stop_existing_ollama(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_OLLAMA_PORT,
    timeout: float = DEFAULT_STOP_TIMEOUT,
) -> StopReport:
    """Stop the Ollama desktop app / daemons that own the default port.

    Only Ollama processes are touched.  The caller must verify the port is
    free afterwards; :class:`OllamaTakeover` does that and restores the
    desktop app when the takeover cannot proceed.
    """
    monitor_host = _loopback(host)
    processes = _iter_ollama_processes()
    # Stop the desktop app first (it may own the daemon), then explicit
    # ``ollama serve`` processes.  Chat clients are left alone: they exit on
    # their own once their daemon disappears.
    targets = [item for item in processes if item.is_desktop_app]
    targets.extend(item for item in processes if item.is_server and item not in targets)

    report = StopReport()
    for target in targets:
        LOG.info("Stopping %s", target.describe())
        if _terminate_process(target):
            report.stopped.append(target)

    # Wait for the port to be released.  Windows can keep a socket in
    # TIME_WAIT for a moment after a hard kill.
    deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < deadline:
        if not _port_open(monitor_host, port):
            break
        time.sleep(0.2)
    if _port_open(monitor_host, port):
        LOG.warning(
            "Port %s is still in use after stopping %s",
            port,
            report.summary,
        )
    return report


# ---------------------------------------------------------------------------
# Managed daemon lifecycle
# ---------------------------------------------------------------------------


def _loopback(host: str) -> str:
    value = (host or "").strip()
    if value in {"", "0.0.0.0", "::", "[::]", "*"}:
        return "127.0.0.1"
    return value


def _port_open(host: str, port: int, *, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((_loopback(host), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _ollama_version(url: str, *, timeout: float = 1.5) -> str | None:
    """Return the Ollama version string if ``url`` answers ``/api/version``."""
    try:
        import httpx  # noqa: PLC0415 - optional at import time, always installed with the proxy
    except ImportError:  # pragma: no cover - declared dependency
        return None
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/api/version", timeout=timeout, trust_env=False
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - any failure means "not answering yet"
        return None
    if isinstance(payload, dict):
        version = payload.get("version")
        if version:
            return str(version)
    return None


def _amnesia_already_running(host: str, port: int) -> bool:
    """True when something that looks like an Amnesia proxy owns the port."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return False
    try:
        response = httpx.get(
            f"http://{_loopback(host)}:{int(port)}/amnesia/health",
            timeout=1.5,
            trust_env=False,
        )
    except Exception:  # noqa: BLE001
        return False
    return response.status_code == 200


def _ollama_binary() -> str:
    """Locate the ``ollama`` executable."""
    candidates: list[str | None] = [shutil.which("ollama")]
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(str(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"))
    elif sys.platform == "darwin":
        candidates.extend(
            [
                "/usr/local/bin/ollama",
                "/opt/homebrew/bin/ollama",
                "/Applications/Ollama.app/Contents/Resources/ollama",
            ]
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise TakeoverError(
        "Could not find the 'ollama' executable. Install Ollama from "
        "https://ollama.com/download and try again."
    )


def _desktop_app_binary() -> str | None:
    """Locate the Ollama desktop/tray app, if this platform has one."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama app.exe"
            if candidate.is_file():
                return str(candidate)
        return None
    if sys.platform == "darwin":
        candidate = Path("/Applications/Ollama.app")
        if candidate.exists():
            return str(candidate)
    return None


def _tail(path: Path | None, lines: int = 12) -> str:
    if path is None:
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _read_pid(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return int(value) if value else None
    except (OSError, ValueError):
        return None


def _write_pid(path: Path, pid: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(int(pid)), encoding="ascii")
    except OSError:  # pragma: no cover - best effort bookkeeping
        LOG.debug("Could not write takeover pid file %s", path, exc_info=True)


def _clear_pid(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _terminate_stale_managed_daemon(pid: int, *, port: int, timeout: float) -> bool:
    """Kill a daemon left behind by a crashed/takeover-killed previous run.

    Only a process that is demonstrably an Ollama server is touched, so a
    recycled PID can never make this kill an unrelated program.
    """
    try:
        import psutil  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return False
    try:
        handle = psutil.Process(pid)
        name = (handle.name() or "").lower()
        cmdline = [str(part).lower() for part in handle.cmdline()]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False
    haystack = " ".join([name, *cmdline])
    if "ollama" not in haystack or "serve" not in haystack:
        LOG.debug("PID %s in the stale takeover file is not an Ollama server; ignoring", pid)
        return False
    LOG.info("Stopping stale managed Ollama daemon from a previous run (pid=%s)", pid)
    try:
        handle.terminate()
        try:
            handle.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            handle.kill()
            handle.wait(timeout=5.0)
    except (psutil.NoSuchProcess, OSError):
        pass
    except psutil.AccessDenied:  # pragma: no cover - unusual Windows state
        LOG.warning("Access denied stopping stale Ollama daemon pid=%s", pid)
        return False
    deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < deadline:
        if not _port_open("127.0.0.1", port):
            break
        time.sleep(0.2)
    return True


def _spawn_daemon(
    binary: str,
    *,
    host: str,
    port: int,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    """Start ``ollama serve`` bound to a private loopback port."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{_loopback(host)}:{int(port)}"
    creationflags = 0
    start_new_session = False
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:  # pragma: no cover - POSIX
        start_new_session = True
    with open(log_path, "ab", buffering=0) as log_file:
        process = subprocess.Popen(
            [binary, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    LOG.info("Started managed Ollama daemon pid=%s on %s:%s", process.pid, host, port)
    return process


def _terminate_daemon(process: subprocess.Popen[bytes] | None, *, timeout: float = 15.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:  # pragma: no cover - already gone
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn daemon
        LOG.warning("Managed Ollama daemon did not stop in %.0fs; killing it", timeout)
        try:
            process.kill()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            LOG.warning("Could not kill the managed Ollama daemon")


def _restart_desktop_app() -> bool:
    """Relaunch the Ollama desktop app after Amnesia stopped it."""
    binary = _desktop_app_binary()
    if not binary:
        LOG.warning("Ollama desktop app binary not found; start Ollama manually if needed.")
        return False
    stdout: Any = subprocess.DEVNULL
    stderr: Any = subprocess.DEVNULL
    creationflags = 0
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    try:
        subprocess.Popen(
            [binary],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError as exc:  # pragma: no cover
        LOG.warning("Could not restart the Ollama desktop app: %s", exc)
        return False
    LOG.info("Restarted the Ollama desktop app")
    return True


def _wait_until_healthy(
    url: str,
    *,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
    log_path: Path | None = None,
) -> str:
    """Wait until ``url`` answers ``/api/version``; return the version."""
    deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise TakeoverError(
                "The managed Ollama daemon exited before it became ready "
                f"(exit code {process.returncode}).\n{_tail(log_path)}".strip()
            )
        version = _ollama_version(url)
        if version:
            return version
        time.sleep(HEALTH_INTERVAL)
    detail = _tail(log_path)
    suffix = f"\nLast daemon output:\n{detail}" if detail else ""
    raise TakeoverError(
        f"Timed out after {timeout:.0f}s waiting for Ollama at {url}.{suffix}"
    )


# ---------------------------------------------------------------------------
# The takeover context manager
# ---------------------------------------------------------------------------


class OllamaTakeover:
    """Swap the standard Ollama port for Amnesia, with a managed daemon.

    Typical use is through the CLI (``amnesia serve --takeover``)::

        takeover = OllamaTakeover()
        takeover.start()
        try:
            ...  # Amnesia listens on 127.0.0.1:11434 here
        finally:
            takeover.stop()

    Parameters
    ----------
    host:
        Interface Amnesia will bind (the old Ollama port).  Wildcards are
        normalized to loopback for health checks.
    port:
        Port Amnesia binds; defaults to Ollama's standard ``11434``.
    upstream_port:
        Private port for the managed Ollama daemon; defaults to ``11435``.
    upstream_url:
        When given, no daemon is spawned and this URL is used as the upstream
        (e.g. a remote Ollama).  Port 11434 must still be free.
    stop_existing:
        Stop an Ollama app/daemon that owns the target port (default true).
    restart_desktop:
        Relaunch the Ollama desktop app when it was stopped (default true).
    log_dir:
        Where the managed daemon log is written (default ``~/.amnesia/logs``).
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_OLLAMA_PORT,
        upstream_port: int = DEFAULT_UPSTREAM_PORT,
        upstream_url: str | None = None,
        stop_existing: bool = True,
        restart_desktop: bool = True,
        log_dir: str | Path | None = None,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT,
    ) -> None:
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.upstream_port = int(upstream_port)
        self._external_upstream = (upstream_url or "").strip().rstrip("/") or None
        self.stop_existing = bool(stop_existing)
        self.restart_desktop = bool(restart_desktop)
        self.start_timeout = float(start_timeout)
        self.stop_timeout = float(stop_timeout)
        if not (1 <= self.port <= 65535):
            raise TakeoverError(f"Invalid takeover port {port}; expected 1-65535.")
        if not (1 <= self.upstream_port <= 65535):
            raise TakeoverError(f"Invalid upstream port {upstream_port}; expected 1-65535.")
        if self._external_upstream is None and self.upstream_port == self.port:
            raise TakeoverError(
                "The upstream port must differ from the port Amnesia binds "
                f"(both were {self.port})."
            )

        base_dir = Path(log_dir).expanduser() if log_dir else Path.home() / ".amnesia" / "logs"
        self.log_path = base_dir / "ollama-takeover.log"
        self.pid_path = base_dir / "ollama-takeover.pid"

        self.process: subprocess.Popen[bytes] | None = None
        self.version: str | None = None
        self.stopped: StopReport | None = None
        self.desktop_restarted = False
        self._desktop_restore_pending = False
        self._started = False
        self._atexit_registered = False

    # -- public properties -------------------------------------------------
    @property
    def upstream_url(self) -> str:
        if self._external_upstream:
            return self._external_upstream
        return f"http://{_loopback(self.host)}:{self.upstream_port}"

    @property
    def listen_url(self) -> str:
        return f"http://{_loopback(self.host)}:{self.port}"

    @property
    def managed(self) -> bool:
        return self._external_upstream is None

    @property
    def takeover_active(self) -> bool:
        return self._started

    def start(self) -> "OllamaTakeover":
        """Prepare the port swap; raises :class:`TakeoverError` on failure."""
        if self._started:
            return self
        try:
            self._acquire_port()
            self._start_upstream()
        except BaseException:
            self._cleanup_after_failure()
            raise
        self._started = True
        self._register_atexit()
        return self

    def stop(self) -> None:
        """Stop the managed daemon and restore the desktop app (best effort)."""
        self._started = False
        _terminate_daemon(self.process, timeout=self.stop_timeout)
        self.process = None
        _clear_pid(self.pid_path)
        if self._desktop_restore_pending:
            self._desktop_restore_pending = False
            if self.restart_desktop:
                self.desktop_restarted = _restart_desktop_app()
        if self._atexit_registered:
            atexit.unregister(self.stop)
            self._atexit_registered = False

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "OllamaTakeover":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------
    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def _acquire_port(self) -> None:
        monitor_host = _loopback(self.host)
        if not _port_open(monitor_host, self.port):
            return
        if _amnesia_already_running(monitor_host, self.port):
            raise TakeoverError(
                f"Port {self.port} is already serving an Amnesia proxy. Stop the other "
                "Amnesia instance first, or let this one use a different --port."
            )
        version = _ollama_version(f"http://{monitor_host}:{self.port}")
        if version is None:
            owner = _port_owner_pid(self.port)
            detail = f" (pid {owner})" if owner else ""
            raise TakeoverError(
                f"Port {self.port} is already in use by another program{detail}. "
                "Stop it first, or pick another port with --port."
            )
        if not self.stop_existing:
            raise TakeoverError(
                f"Ollama {version} is already running on port {self.port}. Quit it "
                "first (tray icon > Quit Ollama), or let takeover stop it by removing "
                "the --keep-ollama flag."
            )
        LOG.info("Ollama %s is running on port %s; stopping it", version, self.port)
        self.stopped = stop_existing_ollama(
            host=monitor_host, port=self.port, timeout=self.stop_timeout
        )
        self._desktop_restore_pending = bool(self.stopped.desktop_stopped and self.restart_desktop)
        LOG.info("Stopped: %s", self.stopped.summary)
        if _port_open(monitor_host, self.port):
            owner = _port_owner_pid(self.port)
            detail = f" (pid {owner})" if owner else ""
            raise TakeoverError(
                f"Port {self.port} is still in use after stopping Ollama{detail}. "
                "Quit the Ollama app from the system tray, then run the command again."
            )

    def _start_upstream(self) -> None:
        if not self.managed:
            version = _ollama_version(self.upstream_url)
            if version is None:
                raise TakeoverError(
                    f"The configured upstream Ollama at {self.upstream_url} is not "
                    "answering /api/version. Start it first."
                )
            self.version = version
            LOG.info("Using existing Ollama %s at %s", version, self.upstream_url)
            return

        upstream_host = _loopback(self.host)
        self._recover_stale_daemon()
        if _port_open(upstream_host, self.upstream_port):
            raise TakeoverError(
                f"Upstream port {self.upstream_port} is already in use. Start Amnesia "
                f"with --upstream-port <free-port> (e.g. --upstream-port 11436)."
            )
        binary = _ollama_binary()
        self.process = _spawn_daemon(
            binary, host=upstream_host, port=self.upstream_port, log_path=self.log_path
        )
        _write_pid(self.pid_path, self.process.pid)
        try:
            self.version = _wait_until_healthy(
                self.upstream_url,
                timeout=self.start_timeout,
                process=self.process,
                log_path=self.log_path,
            )
        except TakeoverError:
            _terminate_daemon(self.process, timeout=5.0)
            self.process = None
            _clear_pid(self.pid_path)
            raise
        LOG.info(
            "Managed Ollama %s is ready at %s (log: %s)",
            self.version,
            self.upstream_url,
            self.log_path,
        )

    def _cleanup_after_failure(self) -> None:
        _terminate_daemon(self.process, timeout=5.0)
        self.process = None
        _clear_pid(self.pid_path)
        if self._desktop_restore_pending:
            self._desktop_restore_pending = False
            if self.restart_desktop:
                _restart_desktop_app()
        self._started = False
        if self._atexit_registered:
            atexit.unregister(self.stop)
            self._atexit_registered = False

    def _recover_stale_daemon(self) -> None:
        """Remove a managed daemon left behind by a previous crashed run."""
        pid = _read_pid(self.pid_path)
        if pid is None:
            return
        _terminate_stale_managed_daemon(
            pid, port=self.upstream_port, timeout=self.stop_timeout
        )
        _clear_pid(self.pid_path)


__all__ = [
    "DEFAULT_OLLAMA_PORT",
    "DEFAULT_STOP_TIMEOUT",
    "DEFAULT_UPSTREAM_PORT",
    "OllamaProcess",
    "OllamaTakeover",
    "StopReport",
    "TakeoverError",
    "stop_existing_ollama",
]
