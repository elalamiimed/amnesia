"""Configuration for Amnesia.

Precedence (lowest to highest):
    built-in defaults  ->  ~/.amnesia/config.json  ->  environment  ->  explicit overrides

Everything can be controlled from the environment, which keeps the CLI, the
proxy and tests easy to configure:

    AMNESIA_HOME              default ~/.amnesia
    AMNESIA_DB                full path to the SQLite file
    AMNESIA_OLLAMA_URL        default http://127.0.0.1:11434
    AMNESIA_MODEL             default chat model
    AMNESIA_EMBED_MODEL       default nomic-embed-text
    AMNESIA_MEMORY_ENABLED    remembered/injected on/off
    AMNESIA_INJECT_CONTEXT    inject memory into model prompts
    AMNESIA_EXTRACTION        automatic fact/activity extraction on/off
    AMNESIA_EXTRACTION_MODEL  model used for distillation (default chat model)
    AMNESIA_EXTRACTION_EVERY  turns between extractions
    AMNESIA_CONTEXT_WINDOW    token window used for budget planning
    AMNESIA_MEMORY_BUDGET     tokens reserved for memory context
    AMNESIA_PROXY_HOST        default 127.0.0.1
    AMNESIA_PROXY_PORT        default 11435
    AMNESIA_API_TOKEN         optional bearer token for the HTTP API
    AMNESIA_REDACT            redact secrets before storing (default on)
    AMNESIA_SESSION_GAP       minutes before a new session starts automatically
    AMNESIA_LOG_LEVEL         default INFO
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

LOG = logging.getLogger("amnesia.config")

DEFAULT_DB_NAME = "amnesia.db"
DEFAULT_CONFIG_NAME = "config.json"

EMBED_MODEL_HINTS = (
    "embed",
    "embedding",
    "bge",
    "e5-",
    "gte",
    "minilm",
    "nomic",
    "mxbai",
)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("Ignoring invalid integer for %s=%r", name, raw)
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        LOG.warning("Ignoring invalid number for %s=%r", name, raw)
        return default


def normalize_ollama_url(raw: str | None) -> str:
    """Normalize OLLAMA_HOST-style values into an HTTP base URL.

    Accepts: '11434', '0.0.0.0:11434', '127.0.0.1:11434', 'http://host:port',
    'https://host'. Wildcard binds are rewritten to loopback for clients.
    """
    value = (raw or "").strip()
    if not value:
        return "http://127.0.0.1:11434"
    if "://" not in value:
        value = "http://" + value
    scheme, _, rest = value.partition("://")
    rest = rest.rstrip("/")
    host, _, port = rest.partition(":")
    if host in {"0.0.0.0", "::", "[::]", "*"}:
        host = "127.0.0.1"
    if not port:
        port = "11434"
    return f"{scheme}://{host}:{port}"


def is_embedding_model_name(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in EMBED_MODEL_HINTS)


@dataclass
class Settings:
    """Resolved Amnesia settings. Create with :func:`load_settings`."""

    home: Path = field(default_factory=lambda: Path.home() / ".amnesia")
    db_path: Path | None = None
    ollama_url: str = "http://127.0.0.1:11434"

    # Proxy / HTTP API
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 11435
    api_token: str | None = None
    request_timeout: float = 600.0

    # Models
    chat_model: str | None = None
    embed_model: str = "nomic-embed-text"
    extraction_model: str | None = None
    context_window: int = 8192

    # Memory behaviour
    memory_enabled: bool = True
    inject_context: bool = True
    extraction_enabled: bool = True
    extraction_every_n_turns: int = 4
    extraction_min_chars: int = 40
    summarization_enabled: bool = True
    redact_secrets: bool = True

    # Budgets (tokens)
    memory_token_budget: int = 1200
    profile_token_budget: int = 350
    recent_turns: int = 6

    # Sessions
    session_gap_minutes: int = 45

    # UX / logging
    log_level: str = "INFO"
    timezone: str = ""  # empty = system local time

    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def database(self) -> Path:
        return Path(self.db_path) if self.db_path else Path(self.home) / DEFAULT_DB_NAME

    @property
    def config_file(self) -> Path:
        return Path(self.home) / DEFAULT_CONFIG_NAME

    @property
    def effective_chat_model(self) -> str | None:
        return self.chat_model

    @property
    def effective_extraction_model(self) -> str | None:
        return self.extraction_model or self.chat_model

    # -- IO --------------------------------------------------------------
    def ensure_home(self) -> Path:
        home = Path(self.home)
        home.mkdir(parents=True, exist_ok=True)
        return home

    def to_dict(self, *, redact_token: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["home"] = str(self.home)
        data["db_path"] = str(self.database)
        if redact_token and data.get("api_token"):
            data["api_token"] = "***"
        return data

    def save(self) -> Path:
        """Persist the current settings (minus secrets) to config.json."""
        self.ensure_home()
        payload = self.to_dict(redact_token=True)
        path = Path(self.home) / DEFAULT_CONFIG_NAME
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def clone(self, **overrides: Any) -> "Settings":
        data = asdict(self)
        data.update(overrides)
        return Settings(**data)


_SETTING_NAMES = {f.name for f in fields(Settings)}


def _read_config_file(home: Path) -> dict[str, Any]:
    path = home / DEFAULT_CONFIG_NAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOG.warning("Could not read %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    # Drop unknown keys silently to stay forward compatible.
    return {key: value for key, value in raw.items() if key in _SETTING_NAMES}


def _load_dotenv_if_available(home: Path) -> None:
    """Load a .env file if python-dotenv is installed (optional dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is declared but optional at runtime
        return
    candidates = [Path.cwd() / ".env", home / ".env"]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    mapping = {
        "AMNESIA_OLLAMA_URL": ("ollama_url", str),
        "AMNESIA_PROXY_HOST": ("proxy_host", str),
        "AMNESIA_API_TOKEN": ("api_token", str),
        "AMNESIA_MODEL": ("chat_model", str),
        "AMNESIA_EMBED_MODEL": ("embed_model", str),
        "AMNESIA_EXTRACTION_MODEL": ("extraction_model", str),
        "AMNESIA_LOG_LEVEL": ("log_level", str),
        "AMNESIA_TIMEZONE": ("timezone", str),
        "AMNESIA_DB": ("db_path", str),
        "AMNESIA_CONTEXT_WINDOW": ("context_window", int),
        "AMNESIA_MEMORY_BUDGET": ("memory_token_budget", int),
        "AMNESIA_PROFILE_BUDGET": ("profile_token_budget", int),
        "AMNESIA_RECENT_TURNS": ("recent_turns", int),
        "AMNESIA_EXTRACTION_EVERY": ("extraction_every_n_turns", int),
        "AMNESIA_EXTRACTION_MIN_CHARS": ("extraction_min_chars", int),
        "AMNESIA_SESSION_GAP": ("session_gap_minutes", int),
        "AMNESIA_REQUEST_TIMEOUT": ("request_timeout", float),
    }
    for env_name, (attr, caster) in mapping.items():
        raw = _env(env_name)
        if raw is None:
            continue
        if attr == "ollama_url":
            overrides[attr] = normalize_ollama_url(raw)
        elif attr == "db_path":
            overrides[attr] = Path(raw).expanduser()
        elif caster is int:
            try:
                overrides[attr] = int(raw)
            except ValueError:
                LOG.warning("Ignoring invalid integer for %s=%r", env_name, raw)
        elif caster is float:
            try:
                overrides[attr] = float(raw)
            except ValueError:
                LOG.warning("Ignoring invalid number for %s=%r", env_name, raw)
        else:
            overrides[attr] = raw

    bool_mapping = {
        "AMNESIA_MEMORY_ENABLED": "memory_enabled",
        "AMNESIA_INJECT_CONTEXT": "inject_context",
        "AMNESIA_EXTRACTION": "extraction_enabled",
        "AMNESIA_SUMMARIZATION": "summarization_enabled",
        "AMNESIA_REDACT": "redact_secrets",
    }
    for env_name, attr in bool_mapping.items():
        raw = _env(env_name)
        if raw is not None:
            overrides[attr] = raw.lower() not in {"0", "false", "no", "off", "disabled"}
    return overrides


def load_settings(
    home: str | Path | None = None,
    *,
    config_file: str | Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build a Settings object from defaults, config file, env and overrides."""
    resolved_home = Path(
        home or _env("AMNESIA_HOME") or (Path.home() / ".amnesia")
    ).expanduser()
    _load_dotenv_if_available(resolved_home)

    if config_file is not None:
        cfg_path = Path(config_file).expanduser()
        data = _read_config_file(cfg_path.parent) if cfg_path.exists() else {}
        if cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = {k: v for k, v in raw.items() if k in _SETTING_NAMES}
            except (OSError, ValueError):
                pass
    else:
        data = _read_config_file(resolved_home)

    settings = Settings(home=resolved_home, **{k: v for k, v in data.items() if k != "home"})
    if "db_path" in data and data["db_path"]:
        settings.db_path = Path(str(data["db_path"])).expanduser()

    env_data = _env_overrides()
    for key, value in env_data.items():
        setattr(settings, key, value)

    if settings.db_path is not None:
        settings.db_path = Path(settings.db_path).expanduser()
    if "ollama_url" in overrides and overrides["ollama_url"]:
        overrides["ollama_url"] = normalize_ollama_url(overrides["ollama_url"])

    for key, value in overrides.items():
        if key not in _SETTING_NAMES:
            raise TypeError(f"Unknown Amnesia setting: {key}")
        if key == "home" and value is not None:
            value = Path(value).expanduser()
        if key == "db_path" and value is not None:
            value = Path(value).expanduser()
        setattr(settings, key, value)

    settings.ollama_url = normalize_ollama_url(settings.ollama_url)
    return settings


def configure_logging(settings: Settings) -> None:
    """Idempotent, non-noisy logging setup for CLI and proxy."""
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    root = logging.getLogger("amnesia")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)
