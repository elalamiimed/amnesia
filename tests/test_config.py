"""Unit tests for configuration loading and precedence."""

from __future__ import annotations

import json

import pytest

from amnesia.config import (
    Settings,
    is_embedding_model_name,
    load_settings,
    normalize_ollama_url,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "http://127.0.0.1:11434"),
        ("", "http://127.0.0.1:11434"),
        ("11434", "http://127.0.0.1:11434"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"),
        ("http://localhost:1234/", "http://localhost:1234"),
        ("https://ollama.example:443", "https://ollama.example:443"),
        ("http://0.0.0.0:5555", "http://127.0.0.1:5555"),
    ],
)
def test_normalize_ollama_url(raw, expected):
    assert normalize_ollama_url(raw) == expected


def test_embedding_model_name_detection():
    assert is_embedding_model_name("nomic-embed-text:latest")
    assert is_embedding_model_name("mxbai-embed-large")
    assert is_embedding_model_name("bge-m3")
    assert not is_embedding_model_name("dolphin3:latest")
    assert not is_embedding_model_name("qwen2.5-coder:7b")


def test_default_settings(tmp_path):
    settings = load_settings(home=tmp_path / "home")
    assert settings.home == tmp_path / "home"
    assert settings.database == tmp_path / "home" / "amnesia.db"
    assert settings.ollama_url == "http://127.0.0.1:11434"
    assert settings.proxy_port == 11435
    assert settings.embed_model == "nomic-embed-text"
    assert settings.memory_enabled is True
    assert settings.redact_secrets is True
    assert settings.effective_extraction_model is None


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("AMNESIA_OLLAMA_URL", "0.0.0.0:12345")
    monkeypatch.setenv("AMNESIA_MODEL", "dolphin3:latest")
    monkeypatch.setenv("AMNESIA_EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("AMNESIA_PROXY_PORT", "1234")
    monkeypatch.setenv("AMNESIA_MEMORY_ENABLED", "0")
    monkeypatch.setenv("AMNESIA_REDACT", "false")
    monkeypatch.setenv("AMNESIA_MEMORY_BUDGET", "777")
    monkeypatch.setenv("AMNESIA_DB", str(tmp_path / "custom.db"))
    settings = load_settings(home=tmp_path / "home")
    assert settings.ollama_url == "http://127.0.0.1:12345"
    assert settings.chat_model == "dolphin3:latest"
    assert settings.embed_model == "mxbai-embed-large"
    assert settings.proxy_port == 1234
    assert settings.memory_enabled is False
    assert settings.redact_secrets is False
    assert settings.memory_token_budget == 777
    assert settings.database == tmp_path / "custom.db"


def test_invalid_env_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("AMNESIA_PROXY_PORT", "not-a-number")
    monkeypatch.setenv("AMNESIA_MEMORY_BUDGET", "nope")
    settings = load_settings(home=tmp_path / "home")
    assert settings.proxy_port == 11435
    assert settings.memory_token_budget == 1200


def test_explicit_overrides_beat_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AMNESIA_MODEL", "from-env")
    settings = load_settings(home=tmp_path / "home", chat_model="from-override")
    assert settings.chat_model == "from-override"


def test_unknown_override_raises(tmp_path):
    with pytest.raises(TypeError):
        load_settings(home=tmp_path / "home", not_a_real_setting=1)


def test_config_file_roundtrip(tmp_path):
    home = tmp_path / "home"
    settings = load_settings(home=home)
    settings.chat_model = "dolphin3:latest"
    settings.embed_model = "mxbai-embed-large"
    settings.memory_token_budget = 900
    settings.api_token = "super-secret-token"
    path = settings.save()
    assert path.exists()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["api_token"] == "***"  # never persist secrets

    reloaded = load_settings(home=home)
    assert reloaded.chat_model == "dolphin3:latest"
    assert reloaded.embed_model == "mxbai-embed-large"
    assert reloaded.memory_token_budget == 900
    assert reloaded.api_token is None  # token was not persisted


def test_clone_creates_independent_copy(tmp_path):
    settings = load_settings(home=tmp_path / "home")
    clone = settings.clone(memory_enabled=False, chat_model="custom")
    assert clone.memory_enabled is False
    assert clone.chat_model == "custom"
    assert settings.memory_enabled is True
    assert settings.chat_model is None


def test_settings_to_dict_redacts_token(tmp_path):
    settings = Settings(home=tmp_path / "home", api_token="secret")
    payload = settings.to_dict()
    assert payload["api_token"] == "***"
    assert payload["home"] == str(tmp_path / "home")
    assert payload["db_path"].endswith("amnesia.db")


def test_ensure_home_creates_directory(tmp_path):
    home = tmp_path / "nested" / "amnesia"
    settings = Settings(home=home)
    created = settings.ensure_home()
    assert created == home
    assert home.exists()
