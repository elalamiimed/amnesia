"""Tests for the Amnesia MCP server tools.

The MCP protocol handshake is not exercised: the exported async tool functions
are called directly against the shared fixture service, and the FastMCP server
is only checked for correct tool registration. No network, no sleeps.
"""

from __future__ import annotations

import json

import pytest

from amnesia import mcp_server
from amnesia.mcp_server import (
    TOOL_NAMES,
    build_server,
    get_service,
    memory_log_event,
    memory_profile,
    memory_remember,
    memory_search,
    memory_stats,
    memory_timeline,
    set_service,
)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def test_tool_names_are_the_documented_contract() -> None:
    assert TOOL_NAMES == (
        "memory_search",
        "memory_remember",
        "memory_profile",
        "memory_log_event",
        "memory_timeline",
        "memory_stats",
    )


@pytest.mark.asyncio
async def test_build_server_registers_all_tools(service) -> None:
    server = build_server(service)
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == set(TOOL_NAMES)


# ---------------------------------------------------------------------------
# memory_remember / memory_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_and_search_roundtrip(service) -> None:
    saved = await memory_remember(
        "Prefers dark mode and Vim keybindings",
        kind="preference",
        key="editor",
        importance=0.9,
        service=service,
    )
    memory = saved["memory"]
    assert memory["id"] > 0
    assert memory["kind"] == "preference"
    assert memory["key"] == "editor"
    assert memory["content"] == "Prefers dark mode and Vim keybindings"
    assert memory["importance"] == pytest.approx(0.9)
    json.dumps(saved)  # must be JSON-serializable

    result = await memory_search("dark mode", service=service)
    assert result["query"] == "dark mode"
    assert result["count"] >= 1
    assert any(hit["id"] == memory["id"] for hit in result["hits"])
    json.dumps(result)


@pytest.mark.asyncio
async def test_remember_key_upserts(service) -> None:
    first = await memory_remember("Uses VS Code", key="editor", service=service)
    second = await memory_remember("Uses Vim keybindings", key="editor", service=service)
    assert first["memory"]["id"] == second["memory"]["id"]
    assert second["memory"]["content"] == "Uses Vim keybindings"


@pytest.mark.asyncio
async def test_remember_clamps_importance(service) -> None:
    saved = await memory_remember("Extremely important fact", importance=5.0, service=service)
    assert saved["memory"]["importance"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_remember_validates_input(service) -> None:
    with pytest.raises(ValueError):
        await memory_remember("   ", service=service)
    with pytest.raises(ValueError):
        await memory_remember("valid text", kind="banana", service=service)


@pytest.mark.asyncio
async def test_search_validates_query(service) -> None:
    with pytest.raises(ValueError):
        await memory_search("   ", service=service)


@pytest.mark.asyncio
async def test_search_limit_is_clamped(service) -> None:
    for index in range(5):
        await memory_remember(f"Fact number {index} about testing", service=service)
    result = await memory_search("testing", limit=1000, service=service)
    assert result["count"] <= 100


# ---------------------------------------------------------------------------
# memory_profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_returns_identity_facts(service) -> None:
    await memory_remember("User is a software engineer", kind="identity", service=service)
    await memory_remember("Prefers concise answers", kind="preference", service=service)
    await memory_remember("Short-lived episode", kind="episode", service=service)

    payload = await memory_profile(service=service)
    assert payload["count"] == 2
    kinds = {fact["kind"] for fact in payload["facts"]}
    assert kinds == {"identity", "preference"}
    for fact in payload["facts"]:
        assert {"id", "kind", "content", "created_at"} <= set(fact)


# ---------------------------------------------------------------------------
# memory_log_event / memory_timeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_event_and_timeline(service) -> None:
    saved = await memory_log_event(
        "Wrote the MCP tests",
        kind="task",
        details="Covered all six tools",
        project="amnesia",
        service=service,
    )
    event = saved["event"]
    assert event["kind"] == "task"
    assert event["project"] == "amnesia"
    assert event["details"] == "Covered all six tools"
    json.dumps(saved)

    timeline = await memory_timeline(days=1, service=service)
    assert timeline["days"] == 1
    assert timeline["count"] == 1
    assert timeline["events"][0]["id"] == event["id"]
    json.dumps(timeline)


@pytest.mark.asyncio
async def test_log_event_validates_title(service) -> None:
    with pytest.raises(ValueError):
        await memory_log_event("   ", service=service)


@pytest.mark.asyncio
async def test_search_include_events_toggle(service) -> None:
    await memory_log_event(
        "Deployed the proxy", kind="project", project="amnesia", service=service
    )
    with_events = await memory_search("deployed proxy", include_events=True, service=service)
    without_events = await memory_search(
        "deployed proxy", include_events=False, service=service
    )
    assert any(hit["kind"] == "event" for hit in with_events["hits"])
    assert all(hit["kind"] != "event" for hit in without_events["hits"])


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_payload(service) -> None:
    await memory_remember("A remembered fact", service=service)
    await memory_log_event("An activity event", service=service)

    payload = await memory_stats(service=service)
    assert payload["memories"] >= 1
    assert payload["events"] >= 1
    assert payload["db_size_human"]
    assert payload["db_path"].endswith("amnesia.db")
    assert payload["ollama_available"] is True
    assert payload["memory_enabled"] is True
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Service plumbing / server dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_service_roundtrip(service) -> None:
    set_service(service)
    try:
        assert get_service() is service
    finally:
        set_service(None)
    assert get_service() is None


@pytest.mark.asyncio
async def test_call_tool_through_fastmcp(service) -> None:
    server = build_server(service)
    result = await server.call_tool(
        "memory_remember",
        {"content": "Called through the MCP dispatch layer", "kind": "fact"},
    )
    rendered = json.dumps(result, default=str)
    assert "Called through the MCP dispatch layer" in rendered

    found = await memory_search("dispatch layer", service=service)
    assert found["count"] >= 1


def test_main_without_mcp_package(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(mcp_server, "MCP_AVAILABLE", False)
    assert mcp_server.main() == 1
    captured = capsys.readouterr()
    assert "mcp" in captured.err.lower()


def test_build_server_without_mcp_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "MCP_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        build_server(None)
