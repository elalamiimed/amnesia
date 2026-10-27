"""Unit tests for the async Ollama HTTP client (against the fake daemon)."""

from __future__ import annotations

import pytest

from amnesia.ollama import OllamaClient, OllamaError, normalize_messages


@pytest.mark.asyncio
async def test_version_and_tags(fake_ollama):
    client = fake_ollama.ollama_client
    assert await client.version() == fake_ollama.version
    models = await client.tags()
    assert len(models) == len(fake_ollama.models)
    names = await client.model_names()
    assert "dolphin3:latest" in names
    assert await client.is_available() is True


@pytest.mark.asyncio
async def test_resolve_chat_model(fake_ollama):
    client = fake_ollama.ollama_client
    assert (
        await client.resolve_chat_model("R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m")
        == "R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m"
    )
    # Unknown preferred model -> first non-embedding model.
    assert await client.resolve_chat_model("does-not-exist") == "dolphin3:latest"
    assert await client.resolve_chat_model(None) == "dolphin3:latest"


@pytest.mark.asyncio
async def test_resolve_chat_model_without_chat_models(fake_ollama):
    fake_ollama.models = ["nomic-embed-text:latest"]
    client = fake_ollama.ollama_client
    with pytest.raises(OllamaError):
        await client.resolve_chat_model(None)


@pytest.mark.asyncio
async def test_chat_and_chat_text(fake_ollama):
    fake_ollama.queue_reply("Hello from the model")
    client = fake_ollama.ollama_client
    data = await client.chat("dolphin3:latest", [{"role": "user", "content": "hi"}])
    assert data["message"]["content"] == "Hello from the model"

    fake_ollama.queue_reply("Second reply")
    text = await client.chat_text("dolphin3:latest", [{"role": "user", "content": "hi"}])
    assert text == "Second reply"


@pytest.mark.asyncio
async def test_chat_json_parses_model_output(fake_ollama):
    fake_ollama.queue_reply('```json\n{"facts": [{"content": "likes python"}], "events": []}\n```')
    client = fake_ollama.ollama_client
    parsed = await client.chat_json(
        "dolphin3:latest", [{"role": "user", "content": "extract"}]
    )
    assert parsed["facts"][0]["content"] == "likes python"


@pytest.mark.asyncio
async def test_chat_json_rejects_invalid_output(fake_ollama):
    fake_ollama.queue_reply("this is not json at all")
    client = fake_ollama.ollama_client
    with pytest.raises(OllamaError):
        await client.chat_json("dolphin3:latest", [{"role": "user", "content": "extract"}])


@pytest.mark.asyncio
async def test_chat_stream_yields_chunks(fake_ollama):
    fake_ollama.queue_reply("streaming reply words")
    fake_ollama.stream_split_words = True
    client = fake_ollama.ollama_client
    chunks = []
    async for chunk in client.chat_stream(
        "dolphin3:latest", [{"role": "user", "content": "hi"}]
    ):
        chunks.append(chunk)
    assert chunks
    assert chunks[-1]["done"] is True
    content = "".join(chunk.get("message", {}).get("content", "") for chunk in chunks)
    assert "streaming reply words" in content


@pytest.mark.asyncio
async def test_embeddings(fake_ollama):
    client = fake_ollama.ollama_client
    vectors = await client.embeddings("nomic-embed-text", ["alpha", "beta"])
    assert len(vectors) == 2
    assert len(vectors[0]) == fake_ollama.embed_dim
    assert vectors[0] != vectors[1]


@pytest.mark.asyncio
async def test_generate(fake_ollama):
    fake_ollama.queue_reply("generated text")
    client = fake_ollama.ollama_client
    data = await client.generate("dolphin3:latest", "say something")
    assert data["response"] == "generated text"


@pytest.mark.asyncio
async def test_http_error_raises_with_status(fake_ollama):
    fake_ollama.fail_status = 500
    client = fake_ollama.ollama_client
    with pytest.raises(OllamaError) as error_info:
        await client.chat("dolphin3:latest", [{"role": "user", "content": "hi"}])
    assert error_info.value.status_code == 500


@pytest.mark.asyncio
async def test_connection_error_raises(fake_ollama):
    fake_ollama.fail_connection = True
    client = fake_ollama.ollama_client
    with pytest.raises(OllamaError):
        await client.version()
    assert await client.is_available() is False


def test_normalize_messages_filters_and_preserves():
    messages = [
        {
            "role": "user",
            "content": "look",
            "images": ["base64"],
            "unknown": "drop me",
        },
        {"role": "assistant", "content": "", "tool_calls": [{"function": "x"}]},
    ]
    normalized = normalize_messages(messages)
    assert normalized[0] == {"role": "user", "content": "look", "images": ["base64"]}
    assert "tool_calls" in normalized[1]
    assert "unknown" not in normalized[0]


@pytest.mark.asyncio
async def test_close_non_owned_client_keeps_injected_client(fake_ollama):
    client = fake_ollama.ollama_client
    await client.close()
    # Injected clients are owned by the caller; a close is a no-op.
    assert await client.version() == fake_ollama.version
