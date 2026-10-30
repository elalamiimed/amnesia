"""Unit tests for embeddings and vector helpers."""

from __future__ import annotations

import math

import pytest

from amnesia.config import load_settings
from amnesia.embeddings import (
    EmbeddingError,
    HashEmbedder,
    NullEmbedder,
    OllamaEmbedder,
    cosine_similarity,
    create_embedder,
    normalize_vector,
    pack_vector,
    unpack_vector,
)


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=128)
    first = embedder._vector("memory for local models")
    second = embedder._vector("memory for local models")
    different = embedder._vector("something completely unrelated")
    assert first == second
    assert first != different
    assert embedder.dim == 128
    assert _norm(first) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_hash_embedder_batch_and_probe():
    embedder = HashEmbedder()
    vectors = await embedder.embed(["one", "two", "three"])
    assert len(vectors) == 3
    assert len(vectors[0]) == 512
    assert await embedder.probe() is True


def test_normalize_vector_handles_zero():
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    normalized = normalize_vector([3.0, 4.0])
    assert normalized == pytest.approx([0.6, 0.8])


def test_pack_unpack_roundtrip():
    vector = [0.5, -0.25, 0.125]
    blob = pack_vector(vector)
    restored = unpack_vector(blob)
    assert restored == pytest.approx(vector, abs=1e-6)
    assert unpack_vector(None) == []


def test_cosine_similarity_ranges():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_null_embedder():
    embedder = NullEmbedder()
    assert await embedder.embed(["a", "b"]) == [[], []]
    assert await embedder.embed_one("a") == []
    assert await embedder.probe() is True


@pytest.mark.asyncio
async def test_create_embedder_disabled(settings):
    disabled = settings.clone(embed_model="none")
    embedder = await create_embedder(disabled)
    assert isinstance(embedder, NullEmbedder)
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_uses_api(fake_ollama, settings):
    embedder = OllamaEmbedder(
        fake_ollama.base_url, "nomic-embed-text", client=fake_ollama.upstream_client
    )
    vectors = await embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert embedder.dim == fake_ollama.embed_dim
    assert _norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)
    assert fake_ollama.embed_calls[-1] == ["hello", "world"]


@pytest.mark.asyncio
async def test_ollama_embedder_single(fake_ollama):
    embedder = OllamaEmbedder(
        fake_ollama.base_url, "nomic-embed-text", client=fake_ollama.upstream_client
    )
    vector = await embedder.embed_one("single")
    assert len(vector) == fake_ollama.embed_dim
    assert embedder.dim == fake_ollama.embed_dim


@pytest.mark.asyncio
async def test_ollama_embedder_raises_on_failure(fake_ollama):
    fake_ollama.fail_status = 500
    embedder = OllamaEmbedder(
        fake_ollama.base_url, "nomic-embed-text", client=fake_ollama.upstream_client
    )
    with pytest.raises(EmbeddingError):
        await embedder.embed(["will fail"])


@pytest.mark.asyncio
async def test_ollama_embedder_probe_failure_returns_false(fake_ollama):
    fake_ollama.fail_status = 500
    embedder = OllamaEmbedder(
        fake_ollama.base_url, "nomic-embed-text", client=fake_ollama.upstream_client
    )
    assert await embedder.probe() is False


class _BrokenEmbedder(NullEmbedder):
    """Primary backend that always fails, for resilience tests."""

    name = "broken"
    dim = None

    async def embed(self, texts):
        raise RuntimeError("primary backend down")

    async def probe(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_resilient_embedder_falls_back_on_probe_failure():
    from amnesia.embeddings import ResilientEmbedder

    fallback = HashEmbedder()
    resilient = ResilientEmbedder(_BrokenEmbedder(), fallback)
    assert await resilient.probe() is True
    assert resilient.active is fallback
    assert resilient.name == fallback.name
    vectors = await resilient.embed(["hello"])
    assert len(vectors[0]) == fallback.dim


@pytest.mark.asyncio
async def test_resilient_embedder_switches_when_primary_dies():
    from amnesia.embeddings import ResilientEmbedder

    class _FlakyEmbedder(HashEmbedder):
        name = "flaky:512"

        async def embed(self, texts):
            raise RuntimeError("vector backend crashed")

    fallback = HashEmbedder()
    resilient = ResilientEmbedder(_FlakyEmbedder(), fallback)
    vectors = await resilient.embed(["resilience"])
    assert len(vectors[0]) == fallback.dim
    assert resilient.active is fallback
