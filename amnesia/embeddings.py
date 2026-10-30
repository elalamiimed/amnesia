"""Embeddings for semantic memory search.

Primary path: Ollama embeddings (default ``nomic-embed-text``, 768 dims).
Fallback: a dependency-free deterministic hashing embedder so Amnesia still
works - keyword-flavoured but entirely offline - when no embedding model is
installed. Embeddings are L2-normalized so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

import httpx

from .config import Settings, normalize_ollama_url

LOG = logging.getLogger("amnesia.embeddings")

DEFAULT_EMBED_MODEL = "nomic-embed-text"
HASH_EMBED_DIM = 512


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        return [0.0] * len(vector)
    return [float(value) / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity; assumes reasonably normalized inputs but normalizes anyway."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        left_norm += fa * fa
        right_norm += fb * fb
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a float vector into a compact float32 BLOB for SQLite."""
    import numpy as np

    return np.asarray(list(vector), dtype=np.float32).tobytes()


def unpack_vector(blob: bytes | memoryview | None) -> list[float]:
    if blob is None:
        return []
    import numpy as np

    return np.frombuffer(bytes(blob), dtype=np.float32).astype(float).tolist()


# ---------------------------------------------------------------------------
# Embedder interface
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """Minimal async embedding interface used across the codebase."""

    name: str = "none"
    dim: int | None = None

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    async def probe(self) -> bool:
        """Verify the backend works and discover the vector dimension."""
        try:
            vector = await self.embed_one("amnesia memory probe")
            if not vector:
                return False
            self.dim = len(vector)
            return True
        except Exception as exc:  # noqa: BLE001 - probe reports availability only
            LOG.debug("Embedder probe failed for %s: %s", self.name, exc)
            return False

    async def close(self) -> None:
        """Release network resources if any."""
        return None

    @property
    def ready(self) -> bool:
        return self.dim is not None


class NullEmbedder(Embedder):
    """Explicitly disabled embeddings (keyword-only retrieval)."""

    name = "none"
    dim = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return []

    async def probe(self) -> bool:
        return True


class HashEmbedder(Embedder):
    """Deterministic offline embedder: signed hashing of unigrams + bigrams.

    This is not semantic in the model sense, but it is stable, free, and gives
    useful fuzzy lexical similarity, which keeps retrieval working on machines
    without an embedding model installed.
    """

    def __init__(self, dim: int = HASH_EMBED_DIM) -> None:
        self.dim = int(dim)
        self.name = f"hash:{self.dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = (text or "").lower()
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = self._tokens(text)
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend((f"{a}_{b}", 0.6) for a, b in zip(tokens, tokens[1:]))
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign * weight
        return normalize_vector(vector)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def probe(self) -> bool:
        return True


class OllamaEmbedder(Embedder):
    """Embeddings served by the local Ollama daemon (``/api/embed``)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = DEFAULT_EMBED_MODEL,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = normalize_ollama_url(base_url)
        self.model = model
        self.name = f"ollama:{model}"
        self.dim = None
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(connect=10.0, read=self._timeout, write=30.0, pool=10.0),
            )
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = [str(text) for text in texts]
        if not payload:
            return []
        response = await self._get_client().post(
            "/api/embed",
            json={"model": self.model, "input": payload, "truncate": True},
        )
        if response.status_code == 404:
            # Older Ollama builds only expose /api/embeddings (one prompt per call).
            return [await self._legacy_embed(text) for text in payload]
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Ollama embed failed ({response.status_code}): {response.text[:200]}"
            )
        data = response.json()
        vectors = data.get("embeddings") or []
        if len(vectors) != len(payload):
            raise EmbeddingError(
                f"Ollama returned {len(vectors)} embeddings for {len(payload)} inputs"
            )
        normalized = [normalize_vector(vector) for vector in vectors]
        if normalized and normalized[0]:
            self.dim = len(normalized[0])
        return normalized

    async def _legacy_embed(self, text: str) -> list[float]:
        response = await self._get_client().post(
            "/api/embeddings", json={"model": self.model, "prompt": text}
        )
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Ollama legacy embed failed ({response.status_code}): {response.text[:200]}"
            )
        vector = response.json().get("embedding") or []
        if not vector:
            raise EmbeddingError("Ollama returned an empty embedding")
        return normalize_vector(vector)

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


class ResilientEmbedder(Embedder):
    """Use a primary embedder and permanently fall back if it stops working.

    Switching is immediate so the ``name``/``dim`` written alongside stored
    vectors always matches the vectors actually produced.
    """

    def __init__(self, primary: Embedder, fallback: Embedder | None = None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active: Embedder = primary
        self.name = primary.name
        self.dim = primary.dim

    @property
    def active(self) -> Embedder:
        return self._active

    async def _adopt_fallback(self, reason: Exception) -> None:
        assert self._fallback is not None
        await self._fallback.probe()
        LOG.warning(
            "Switching embeddings from %s to %s (%s)",
            self._active.name,
            self._fallback.name,
            reason,
        )
        self._active = self._fallback
        self.name = self._fallback.name
        self.dim = self._fallback.dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vectors = await self._active.embed(texts)
            if vectors and vectors[0] and self.dim is None:
                self.dim = len(vectors[0])
            return vectors
        except Exception as exc:  # noqa: BLE001 - fallback path
            if self._fallback is None or self._active is self._fallback:
                raise
            await self._adopt_fallback(exc)
            return await self._active.embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    async def probe(self) -> bool:
        ok = await self._primary.probe()
        if ok:
            self._active = self._primary
            self.dim = self._primary.dim
            self.name = self._primary.name
            return True
        if self._fallback is not None:
            await self._adopt_fallback(EmbeddingError("primary embedder probe failed"))
            return True
        return False

    async def close(self) -> None:
        await self._primary.close()
        if self._fallback is not None:
            await self._fallback.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_embedder(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    allow_fallback: bool = True,
) -> Embedder:
    """Build the best available embedder for the current machine."""
    model = (settings.embed_model or "").strip()
    if model.lower() in {"", "none", "off", "disabled", "0"}:
        LOG.info("Embeddings disabled by configuration")
        return NullEmbedder()

    primary: Embedder = OllamaEmbedder(
        settings.ollama_url,
        model,
        timeout=min(settings.request_timeout, 120.0),
        client=client,
    )
    if await primary.probe():
        LOG.info("Using Ollama embeddings: %s (%s dims)", primary.name, primary.dim)
        return primary

    if not allow_fallback:
        raise EmbeddingError(
            f"Embedding model {model!r} is not available. Run: ollama pull {model}"
        )
    LOG.warning(
        "Embedding model %r unavailable; using deterministic hash embeddings. "
        "Run `ollama pull %s` for semantic memory.",
        model,
        model,
    )
    fallback = HashEmbedder()
    await fallback.probe()
    return fallback
