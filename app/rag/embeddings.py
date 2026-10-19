"""Embeddings asíncronos: OpenAI en producción, hashing determinístico para demo y tests."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.config import Settings
from app.rag.text import tokenize


class AsyncEmbedder(Protocol):
    """Lo único que el RAG necesita de un modelo de embeddings."""

    @property
    def dimension(self) -> int: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Bolsa de palabras proyectada con hashing y normalizada (norma 1).

    No entiende sinónimos, pero es determinística, gratuita y sin red: permite levantar el
    sistema completo y testear la recuperación sin API keys.
    """

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    """Adaptador sobre OpenAIEmbeddings, que ya usa el cliente async oficial de OpenAI."""

    _DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, settings: Settings) -> None:
        if settings.embedding_model not in self._DIMENSIONS:
            raise ValueError(f"EMBEDDING_MODEL no soportado: {settings.embedding_model}")
        self._dimension = self._DIMENSIONS[settings.embedding_model]
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.openai_api_key),
            max_retries=settings.llm_max_retries,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(text)


def create_embedder(settings: Settings) -> AsyncEmbedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return HashEmbedder()
