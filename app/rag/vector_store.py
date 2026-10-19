"""Bases vectoriales intercambiables: Pinecone (async nativo) o local sobre Redis."""

from __future__ import annotations

import json
import math
from typing import Any, Protocol

from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.async_client.async_index import AsyncIndex
from redis.asyncio import Redis

from app.config import Settings
from app.rag.models import Chunk, SearchFilter

VectorHit = tuple[str, float]


class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def query(
        self, vector: list[float], k: int, search_filter: SearchFilter
    ) -> list[VectorHit]: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def close(self) -> None: ...


class LocalVectorStore:
    """Coseno exacto en memoria, persistido en un hash de Redis.

    Alcanza para miles de chunks y permite correr todo sin cuenta de Pinecone. Guarda la
    metadata mínima para filtrar sin pedirle los chunks al ChunkStore.
    """

    def __init__(self, redis: Redis, prefix: str = "kh:") -> None:
        self.redis = redis
        self._key = f"{prefix}vectors"

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        mapping = {
            c.id: json.dumps({"v": v, "module": c.module, "doc_type": c.doc_type.value})
            for c, v in zip(chunks, vectors, strict=True)
        }
        await self.redis.hset(self._key, mapping=mapping)  # type: ignore[arg-type]

    async def query(
        self, vector: list[float], k: int, search_filter: SearchFilter
    ) -> list[VectorHit]:
        raw = await self.redis.hgetall(self._key)
        modules = set(search_filter.modules or [])
        doc_types = {t.value for t in search_filter.doc_types or []}
        hits: list[VectorHit] = []
        for chunk_id, payload in raw.items():
            item = json.loads(payload)
            if modules and item["module"] not in modules:
                continue
            if doc_types and item["doc_type"] not in doc_types:
                continue
            hits.append((str(chunk_id), _cosine(vector, item["v"])))
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await self.redis.hdel(self._key, *ids)

    async def close(self) -> None:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class PineconeVectorStore:
    """Cliente async oficial (PineconeAsyncio): ninguna llamada bloquea el event loop."""

    def __init__(self, client: PineconeAsyncio, index: AsyncIndex, namespace: str) -> None:
        self._client = client
        self._index = index
        self.namespace = namespace

    @classmethod
    async def connect(cls, settings: Settings) -> PineconeVectorStore:
        client = PineconeAsyncio(api_key=settings.pinecone_api_key)
        try:
            description = await client.describe_index(settings.pinecone_index)
        except Exception:
            await client.close()
            raise
        # El host sale de la API, nunca hardcodeado: cambia si el índice se recrea.
        if not description.host:
            await client.close()
            raise RuntimeError(f"El índice {settings.pinecone_index} todavía no tiene host")
        index = client.IndexAsyncio(host=description.host)
        return cls(client, index, settings.pinecone_namespace)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        records: list[dict[str, Any]] = [
            {"id": c.id, "values": v, "metadata": c.vector_metadata()}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # 100 por request: rango recomendado por Pinecone y lejos del límite de 2 MB.
        await self._index.upsert(
            vectors=records, namespace=self.namespace, batch_size=100, show_progress=False
        )

    async def query(
        self, vector: list[float], k: int, search_filter: SearchFilter
    ) -> list[VectorHit]:
        response = await self._index.query(
            vector=vector,
            top_k=k,
            namespace=self.namespace,
            filter=search_filter.to_pinecone(),
            include_metadata=False,
        )
        return [(m.id, float(m.score or 0.0)) for m in response.matches]

    async def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), 1000):
            await self._index.delete(ids=ids[start : start + 1000], namespace=self.namespace)

    async def close(self) -> None:
        await self._index.close()
        await self._client.close()


async def ensure_pinecone_index(settings: Settings, dimension: int) -> bool:
    """Crea el índice serverless si no existe. Devuelve True si lo creó."""

    async with PineconeAsyncio(api_key=settings.pinecone_api_key) as client:
        if await client.has_index(settings.pinecone_index):
            description = await client.describe_index(settings.pinecone_index)
            if description.dimension != dimension:
                raise RuntimeError(
                    f"El índice {settings.pinecone_index} tiene dimensión "
                    f"{description.dimension} y el modelo de embeddings da {dimension}."
                )
            return False
        await client.create_index(
            name=settings.pinecone_index,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        return True


async def create_vector_store(settings: Settings, redis: Redis) -> VectorStore:
    if settings.vector_store == "pinecone":
        return await PineconeVectorStore.connect(settings)
    return LocalVectorStore(redis, settings.key_prefix)
