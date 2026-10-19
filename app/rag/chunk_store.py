"""Catálogo de documentos y chunks en Redis: la fuente de verdad del texto.

BM25 necesita el corpus completo en memoria y Pinecone (vectores densos) no busca por palabras,
así que el texto se guarda acá. Cada alta o baja incrementa `corpus:version`; los procesos que
cachean el índice BM25 lo reconstruyen solo cuando la versión cambió.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.rag.models import Chunk, DocumentMetadata


class DocumentRecord(BaseModel):
    """Documento indexado, con trazabilidad de quién lo aprobó."""

    metadata: DocumentMetadata
    chunk_ids: list[str]
    approved_by: str = "seed"
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChunkStore:
    def __init__(self, redis: Redis, prefix: str = "kh:") -> None:
        self.redis = redis
        self._chunks = f"{prefix}chunks"
        self._docs = f"{prefix}docs"
        self._version = f"{prefix}corpus:version"

    async def version(self) -> int:
        return int(await self.redis.get(self._version) or 0)

    async def save_document(self, record: DocumentRecord, chunks: list[Chunk]) -> list[str]:
        """Reemplaza atómicamente un documento. Devuelve los IDs de chunks que sobraron.

        Si una versión nueva del archivo tiene menos chunks, los IDs viejos de más hay que
        borrarlos también de la base vectorial; por eso se devuelven.
        """

        previous = await self.get_document(record.metadata.doc_id)
        stale = sorted(set(previous.chunk_ids) - set(record.chunk_ids)) if previous else []
        async with self.redis.pipeline(transaction=True) as pipe:
            if stale:
                pipe.hdel(self._chunks, *stale)
            if chunks:
                pipe.hset(self._chunks, mapping={c.id: c.model_dump_json() for c in chunks})
            pipe.hset(self._docs, record.metadata.doc_id, record.model_dump_json())
            pipe.incr(self._version)
            await pipe.execute()
        return stale

    async def delete_document(self, doc_id: str) -> list[str]:
        record = await self.get_document(doc_id)
        if record is None:
            return []
        async with self.redis.pipeline(transaction=True) as pipe:
            if record.chunk_ids:
                pipe.hdel(self._chunks, *record.chunk_ids)
            pipe.hdel(self._docs, doc_id)
            pipe.incr(self._version)
            await pipe.execute()
        return record.chunk_ids

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        raw = await self.redis.hget(self._docs, doc_id)
        return DocumentRecord.model_validate_json(raw) if raw else None

    async def list_documents(self) -> list[DocumentRecord]:
        raw = await self.redis.hgetall(self._docs)
        records = [DocumentRecord.model_validate_json(v) for v in raw.values()]
        return sorted(records, key=lambda r: (r.metadata.module, r.metadata.source))

    async def get_chunks(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        raw = await self.redis.hmget(self._chunks, ids)
        return [Chunk.model_validate_json(r) for r in raw if r]

    async def all_chunks(self) -> list[Chunk]:
        raw = await self.redis.hgetall(self._chunks)
        chunks = [Chunk.model_validate_json(v) for v in raw.values()]
        return sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index))
