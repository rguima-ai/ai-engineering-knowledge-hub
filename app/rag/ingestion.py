"""Ingesta en dos fases: preparar (parsear + fragmentar) e indexar (embeddings + bases).

Están separadas porque entre una y otra puede haber una aprobación humana: el documento se
analiza al subirlo, pero no entra a la base compartida hasta que alguien lo aprueba.
"""

from __future__ import annotations

import asyncio
import logging

from app.rag.chunk_store import ChunkStore, DocumentRecord
from app.rag.chunking import TokenChunker
from app.rag.embeddings import AsyncEmbedder
from app.rag.loaders import parse_document
from app.rag.models import Chunk, ParsedDocument
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        store: ChunkStore,
        vectors: VectorStore,
        embedder: AsyncEmbedder,
        chunker: TokenChunker,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.chunker = chunker

    async def prepare(
        self,
        data: bytes,
        source: str,
        *,
        module: int | None = None,
        topic: str = "",
        uploaded_by: str = "seed",
    ) -> tuple[ParsedDocument, list[Chunk]]:
        """Parsear un PDF y tokenizar es CPU: se hace en un hilo para no frenar el loop."""

        def _work() -> tuple[ParsedDocument, list[Chunk]]:
            document = parse_document(
                data, source, module=module, topic=topic, uploaded_by=uploaded_by
            )
            return document, self.chunker.chunk(document)

        return await asyncio.to_thread(_work)

    async def index(
        self, document: ParsedDocument, chunks: list[Chunk], *, approved_by: str = "seed"
    ) -> DocumentRecord:
        """Idempotente: reindexar el mismo archivo reemplaza sus chunks sin duplicar."""

        vectors = await self.embedder.aembed_documents([c.text for c in chunks])
        # Primero la base vectorial y después Redis: si el upsert falla, el corpus de BM25
        # no queda apuntando a vectores que no existen.
        await self.vectors.upsert(chunks, vectors)
        record = DocumentRecord(
            metadata=document.metadata,
            chunk_ids=[c.id for c in chunks],
            approved_by=approved_by,
        )
        stale = await self.store.save_document(record, chunks)
        await self.vectors.delete(stale)
        logger.info(
            "documento indexado doc_id=%s source=%s chunks=%s module=%s",
            document.metadata.doc_id,
            document.metadata.source,
            len(chunks),
            document.metadata.module,
        )
        return record

    async def delete(self, doc_id: str) -> int:
        ids = await self.store.delete_document(doc_id)
        await self.vectors.delete(ids)
        return len(ids)
