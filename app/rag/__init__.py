"""RAG híbrido: loaders, chunking, embeddings, bases vectoriales y recuperación."""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import Settings
from app.rag.chunk_store import ChunkStore
from app.rag.chunking import TokenChunker
from app.rag.embeddings import AsyncEmbedder, create_embedder
from app.rag.ingestion import IngestionService
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore, create_vector_store


@dataclass
class RAGComponents:
    embedder: AsyncEmbedder
    store: ChunkStore
    vectors: VectorStore
    retriever: HybridRetriever
    ingestion: IngestionService

    async def close(self) -> None:
        await self.vectors.close()


async def build_rag(settings: Settings, redis: Redis) -> RAGComponents:
    """Arma el RAG con el mismo embedder para indexar y para consultar."""

    embedder = create_embedder(settings)
    store = ChunkStore(redis, settings.key_prefix)
    vectors = await create_vector_store(settings, redis)
    return RAGComponents(
        embedder=embedder,
        store=store,
        vectors=vectors,
        retriever=HybridRetriever(store, vectors, embedder, bm25_weight=settings.bm25_weight),
        ingestion=IngestionService(
            store, vectors, embedder, TokenChunker(settings.chunk_size, settings.chunk_overlap)
        ),
    )
