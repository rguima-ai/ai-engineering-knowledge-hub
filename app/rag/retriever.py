"""Recuperación híbrida: BM25 (palabras exactas) + vectores (significado), fusionados con RRF.

En la Pre-entrega 4 la fusión la hacía EnsembleRetriever y la consulta a Pinecone corría en
un hilo. Acá la fusión es propia (Reciprocal Rank Fusion ponderado, la misma fórmula) y todo
el camino es async: embeddings, Pinecone y Redis.
"""

from __future__ import annotations

import asyncio
import logging

from rank_bm25 import BM25Okapi

from app.rag.chunk_store import ChunkStore
from app.rag.embeddings import AsyncEmbedder
from app.rag.models import Chunk, RetrievedChunk, SearchQuery
from app.rag.text import tokenize
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Constante estándar de RRF: amortigua la diferencia entre el puesto 1 y el 2.
RRF_K = 60


class _BM25Index:
    def __init__(self, version: int, chunks: list[Chunk]) -> None:
        self.version = version
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None

    def search(self, query: SearchQuery, limit: int) -> list[tuple[str, float]]:
        terms = tokenize(query.query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            (
                (chunk.id, float(score))
                for chunk, score in zip(self.chunks, scores, strict=True)
                if score > 0 and query.filter.matches(chunk)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:limit]


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        vectors: VectorStore,
        embedder: AsyncEmbedder,
        *,
        bm25_weight: float = 0.5,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self._index: _BM25Index | None = None
        self._lock = asyncio.Lock()

    async def _bm25(self) -> _BM25Index:
        """Reconstruye BM25 solo si otro proceso cambió el corpus (ej. un upload aprobado)."""

        version = await self.store.version()
        if self._index is not None and self._index.version == version:
            return self._index
        async with self._lock:
            if self._index is None or self._index.version != version:
                chunks = await self.store.all_chunks()
                self._index = _BM25Index(version, chunks)
                logger.info("BM25 reconstruido version=%s chunks=%s", version, len(chunks))
        return self._index

    async def search(self, query: SearchQuery) -> list[RetrievedChunk]:
        # Cada rama trae el doble de candidatos: la fusión necesita margen para reordenar.
        candidates = query.k * 2
        bm25_index, vector = await asyncio.gather(
            self._bm25(), self.embedder.aembed_query(query.query)
        )
        vector_hits = await self.vectors.query(vector, candidates, query.filter)
        bm25_hits = bm25_index.search(query, candidates)

        fused: dict[str, float] = {}
        for weight, hits in ((self.bm25_weight, bm25_hits), (1 - self.bm25_weight, vector_hits)):
            for rank, (chunk_id, _) in enumerate(hits, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (RRF_K + rank)

        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[: query.k]
        chunks = {c.id: c for c in await self.store.get_chunks(top_ids)}
        # Un ID que está en Pinecone pero ya no en Redis (borrado a medias) se descarta.
        return [
            RetrievedChunk.from_chunk(chunks[cid], fused[cid]) for cid in top_ids if cid in chunks
        ]
