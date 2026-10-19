"""Recuperador híbrido: BM25 (palabras exactas) + Pinecone (significado), fusionados con RRF."""
import asyncio
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import (AsyncCallbackManagerForRetrieverRun,
                                      CallbackManagerForRetrieverRun)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from ingest import load_chunks
from schemas import RetrievedChunk

Mode = Literal["hybrid", "bm25", "vector"]

# Palabras vacías del español (sin tildes: se comparan después de normalizar).
STOPWORDS = frozenset("""
a al algo algun alguna ante como con contra cual cuales cuando de del desde donde e el ella ellos
en entre era es esa ese eso esta este esto estos hay la las le les lo los mas me mi mis muy ni no
nos o otra otro para pero por porque puede que se sea ser si sin sobre son su sus tambien tan te
tiene tienen todo todos tu un una uno unos y ya yo hace hacer hago sirve uso
""".split())

_TOKEN = re.compile(r"[a-z0-9_]+")


def preprocess(text: str) -> list[str]:
    """Tokenizador para BM25: minúsculas, sin tildes, sin signos y sin palabras vacías.

    El default de BM25Retriever es text.split(): "¿Qué" no coincide con "que", "lru_cache?" no
    coincide con "lru_cache" y "función" no coincide con "funcion". "asyncio.gather" se parte en
    "asyncio" y "gather", así matchea tanto la forma con módulo como la suelta.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS]


# ---------- filtros de metadatos (el mismo filtro para las dos ramas) ----------
_OPS = {
    "$eq": lambda v, x: v == x,
    "$ne": lambda v, x: v != x,
    "$in": lambda v, x: v in x,
    "$nin": lambda v, x: v not in x,
    "$gt": lambda v, x: v > x,
    "$gte": lambda v, x: v >= x,
    "$lt": lambda v, x: v < x,
    "$lte": lambda v, x: v <= x,
}


def matches_filter(metadata: dict[str, Any], filtro: dict[str, Any] | None) -> bool:
    """Evalúa en memoria el subconjunto del lenguaje de filtros de Pinecone que usamos.

    {"category": "funcional"} equivale a {"category": {"$eq": "funcional"}}; varios campos se combinan con AND.
    """
    if not filtro:
        return True
    for campo, condicion in filtro.items():
        if campo not in metadata:
            return False
        if not isinstance(condicion, dict):
            condicion = {"$eq": condicion}
        for op, valor in condicion.items():
            if op not in _OPS:
                raise ValueError(f"Operador de filtro no soportado: {op}")
            if not _OPS[op](metadata[campo], valor):
                return False
    return True


class PineconeRetriever(BaseRetriever):
    """Búsqueda vectorial con el SDK nativo de Pinecone, con namespace y filtro de metadatos opcional."""

    index: Any
    embeddings: Any
    namespace: str = config.NAMESPACE
    k: int = config.TOP_K
    filter: dict | None = None

    def _to_documents(self, res) -> list[Document]:
        # El texto vuelve desde la metadata: no hace falta otra base para recuperarlo.
        return [Document(page_content=m.metadata["text"],
                         metadata={**m.metadata, "id": m.id, "score": m.score})
                for m in res.matches]

    def _query(self, vector: list[float]):
        return self.index.query(vector=vector, top_k=self.k, namespace=self.namespace,
                                filter=self.filter, include_metadata=True)

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self._to_documents(self._query(self.embeddings.embed_query(query)))

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]:
        vector = await self.embeddings.aembed_query(query)
        return self._to_documents(await asyncio.to_thread(self._query, vector))


class RAGSystem:
    """Encapsula BM25 + Pinecone + EnsembleRetriever y devuelve el top-k sin duplicados."""

    def __init__(self, index, embeddings, k: int = config.TOP_K,
                 weights: tuple[float, float] = config.HYBRID_WEIGHTS,
                 namespace: str = config.NAMESPACE, filter: dict | None = None,
                 data_dir: str | Path = config.DATA_DIR):
        # El corpus de BM25 se arma con el MISMO chunking e IDs que la ingesta. BM25 necesita el texto
        # en memoria: Pinecone con vectores densos no hace búsqueda por palabras.
        docs = [Document(page_content=c.text, metadata={**c.model_dump(), "id": c.vector_id})
                for c in load_chunks(data_dir)]
        # Mismo filtro en las dos ramas: si solo filtrara Pinecone, BM25 colaría chunks de otras categorías.
        docs = [d for d in docs if matches_filter(d.metadata, filter)]
        if not docs:
            raise ValueError(f"Ningún chunk cumple el filtro {filter}")

        self.k = k
        self.weights = weights
        self.bm25 = BM25Retriever.from_documents(docs, preprocess_func=preprocess, k=k)
        self.vector = PineconeRetriever(index=index, embeddings=embeddings, namespace=namespace,
                                        k=k, filter=filter)

    def _ensemble(self, weights: tuple[float, float]) -> EnsembleRetriever:
        # id_key="id": fusiona por ID de chunk. Sin él compara el texto completo, y dos chunks con el
        # mismo texto (o el mismo chunk con espacios distintos) se contarían mal.
        return EnsembleRetriever(retrievers=[self.bm25, self.vector], weights=list(weights), id_key="id")

    def _retriever(self, mode: Mode, weights: tuple[float, float] | None) -> BaseRetriever:
        if mode == "hybrid":
            return self._ensemble(weights or self.weights)
        if mode == "bm25":
            return self.bm25
        if mode == "vector":
            return self.vector
        raise ValueError(f"mode inválido: {mode!r}. Opciones: hybrid, bm25, vector")

    def _top(self, docs: list[Document]) -> list[RetrievedChunk]:
        # El ensemble devuelve hasta 2k (k de cada rama): se deduplica por ID y se recorta a k.
        vistos: set[str] = set()
        out: list[RetrievedChunk] = []
        for d in docs:
            if d.metadata["id"] in vistos:
                continue
            vistos.add(d.metadata["id"])
            out.append(RetrievedChunk(id=d.metadata["id"], source=d.metadata["source"],
                                      chunk_id=d.metadata["chunk_id"],
                                      category=d.metadata["category"], text=d.page_content))
            if len(out) == self.k:
                break
        return out

    def retrieve(self, query: str, mode: Mode = "hybrid",
                 weights: tuple[float, float] | None = None) -> list[RetrievedChunk]:
        return self._top(self._retriever(mode, weights).invoke(query))

    async def aretrieve(self, query: str, mode: Mode = "hybrid",
                        weights: tuple[float, float] | None = None) -> list[RetrievedChunk]:
        return self._top(await self._retriever(mode, weights).ainvoke(query))
