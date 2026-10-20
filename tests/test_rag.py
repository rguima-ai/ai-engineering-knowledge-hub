from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.tools import build_search_tool, search
from app.rag import RAGComponents
from app.rag.models import Chunk, DocType, RetrievedChunk, SearchFilter, SearchQuery
from app.rag.vector_store import PineconeVectorStore
from app.seed import discover
from tests.conftest import ROOT

DOCS = {
    "modulo_04_rag/hibrido.md": (
        "# Búsqueda híbrida\n\nBM25 encuentra coincidencias exactas como lru_cache. "
        "Reciprocal Rank Fusion combina los rankings de BM25 y Pinecone."
    ),
    "modulo_05_langgraph/checkpointer.md": (
        "# Checkpointer\n\nAsyncRedisSaver guarda el estado del grafo por thread_id para "
        "que la conversación sobreviva a un reinicio."
    ),
    "modulo_06_multi/supervisor.md": (
        "# Supervisor\n\nEl Supervisor decide qué especialista trabaja después y revisa la calidad."
    ),
}


async def _index_all(rag: RAGComponents) -> None:
    for source, text in DOCS.items():
        document, chunks = await rag.ingestion.prepare(text.encode(), source)
        await rag.ingestion.index(document, chunks)


async def test_busqueda_hibrida_encuentra_por_palabra_exacta(rag: RAGComponents) -> None:
    await _index_all(rag)
    results = await rag.retriever.search(SearchQuery(query="¿Qué hace AsyncRedisSaver?", k=2))
    assert results[0].source == "modulo_05_langgraph/checkpointer.md"
    assert results[0].module == 5


async def test_el_filtro_aplica_a_las_dos_ramas(rag: RAGComponents) -> None:
    await _index_all(rag)
    query = SearchQuery(query="Supervisor checkpointer BM25", filter=SearchFilter(modules=[4]), k=5)
    results = await rag.retriever.search(query)
    assert results
    assert {r.module for r in results} == {4}


async def test_reindexar_es_idempotente_y_borra_chunks_sobrantes(rag: RAGComponents) -> None:
    source = "modulo_03/largo.md"
    long_text = "\n\n".join(f"Sección {i}. " + "embeddings chunking " * 60 for i in range(4))
    document, chunks = await rag.ingestion.prepare(long_text.encode(), source)
    await rag.ingestion.index(document, chunks)
    assert len(chunks) > 1

    short, short_chunks = await rag.ingestion.prepare(b"# Corto\n\nembeddings", source)
    await rag.ingestion.index(short, short_chunks)

    stored = await rag.store.all_chunks()
    assert [c.id for c in stored] == [c.id for c in short_chunks]
    vectors = await rag.vectors.query([1.0] + [0.0] * 511, 50, SearchFilter())
    assert {cid for cid, _ in vectors} == {c.id for c in short_chunks}


async def test_bm25_se_reconstruye_cuando_cambia_el_corpus(rag: RAGComponents) -> None:
    await _index_all(rag)
    before = await rag.retriever.search(SearchQuery(query="pydantic_settings", k=3))
    assert all(r.source != "modulo_07/config.md" for r in before)
    document, chunks = await rag.ingestion.prepare(
        b"# Config\n\npydantic_settings lee el archivo .env", "modulo_07/config.md"
    )
    await rag.ingestion.index(document, chunks)
    results = await rag.retriever.search(SearchQuery(query="pydantic_settings", k=1))
    assert results[0].source == "modulo_07/config.md"


async def test_borrar_documento(rag: RAGComponents) -> None:
    await _index_all(rag)
    doc_id = (await rag.store.list_documents())[0].metadata.doc_id
    removed = await rag.ingestion.delete(doc_id)
    assert removed >= 1
    assert await rag.store.get_document(doc_id) is None


def test_los_sample_docs_cubren_todos_los_tipos_y_modulos() -> None:
    files = discover([ROOT / "sample_docs"])
    suffixes = {path.suffix for path, _ in files}
    assert {".md", ".py", ".ipynb", ".html"} <= suffixes
    assert all(source.startswith("sample_docs/") for _, source in files)


class _FakeMatch:
    def __init__(self, id: str, score: float) -> None:
        self.id = id
        self.score = score


class _FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(("upsert", kwargs))

    async def query(self, **kwargs: Any) -> Any:
        self.calls.append(("query", kwargs))
        return type("R", (), {"matches": [_FakeMatch("a::0", 0.9)]})()

    async def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))


async def test_pinecone_store_traduce_filtros_y_metadata() -> None:
    index = _FakeIndex()
    store = PineconeVectorStore(client=None, index=index, namespace="ns")  # type: ignore[arg-type]
    chunk = Chunk(
        id="a::0",
        doc_id="a",
        chunk_index=0,
        text="t",
        source="s",
        title="T",
        doc_type=DocType.CODE,
        module=5,
    )
    await store.upsert([chunk], [[0.1, 0.2]])
    hits = await store.query([0.1, 0.2], 3, SearchFilter(modules=[5], doc_types=[DocType.CODE]))
    assert hits == [("a::0", 0.9)]
    upsert = index.calls[0][1]
    assert upsert["namespace"] == "ns"
    assert upsert["vectors"][0]["metadata"]["module"] == 5
    assert "id" not in upsert["vectors"][0]["metadata"]
    assert index.calls[1][1]["filter"] == {
        "$and": [{"module": {"$in": [5]}}, {"doc_type": {"$in": ["code"]}}]
    }


async def test_herramienta_de_busqueda_valida_entrada_y_salida(rag: RAGComponents) -> None:
    """El RAG expuesto como tool: rechaza entradas inválidas y devuelve RetrievedChunk."""

    await _index_all(rag)
    tool = build_search_tool(rag.retriever)
    assert tool.name == "search_course_material"

    results = await search(tool, SearchQuery(query="¿Qué hace AsyncRedisSaver?", k=2))
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].source == "modulo_05_langgraph/checkpointer.md"

    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "x"})  # menos de 2 caracteres
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "checkpointer", "k": 99})  # k fuera de rango
