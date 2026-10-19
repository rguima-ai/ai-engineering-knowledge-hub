from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.backends import SimulatedBackend
from app.agents.ingest_graph import build_ingest_graph
from app.rag import RAGComponents
from app.rag.models import SearchQuery
from app.rag.uploads import UploadStore

GUIDE = (
    "# Guía de streaming\n\nPara hacer streaming de tokens con LangChain se usa astream. "
    "Cada chunk llega apenas el modelo lo genera, lo que baja el tiempo hasta el primer token. "
    "Conviene combinarlo con un timeout por llamada y reintentos con backoff exponencial."
).encode()


def _graph(rag: RAGComponents, redis: Any) -> tuple[Any, UploadStore]:
    uploads = UploadStore(redis)
    graph = build_ingest_graph(
        SimulatedBackend(),
        rag.ingestion,
        uploads,
        rag.store,
        rag.vectors,
        rag.embedder,
        checkpointer=InMemorySaver(),
    )
    return graph, uploads


async def _start(graph: Any, uploads: UploadStore, upload_id: str, data: bytes) -> dict[str, Any]:
    await uploads.save(upload_id, data)
    config = {"configurable": {"thread_id": upload_id}}
    await graph.ainvoke(
        {
            "upload_id": upload_id,
            "filename": "streaming.md",
            "uploaded_by": "ana",
            "module_hint": None,
        },
        config=config,
    )
    return config


async def test_se_pausa_hasta_la_aprobacion_y_despues_indexa(
    rag: RAGComponents, redis: Any
) -> None:
    graph, uploads = _graph(rag, redis)
    config = await _start(graph, uploads, "up1", GUIDE)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("human_review",)
    payload = snapshot.tasks[0].interrupts[0].value
    assert payload["type"] == "document_review"
    assert payload["stats"]["chunks"] == 1
    assert await rag.store.list_documents() == []  # nada indexado antes de aprobar

    await graph.ainvoke(
        Command(resume={"approved": True, "reviewer": "rocco", "module": 1}), config=config
    )
    result = (await graph.aget_state(config)).values["result"]
    assert result["status"] == "indexed"
    assert result["module"] == 1
    hits = await rag.retriever.search(SearchQuery(query="streaming astream", k=1))
    assert hits[0].source == "uploads/streaming.md"
    record = await rag.store.get_document(result["doc_id"])
    assert record is not None and record.approved_by == "rocco"
    assert await redis.get("kh:upload:up1") is None


async def test_rechazo_no_indexa_y_borra_el_archivo(rag: RAGComponents, redis: Any) -> None:
    graph, uploads = _graph(rag, redis)
    config = await _start(graph, uploads, "up2", GUIDE)
    await graph.ainvoke(
        Command(resume={"approved": False, "reviewer": "rocco", "comment": "fuera de tema"}),
        config=config,
    )
    assert (await graph.aget_state(config)).values["result"]["status"] == "rejected"
    assert await rag.store.list_documents() == []
    assert await redis.get("kh:upload:up2") is None


async def test_detecta_duplicados_de_otro_documento(rag: RAGComponents, redis: Any) -> None:
    document, chunks = await rag.ingestion.prepare(GUIDE, "sample_docs/modulo_01/guia.md")
    await rag.ingestion.index(document, chunks)

    graph, uploads = _graph(rag, redis)
    config = await _start(graph, uploads, "up3", GUIDE)
    payload = (await graph.aget_state(config)).tasks[0].interrupts[0].value
    assert payload["duplicates"][0]["source"] == "sample_docs/modulo_01/guia.md"
    assert payload["duplicates"][0]["similarity"] > 0.99
