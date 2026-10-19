"""Grafo de ingesta: analizar → aprobación humana (interrupt) → indexar o descartar.

Un documento malo en la base compartida empeora las respuestas de todo el equipo, así que
nada entra sin que una persona lo apruebe. El grafo se pausa con interrupt(), el
checkpointer guarda dónde quedó y POST /documents/{id}/review lo reanuda con
Command(resume=decisión), aunque el worker se haya reiniciado en el medio.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.agents.backends import AgentBackend
from app.agents.schemas import DocumentAnalysis, DuplicateCandidate, HumanDecision
from app.rag.chunk_store import ChunkStore
from app.rag.embeddings import AsyncEmbedder
from app.rag.ingestion import IngestionService
from app.rag.models import SearchFilter
from app.rag.uploads import UploadStore
from app.rag.vector_store import VectorStore

# Coseno a partir del cual un chunk se considera casi idéntico a otro ya indexado.
DUPLICATE_THRESHOLD = 0.92
EXCERPT_CHARS = 4_000


class IngestState(TypedDict, total=False):
    upload_id: str
    filename: str
    uploaded_by: str
    module_hint: int | None
    analysis: dict[str, Any]
    duplicates: list[dict[str, Any]]
    stats: dict[str, int]
    decision: dict[str, Any]
    result: dict[str, Any]


def source_for(filename: str) -> str:
    """La fuente define el doc_id: volver a subir el mismo nombre reemplaza la versión anterior."""

    return f"uploads/{filename}"


def build_ingest_graph(
    backend: AgentBackend,
    ingestion: IngestionService,
    uploads: UploadStore,
    store: ChunkStore,
    vectors: VectorStore,
    embedder: AsyncEmbedder,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[IngestState, None, Any, Any]:
    async def analyze(state: IngestState) -> dict[str, Any]:
        data = await uploads.load(state["upload_id"])
        document, chunks = await ingestion.prepare(
            data,
            source_for(state["filename"]),
            module=state.get("module_hint"),
            uploaded_by=state["uploaded_by"],
        )
        analysis = await backend.analyze_document(
            state["filename"], document.text[:EXCERPT_CHARS], state.get("module_hint")
        )
        duplicates = await _find_duplicates(document.metadata.doc_id, [c.text for c in chunks[:3]])
        return {
            "analysis": analysis.model_dump(mode="json"),
            "duplicates": [d.model_dump(mode="json") for d in duplicates],
            "stats": {"chunks": len(chunks), "characters": len(document.text)},
        }

    async def _find_duplicates(doc_id: str, texts: list[str]) -> list[DuplicateCandidate]:
        best: dict[str, float] = {}
        for vector in await embedder.aembed_documents(texts):
            for chunk_id, score in await vectors.query(vector, 3, SearchFilter()):
                other = chunk_id.split("::", 1)[0]
                # El mismo doc_id es una versión anterior del mismo archivo: no es duplicado.
                if other != doc_id and score >= DUPLICATE_THRESHOLD:
                    best[other] = max(score, best.get(other, 0.0))
        candidates = []
        for other, score in sorted(best.items(), key=lambda item: -item[1]):
            record = await store.get_document(other)
            if record:
                candidates.append(
                    DuplicateCandidate(
                        doc_id=other, source=record.metadata.source, similarity=round(score, 3)
                    )
                )
        return candidates

    def human_review(state: IngestState) -> dict[str, Any]:
        # Todo lo que va antes del interrupt se re-ejecuta al reanudar: por eso este nodo
        # no tiene efectos, solo pregunta. El trabajo caro quedó guardado en el nodo anterior.
        response = interrupt(
            {
                "type": "document_review",
                "question": "¿Aprobás sumar este documento a la base compartida?",
                "filename": state["filename"],
                "uploaded_by": state["uploaded_by"],
                "analysis": state["analysis"],
                "duplicates": state.get("duplicates", []),
                "stats": state.get("stats", {}),
            }
        )
        decision = HumanDecision.model_validate(response)
        return {"decision": decision.model_dump(mode="json")}

    async def apply_decision(state: IngestState) -> dict[str, Any]:
        decision = HumanDecision.model_validate(state["decision"])
        analysis = DocumentAnalysis.model_validate(state["analysis"])
        if not decision.approved:
            await uploads.delete(state["upload_id"])
            return {"result": {"status": "rejected", "reviewer": decision.reviewer}}

        data = await uploads.load(state["upload_id"])
        module = decision.module if decision.module is not None else analysis.module
        document, chunks = await ingestion.prepare(
            data,
            source_for(state["filename"]),
            module=module,
            topic=decision.topic or analysis.topic,
            uploaded_by=state["uploaded_by"],
        )
        record = await ingestion.index(document, chunks, approved_by=decision.reviewer)
        await uploads.delete(state["upload_id"])
        return {
            "result": {
                "status": "indexed",
                "doc_id": record.metadata.doc_id,
                "source": record.metadata.source,
                "module": record.metadata.module,
                "chunks": len(record.chunk_ids),
                "reviewer": decision.reviewer,
            }
        }

    builder = StateGraph(IngestState)
    builder.add_node("analyze", analyze)
    builder.add_node("human_review", human_review)
    builder.add_node("apply_decision", apply_decision)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "human_review")
    builder.add_edge("human_review", "apply_decision")
    builder.add_edge("apply_decision", END)
    return builder.compile(checkpointer=checkpointer, name="ingest")
