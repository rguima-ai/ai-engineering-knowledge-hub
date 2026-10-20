"""Grafo del chat: Supervisor + Buscador RAG + Revisor de código + Tutor + Validador.

El Supervisor es un LLM, pero decide dentro de una política escrita en código:
`allowed_routes` calcula qué pasos tienen sentido según el estado (no se responde sin buscar,
no se termina sin validar la última versión de la respuesta, hay un tope de revisiones) y el
LLM elige entre esas opciones. Si hay una sola opción no se le pregunta nada: ahorra una
llamada y latencia. Si elige algo fuera de la lista, se corrige y queda registrado.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.backends import AgentBackend
from app.agents.schemas import (
    ChatRequest,
    ChatResult,
    Citation,
    CodeReview,
    RequestPlan,
    Route,
    SupervisorDecision,
    TutorAnswer,
    ValidationReport,
)
from app.agents.tools import build_search_tool, search
from app.config import Settings
from app.rag.models import RetrievedChunk, SearchFilter, SearchQuery
from app.rag.retriever import HybridRetriever

MAX_RETRIEVAL_ROUNDS = 2
MAX_CONTEXT_CHUNKS = 10
_CITATION = re.compile(r"\[(\d{1,2})\]")
# "No encontré esto en el material" y variantes como "No encontré información sobre X en el
# material del curso" (el LLM no siempre repite la frase literal del prompt).
_ADMITS_UNKNOWN = re.compile(
    r"no (encontr[eé]|est[aá]|aparece)\b[^.]{0,200}?\ben el material", re.IGNORECASE
)


class ChatState(MessagesState, total=False):
    """`messages` acumula la conversación entre turnos; el resto se reinicia en cada turno."""

    request: dict[str, Any]
    plan: dict[str, Any] | None
    queries: list[str]
    retrieval_rounds: int
    retrieved: list[dict[str, Any]]
    code_review: dict[str, Any] | None
    draft: dict[str, Any] | None
    draft_version: int
    validation: dict[str, Any] | None
    validated_version: int
    revisions: int
    agents_used: list[str]
    supervisor_steps: int
    supervisor_log: list[dict[str, Any]]
    next_agent: Route
    result: dict[str, Any] | None


def new_turn(request: ChatRequest) -> dict[str, Any]:
    """Entrada de un turno nuevo: agrega el mensaje y limpia el estado del turno anterior.

    Sin este reinicio, el checkpointer traería el borrador y la validación de la pregunta
    anterior y el Supervisor creería que ya está todo hecho.
    """

    return {
        "messages": [HumanMessage(content=request.question)],
        "request": request.model_dump(mode="json"),
        "plan": None,
        "queries": [],
        "retrieval_rounds": 0,
        "retrieved": [],
        "code_review": None,
        "draft": None,
        "draft_version": 0,
        "validation": None,
        "validated_version": 0,
        "revisions": 0,
        "agents_used": [],
        "supervisor_steps": 0,
        "supervisor_log": [],
        "result": None,
    }


def _context(state: ChatState) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(c) for c in state.get("retrieved") or []]


def _history(state: ChatState) -> Sequence[BaseMessage]:
    """Mensajes de turnos anteriores (el último es la pregunta actual)."""

    return list(state.get("messages") or [])[:-1]


def allowed_routes(state: ChatState, settings: Settings) -> list[Route]:
    """Pasos válidos según el estado. El primero es el default si el LLM no elige bien."""

    max_supervisor_steps = settings.max_graph_steps // 2 - 1
    if state.get("supervisor_steps", 0) >= max_supervisor_steps:
        return ["FINISH"]

    request = ChatRequest.model_validate(state["request"])
    rounds = state.get("retrieval_rounds", 0)
    search: list[Route] = ["retriever"] if rounds < MAX_RETRIEVAL_ROUNDS else []

    if rounds == 0:
        return ["retriever"]
    if request.code and not state.get("code_review"):
        return ["code_reviewer", *search]
    if not state.get("draft"):
        return ["tutor", *search]
    if state.get("validated_version", 0) != state.get("draft_version", 0):
        return ["validator"]
    validation = state.get("validation") or {}
    if (
        validation.get("verdict") == "needs_revision"
        and state.get("revisions", 0) < settings.max_revisions
    ):
        return ["tutor", *search]
    return ["FINISH"]


def _situation(state: ChatState) -> str:
    """Resumen corto del estado para que el LLM del Supervisor elija con contexto."""

    request = ChatRequest.model_validate(state["request"])
    validation = state.get("validation") or {}
    draft = f"versión {state.get('draft_version')}" if state.get("draft") else "no"
    return (
        f"Pedido: {request.question[:300]}\n"
        f"Trae código: {'sí' if request.code else 'no'}\n"
        f"Búsquedas hechas: {state.get('retrieval_rounds', 0)} "
        f"({len(state.get('retrieved') or [])} fragmentos)\n"
        f"Revisión de código: {'hecha' if state.get('code_review') else 'no'}\n"
        f"Borrador: {draft}\n"
        f"Validación: {validation.get('verdict', 'no')} {validation.get('feedback', '')[:300]}"
    )


def build_chat_graph(
    backend: AgentBackend,
    retriever: HybridRetriever,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[ChatState, None, Any, Any]:
    search_tool = build_search_tool(retriever)

    def used(state: ChatState, agent: str) -> list[str]:
        return [*(state.get("agents_used") or []), agent]

    async def supervisor(state: ChatState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        request = ChatRequest.model_validate(state["request"])
        if state.get("plan") is None:
            plan = await backend.plan(request, _history(state))
            updates["plan"] = plan.model_dump(mode="json")
            updates["queries"] = plan.search_queries
            state = {**state, **updates}  # type: ignore[typeddict-item]

        allowed = allowed_routes(state, settings)
        if len(allowed) == 1:
            decision = SupervisorDecision(next=allowed[0], reason="Único paso permitido.")
            source = "policy"
        else:
            decision = await backend.route(_situation(state), allowed)
            source = "llm"
            if decision.next not in allowed:
                decision = SupervisorDecision(
                    next=allowed[0],
                    reason=f"Corregido: el LLM eligió {decision.next}, fuera de {allowed}.",
                )
                source = "policy_override"

        log_entry = {"next": decision.next, "reason": decision.reason, "source": source}
        return {
            **updates,
            "next_agent": decision.next,
            "supervisor_steps": state.get("supervisor_steps", 0) + 1,
            "supervisor_log": [*(state.get("supervisor_log") or []), log_entry],
        }

    async def retriever_node(state: ChatState) -> dict[str, Any]:
        request = ChatRequest.model_validate(state["request"])
        plan = RequestPlan.model_validate(state["plan"])
        rounds = state.get("retrieval_rounds", 0)
        queries = state.get("queries") or plan.search_queries
        if rounds > 0:
            validation = state.get("validation") or {}
            feedback = validation.get("feedback") or "El material recuperado no alcanza."
            queries = (await backend.refine_queries(request, queries, feedback)).queries

        if request.modules:
            # Lo eligió el usuario: filtro estricto, salvo que no haya nada en esos módulos.
            results = await _search_all(queries, SearchFilter(modules=request.modules))
            if not any(results):
                results = await _search_all(queries, SearchFilter())
        elif plan.modules:
            # Lo infirió el LLM: es una preferencia, no un filtro. Si el módulo está mal inferido
            # o el documento quedó mal clasificado, un filtro estricto lo vuelve invisible.
            results = await _search_all(queries, SearchFilter(modules=plan.modules))
            results += await _search_all(queries, SearchFilter())
        else:
            results = await _search_all(queries, SearchFilter())

        merged = {c.id: c for c in _context(state)}
        for chunk in sorted((c for batch in results for c in batch), key=lambda c: -c.score):
            merged.setdefault(chunk.id, chunk)
        context = list(merged.values())[:MAX_CONTEXT_CHUNKS]
        return {
            "queries": queries,
            "retrieval_rounds": rounds + 1,
            "retrieved": [c.model_dump(mode="json") for c in context],
            "agents_used": used(state, "retriever"),
        }

    async def _search_all(
        queries: list[str], search_filter: SearchFilter
    ) -> list[list[RetrievedChunk]]:
        return list(
            await asyncio.gather(
                *(
                    search(
                        search_tool, SearchQuery(query=q, filter=search_filter, k=settings.top_k)
                    )
                    for q in queries
                )
            )
        )

    async def code_reviewer(state: ChatState) -> dict[str, Any]:
        request = ChatRequest.model_validate(state["request"])
        review = await backend.review_code(request, _context(state))
        return {
            "code_review": review.model_dump(mode="json"),
            "agents_used": used(state, "code_reviewer"),
        }

    async def tutor(state: ChatState) -> dict[str, Any]:
        request = ChatRequest.model_validate(state["request"])
        plan = RequestPlan.model_validate(state["plan"])
        raw_review = state.get("code_review")
        review = CodeReview.model_validate(raw_review) if raw_review else None
        feedback = ""
        revisions = state.get("revisions", 0)
        if state.get("draft"):
            report = ValidationReport.model_validate(state["validation"])
            claims = "; ".join(report.unsupported_claims)
            feedback = f"{report.feedback} Afirmaciones sin respaldo: {claims}".strip()
            revisions += 1
        draft = await backend.tutor(
            request, plan, _context(state), review, feedback, _history(state)
        )
        return {
            "draft": draft.model_dump(mode="json"),
            "draft_version": state.get("draft_version", 0) + 1,
            "revisions": revisions,
            "agents_used": used(state, "tutor"),
        }

    async def validator(state: ChatState) -> dict[str, Any]:
        request = ChatRequest.model_validate(state["request"])
        draft = TutorAnswer.model_validate(state["draft"])
        context = _context(state)
        report = check_citations(draft, len(context))
        if report is None:
            report = await backend.validate(request, draft, context)
        return {
            "validation": report.model_dump(mode="json"),
            "validated_version": state.get("draft_version", 0),
            "agents_used": used(state, "validator"),
        }

    async def finalize(state: ChatState) -> dict[str, Any]:
        result = build_result(state)
        return {
            "result": result.model_dump(mode="json"),
            "messages": [AIMessage(content=result.answer, name="tutor")],
        }

    builder = StateGraph(ChatState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("retriever", retriever_node)
    builder.add_node("code_reviewer", code_reviewer)
    builder.add_node("tutor", tutor)
    builder.add_node("validator", validator)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "retriever": "retriever",
            "code_reviewer": "code_reviewer",
            "tutor": "tutor",
            "validator": "validator",
            "FINISH": "finalize",
        },
    )
    for specialist in ("retriever", "code_reviewer", "tutor", "validator"):
        builder.add_edge(specialist, "supervisor")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="chat")


def check_citations(draft: TutorAnswer, available: int) -> ValidationReport | None:
    """Control determinístico previo al LLM. Devuelve None si las citas son coherentes."""

    cited = {int(n) for n in _CITATION.findall(draft.answer)} | set(draft.sources)
    invalid = sorted(n for n in cited if n < 1 or n > available)
    if invalid:
        return ValidationReport(
            verdict="needs_revision",
            unsupported_claims=[],
            feedback=f"Citaste fragmentos que no existen: {invalid}. Hay {available} fragmentos.",
        )
    if available and not cited and not _ADMITS_UNKNOWN.search(draft.answer):
        return ValidationReport(
            verdict="needs_revision",
            unsupported_claims=[],
            feedback="La respuesta no cita ningún fragmento. Agregá citas [n] o decí que no está.",
        )
    return None


def build_result(state: ChatState) -> ChatResult:
    context = _context(state)
    draft = (
        TutorAnswer.model_validate(state["draft"])
        if state.get("draft")
        else TutorAnswer(answer="No pude generar una respuesta.", sources=[], quiz=[])
    )
    validation = (
        ValidationReport.model_validate(state["validation"]) if state.get("validation") else None
    )
    answer = draft.answer
    if validation and validation.verdict == "needs_revision":
        # Se agotaron las revisiones: se entrega igual, pero sin esconder el problema.
        detail = "; ".join(validation.unsupported_claims) or validation.feedback
        answer += f"\n\n> ⚠️ El Validador no pudo confirmar todo con el material: {detail}"

    cited = sorted({int(n) for n in _CITATION.findall(draft.answer)} | set(draft.sources))
    citations = [
        Citation(
            number=n,
            chunk_id=context[n - 1].id,
            source=context[n - 1].source,
            title=context[n - 1].title,
            module=context[n - 1].module,
            snippet=context[n - 1].text[:280],
        )
        for n in cited
        if 1 <= n <= len(context)
    ]
    return ChatResult(
        answer=answer,
        citations=citations,
        quiz=draft.quiz,
        code_review=(
            CodeReview.model_validate(state["code_review"]) if state.get("code_review") else None
        ),
        validation=validation,
        agents_used=state.get("agents_used") or [],
        revisions=state.get("revisions", 0),
    )
