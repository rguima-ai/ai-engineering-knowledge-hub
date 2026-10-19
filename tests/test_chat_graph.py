from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.backends import SimulatedBackend
from app.agents.chat_graph import build_chat_graph, check_citations, new_turn
from app.agents.schemas import (
    ChatRequest,
    ChatResult,
    CodeReview,
    RequestPlan,
    Route,
    SupervisorDecision,
    TutorAnswer,
)
from app.config import Settings
from app.rag import RAGComponents
from tests.test_rag import DOCS

CODE_WITH_BUGS = """\
import requests
from langgraph.checkpoint.memory import MemorySaver

async def buscar(url):
    return requests.get(url).json()

graph = builder.compile(checkpointer=MemorySaver())
"""


async def _seed(rag: RAGComponents) -> None:
    for source, text in DOCS.items():
        document, chunks = await rag.ingestion.prepare(text.encode(), source)
        await rag.ingestion.index(document, chunks)


async def _ask(
    graph: Any, request: ChatRequest, thread: str = "t1"
) -> tuple[ChatResult, dict[str, Any]]:
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 50}
    await graph.ainvoke(new_turn(request), config=config)
    state = (await graph.aget_state(config)).values
    return ChatResult.model_validate(state["result"]), state


@pytest.fixture
async def seeded(rag: RAGComponents) -> RAGComponents:
    await _seed(rag)
    return rag


def _graph(rag: RAGComponents, settings: Settings, backend: Any = None) -> Any:
    return build_chat_graph(
        backend or SimulatedBackend(), rag.retriever, settings, checkpointer=InMemorySaver()
    )


async def test_pregunta_simple_busca_redacta_y_valida(
    seeded: RAGComponents, settings: Settings
) -> None:
    result, state = await _ask(
        _graph(seeded, settings), ChatRequest(question="¿Para qué sirve AsyncRedisSaver?")
    )
    assert result.agents_used == ["retriever", "tutor", "validator"]
    assert result.validation and result.validation.verdict == "approved"
    assert result.citations
    assert result.citations[0].source == "modulo_05_langgraph/checkpointer.md"
    assert [step["next"] for step in state["supervisor_log"]] == [
        "retriever",
        "tutor",
        "validator",
        "FINISH",
    ]


async def test_con_codigo_pasa_por_el_revisor_antes_del_tutor(
    seeded: RAGComponents, settings: Settings
) -> None:
    request = ChatRequest(question="Revisá mi agente", code=CODE_WITH_BUGS, level="principiante")
    result, _ = await _ask(_graph(seeded, settings), request)
    assert result.agents_used == ["retriever", "code_reviewer", "tutor", "validator"]
    assert result.code_review is not None
    problems = " ".join(i.problem for i in result.code_review.issues)
    assert "requests" in problems and "MemorySaver" in problems
    assert result.answer.startswith("Te lo explico paso a paso.")


async def test_cuestionario_pedido_en_el_texto(seeded: RAGComponents, settings: Settings) -> None:
    result, state = await _ask(
        _graph(seeded, settings),
        ChatRequest(question="Explicame el Supervisor y generame 2 preguntas"),
    )
    assert len(result.quiz) == min(2, len(state["retrieved"]))


async def test_segundo_turno_conserva_historial_y_reinicia_el_turno(
    seeded: RAGComponents, settings: Settings
) -> None:
    graph = _graph(seeded, settings)
    await _ask(graph, ChatRequest(question="¿Qué es BM25?"))
    result, state = await _ask(graph, ChatRequest(question="¿Y el Supervisor?"))
    assert len(state["messages"]) == 4
    assert result.agents_used == ["retriever", "tutor", "validator"]
    assert result.revisions == 0


async def test_filtro_de_modulo_inexistente_vuelve_a_buscar_en_todo(
    seeded: RAGComponents, settings: Settings
) -> None:
    result, _ = await _ask(
        _graph(seeded, settings), ChatRequest(question="checkpointer thread_id", modules=[9])
    )
    assert result.citations


class WrongModulePlanner(SimulatedBackend):
    """El LLM infiere el módulo 6, pero la respuesta está en el 5."""

    async def plan(self, request: ChatRequest, history: Sequence[BaseMessage]) -> RequestPlan:
        plan = await super().plan(request, history)
        return plan.model_copy(update={"modules": [6]})


async def test_modulo_inferido_por_el_llm_no_oculta_otros_documentos(
    seeded: RAGComponents, settings: Settings
) -> None:
    graph = _graph(seeded, settings, WrongModulePlanner())
    result, _ = await _ask(graph, ChatRequest(question="¿Qué hace AsyncRedisSaver?"))
    assert "modulo_05_langgraph/checkpointer.md" in {c.source for c in result.citations}


async def test_sin_documentos_dice_que_no_sabe(rag: RAGComponents, settings: Settings) -> None:
    result, _ = await _ask(_graph(rag, settings), ChatRequest(question="¿Qué es RRF?"))
    assert "No encontré" in result.answer
    assert result.citations == []
    assert result.validation and result.validation.verdict == "approved"


class _BadCitationsBackend(SimulatedBackend):
    """El Tutor cita un fragmento inexistente las primeras `bad_drafts` veces."""

    def __init__(self, bad_drafts: int) -> None:
        self.bad_drafts = bad_drafts
        self.feedbacks: list[str] = []

    async def tutor(
        self,
        request: ChatRequest,
        plan: RequestPlan,
        context: Any,
        review: CodeReview | None,
        feedback: str,
        history: Sequence[BaseMessage],
    ) -> TutorAnswer:
        self.feedbacks.append(feedback)
        if self.bad_drafts > 0:
            self.bad_drafts -= 1
            return TutorAnswer(answer="Dato inventado [99]", sources=[99], quiz=[])
        return await super().tutor(request, plan, context, review, feedback, history)


async def test_cita_invalida_fuerza_una_revision(seeded: RAGComponents, settings: Settings) -> None:
    backend = _BadCitationsBackend(bad_drafts=1)
    result, _ = await _ask(_graph(seeded, settings, backend), ChatRequest(question="¿Qué es BM25?"))
    assert result.revisions == 1
    assert result.agents_used == ["retriever", "tutor", "validator", "tutor", "validator"]
    assert "[99]" in backend.feedbacks[1]
    assert result.validation and result.validation.verdict == "approved"


async def test_revisiones_agotadas_entregan_con_advertencia(
    seeded: RAGComponents, settings: Settings
) -> None:
    result, _ = await _ask(
        _graph(seeded, settings, _BadCitationsBackend(bad_drafts=10)),
        ChatRequest(question="¿Qué es BM25?"),
    )
    assert result.revisions == settings.max_revisions
    assert "⚠️" in result.answer
    assert result.citations == []


class _RebelSupervisor(SimulatedBackend):
    async def route(self, situation: str, allowed: list[Route]) -> SupervisorDecision:
        return SupervisorDecision(next="FINISH", reason="quiero terminar ya")


async def test_eleccion_fuera_de_la_politica_se_corrige(
    seeded: RAGComponents, settings: Settings
) -> None:
    result, state = await _ask(
        _graph(seeded, settings, _RebelSupervisor()), ChatRequest(question="¿Qué es BM25?")
    )
    overrides = [s for s in state["supervisor_log"] if s["source"] == "policy_override"]
    assert overrides and overrides[0]["next"] == "tutor"
    assert result.validation is not None


def test_check_citations() -> None:
    ok = TutorAnswer(answer="Algo [1] y [2]", sources=[1, 2], quiz=[])
    assert check_citations(ok, 2) is None
    sin_citas = TutorAnswer(answer="Algo sin fuentes", sources=[], quiz=[])
    report = check_citations(sin_citas, 3)
    assert report is not None and report.verdict == "needs_revision"
    no_sabe = TutorAnswer(answer="No encontré esto en el material.", sources=[], quiz=[])
    assert check_citations(no_sabe, 3) is None
    # Variante real de gpt-4.1-mini (corrida 4): antes forzaba una revisión innecesaria.
    variante = TutorAnswer(
        answer="No encontré información sobre la capital de Australia en el material del curso.",
        sources=[],
        quiz=[],
    )
    assert check_citations(variante, 3) is None
