"""Integración con el ChatOpenAI real: solo se reemplaza la llamada HTTP (_agenerate).

Los fakes de los otros tests codifican nuestras suposiciones; acá se ejercita el camino real
de with_structured_output (armado del response_format y parseo de la respuesta).
"""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.backends import OpenAIBackend, format_history
from app.agents.chat_graph import build_chat_graph, new_turn
from app.agents.schemas import ChatRequest
from app.agents.schemas import ChatResult as TurnResult
from app.config import Settings
from app.rag import RAGComponents
from tests.test_chat_graph import CODE_WITH_BUGS, _seed

RESPONSES: dict[str, dict[str, Any]] = {
    "RequestPlan": {
        "intent": "Revisar un agente y explicar errores",
        "search_queries": ["checkpointer persistente AsyncRedisSaver"],
        "modules": [5],
        "wants_quiz": True,
        "quiz_count": 1,
    },
    "CodeReview": {
        "summary": "Usa MemorySaver.",
        "issues": [
            {
                "severity": "media",
                "line": 7,
                "problem": "MemorySaver no persiste.",
                "fix": "Usar AsyncRedisSaver.",
                "sources": [1],
            }
        ],
        "strengths": [],
    },
    "TutorAnswer": {
        "answer": "AsyncRedisSaver guarda el estado por thread_id [1].",
        "sources": [1],
        "quiz": [
            {
                "question": "¿Qué guarda?",
                "options": [],
                "answer": "El estado",
                "explanation": "Fragmento [1].",
            }
        ],
    },
    "ValidationReport": {"verdict": "approved", "unsupported_claims": [], "feedback": ""},
}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    responses = dict(RESPONSES)

    async def fake_agenerate(
        self: ChatOpenAI, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        schema = kwargs["response_format"]
        name = schema.__name__ if isinstance(schema, type) else schema["json_schema"]["name"]
        seen.append(name)
        if name == "SupervisorDecision":
            # Elige la primera opción permitida que el Supervisor le pasa en el prompt.
            allowed = json.loads(messages[-1].content)["opciones_permitidas"]
            responses[name] = {"next": allowed[0], "reason": f"Toca {allowed[0]}."}
        # Igual que ChatOpenAI con response_format: el SDK (.parse) deja el objeto en "parsed".
        message = AIMessage(
            content=json.dumps(responses[name]),
            additional_kwargs={"parsed": schema.model_validate(responses[name])},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake_agenerate)
    return seen


async def test_grafo_completo_con_el_backend_openai(
    rag: RAGComponents, settings: Settings, calls: list[str]
) -> None:
    await _seed(rag)
    openai_settings = settings.model_copy(update={"openai_api_key": "sk-test"})
    graph = build_chat_graph(
        OpenAIBackend(openai_settings), rag.retriever, settings, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "x"}, "recursion_limit": 50}
    request = ChatRequest(question="Revisá mi agente", code=CODE_WITH_BUGS, level="principiante")
    await graph.ainvoke(new_turn(request), config=config)
    result = TurnResult.model_validate((await graph.aget_state(config)).values["result"])

    # El Supervisor consulta al LLM solo cuando la política deja más de una opción.
    assert calls == [
        "RequestPlan",
        "SupervisorDecision",
        "CodeReview",
        "SupervisorDecision",
        "TutorAnswer",
        "ValidationReport",
    ]
    assert result.agents_used == ["retriever", "code_reviewer", "tutor", "validator"]
    assert result.citations[0].module == 5
    assert result.quiz[0].answer == "El estado"
    assert result.code_review and result.code_review.issues[0].line == 7


def test_history_keeps_a_full_long_answer() -> None:
    """Regresión: una respuesta de ~4 000 caracteres tiene que llegar entera al próximo turno."""

    answer = "1. estado sin reducer\n" + "x" * 4_000 + "\n5. falta el checkpointer"
    history = format_history([HumanMessage("Revisá mi agente"), AIMessage(answer)])

    assert "5. falta el checkpointer" in history
