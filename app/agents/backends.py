"""Backends de los agentes: OpenAI real y una simulación determinística sin costo.

El grafo solo conoce el Protocol AgentBackend. Así los tests y la demo de infraestructura
recorren exactamente los mismos nodos, aristas y checkpoints que la versión real.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.agents import prompts
from app.agents.schemas import (
    ChatRequest,
    CodeIssue,
    CodeReview,
    DocumentAnalysis,
    QuizQuestion,
    RefinedQueries,
    RequestPlan,
    Route,
    SupervisorDecision,
    TutorAnswer,
    ValidationReport,
)
from app.config import Settings
from app.rag.loaders import infer_module
from app.rag.models import RetrievedChunk


class AgentBackend(Protocol):
    async def plan(self, request: ChatRequest, history: Sequence[BaseMessage]) -> RequestPlan: ...

    async def route(self, situation: str, allowed: list[Route]) -> SupervisorDecision: ...

    async def refine_queries(
        self, request: ChatRequest, previous: list[str], feedback: str
    ) -> RefinedQueries: ...

    async def review_code(
        self, request: ChatRequest, context: list[RetrievedChunk]
    ) -> CodeReview: ...

    async def tutor(
        self,
        request: ChatRequest,
        plan: RequestPlan,
        context: list[RetrievedChunk],
        review: CodeReview | None,
        feedback: str,
        history: Sequence[BaseMessage],
    ) -> TutorAnswer: ...

    async def validate(
        self, request: ChatRequest, draft: TutorAnswer, context: list[RetrievedChunk]
    ) -> ValidationReport: ...

    async def analyze_document(
        self, filename: str, excerpt: str, module_hint: int | None
    ) -> DocumentAnalysis: ...


def format_context(context: list[RetrievedChunk]) -> str:
    """Fragmentos numerados [1..n]: el LLM cita números, el código los mapea a chunk_ids."""

    if not context:
        return "(no se encontraron fragmentos)"
    return "\n\n".join(
        f"[{n}] módulo {c.module} · {c.source}\n{c.text}" for n, c in enumerate(context, start=1)
    )


# Una respuesta con revisión de código y cuestionario ronda los 4 000 caracteres. Con 1 500 el
# historial cortaba la mitad de la respuesta anterior y la repregunta "recordame los errores"
# perdía los últimos (visto en la corrida real 6).
HISTORY_CHARS_PER_MESSAGE = 6_000


def format_history(history: Sequence[BaseMessage], limit: int = 6) -> str:
    recent = list(history)[-limit:]
    if not recent:
        return "(primer mensaje de la conversación)"
    lines = []
    for message in recent:
        role = "usuario" if isinstance(message, HumanMessage) else "tutor"
        lines.append(f"{role}: {str(message.content)[:HISTORY_CHARS_PER_MESSAGE]}")
    return "\n".join(lines)


T = TypeVar("T", bound=BaseModel)


class OpenAIBackend:
    """Cada agente es el mismo modelo con otro prompt y otro esquema de salida."""

    def __init__(self, settings: Settings) -> None:
        self._model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=SecretStr(settings.openai_api_key),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def _ask(self, schema: type[T], system: str, payload: dict[str, Any], name: str) -> T:
        runnable = self._model.with_structured_output(schema).with_config(run_name=name)
        result = await runnable.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=1)),
            ]
        )
        return schema.model_validate(result)

    async def plan(self, request: ChatRequest, history: Sequence[BaseMessage]) -> RequestPlan:
        return await self._ask(
            RequestPlan,
            prompts.SUPERVISOR_PLAN,
            {
                "conversacion_previa": format_history(history),
                "pedido": request.question,
                "trae_codigo": request.code is not None,
                "modulos_elegidos_por_el_usuario": request.modules,
            },
            "supervisor.plan",
        )

    async def route(self, situation: str, allowed: list[Route]) -> SupervisorDecision:
        return await self._ask(
            SupervisorDecision,
            prompts.SUPERVISOR_ROUTE,
            {"situacion": situation, "opciones_permitidas": allowed},
            "supervisor.route",
        )

    async def refine_queries(
        self, request: ChatRequest, previous: list[str], feedback: str
    ) -> RefinedQueries:
        return await self._ask(
            RefinedQueries,
            prompts.REFINE_QUERIES,
            {"pedido": request.question, "busquedas_previas": previous, "feedback": feedback},
            "retriever.refine",
        )

    async def review_code(self, request: ChatRequest, context: list[RetrievedChunk]) -> CodeReview:
        return await self._ask(
            CodeReview,
            prompts.CODE_REVIEWER,
            {
                "pedido": request.question,
                "codigo_del_alumno": request.code,
                "fragmentos": format_context(context),
            },
            "code_reviewer",
        )

    async def tutor(
        self,
        request: ChatRequest,
        plan: RequestPlan,
        context: list[RetrievedChunk],
        review: CodeReview | None,
        feedback: str,
        history: Sequence[BaseMessage],
    ) -> TutorAnswer:
        return await self._ask(
            TutorAnswer,
            prompts.TUTOR,
            {
                "conversacion_previa": format_history(history),
                "pedido": request.question,
                "nivel": request.level,
                "quiz_count": plan.quiz_count if plan.wants_quiz else 0,
                "revision_de_codigo": review.model_dump() if review else None,
                "feedback_del_validador": feedback or None,
                "fragmentos": format_context(context),
            },
            "tutor",
        )

    async def validate(
        self, request: ChatRequest, draft: TutorAnswer, context: list[RetrievedChunk]
    ) -> ValidationReport:
        return await self._ask(
            ValidationReport,
            prompts.VALIDATOR,
            {
                "pedido": request.question,
                "respuesta": draft.answer,
                "fragmentos": format_context(context),
            },
            "validator",
        )

    async def analyze_document(
        self, filename: str, excerpt: str, module_hint: int | None
    ) -> DocumentAnalysis:
        return await self._ask(
            DocumentAnalysis,
            prompts.DOCUMENT_ANALYST,
            {"archivo": filename, "modulo_sugerido": module_hint, "comienzo": excerpt},
            "document_analyst",
        )


# ---------------------------------------------------------------------------
# Simulación determinística
# ---------------------------------------------------------------------------

_QUIZ_WORDS = ("pregunta", "cuestionario", "quiz", "ejercicio", "evaluame", "evaluá")
_QUIZ_COUNT = re.compile(r"(\d{1,2})\s+(?:preguntas|ejercicios)")
# (patrón, severidad, problema, corrección): errores típicos que marca el curso.
_CODE_RULES = [
    (
        r"\brequests\.(get|post)\(",
        "alta",
        "Llamada HTTP bloqueante (requests) que congela el event loop.",
        "Usar httpx.AsyncClient o el SDK async oficial con await.",
    ),
    (
        r"\btime\.sleep\(",
        "alta",
        "time.sleep bloquea el event loop.",
        "Usar await asyncio.sleep(...).",
    ),
    (
        r"\bMemorySaver\(",
        "media",
        "MemorySaver pierde las conversaciones al reiniciar el proceso.",
        "Usar un checkpointer persistente (AsyncSqliteSaver o AsyncRedisSaver).",
    ),
    (
        r"\.invoke\(",
        "baja",
        "Se usa invoke síncrono.",
        "Dentro de código async usar ainvoke.",
    ),
]


def _first_sentence(text: str, limit: int = 220) -> str:
    clean = " ".join(line for line in text.splitlines() if not line.startswith("#")).strip()
    sentence = re.split(r"(?<=[.!?])\s", clean, maxsplit=1)[0]
    return sentence[:limit].rstrip()


class SimulatedBackend:
    """Reglas simples en lugar de un LLM: reproducible, gratis y sin red."""

    async def plan(self, request: ChatRequest, history: Sequence[BaseMessage]) -> RequestPlan:
        lowered = request.question.lower()
        wants_quiz = any(word in lowered for word in _QUIZ_WORDS)
        count = _QUIZ_COUNT.search(lowered)
        return RequestPlan(
            intent=f"Responder: {request.question[:120]}",
            search_queries=[request.question[:300]],
            modules=request.modules or [],
            wants_quiz=wants_quiz,
            quiz_count=min(int(count.group(1)), 10) if count else (3 if wants_quiz else 0),
        )

    async def route(self, situation: str, allowed: list[Route]) -> SupervisorDecision:
        return SupervisorDecision(next=allowed[0], reason="Política por defecto (simulación).")

    async def refine_queries(
        self, request: ChatRequest, previous: list[str], feedback: str
    ) -> RefinedQueries:
        return RefinedQueries(queries=[f"{request.question[:200]} {feedback[:80]}".strip()])

    async def review_code(self, request: ChatRequest, context: list[RetrievedChunk]) -> CodeReview:
        code = request.code or ""
        issues: list[CodeIssue] = []
        for pattern, severity, problem, fix in _CODE_RULES:
            for number, line in enumerate(code.splitlines(), start=1):
                if re.search(pattern, line):
                    issues.append(
                        CodeIssue(
                            severity=severity,  # type: ignore[arg-type]
                            line=number,
                            problem=problem,
                            fix=fix,
                            sources=[1] if context else [],
                        )
                    )
                    break
        strengths = ["Usa async def."] if "async def" in code else []
        return CodeReview(
            summary=f"Se encontraron {len(issues)} problema(s) en el código.",
            issues=issues,
            strengths=strengths,
        )

    async def tutor(
        self,
        request: ChatRequest,
        plan: RequestPlan,
        context: list[RetrievedChunk],
        review: CodeReview | None,
        feedback: str,
        history: Sequence[BaseMessage],
    ) -> TutorAnswer:
        if not context:
            return TutorAnswer(
                answer="No encontré esto en el material del curso.", sources=[], quiz=[]
            )
        top = context[:3]
        intro = {
            "principiante": "Te lo explico paso a paso.",
            "intermedio": "Resumen de lo que dice el material:",
            "avanzado": "Detalle técnico según el material:",
        }[request.level]
        lines = [intro, ""]
        lines += [f"- {_first_sentence(c.text)} [{n}]" for n, c in enumerate(top, start=1)]
        if review and review.issues:
            lines += ["", "**Correcciones a tu código:**"]
            lines += [f"- Línea {i.line}: {i.problem} {i.fix} [1]" for i in review.issues]
        quiz = (
            [
                QuizQuestion(
                    question=f"¿Qué explica el material sobre «{c.title}»?",
                    options=[],
                    answer=_first_sentence(c.text, 160),
                    explanation=f"Sale del fragmento [{n}] ({c.source}).",
                )
                for n, c in enumerate(context[: plan.quiz_count], start=1)
            ]
            if plan.wants_quiz
            else []
        )
        return TutorAnswer(answer="\n".join(lines), sources=list(range(1, len(top) + 1)), quiz=quiz)

    async def validate(
        self, request: ChatRequest, draft: TutorAnswer, context: list[RetrievedChunk]
    ) -> ValidationReport:
        return ValidationReport(verdict="approved", unsupported_claims=[], feedback="")

    async def analyze_document(
        self, filename: str, excerpt: str, module_hint: int | None
    ) -> DocumentAnalysis:
        issues = ["El documento es muy corto."] if len(excerpt) < 200 else []
        if re.search(r"\bsk-[A-Za-z0-9]{10,}|pcsk_", excerpt):
            issues.append("Parece contener una API key.")
        return DocumentAnalysis(
            module=module_hint if module_hint is not None else infer_module(filename),
            topic=_first_sentence(excerpt, 80) or filename,
            summary=_first_sentence(excerpt, 300),
            difficulty="intermedio",
            quality_issues=issues,
        )


def create_backend(settings: Settings) -> AgentBackend:
    settings.validate_runtime()
    if settings.agent_mode == "openai":
        return OpenAIBackend(settings)
    return SimulatedBackend()
