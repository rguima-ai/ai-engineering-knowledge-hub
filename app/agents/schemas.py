"""Contratos Pydantic de los agentes: lo que entra al grafo y lo que devuelve cada especialista.

Los modelos que produce un LLM (with_structured_output) no tienen defaults: el modo
json_schema estricto de OpenAI exige que todos los campos sean requeridos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["principiante", "intermedio", "avanzado"]
Route = Literal["retriever", "code_reviewer", "tutor", "validator", "FINISH"]


# ---------- entrada del chat ----------
class ChatRequest(BaseModel):
    """Pedido de un usuario en un turno de la conversación."""

    question: str = Field(min_length=3, max_length=4_000)
    code: str | None = Field(
        default=None, max_length=30_000, description="Código del alumno a revisar (opcional)"
    )
    level: Level = "intermedio"
    modules: list[int] | None = Field(
        default=None, max_length=20, description="Restringe la búsqueda a estos módulos"
    )


# ---------- Supervisor ----------
class RequestPlan(BaseModel):
    """Cómo interpreta el Supervisor el pedido. Se arma una vez por turno."""

    intent: str = Field(description="Qué quiere el usuario, en una oración")
    search_queries: list[str] = Field(
        min_length=1, max_length=3, description="Búsquedas autocontenidas para el material"
    )
    modules: list[int] = Field(
        description="Módulos del curso relevantes (1-7); vacío si no se puede inferir"
    )
    wants_quiz: bool
    quiz_count: int = Field(ge=0, le=10)


class SupervisorDecision(BaseModel):
    """Elección del próximo paso entre las opciones que la política permite."""

    next: Route
    reason: str = Field(description="Por qué ese paso, en una oración")


class RefinedQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


# ---------- especialistas ----------
class CodeIssue(BaseModel):
    severity: Literal["alta", "media", "baja"]
    line: int | None = Field(description="Línea aproximada del problema, si aplica")
    problem: str
    fix: str
    sources: list[int] = Field(description="Números de fragmento [n] que respaldan la corrección")


class CodeReview(BaseModel):
    summary: str
    issues: list[CodeIssue]
    strengths: list[str]


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="2 a 4 opciones; vacío si la pregunta es abierta")
    answer: str
    explanation: str


class TutorAnswer(BaseModel):
    answer: str = Field(description="Respuesta en Markdown con citas [n] al material")
    sources: list[int] = Field(description="Números de fragmento [n] citados en la respuesta")
    quiz: list[QuizQuestion]


class ValidationReport(BaseModel):
    verdict: Literal["approved", "needs_revision"]
    unsupported_claims: list[str] = Field(
        description="Afirmaciones de la respuesta que ningún fragmento respalda"
    )
    feedback: str = Field(description="Qué tiene que corregir el Tutor; vacío si se aprueba")


# ---------- salida del chat ----------
class Citation(BaseModel):
    number: int
    chunk_id: str
    source: str
    title: str
    module: int
    snippet: str


class ChatResult(BaseModel):
    """Respuesta final del turno, la que devuelve la API."""

    answer: str
    citations: list[Citation]
    quiz: list[QuizQuestion]
    code_review: CodeReview | None
    validation: ValidationReport | None
    agents_used: list[str]
    revisions: int


# ---------- ingesta con aprobación humana ----------
class DocumentAnalysis(BaseModel):
    """Lo que el agente de ingesta le muestra al revisor humano antes de indexar."""

    module: int = Field(ge=0, le=20, description="Módulo del curso; 0 si es general")
    topic: str = Field(max_length=120)
    summary: str
    difficulty: Literal["principiante", "intermedio", "avanzado"]
    quality_issues: list[str] = Field(
        description="Problemas que el revisor debería mirar (texto roto, fuera de tema, etc.)"
    )


class DuplicateCandidate(BaseModel):
    doc_id: str
    source: str
    similarity: float


class HumanDecision(BaseModel):
    """Decisión del revisor; puede corregir el módulo y el tema detectados."""

    approved: bool
    reviewer: str = Field(min_length=2, max_length=80)
    comment: str = Field(default="", max_length=500)
    module: int | None = Field(default=None, ge=0, le=20)
    topic: str | None = Field(default=None, max_length=120)
