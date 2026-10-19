"""Contratos Pydantic de la API y de los registros que se guardan en Redis."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.models import DocType

USER_ID_PATTERN = r"^[A-Za-z0-9_.@-]{2,60}$"


def chat_thread(conversation_id: str) -> str:
    """thread_id del checkpointer para una conversación."""

    return f"chat:{conversation_id}"


def ingest_thread(job_id: str) -> str:
    return f"ingest:{job_id}"


def _now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


class JobKind(StrEnum):
    CHAT = "chat"
    INGEST = "ingest"


class JobRecord(BaseModel):
    """Trabajo persistido en Redis. `payload` es el ChatRequest o los datos del upload."""

    job_id: str
    kind: JobKind
    user_id: str
    conversation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    approval_request: dict[str, Any] | None = None
    approval_decision: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ConversationCreate(BaseModel):
    title: str = Field(default="Nueva conversación", min_length=1, max_length=120)


class ChatMessageView(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationDetail(Conversation):
    """Los mensajes salen del checkpointer de LangGraph, no de una copia aparte."""

    messages: list[ChatMessageView]


class DocumentView(BaseModel):
    doc_id: str
    source: str
    title: str
    doc_type: DocType
    module: int
    topic: str
    chunks: int
    uploaded_by: str
    approved_by: str
    indexed_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: Literal["connected", "unavailable"]
    agent_mode: str
    vector_store: str
    embedding_provider: str
    documents: int
