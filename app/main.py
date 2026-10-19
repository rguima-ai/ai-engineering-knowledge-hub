"""API FastAPI: valida, persiste y encola. Nunca ejecuta un LLM dentro del request."""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import PurePath
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.agents.schemas import ChatRequest
from app.config import Settings, get_settings
from app.rag.chunk_store import ChunkStore
from app.rag.loaders import UnsupportedFileError, doc_type_for
from app.rag.uploads import UploadStore
from app.redis_client import create_redis
from app.repositories import (
    ConversationBusyError,
    ConversationRepository,
    InvalidJobTransitionError,
    JobRepository,
)
from app.schemas import (
    USER_ID_PATTERN,
    ChatMessageView,
    Conversation,
    ConversationCreate,
    ConversationDetail,
    DocumentView,
    HealthResponse,
    JobCreated,
    JobKind,
    JobRecord,
    JobStatus,
    chat_thread,
)

RedisFactory = Callable[[str], Redis]
CheckpointerFactory = Callable[[Settings], AbstractAsyncContextManager[BaseCheckpointSaver[Any]]]

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class ReviewDecision(BaseModel):
    """Lo que manda el revisor; su identidad sale del header, no del body."""

    approved: bool
    comment: str = Field(default="", max_length=500)
    module: int | None = Field(default=None, ge=0, le=20)
    topic: str | None = Field(default=None, max_length=120)


def default_redis_factory(url: str) -> Redis:
    return create_redis(url)


@asynccontextmanager
async def redis_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as saver:
        await saver.asetup()
        yield saver


class Services:
    """Dependencias armadas en el lifespan, una sola vez por proceso."""

    def __init__(self, settings: Settings, redis: Redis, checkpointer: BaseCheckpointSaver[Any]):
        self.settings = settings
        self.redis = redis
        self.checkpointer = checkpointer
        self.jobs = JobRepository(
            redis,
            prefix=settings.key_prefix,
            queue_name=settings.queue_name,
            ttl_seconds=settings.job_ttl_seconds,
        )
        self.conversations = ConversationRepository(
            redis, prefix=settings.key_prefix, lock_seconds=settings.conversation_lock_seconds
        )
        self.documents = ChunkStore(redis, settings.key_prefix)
        self.uploads = UploadStore(redis, settings.key_prefix, settings.job_ttl_seconds)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]
UserId = Annotated[
    str,
    Header(
        alias="X-User-Id",
        pattern=USER_ID_PATTERN,
        description="Identificador del integrante del equipo (sin autenticación: ver README).",
    ),
]


def safe_filename(name: str | None) -> str:
    base = PurePath(name or "").name
    cleaned = _UNSAFE_FILENAME.sub("_", base).strip("._")
    if not cleaned:
        raise HTTPException(status_code=422, detail="Nombre de archivo inválido.")
    return cleaned[:120]


def create_app(
    settings: Settings | None = None,
    redis_factory: RedisFactory = default_redis_factory,
    checkpointer_factory: CheckpointerFactory = redis_checkpointer,
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = redis_factory(resolved.redis_url)
        try:
            async with checkpointer_factory(resolved) as checkpointer:
                app.state.services = Services(resolved, redis, checkpointer)
                # Sin Phoenix en la API: no llama a LLMs y el polling taparía las trazas útiles.
                yield
        finally:
            await redis.aclose()

    app = FastAPI(
        title="AI Engineering Knowledge Hub",
        version="1.0.0",
        description=(
            "Tutor multi-agente sobre el material del curso. Las preguntas y los documentos "
            "se procesan en segundo plano: cada POST devuelve un `job_id` para consultar en "
            "`GET /jobs/{job_id}`."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RedisConnectionError)
    @app.exception_handler(RedisTimeoutError)
    async def redis_unavailable(request: Request, exc: Exception) -> JSONResponse:
        """Redis caído es transitorio: 503 + Retry-After le dice al cliente que reintente."""

        return JSONResponse(
            status_code=503,
            content={"detail": "Almacenamiento no disponible momentáneamente. Reintentá."},
            headers={"Retry-After": "5"},
        )

    # ---------------- operación ----------------
    @app.get("/health", response_model=HealthResponse, tags=["operación"])
    async def health(services: ServicesDep) -> HealthResponse:
        try:
            redis_ok = await services.jobs.ping()
            documents = len(await services.documents.list_documents())
        except Exception:
            redis_ok, documents = False, 0
        return HealthResponse(
            status="ok" if redis_ok else "degraded",
            redis="connected" if redis_ok else "unavailable",
            agent_mode=resolved.agent_mode,
            vector_store=resolved.vector_store,
            embedding_provider=resolved.embedding_provider,
            documents=documents,
        )

    # ---------------- conversaciones ----------------
    @app.post(
        "/conversations",
        response_model=Conversation,
        status_code=status.HTTP_201_CREATED,
        tags=["conversaciones"],
    )
    async def create_conversation(
        payload: ConversationCreate, user_id: UserId, services: ServicesDep
    ) -> Conversation:
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()), user_id=user_id, title=payload.title
        )
        await services.conversations.create(conversation)
        return conversation

    @app.get("/conversations", response_model=list[Conversation], tags=["conversaciones"])
    async def list_conversations(user_id: UserId, services: ServicesDep) -> list[Conversation]:
        return await services.conversations.list_for_user(user_id)

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationDetail,
        tags=["conversaciones"],
    )
    async def get_conversation(
        conversation_id: str, user_id: UserId, services: ServicesDep
    ) -> ConversationDetail:
        conversation = await services.conversations.get_owned(conversation_id, user_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")
        snapshot = await services.checkpointer.aget_tuple(
            {"configurable": {"thread_id": chat_thread(conversation_id), "checkpoint_ns": ""}}
        )
        raw = snapshot.checkpoint["channel_values"].get("messages", []) if snapshot else []
        messages = [
            ChatMessageView(
                role="user" if isinstance(m, HumanMessage) else "assistant",
                content=str(m.content),
            )
            for m in raw
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        return ConversationDetail(**conversation.model_dump(), messages=messages)

    @app.post(
        "/conversations/{conversation_id}/messages",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["conversaciones"],
        responses={409: {"description": "La conversación ya tiene un turno en curso."}},
    )
    async def send_message(
        conversation_id: str, payload: ChatRequest, user_id: UserId, services: ServicesDep
    ) -> JobCreated:
        if await services.conversations.get_owned(conversation_id, user_id) is None:
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")
        job_id = str(uuid.uuid4())
        try:
            await services.conversations.acquire(conversation_id, job_id)
        except ConversationBusyError as exc:
            raise HTTPException(
                status_code=409, detail="Esperá a que termine la respuesta anterior."
            ) from exc
        record = JobRecord(
            job_id=job_id,
            kind=JobKind.CHAT,
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload.model_dump(mode="json"),
        )
        try:
            await services.jobs.create_and_enqueue(record)
        except Exception:
            await services.conversations.release(conversation_id, job_id)
            raise
        return JobCreated(job_id=job_id, status=record.status, status_url=f"/jobs/{job_id}")

    # ---------------- jobs ----------------
    @app.get("/jobs/{job_id}", response_model=JobRecord, tags=["jobs"])
    async def get_job(job_id: str, user_id: UserId, services: ServicesDep) -> JobRecord:
        record = await services.jobs.get(job_id)
        if record is None or (record.user_id != user_id and user_id not in resolved.admins):
            raise HTTPException(status_code=404, detail="Job no encontrado.")
        return record

    # ---------------- documentos ----------------
    @app.get("/documents", response_model=list[DocumentView], tags=["documentos"])
    async def list_documents(services: ServicesDep) -> list[DocumentView]:
        return [
            DocumentView(
                **r.metadata.model_dump(
                    include={
                        "doc_id",
                        "source",
                        "title",
                        "doc_type",
                        "module",
                        "topic",
                        "uploaded_by",
                    }
                ),
                chunks=len(r.chunk_ids),
                approved_by=r.approved_by,
                indexed_at=r.indexed_at,
            )
            for r in await services.documents.list_documents()
        ]

    @app.post(
        "/documents",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documentos"],
        responses={
            413: {"description": "Archivo demasiado grande."},
            415: {"description": "Tipo de archivo no soportado."},
        },
    )
    async def upload_document(
        user_id: UserId,
        services: ServicesDep,
        file: Annotated[UploadFile, File(description="PDF, HTML, MD, PY, IPYNB o TXT")],
        module: Annotated[int | None, Form(ge=0, le=20)] = None,
    ) -> JobCreated:
        filename = safe_filename(file.filename)
        try:
            doc_type_for(filename)
        except UnsupportedFileError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        # Se lee como máximo un byte de más: alcanza para saber si se pasó del límite.
        data = await file.read(resolved.max_upload_bytes + 1)
        if len(data) > resolved.max_upload_bytes:
            raise HTTPException(
                status_code=413, detail=f"El máximo es {resolved.max_upload_mb} MB."
            )
        if not data:
            raise HTTPException(status_code=422, detail="El archivo está vacío.")

        job_id = str(uuid.uuid4())
        await services.uploads.save(job_id, data)
        record = JobRecord(
            job_id=job_id,
            kind=JobKind.INGEST,
            user_id=user_id,
            payload={"filename": filename, "module": module, "bytes": len(data)},
        )
        await services.jobs.create_and_enqueue(record)
        return JobCreated(job_id=job_id, status=record.status, status_url=f"/jobs/{job_id}")

    @app.get("/documents/reviews", response_model=list[JobRecord], tags=["documentos"])
    async def pending_reviews(user_id: UserId, services: ServicesDep) -> list[JobRecord]:
        if user_id not in resolved.admins:
            raise HTTPException(status_code=403, detail="Solo un admin revisa documentos.")
        return await services.jobs.pending_reviews()

    @app.post(
        "/documents/reviews/{job_id}",
        response_model=JobRecord,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documentos"],
    )
    async def review_document(
        job_id: str, decision: ReviewDecision, user_id: UserId, services: ServicesDep
    ) -> JobRecord:
        if user_id not in resolved.admins:
            raise HTTPException(status_code=403, detail="Solo un admin revisa documentos.")
        record = await services.jobs.get(job_id)
        if record is None or record.kind is not JobKind.INGEST:
            raise HTTPException(status_code=404, detail="Documento no encontrado.")
        if record.status is not JobStatus.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"El documento está en {record.status}.")
        try:
            return await services.jobs.submit_approval(
                job_id, {**decision.model_dump(mode="json"), "reviewer": user_id}
            )
        except InvalidJobTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
