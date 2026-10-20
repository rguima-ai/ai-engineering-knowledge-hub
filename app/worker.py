"""Worker: consume la cola de Redis y ejecuta el grafo de chat o el de ingesta.

La API nunca corre un LLM dentro del request HTTP (patrón de la Pre-entrega 7): registra el
job, lo encola y devuelve 202. Este proceso hace el trabajo pesado y persiste el resultado.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.agents.backends import create_backend
from app.agents.chat_graph import build_chat_graph, new_turn
from app.agents.ingest_graph import build_ingest_graph
from app.agents.schemas import ChatRequest
from app.config import Settings, get_settings
from app.observability import configure_observability, set_span_attributes
from app.rag import build_rag
from app.rag.uploads import UploadStore
from app.redis_client import create_redis
from app.repositories import ConversationRepository, JobRepository
from app.schemas import JobKind, JobRecord, JobStatus, chat_thread, ingest_thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for noisy in ("httpx", "httpcore", "openai", "pinecone"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("worker")

DEFAULT_TITLE = "Nueva conversación"


def job_input(record: JobRecord) -> str:
    """Texto que Phoenix muestra como entrada del job (columna input y 'first input')."""

    if record.kind is JobKind.CHAT:
        return str(record.payload.get("question", ""))
    filename = record.payload.get("filename", "")
    if record.approval_decision is not None:
        verdict = "aprobado" if record.approval_decision.get("approved") else "rechazado"
        reviewer = record.approval_decision.get("reviewer", "?")
        return f"Revisión de {filename}: {verdict} por {reviewer}"
    return f"Carga de {filename} por {record.user_id}"


class Worker:
    def __init__(
        self,
        jobs: JobRepository,
        conversations: ConversationRepository,
        chat_graph: Any,
        ingest_graph: Any,
        tracer: Tracer,
        settings: Settings,
    ) -> None:
        self.jobs = jobs
        self.conversations = conversations
        self.chat_graph = chat_graph
        self.ingest_graph = ingest_graph
        self.tracer = tracer
        self.settings = settings

    def _config(self, thread_id: str, record: JobRecord) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            # Margen sobre el tope propio del Supervisor: el límite de LangGraph es la red final.
            "recursion_limit": self.settings.max_graph_steps + 6,
            "metadata": {"job_id": record.job_id, "user_id": record.user_id},
            "tags": ["knowledge-hub", record.kind.value, self.settings.agent_mode],
        }

    async def process_job(self, job_id: str) -> None:
        record = await self.jobs.claim(job_id)
        if record is None:
            logger.warning("job=%s ignorado: no existe, venció o no está PENDING", job_id)
            return

        # Mismo valor que el thread_id: OpenInference ya usa ese metadato como sesión.
        session = (
            chat_thread(record.conversation_id) if record.conversation_id else ingest_thread(job_id)
        )
        with (
            self.tracer.start_as_current_span(f"job.{record.kind.value}") as span,
            # Phoenix agrupa por session_id: todos los turnos de una conversación juntos.
            using_attributes(session_id=session, user_id=record.user_id),
        ):
            set_span_attributes(
                span,
                {
                    # Sin esto Phoenix muestra el job como kind "unknown" y sin input/output.
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    SpanAttributes.INPUT_VALUE: job_input(record),
                    "job.id": job_id,
                    "job.kind": record.kind.value,
                    "job.resume": record.approval_decision is not None,
                    "agent_mode": self.settings.agent_mode,
                },
            )
            try:
                if record.kind is JobKind.CHAT:
                    outcome, output = await self._run_chat(record)
                else:
                    outcome, output = await self._run_ingest(record)
                span.set_attribute("job.outcome", outcome)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)
                span.set_status(Status(StatusCode.OK))
                logger.info("job=%s kind=%s outcome=%s", job_id, record.kind.value, outcome)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("job=%s falló", job_id)
                await self.jobs.update(
                    job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            finally:
                if record.kind is JobKind.CHAT and record.conversation_id:
                    await self.conversations.release(record.conversation_id, job_id)

    async def _run_chat(self, record: JobRecord) -> tuple[str, str]:
        assert record.conversation_id is not None
        request = ChatRequest.model_validate(record.payload)
        config = self._config(chat_thread(record.conversation_id), record)
        await self.chat_graph.ainvoke(new_turn(request), config=config)
        state = (await self.chat_graph.aget_state(config)).values
        result = state.get("result")
        if result is None:
            raise RuntimeError("El grafo terminó sin resultado.")

        conversation = await self.conversations.get(record.conversation_id)
        title = None
        if conversation and conversation.title == DEFAULT_TITLE:
            title = request.question[:60]
        await self.conversations.touch(record.conversation_id, title)
        await self.jobs.update(record.job_id, status=JobStatus.DONE, result=result)
        return "done", str(result.get("answer", ""))

    async def _run_ingest(self, record: JobRecord) -> tuple[str, str]:
        config = self._config(ingest_thread(record.job_id), record)
        graph_input: Any
        if record.approval_decision is not None:
            graph_input = Command(resume=record.approval_decision)
        else:
            graph_input = {
                "upload_id": record.job_id,
                "filename": record.payload["filename"],
                "uploaded_by": record.user_id,
                "module_hint": record.payload.get("module"),
            }
        await self.ingest_graph.ainvoke(graph_input, config=config)
        snapshot = await self.ingest_graph.aget_state(config)
        if snapshot.next:
            interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
            await self.jobs.update(
                record.job_id,
                status=JobStatus.WAITING_APPROVAL,
                approval_request=interrupts[0] if interrupts else {},
            )
            analysis = (interrupts[0] if interrupts else {}).get("analysis") or {}
            return "waiting_approval", (
                "Esperando aprobación de un admin. "
                f"Módulo {analysis.get('module')}: {analysis.get('summary', '')}"
            )
        result = snapshot.values.get("result") or {}
        await self.jobs.update(record.job_id, status=JobStatus.DONE, result=result)
        if result.get("status") != "indexed":
            return "done", f"Documento rechazado por {result.get('reviewer')}"
        return "done", (
            f"Indexado {result['source']}: {result['chunks']} chunks, módulo {result['module']}"
        )

    async def run_forever(self) -> None:
        """Procesa hasta WORKER_CONCURRENCY jobs a la vez en este proceso.

        Todo el trabajo es espera de red (LLM, Pinecone, Redis), así que un solo event loop
        atiende varios turnos en paralelo. El semáforo se toma antes del BLPOP: el worker no
        saca de la cola más jobs de los que puede procesar, y el resto queda para otra réplica.
        """

        logger.info(
            "worker listo; esperando jobs en %s concurrency=%s",
            self.jobs.queue_name,
            self.settings.worker_concurrency,
        )
        slots = asyncio.Semaphore(self.settings.worker_concurrency)
        running: set[asyncio.Task[None]] = set()
        try:
            while True:
                await slots.acquire()
                try:
                    job_id = await self.jobs.dequeue(self.settings.worker_poll_timeout_seconds)
                except (RedisConnectionError, RedisTimeoutError):
                    slots.release()
                    logger.warning("Redis no disponible; reintento en 1 s")
                    await asyncio.sleep(1)
                    continue
                if job_id is None:
                    slots.release()
                    continue
                task = asyncio.create_task(self._process_safely(job_id))
                running.add(task)
                task.add_done_callback(running.discard)
                task.add_done_callback(lambda _: slots.release())
        finally:
            # SIGTERM: los jobs en curso se cancelan; el checkpoint conserva el último paso.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def _process_safely(self, job_id: str) -> None:
        try:
            await self.process_job(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Ej.: Redis cae mientras se guarda FAILED. El loop sigue vivo.
            logger.exception("error no recuperado con el job %s", job_id)


async def main() -> None:
    settings = get_settings()
    settings.validate_runtime()
    tracer = configure_observability(settings)
    # redis-py 8 usa socket_timeout=5 s: un BLPOP de 5 s se cortaría del lado del cliente.
    redis = create_redis(
        settings.redis_url, socket_timeout=settings.worker_poll_timeout_seconds + 5
    )
    rag = await build_rag(settings, redis)
    jobs = JobRepository(
        redis,
        prefix=settings.key_prefix,
        queue_name=settings.queue_name,
        ttl_seconds=settings.job_ttl_seconds,
    )
    conversations = ConversationRepository(
        redis, prefix=settings.key_prefix, lock_seconds=settings.conversation_lock_seconds
    )
    uploads = UploadStore(redis, settings.key_prefix, settings.job_ttl_seconds)
    backend = create_backend(settings)
    try:
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
            await checkpointer.asetup()
            chat_graph = build_chat_graph(backend, rag.retriever, settings, checkpointer)
            ingest_graph = build_ingest_graph(
                backend, rag.ingestion, uploads, rag.store, rag.vectors, rag.embedder, checkpointer
            )
            worker = Worker(jobs, conversations, chat_graph, ingest_graph, tracer, settings)
            await _run_until_signal(worker)
    finally:
        # El exportador manda las trazas en lotes cada pocos segundos: sin este flush, las
        # del último job antes de un `docker compose restart` se perdían.
        flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if flush is not None:
            flush()
        await rag.close()
        await redis.aclose()


async def _run_until_signal(worker: Worker) -> None:
    """SIGTERM (docker stop/restart) cancela el loop en lugar de matar el proceso de golpe.

    Si llega en medio de un job, ese job queda RUNNING; el checkpoint conserva el último paso
    completo del grafo.
    """

    task = asyncio.create_task(worker.run_forever())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        logger.info("worker detenido por señal; cerrando ordenadamente")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
