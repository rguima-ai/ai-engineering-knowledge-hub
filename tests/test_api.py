"""Circuito completo API → Redis → worker → grafo → API, sin red.

La API corre en el TestClient (su propio event loop) y el worker en el loop del test; los dos
hablan con el mismo FakeServer de Redis y comparten el checkpointer, como en Docker comparten
el Redis real.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.agents.backends import SimulatedBackend
from app.agents.chat_graph import build_chat_graph
from app.agents.ingest_graph import build_ingest_graph
from app.config import Settings
from app.main import create_app
from app.rag import RAGComponents, build_rag
from app.rag.uploads import UploadStore
from app.repositories import ConversationRepository, JobRepository
from app.worker import Worker
from tests.test_chat_graph import CODE_WITH_BUGS, _seed

USER = {"X-User-Id": "ana"}
ADMIN = {"X-User-Id": "admin"}


class Harness:
    def __init__(self, client: TestClient, worker: Worker, rag: RAGComponents) -> None:
        self.client = client
        self.worker = worker
        self.rag = rag

    async def drain(self) -> None:
        """Procesa todo lo encolado, como haría el worker en su loop."""

        jobs = self.worker.jobs
        while await jobs.redis.llen(jobs.queue_name):  # type: ignore[misc]
            job_id = await jobs.dequeue(1)
            assert job_id is not None
            await self.worker.process_job(job_id)

    def ask(self, conversation_id: str, headers: dict[str, str] = USER, **body: Any) -> Any:
        return self.client.post(
            f"/conversations/{conversation_id}/messages", json=body, headers=headers
        )


@pytest.fixture
async def harness(settings: Settings) -> AsyncIterator[Harness]:
    server = FakeServer()
    checkpointer = InMemorySaver()

    @asynccontextmanager
    async def shared_checkpointer(_: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
        yield checkpointer

    app = create_app(
        settings,
        redis_factory=lambda url: FakeAsyncRedis(server=server, decode_responses=True),
        checkpointer_factory=shared_checkpointer,
    )
    redis = FakeAsyncRedis(server=server, decode_responses=True)
    rag = await build_rag(settings, redis)
    await _seed(rag)
    backend = SimulatedBackend()
    worker = Worker(
        JobRepository(
            redis,
            prefix=settings.key_prefix,
            queue_name=settings.queue_name,
            ttl_seconds=settings.job_ttl_seconds,
        ),
        ConversationRepository(redis, prefix=settings.key_prefix, lock_seconds=60),
        build_chat_graph(backend, rag.retriever, settings, checkpointer),
        build_ingest_graph(
            backend,
            rag.ingestion,
            UploadStore(redis),
            rag.store,
            rag.vectors,
            rag.embedder,
            checkpointer,
        ),
        trace.get_tracer("test"),
        settings,
    )
    with TestClient(app) as client:
        yield Harness(client, worker, rag)
    await redis.aclose()


@pytest.fixture
def conversation(harness: Harness) -> Iterator[str]:
    response = harness.client.post("/conversations", json={}, headers=USER)
    assert response.status_code == 201
    yield response.json()["conversation_id"]


def test_health(harness: Harness) -> None:
    body = harness.client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 3


def test_interfaz_web_se_sirve_en_la_raiz(harness: Harness) -> None:
    response = harness.client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Knowledge Hub</title>" in response.text
    # Es un cliente de la API, no una ruta documentada de la API.
    assert "/" not in harness.client.get("/openapi.json").json()["paths"]


async def test_pregunta_completa_y_historial(harness: Harness, conversation: str) -> None:
    created = harness.ask(conversation, question="¿Qué hace AsyncRedisSaver?", level="avanzado")
    assert created.status_code == 202
    job_url = created.json()["status_url"]
    assert harness.client.get(job_url, headers=USER).json()["status"] == "PENDING"

    await harness.drain()
    job = harness.client.get(job_url, headers=USER).json()
    assert job["status"] == "DONE", job["error"]
    assert job["result"]["citations"][0]["source"] == "modulo_05_langgraph/checkpointer.md"

    detail = harness.client.get(f"/conversations/{conversation}", headers=USER).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"] == "¿Qué hace AsyncRedisSaver?"

    listed = harness.client.get("/conversations", headers=USER).json()
    assert [c["conversation_id"] for c in listed] == [conversation]


async def test_un_turno_a_la_vez_por_conversacion(harness: Harness, conversation: str) -> None:
    assert harness.ask(conversation, question="Primera pregunta").status_code == 202
    busy = harness.ask(conversation, question="Segunda pregunta")
    assert busy.status_code == 409
    await harness.drain()
    assert harness.ask(conversation, question="Ahora sí, segunda").status_code == 202


async def test_revision_de_codigo_por_api(harness: Harness, conversation: str) -> None:
    job_url = harness.ask(conversation, question="Revisá mi agente", code=CODE_WITH_BUGS).json()[
        "status_url"
    ]
    await harness.drain()
    result = harness.client.get(job_url, headers=USER).json()["result"]
    assert "code_reviewer" in result["agents_used"]
    assert result["code_review"]["issues"]


def test_conversacion_ajena_es_404(harness: Harness, conversation: str) -> None:
    other = {"X-User-Id": "beto"}
    assert harness.client.get(f"/conversations/{conversation}", headers=other).status_code == 404
    assert harness.ask(conversation, headers=other, question="hola hola").status_code == 404


def test_validacion_de_entrada(harness: Harness, conversation: str) -> None:
    assert harness.ask(conversation, question="").status_code == 422
    assert harness.ask(conversation, question="hola", level="experto").status_code == 422
    no_user = harness.client.post("/conversations", json={})
    assert no_user.status_code == 422
    bad_user = harness.client.post("/conversations", json={}, headers={"X-User-Id": "a b"})
    assert bad_user.status_code == 422


async def test_upload_aprobacion_y_busqueda(harness: Harness, conversation: str) -> None:
    content = b"# Streaming\n\nastream entrega tokens."
    files = {"file": ("../../streaming guía.md", content, "text/markdown")}
    created = harness.client.post("/documents", files=files, data={"module": "1"}, headers=USER)
    assert created.status_code == 202
    job_url = created.json()["status_url"]

    await harness.drain()
    job = harness.client.get(job_url, headers=USER).json()
    assert job["status"] == "WAITING_APPROVAL"
    assert job["approval_request"]["filename"] == "streaming_gu_a.md"

    review_url = f"/documents/reviews/{job['job_id']}"
    assert harness.client.post(review_url, json={"approved": True}, headers=USER).status_code == 403
    pending = harness.client.get("/documents/reviews", headers=ADMIN).json()
    assert [p["job_id"] for p in pending] == [job["job_id"]]

    approved = harness.client.post(
        review_url, json={"approved": True, "topic": "streaming"}, headers=ADMIN
    )
    assert approved.status_code == 202
    await harness.drain()

    job = harness.client.get(job_url, headers=USER).json()
    assert job["status"] == "DONE" and job["result"]["status"] == "indexed"
    assert job["result"]["reviewer"] == "admin"
    assert harness.client.get("/documents/reviews", headers=ADMIN).json() == []
    docs = {d["source"]: d for d in harness.client.get("/documents").json()}
    assert docs["uploads/streaming_gu_a.md"]["approved_by"] == "admin"
    assert docs["uploads/streaming_gu_a.md"]["topic"] == "streaming"

    answer_url = harness.ask(conversation, question="¿Qué entrega astream?").json()["status_url"]
    await harness.drain()
    citations = harness.client.get(answer_url, headers=USER).json()["result"]["citations"]
    assert citations[0]["source"] == "uploads/streaming_gu_a.md"


def test_upload_rechaza_extension_y_tamano(harness: Harness, settings: Settings) -> None:
    png = harness.client.post("/documents", files={"file": ("x.png", b"123")}, headers=USER)
    assert png.status_code == 415
    huge = b"a" * (settings.max_upload_bytes + 1)
    big = harness.client.post("/documents", files={"file": ("x.md", huge)}, headers=USER)
    assert big.status_code == 413


async def test_un_error_del_grafo_queda_como_failed(
    harness: Harness, conversation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(SimulatedBackend, "plan", boom)
    job_url = harness.ask(conversation, question="¿Qué es BM25?").json()["status_url"]
    await harness.drain()
    job = harness.client.get(job_url, headers=USER).json()
    assert job["status"] == "FAILED"
    assert "LLM caído" in job["error"]
    # El lock se liberó: la conversación sigue usable.
    assert harness.ask(conversation, question="Otra vez").status_code == 202


def test_redis_caido_responde_503_con_retry_after(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.repositories import ConversationRepository as Repo

    async def down(*args: Any, **kwargs: Any) -> Any:
        raise RedisConnectionError("Connection closed by server.")

    monkeypatch.setattr(Repo, "list_for_user", down)
    response = harness.client.get("/conversations", headers=USER)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


# ---------- trazas: el span raíz de cada job tiene que ser legible en Phoenix ----------
# Sin kind/input/output Phoenix muestra el job como "unknown" y las sesiones con "--".
@pytest.fixture
def spans(harness: Harness) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    harness.worker.tracer = provider.get_tracer("test")
    return exporter


def _job_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name.startswith("job.")]


async def test_span_de_chat_con_pregunta_y_respuesta(
    harness: Harness, conversation: str, spans: InMemorySpanExporter
) -> None:
    created = harness.ask(conversation, question="¿Qué hace AsyncRedisSaver?")
    await harness.drain()

    (span,) = _job_spans(spans)
    attrs = span.attributes or {}
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert attrs["input.value"] == "¿Qué hace AsyncRedisSaver?"
    assert attrs["output.value"]
    assert span.status.status_code is StatusCode.OK
    # El job guarda su trace_id para que la interfaz arme el link a Phoenix.
    job = harness.client.get(f"/jobs/{created.json()['job_id']}", headers=USER).json()
    assert job["trace_id"] == format(span.context.trace_id, "032x")


async def test_span_de_ingesta_antes_y_despues_de_aprobar(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    upload = harness.client.post(
        "/documents",
        files={"file": ("apunte.md", b"# Streaming\n\nastream_events emite eventos parciales.")},
        headers=USER,
    )
    await harness.drain()
    harness.client.post(
        f"/documents/reviews/{upload.json()['job_id']}", json={"approved": True}, headers=ADMIN
    )
    await harness.drain()

    waiting, resumed = _job_spans(spans)
    assert (waiting.attributes or {})["input.value"] == "Carga de apunte.md por ana"
    assert str((waiting.attributes or {})["output.value"]).startswith("Esperando aprobación")
    assert (resumed.attributes or {})["input.value"] == "Revisión de apunte.md: aprobado por admin"
    assert str((resumed.attributes or {})["output.value"]).startswith("Indexado uploads/apunte.md")


# ---------- concurrencia del worker ----------
async def test_worker_procesa_jobs_en_paralelo_con_tope(harness: Harness) -> None:
    """Con WORKER_CONCURRENCY=2, cinco jobs se procesan de a dos, nunca de a uno ni de a tres."""

    worker = harness.worker
    worker.settings = worker.settings.model_copy(
        update={"worker_concurrency": 2, "worker_poll_timeout_seconds": 1}
    )
    in_flight = peak = 0
    done: list[str] = []
    all_done = asyncio.Event()

    async def slow_job(job_id: str) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)  # espera de red simulada
        in_flight -= 1
        done.append(job_id)
        if len(done) == 5:
            all_done.set()

    worker.process_job = slow_job  # type: ignore[method-assign]
    await worker.jobs.redis.rpush(worker.jobs.queue_name, *[f"j{i}" for i in range(5)])  # type: ignore[misc]

    loop = asyncio.create_task(worker.run_forever())
    async with asyncio.timeout(5):
        await all_done.wait()
    loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop

    assert sorted(done) == [f"j{i}" for i in range(5)]
    assert peak == 2
