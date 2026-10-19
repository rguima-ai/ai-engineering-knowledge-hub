"""Persistencia en Redis: trabajos (cola + estado) y conversaciones por usuario.

El repositorio de jobs viene de la Pre-entrega 7: el estado se escribe antes de publicar el
ID en la cola, y las transiciones (PENDING→RUNNING, WAITING_APPROVAL→PENDING) son atómicas
con WATCH/MULTI para que dos workers nunca procesen el mismo job.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.schemas import Conversation, JobKind, JobRecord, JobStatus


class JobNotFoundError(LookupError):
    """El job no existe o venció."""


class InvalidJobTransitionError(RuntimeError):
    """El estado actual no permite la transición pedida."""


class ConversationBusyError(RuntimeError):
    """La conversación ya tiene un turno en curso."""


class JobRepository:
    def __init__(self, redis: Redis, *, prefix: str, queue_name: str, ttl_seconds: int) -> None:
        self.redis = redis
        self.queue_name = queue_name
        self.ttl_seconds = ttl_seconds
        self._prefix = f"{prefix}job:"
        self._pending_reviews = f"{prefix}pending_reviews"

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    async def create_and_enqueue(self, record: JobRecord) -> None:
        """Persiste PENDING antes de publicar el ID: el worker nunca lee un job inexistente."""

        key = self._key(record.job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            if await pipe.exists(key):
                raise InvalidJobTransitionError(f"El job {record.job_id} ya existe.")
            pipe.multi()  # type: ignore[no-untyped-call]
            pipe.set(key, record.model_dump_json(), ex=self.ttl_seconds)
            pipe.rpush(self.queue_name, record.job_id)
            await pipe.execute()

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self.redis.get(self._key(job_id))
        return JobRecord.model_validate_json(raw) if raw else None

    async def update(self, job_id: str, **changes: Any) -> JobRecord:
        record = await self.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        changes["updated_at"] = datetime.now(UTC)
        updated = record.model_copy(update=changes)
        await self.redis.set(self._key(job_id), updated.model_dump_json(), ex=self.ttl_seconds)
        if updated.kind is JobKind.INGEST:
            if updated.status is JobStatus.WAITING_APPROVAL:
                await self.redis.sadd(self._pending_reviews, job_id)
            else:
                await self.redis.srem(self._pending_reviews, job_id)
        return updated

    async def pending_reviews(self) -> list[JobRecord]:
        ids = sorted(str(i) for i in await self.redis.smembers(self._pending_reviews))
        records = [await self.get(job_id) for job_id in ids]
        return [r for r in records if r and r.status is JobStatus.WAITING_APPROVAL]

    async def dequeue(self, timeout_seconds: int) -> str | None:
        item = await self.redis.blpop([self.queue_name], timeout=timeout_seconds)
        if item is None:
            return None
        return str(item[1])

    async def _transition(
        self,
        job_id: str,
        expected: JobStatus,
        changes: dict[str, Any],
        *,
        enqueue: bool,
        retries: int = 3,
    ) -> JobRecord | None:
        key = self._key(job_id)
        for _ in range(retries):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        raise JobNotFoundError(job_id)
                    current = JobRecord.model_validate_json(raw)
                    if current.status is not expected:
                        return None
                    updated = current.model_copy(
                        update={**changes, "updated_at": datetime.now(UTC)}
                    )
                    pipe.multi()  # type: ignore[no-untyped-call]
                    pipe.set(key, updated.model_dump_json(), ex=self.ttl_seconds)
                    if enqueue:
                        pipe.rpush(self.queue_name, job_id)
                    await pipe.execute()
                    return updated
                except WatchError:
                    continue
        raise InvalidJobTransitionError("No se pudo actualizar el job por escrituras concurrentes.")

    async def claim(self, job_id: str) -> JobRecord | None:
        """PENDING → RUNNING. None si no existe o ya lo tomó otro worker."""

        try:
            return await self._transition(
                job_id,
                JobStatus.PENDING,
                {"status": JobStatus.RUNNING, "error": None},
                enqueue=False,
            )
        except JobNotFoundError:
            return None

    async def submit_approval(self, job_id: str, decision: dict[str, Any]) -> JobRecord:
        """WAITING_APPROVAL → PENDING y reencola, en una sola transacción."""

        updated = await self._transition(
            job_id,
            JobStatus.WAITING_APPROVAL,
            {"status": JobStatus.PENDING, "approval_decision": decision},
            enqueue=True,
        )
        if updated is None:
            raise InvalidJobTransitionError("El documento no está esperando revisión.")
        await self.redis.srem(self._pending_reviews, job_id)
        return updated

    async def ping(self) -> bool:
        return bool(await self.redis.ping())


class ConversationRepository:
    def __init__(self, redis: Redis, *, prefix: str, lock_seconds: int) -> None:
        self.redis = redis
        self.lock_seconds = lock_seconds
        self._prefix = prefix

    def _key(self, conversation_id: str) -> str:
        return f"{self._prefix}conversation:{conversation_id}"

    def _user_index(self, user_id: str) -> str:
        return f"{self._prefix}user:{user_id}:conversations"

    def _lock(self, conversation_id: str) -> str:
        return f"{self._prefix}conversation:{conversation_id}:lock"

    async def create(self, conversation: Conversation) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self._key(conversation.conversation_id), conversation.model_dump_json())
            pipe.zadd(
                self._user_index(conversation.user_id),
                {conversation.conversation_id: conversation.updated_at.timestamp()},
            )
            await pipe.execute()

    async def get(self, conversation_id: str) -> Conversation | None:
        raw = await self.redis.get(self._key(conversation_id))
        return Conversation.model_validate_json(raw) if raw else None

    async def get_owned(self, conversation_id: str, user_id: str) -> Conversation | None:
        """Una conversación de otro usuario se trata igual que una inexistente (404)."""

        conversation = await self.get(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return None
        return conversation

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[Conversation]:
        ids = await self.redis.zrevrange(self._user_index(user_id), 0, limit - 1)
        conversations = [await self.get(str(cid)) for cid in ids]
        return [c for c in conversations if c is not None]

    async def touch(self, conversation_id: str, title: str | None = None) -> None:
        conversation = await self.get(conversation_id)
        if conversation is None:
            return
        changes: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if title:
            changes["title"] = title
        updated = conversation.model_copy(update=changes)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self._key(conversation_id), updated.model_dump_json())
            pipe.zadd(
                self._user_index(updated.user_id),
                {conversation_id: updated.updated_at.timestamp()},
            )
            await pipe.execute()

    async def acquire(self, conversation_id: str, job_id: str) -> None:
        """Un turno a la vez: dos turnos simultáneos pisarían el mismo checkpoint.

        El vencimiento evita que un worker caído deje la conversación bloqueada para siempre.
        """

        ok = await self.redis.set(
            self._lock(conversation_id), job_id, nx=True, ex=self.lock_seconds
        )
        if not ok:
            raise ConversationBusyError(conversation_id)

    async def release(self, conversation_id: str, job_id: str) -> None:
        """Libera solo si el lock sigue siendo de este job (pudo vencer y tomarlo otro)."""

        key = self._lock(conversation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.get(key) != job_id:
                    return
                pipe.multi()  # type: ignore[no-untyped-call]
                pipe.delete(key)
                await pipe.execute()
            except WatchError:
                return
