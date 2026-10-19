"""Archivos subidos que esperan aprobación. Viven en Redis con vencimiento.

La API recibe el archivo y el worker lo procesa en otro proceso (y en Docker, en otro
contenedor): Redis es el disco compartido entre los dos. Se guarda en base64 porque los
clientes usan decode_responses=True.
"""

from __future__ import annotations

import base64

from redis.asyncio import Redis


class UploadNotFoundError(LookupError):
    """El archivo venció o ya fue procesado."""


class UploadStore:
    def __init__(self, redis: Redis, prefix: str = "kh:", ttl_seconds: int = 86_400) -> None:
        self.redis = redis
        self.prefix = f"{prefix}upload:"
        self.ttl_seconds = ttl_seconds

    async def save(self, upload_id: str, data: bytes) -> None:
        encoded = base64.b64encode(data).decode("ascii")
        await self.redis.set(f"{self.prefix}{upload_id}", encoded, ex=self.ttl_seconds)

    async def load(self, upload_id: str) -> bytes:
        encoded = await self.redis.get(f"{self.prefix}{upload_id}")
        if encoded is None:
            raise UploadNotFoundError(upload_id)
        return base64.b64decode(encoded)

    async def delete(self, upload_id: str) -> None:
        await self.redis.delete(f"{self.prefix}{upload_id}")
