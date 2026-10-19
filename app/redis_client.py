"""Cliente Redis compartido por API, worker y seed, preparado para que Redis se reinicie.

Sin esto, después de un reinicio de Redis la primera request usaba una conexión muerta del
pool y la API respondía 500 (lo vimos al reconstruir el stack con Docker Compose).
"""

from __future__ import annotations

from typing import Any

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialWithJitterBackoff


def create_redis(url: str, **kwargs: Any) -> Redis:
    return Redis.from_url(
        url,
        decode_responses=True,
        # Antes de reutilizar una conexión ociosa la verifica con PING.
        health_check_interval=15,
        # 6 reintentos con backoff de hasta 2 s cubren el arranque de Redis (carga del AOF).
        retry=Retry(ExponentialWithJitterBackoff(cap=2.0, base=0.1), retries=6),
        **kwargs,
    )
