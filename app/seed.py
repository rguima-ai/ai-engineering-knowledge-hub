"""Ingesta inicial: indexa sample_docs/ y, si existe, knowledge_base/ (material privado).

    python -m app.seed                 # carpetas por defecto
    python -m app.seed otra/carpeta    # carpetas propias

Es idempotente: el ID de cada documento sale de su ruta, así que correrlo dos veces reemplaza
los chunks en lugar de duplicarlos. Los documentos sembrados no pasan por aprobación humana:
los cargó quien administra el repo.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.rag import build_rag
from app.rag.embeddings import create_embedder
from app.rag.loaders import EXTENSIONS, EmptyDocumentError
from app.rag.vector_store import ensure_pinecone_index
from app.redis_client import create_redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("seed")

DEFAULT_DIRS = ("sample_docs", "knowledge_base")


def discover(roots: list[Path]) -> list[tuple[Path, str]]:
    """(ruta en disco, ruta relativa que se guarda como fuente)."""

    files: list[tuple[Path, str]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                files.append((path, path.relative_to(root.parent).as_posix()))
    return files


async def seed(roots: list[Path]) -> int:
    settings = get_settings()
    settings.validate_runtime()
    if settings.vector_store == "pinecone":
        created = await ensure_pinecone_index(settings, create_embedder(settings).dimension)
        logger.info("índice Pinecone %s %s", settings.pinecone_index, "creado" if created else "ok")

    redis = create_redis(settings.redis_url)
    rag = await build_rag(settings, redis)
    indexed = 0
    try:
        for path, source in discover(roots):
            try:
                document, chunks = await rag.ingestion.prepare(path.read_bytes(), source)
            except EmptyDocumentError:
                logger.warning("salteado (sin texto) %s", source)
                continue
            await rag.ingestion.index(document, chunks)
            indexed += 1
    finally:
        await rag.close()
        await redis.aclose()
    logger.info("ingesta terminada: %s documentos", indexed)
    return indexed


def main() -> None:
    roots = [Path(arg) for arg in sys.argv[1:]] or [Path(d) for d in DEFAULT_DIRS]
    asyncio.run(seed(roots))


if __name__ == "__main__":
    main()
