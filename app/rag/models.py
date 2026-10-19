"""Contratos Pydantic del RAG: documentos, chunks, filtros y resultados de búsqueda."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocType(StrEnum):
    """Tipos de archivo que entiende la ingesta."""

    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    NOTEBOOK = "notebook"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    """Metadatos de un documento; viajan a cada chunk y sirven de filtro."""

    doc_id: str = Field(min_length=8, max_length=64)
    source: str = Field(min_length=1, max_length=300, description="Ruta o nombre original")
    title: str = Field(min_length=1, max_length=200)
    doc_type: DocType
    module: int = Field(default=0, ge=0, le=20, description="0 = general / sin módulo")
    topic: str = Field(default="", max_length=120)
    uploaded_by: str = Field(default="seed", max_length=80)

    @staticmethod
    def make_id(source: str) -> str:
        """ID estable por ruta: reingestar el mismo archivo reemplaza sus chunks."""

        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class ParsedDocument(BaseModel):
    """Texto plano extraído de un archivo, listo para fragmentar."""

    metadata: DocumentMetadata
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    """Fragmento indexado. El texto vive acá y en la metadata de Pinecone."""

    id: str
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    source: str
    title: str
    doc_type: DocType
    module: int
    topic: str = ""

    @classmethod
    def build(cls, metadata: DocumentMetadata, index: int, text: str) -> Chunk:
        return cls(
            id=f"{metadata.doc_id}::{index}",
            doc_id=metadata.doc_id,
            chunk_index=index,
            text=text,
            source=metadata.source,
            title=metadata.title,
            doc_type=metadata.doc_type,
            module=metadata.module,
            topic=metadata.topic,
        )

    def vector_metadata(self) -> dict[str, Any]:
        """Metadata plana para Pinecone (solo str, int, float, bool o list[str])."""

        return self.model_dump(mode="json", exclude={"id"})


class SearchFilter(BaseModel):
    """Filtro validado; se aplica igual a BM25 y a la rama vectorial."""

    modules: list[int] | None = Field(default=None, max_length=20)
    doc_types: list[DocType] | None = Field(default=None, max_length=6)

    @field_validator("modules")
    @classmethod
    def _valid_modules(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(m < 0 or m > 20 for m in value):
            raise ValueError("Los módulos van de 0 a 20")
        return value or None

    def matches(self, chunk: Chunk) -> bool:
        if self.modules and chunk.module not in self.modules:
            return False
        if self.doc_types and chunk.doc_type not in self.doc_types:
            return False
        return True

    def to_pinecone(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.modules:
            clauses.append({"module": {"$in": self.modules}})
        if self.doc_types:
            clauses.append({"doc_type": {"$in": [t.value for t in self.doc_types]}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class SearchQuery(BaseModel):
    """Entrada de la herramienta de búsqueda que usan los agentes."""

    query: str = Field(min_length=2, max_length=1_000)
    filter: SearchFilter = Field(default_factory=SearchFilter)
    k: int = Field(default=6, ge=1, le=20)


class RetrievedChunk(BaseModel):
    """Chunk devuelto por la búsqueda, con su puntaje de fusión."""

    id: str
    source: str
    title: str
    module: int
    doc_type: DocType
    text: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> RetrievedChunk:
        return cls(
            id=chunk.id,
            source=chunk.source,
            title=chunk.title,
            module=chunk.module,
            doc_type=chunk.doc_type,
            text=chunk.text,
            score=round(score, 6),
        )
