"""Configuración validada y centralizada: el único lugar que lee variables de entorno."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables de entorno de API, worker, RAG y observabilidad."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- infraestructura ----------
    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "kh:"
    queue_name: str = "kh:jobs"
    worker_poll_timeout_seconds: int = Field(default=5, ge=1, le=60)
    job_ttl_seconds: int = Field(default=86_400, ge=300)
    conversation_lock_seconds: int = Field(default=300, ge=30)
    # Usuarios (X-User-Id) que pueden aprobar documentos, separados por coma.
    admin_users: str = "admin"

    # ---------- modelos ----------
    # simulated: agentes determinísticos sin costo (tests y demo de infraestructura).
    agent_mode: Literal["simulated", "openai"] = "simulated"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ---------- embeddings y base vectorial ----------
    embedding_provider: Literal["hash", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    vector_store: Literal["local", "pinecone"] = "local"
    pinecone_api_key: str = ""
    pinecone_index: str = "ai-engineering-knowledge-hub"
    pinecone_namespace: str = "course"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---------- RAG ----------
    chunk_size: int = Field(default=450, ge=100, le=2_000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    top_k: int = Field(default=6, ge=1, le=20)
    # Peso de BM25 en la fusión; el vectorial recibe 1 - bm25_weight.
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    max_upload_mb: int = Field(default=10, ge=1, le=50)

    # ---------- grafo ----------
    max_graph_steps: int = Field(default=24, ge=8, le=80)
    max_revisions: int = Field(default=1, ge=0, le=3)

    # ---------- observabilidad ----------
    observability_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "ai-engineering-knowledge-hub"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP tiene que ser menor que CHUNK_SIZE")
        return self

    def validate_runtime(self) -> None:
        """Falla al arrancar, no en la primera pregunta, si falta una credencial."""

        needs_openai = self.agent_mode == "openai" or self.embedding_provider == "openai"
        if needs_openai and not self.openai_api_key:
            raise RuntimeError(
                "AGENT_MODE=openai o EMBEDDING_PROVIDER=openai requieren OPENAI_API_KEY. "
                "Usá AGENT_MODE=simulated y EMBEDDING_PROVIDER=hash para la demo sin costo."
            )
        if self.vector_store == "pinecone" and not self.pinecone_api_key:
            raise RuntimeError("VECTOR_STORE=pinecone requiere PINECONE_API_KEY.")

    @property
    def admins(self) -> frozenset[str]:
        return frozenset(u.strip() for u in self.admin_users.split(",") if u.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Una única configuración validada por proceso."""

    return Settings()
