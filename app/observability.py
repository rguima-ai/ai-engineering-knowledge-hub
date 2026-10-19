"""Phoenix vía OpenTelemetry. Viene de la Pre-entrega 7."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

from app.config import Settings

logger = logging.getLogger(__name__)
_configured = False


def configure_observability(settings: Settings) -> Tracer:
    """Registra Phoenix una sola vez por proceso y devuelve un tracer para spans propios."""

    global _configured
    if not settings.observability_enabled:
        return trace.get_tracer("knowledge-hub.disabled")

    if not _configured:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from phoenix.otel import register

            tracer_provider = register(
                project_name=settings.phoenix_project_name,
                endpoint=settings.phoenix_collector_endpoint,
                protocol="http/protobuf",
                auto_instrument=False,
                batch=True,
            )
            # Solo LangChain: su span de ChatOpenAI ya trae modelo y tokens. Sumar también el
            # instrumentor de OpenAI duplica cada llamada y Phoenix cuenta el costo dos veces.
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            _configured = True
            logger.info("Phoenix activo en %s", settings.phoenix_collector_endpoint)
        except Exception:  # Phoenix caído no tiene que tirar abajo el worker.
            logger.exception("No se pudo inicializar Phoenix; se sigue sin exportar trazas.")

    return trace.get_tracer("knowledge-hub")


def set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """Solo tipos que OpenTelemetry acepta; los None se omiten."""

    for key, value in attributes.items():
        if value is not None and isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
