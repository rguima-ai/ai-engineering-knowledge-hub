"""Pipeline LCEL: prompt | modelo con salida estructurada | verificación, envuelto en reintentos."""
import logging
import os
import re
import time
from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import EntidadesTecnicas

log = logging.getLogger("pipeline")


class SalidaInvalidaError(Exception):
    """El modelo respondió, pero la salida no sirve: cortada, mal formada o no pasa la validación."""


# Qué se reintenta: salidas inválidas + errores transitorios de la API.
# Las clases de langchain_core sirven para los dos proveedores: langchain_openai y
# langchain_anthropic envuelven los errores de sus SDKs en subclases de estas.
# Nunca se reintentan los permanentes (ModelAuthenticationError, ModelInvalidRequestError,
# ModelNotFoundError...): van a fallar igual y solo gastan tiempo y cuota.
ERRORES_REINTENTABLES: tuple[type[Exception], ...] = (
    SalidaInvalidaError,
    ModelRateLimitError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelAPIError,  # 5xx y el 529 "overloaded" de Anthropic
)

# ---------- prompt ----------
SYSTEM = (
    "Sos un analista técnico senior. Tu trabajo es extraer información estructurada "
    "de textos técnicos: logs de error, incidentes o descripciones de arquitectura.\n\n"
    "{instrucciones_formato}"
)

INSTRUCCIONES_FORMATO = """Respondé usando la herramienta EntidadesTecnicas, con estos tres campos:
- tecnologias: lista de tecnologías, frameworks, lenguajes, bases de datos o herramientas que el texto menciona explícitamente, sin repetir. No inventes ni deduzcas tecnologías que no aparecen en el texto. Si el texto no menciona ninguna, devolvé la lista vacía.
- nivel_de_criticidad: exactamente uno de estos valores, en minúscula: baja, media o alta. Usá alta si hay impacto en producción, pérdida de datos o caída del servicio; media si hay degradación o riesgo concreto; baja si es informativo o no hay impacto claro.
- resumen_tecnico: una o dos oraciones en español que resuman el problema o la arquitectura."""

# Variables del prompt: {texto} llega en cada ainvoke; {instrucciones_formato} queda fija con .partial().
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Texto a analizar:\n\n{texto}"),
]).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)

# ---------- Factory de modelos ----------
# Se pisan con OPENAI_MODEL / ANTHROPIC_MODEL en el .env: un modelo retirado se cambia sin tocar código.
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}
MODEL_ENV = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL"}


def _max_tokens() -> int:
    return int(os.getenv("LLM_MAX_TOKENS", "1024"))


def _timeout_s() -> float:
    return float(os.getenv("LLM_TIMEOUT_S", "30"))


def _openai(model: str) -> BaseChatModel:
    # temperature=0: en extracción queremos la misma respuesta para el mismo texto.
    # max_retries=0: los reintentos son solo los de with_retry; si no, se multiplican (3 x 3 = 9).
    return ChatOpenAI(model=model, temperature=0, max_tokens=_max_tokens(),
                      timeout=_timeout_s(), max_retries=0)


def _anthropic(model: str) -> BaseChatModel:
    # Sin temperature a propósito: si se la pasamos, LangChain la manda a la API y
    # los Claude recientes la rechazan con un 400.
    return ChatAnthropic(model=model, max_tokens=_max_tokens(),
                         timeout=_timeout_s(), max_retries=0)


# Registro proveedor -> constructor. Sumar un proveedor es agregar una línea acá.
_REGISTRY: dict[str, Callable[[str], BaseChatModel]] = {"openai": _openai, "anthropic": _anthropic}


def get_model(provider: str | None = None) -> BaseChatModel:
    """Usa el proveedor pedido o, si no se pasa ninguno, el de LLM_PROVIDER (por defecto openai)."""
    elegido = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    constructor = _REGISTRY.get(elegido)
    if constructor is None:
        raise ValueError(f"LLM_PROVIDER inválido: {elegido!r}. Opciones: {', '.join(_REGISTRY)}")
    modelo = os.getenv(MODEL_ENV[elegido]) or DEFAULT_MODELS[elegido]
    log.info("modelo provider=%s model=%s", elegido, modelo)
    return constructor(modelo)


# ---------- verificación ----------
def _resumen_parseo(error: BaseException) -> str:
    # Un ValidationError de Pydantic ocupa varias líneas; para el log alcanza con "campo: mensaje".
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return " ".join(str(error).split())


def _rechazar(motivo: str) -> SalidaInvalidaError:
    log.warning("validacion=rechazada motivo=%s", motivo)
    return SalidaInvalidaError(motivo)


def verificar_salida(resultado: dict) -> EntidadesTecnicas:
    """Recibe {raw, parsed, parsing_error} de with_structured_output(include_raw=True).

    Si algo está mal lanza SalidaInvalidaError, que with_retry reconoce y reintenta.
    """
    raw = resultado.get("raw")
    meta = getattr(raw, "response_metadata", None) or {}

    # Respuesta cortada por límite de tokens: aunque el JSON parseara, podría estar incompleto.
    if meta.get("finish_reason") == "length" or meta.get("stop_reason") == "max_tokens":
        raise _rechazar("respuesta cortada por límite de tokens")

    error = resultado.get("parsing_error")
    if error is not None:
        # JSON mal formado, campos faltantes o una regla de EntidadesTecnicas que no se cumple.
        raise _rechazar(f"no pasó el parseo/validación: {_resumen_parseo(error)}") from error

    parsed = resultado.get("parsed")
    if not isinstance(parsed, EntidadesTecnicas):
        # Pasa, por ejemplo, si el modelo contestó texto libre en vez de usar la herramienta.
        raise _rechazar("el modelo no devolvió una salida estructurada")

    log.info("validacion=ok tecnologias=%d criticidad=%s",
             len(parsed.tecnologias), parsed.nivel_de_criticidad.value)
    return parsed


def _resumen_error(traza: str | None) -> str:
    # run.error es el traceback completo. Nos quedamos con la última línea "Tipo: mensaje"
    # (la última línea a secas puede ser el pie de un error de varias líneas, como los de Pydantic).
    lineas = re.findall(r"^[A-Za-z_][\w.]*: .*$", traza or "", flags=re.MULTILINE)
    return lineas[-1] if lineas else (traza or "?").strip()[:200]


def _log_error_modelo(run) -> None:
    # Se dispara en cada intento fallido de la llamada al modelo (429, timeout, 401...).
    # No guarda estado: cada ejecución loguea lo suyo, aunque haya varias en paralelo.
    log.warning("llamada_al_modelo=fallida error=%s", _resumen_error(run.error))


# ---------- cadena ----------
def build_chain(
    model: BaseChatModel | None = None,
    *,
    provider: str | None = None,
    stop_after_attempt: int = 3,
    backoff_inicial_s: float = 1.0,
) -> Runnable:
    """Arma la cadena completa. `model` permite inyectar un modelo falso en los tests;
    `backoff_inicial_s=0` elimina la espera entre reintentos."""
    model = model or get_model(provider)
    # function_calling explícito: ChatOpenAI usa json_schema por defecto, y ahí el SDK de OpenAI
    # valida DENTRO de la llamada al modelo. Un ValidationError (o LengthFinishReasonError si se
    # corta) saldría crudo, sin pasar por include_raw ni por verificar_salida, y no se reintentaría.
    # Con function_calling el parseo lo hace LangChain después, y es el mismo método que usa Anthropic.
    estructurado = model.with_structured_output(
        EntidadesTecnicas, method="function_calling", include_raw=True
    )
    chain = (
        PROMPT
        | estructurado.with_listeners(on_error=_log_error_modelo)
        | RunnableLambda(verificar_salida)
    )
    # La cadena ENTERA se reintenta: si la verificación rechaza la salida, se vuelve a llamar al modelo.
    return chain.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=backoff_inicial_s > 0,
        # espera ~1 s, 2 s, 4 s... más un jitter de hasta 1 s, con techo de 10 s
        exponential_jitter_params={"initial": backoff_inicial_s, "max": 10, "jitter": backoff_inicial_s},
    )


async def process_text(text: str, chain: Runnable | None = None) -> EntidadesTecnicas:
    """Extrae las entidades técnicas de `text`. Si fallan todos los intentos, loguea y relanza el error."""
    chain = chain or build_chain()
    t0 = time.perf_counter()
    log.info("inicio caracteres=%d", len(text))
    try:
        # La clave "texto" tiene que coincidir con la variable {texto} del prompt.
        resultado = await chain.ainvoke({"texto": text})
    except ERRORES_REINTENTABLES as e:
        log.error("fallo_definitivo tipo=reintentos_agotados error=%s: %s", type(e).__name__, e)
        raise
    except Exception as e:
        log.error("fallo_definitivo tipo=no_reintentable error=%s: %s", type(e).__name__, e)
        raise
    log.info("ok latencia_ms=%.0f resultado=%s",
             (time.perf_counter() - t0) * 1000, resultado.model_dump_json())
    return resultado
