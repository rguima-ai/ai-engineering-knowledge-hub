"""RAG grounded: recuperación en ChromaDB + cadena LCEL prompt | llm | PydanticOutputParser.

    python rag.py                     # corre las dos pruebas (necesita la key del LLM)
    python rag.py --solo-recuperacion # muestra qué fragmentos se recuperan y con qué score (sin keys)
"""
import argparse
import asyncio
import logging
import time

from langchain_core.documents import Document
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

import config
from ingest import ingest
from schemas import NO_LO_SE, RAGResponse, Referencia, RespuestaLLM

log = logging.getLogger("rag")

# Qué se reintenta: JSON mal formado o que no valida (OutputParserException) y errores
# transitorios de la API. Nunca los permanentes (ModelAuthenticationError, ModelInvalidRequestError...):
# van a fallar igual y solo gastan tiempo y cuota.
ERRORES_REINTENTABLES: tuple[type[Exception], ...] = (
    OutputParserException,
    ModelRateLimitError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelAPIError,  # 5xx y el 529 "overloaded" de Anthropic
)

# ---------- prompt ----------
SYSTEM = (
    "Sos un asistente que responde preguntas sobre los apuntes del curso AI Engineering.\n"
    "Tu única fuente de información son los FRAGMENTOS numerados que vienen con cada pregunta.\n\n"
    "Reglas:\n"
    "1. Respondé SOLO con información que esté en los fragmentos. No uses conocimiento general "
    "ni completes con suposiciones, aunque sepas la respuesta.\n"
    '2. Si los fragmentos no contienen la respuesta, respondé exactamente "{no_lo_se}" '
    "y dejá la lista de fragmentos vacía.\n"
    '3. En "fragmentos" poné los números de los fragmentos que usaste: [1] se indica como 1. '
    "No escribas nombres de archivo ni citas dentro de la respuesta.\n"
    "4. Respondé en español, de forma breve y directa.\n\n"
    "{instrucciones_formato}"
)

PARSER = PydanticOutputParser(pydantic_object=RespuestaLLM)

# {contexto} y {pregunta} llegan en cada ainvoke; lo fijo queda con .partial().
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "FRAGMENTOS:\n\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(no_lo_se=NO_LO_SE, instrucciones_formato=PARSER.get_format_instructions())


def formatear_contexto(recuperados: list[tuple[Document, float]]) -> str:
    # Solo número + texto. Si le pasáramos el nombre del archivo, el LLM podría "citarlo" (o inventar uno).
    return "\n\n".join(f"[{i}]\n{doc.page_content}" for i, (doc, _) in enumerate(recuperados, 1))


def build_chain(llm: BaseChatModel, *, stop_after_attempt: int = 3, backoff_inicial_s: float = 1.0) -> Runnable:
    """prompt | llm | parser, reintentado entero: si el parser rechaza la salida, se vuelve a llamar al LLM."""
    return (PROMPT | llm | PARSER).with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=backoff_inicial_s > 0,
        exponential_jitter_params={"initial": backoff_inicial_s, "max": 10, "jitter": backoff_inicial_s},
    )


class RAGPipeline:
    """Recuperación con umbral de relevancia + generación grounded + referencias verificables."""

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseChatModel | None = None,
        *,
        top_k: int = config.TOP_K,
        min_relevance: float = config.MIN_RELEVANCE,
        backoff_inicial_s: float = 1.0,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.min_relevance = min_relevance
        self._llm = llm
        self._backoff_inicial_s = backoff_inicial_s
        self._chain: Runnable | None = None

    @property
    def chain(self) -> Runnable:
        # Se crea recién cuando hace falta: la recuperación sola no necesita API key.
        if self._chain is None:
            self._chain = build_chain(self._llm or config.get_llm(), backoff_inicial_s=self._backoff_inicial_s)
        return self._chain

    async def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Top-k por similitud coseno, descartando lo que no supera min_relevance. Ordenado de mayor a menor."""
        resultados = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=self.top_k)
        relevantes = [(doc, score) for doc, score in resultados if score >= self.min_relevance]
        log.info("recuperacion top_k=%d relevantes=%d umbral=%.2f scores=[%s]",
                 self.top_k, len(relevantes), self.min_relevance,
                 ", ".join(f"{d.metadata.get('chunk_id')}={s:.3f}" for d, s in resultados))
        return relevantes

    def _referencias(self, salida: RespuestaLLM, recuperados: list[tuple[Document, float]]) -> list[Referencia]:
        """Traduce los números que devolvió el LLM a fuente y chunk_id reales."""
        if salida.respuesta == NO_LO_SE:
            if salida.fragmentos:
                log.info("referencias=descartadas motivo=no_lo_se fragmentos=%s", salida.fragmentos)
            return []

        referencias: list[Referencia] = []
        for numero in dict.fromkeys(salida.fragmentos):  # sin duplicados, en el orden del LLM
            if not 1 <= numero <= len(recuperados):
                log.warning("fragmento_inexistente numero=%d disponibles=1..%d", numero, len(recuperados))
                continue
            doc, score = recuperados[numero - 1]
            referencias.append(Referencia(fragmento=numero, fuente=doc.metadata["source"],
                                          chunk_id=doc.metadata["chunk_id"], score=round(score, 4)))
        if not referencias:
            log.warning("respuesta_sin_referencias_validas fragmentos=%s", salida.fragmentos)
        return referencias

    async def answer(self, query: str) -> RAGResponse:
        t0 = time.perf_counter()
        log.info("pregunta %r", query)
        recuperados = await self.retrieve(query)
        if not recuperados:
            # Ningún fragmento es relevante: no hay nada con qué fundamentar una respuesta,
            # así que ni siquiera se llama al LLM (ahorra costo y elimina el riesgo de alucinar).
            log.info("respuesta=no_lo_se motivo=sin_fragmentos_relevantes llm=no_llamado")
            return RAGResponse(respuesta=NO_LO_SE)

        try:
            salida: RespuestaLLM = await self.chain.ainvoke(
                {"contexto": formatear_contexto(recuperados), "pregunta": query}
            )
        except Exception as e:
            log.error("fallo_definitivo error=%s: %s", type(e).__name__, e)
            raise

        respuesta = RAGResponse(respuesta=salida.respuesta, referencias=self._referencias(salida, recuperados))
        log.info("ok latencia_ms=%.0f referencias=%s", (time.perf_counter() - t0) * 1000,
                 [r.chunk_id for r in respuesta.referencias])
        return respuesta


# ---------- API pública ----------
_pipeline: RAGPipeline | None = None
_pipeline_lock = asyncio.Lock()


async def _default_pipeline() -> RAGPipeline:
    global _pipeline
    async with _pipeline_lock:  # si llegan dos preguntas juntas al arrancar, se ingesta una sola vez
        if _pipeline is None:
            # ingest() es síncrono y pesado (embeddings): en un thread, para no bloquear el event loop.
            resultado = await asyncio.to_thread(ingest)
            _pipeline = RAGPipeline(resultado.vectorstore)
    return _pipeline


async def get_rag_response(query: str, pipeline: RAGPipeline | None = None) -> RAGResponse:
    """a) busca por similitud en ChromaDB, b) arma el prompt, c) llama al LLM async, d) parsea a Pydantic."""
    pipeline = pipeline or await _default_pipeline()
    return await pipeline.answer(query)


# ---------- pruebas ----------
PREGUNTA_CON_RESPUESTA = "¿Qué tamaño de chunk y cuánto overlap se recomiendan como práctica común?"
# Está cerca del tema (ChromaDB aparece en los apuntes) pero la respuesta no está: un LLM que
# usa su conocimiento general contestaría igual. Es la prueba de que no alucina.
PREGUNTA_TRAMPA = "¿En qué año se creó ChromaDB y qué empresa lo desarrolla?"
# Fuera de tema: no debería pasar el umbral de relevancia.
PREGUNTA_FUERA_DE_TEMA = "¿Cuál es la receta tradicional de las empanadas salteñas?"


async def mostrar_recuperacion(pipeline: RAGPipeline, preguntas: list[str]) -> None:
    for pregunta in preguntas:
        print(f"\n=== {pregunta}")
        resultados = await pipeline.vectorstore.asimilarity_search_with_relevance_scores(
            pregunta, k=pipeline.top_k)
        for doc, score in resultados:
            marca = "✅" if score >= pipeline.min_relevance else "❌"
            texto = " ".join(doc.page_content.split())[:90]
            print(f"  {marca} {score:.3f}  {doc.metadata['chunk_id']:<40} {texto}...")
        pasan = sum(score >= pipeline.min_relevance for _, score in resultados)
        print(f"  -> {pasan} de {len(resultados)} superan MIN_RELEVANCE={pipeline.min_relevance}")


async def correr(nombre: str, pregunta: str, pipeline: RAGPipeline) -> None:
    print(f"\n=== {nombre}\nPregunta: {pregunta}")
    try:
        resultado = await get_rag_response(pregunta, pipeline)
    except Exception as e:  # answer() ya logueó el detalle; acá solo mostramos que el programa sigue
        print(f"❌ No se pudo responder: {type(e).__name__}: {e}")
        return
    print(resultado.model_dump_json(indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG local sobre los apuntes del curso")
    parser.add_argument("--solo-recuperacion", action="store_true",
                        help="muestra los fragmentos recuperados y sus scores, sin llamar al LLM")
    args = parser.parse_args()
    config.setup_logging()

    resultado = await asyncio.to_thread(ingest)
    print(f"Índice: {resultado.chunks} chunks ({'reindexado' if resultado.reindexed else 'sin cambios'})")
    pipeline = RAGPipeline(resultado.vectorstore)

    if args.solo_recuperacion:
        await mostrar_recuperacion(pipeline, [PREGUNTA_CON_RESPUESTA, PREGUNTA_TRAMPA, PREGUNTA_FUERA_DE_TEMA])
        return

    try:
        pipeline.chain  # crea el LLM ahora, para fallar con un mensaje claro si falta la key
    except Exception as e:
        print(f"⚠️ No se pudo configurar el LLM: {type(e).__name__}: {e}")
        return

    await correr("Prueba 1: pregunta con respuesta en los documentos", PREGUNTA_CON_RESPUESTA, pipeline)
    await correr("Prueba 2: pregunta trampa (sin respuesta en los documentos)", PREGUNTA_TRAMPA, pipeline)


if __name__ == "__main__":
    asyncio.run(main())
