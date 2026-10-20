"""Herramientas de los agentes, con contratos Pydantic de entrada y de salida.

El RAG de las Pre-entregas 3 y 4 queda expuesto como la herramienta del Buscador: el agente no
toca el retriever directamente. La entrada se valida con `SearchQuery` (LangChain la rechaza
antes de ejecutar si no cumple) y la salida se vuelve a validar con `RetrievedChunk` del lado
de quien la usa. En Phoenix cada búsqueda aparece como un span de tipo TOOL con su consulta y
sus resultados.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.rag.models import RetrievedChunk, SearchQuery
from app.rag.retriever import HybridRetriever

SEARCH_TOOL_NAME = "search_course_material"


def build_search_tool(retriever: HybridRetriever) -> BaseTool:
    async def search_course_material(**arguments: Any) -> list[dict[str, Any]]:
        query = SearchQuery.model_validate(arguments)
        return [chunk.model_dump(mode="json") for chunk in await retriever.search(query)]

    return StructuredTool.from_function(
        coroutine=search_course_material,
        name=SEARCH_TOOL_NAME,
        description=(
            "Búsqueda híbrida (BM25 + vectores, fusionados con RRF) en el material del curso. "
            "Devuelve fragmentos con fuente, módulo y puntaje."
        ),
        args_schema=SearchQuery,
    )


async def search(tool: BaseTool, query: SearchQuery) -> list[RetrievedChunk]:
    """Invoca la herramienta y valida lo que devuelve: no se confía en la salida."""

    raw = await tool.ainvoke(query.model_dump(mode="json"))
    return [RetrievedChunk.model_validate(item) for item in raw]
