"""Definición del estado, los nodos y las aristas del agente ReAct."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from react_agent.tools import TOOLS


class ToolBindableModel(Protocol):
    """Interfaz mínima que permite usar ChatOpenAI o un modelo falso en tests."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any: ...


class AgentState(MessagesState):
    """Estado del grafo.

    MessagesState ya define `messages` con el reducer `add_messages`, por eso cada nodo
    agrega mensajes al historial en lugar de borrar los anteriores.
    """


SYSTEM_PROMPT = SystemMessage(
    content=(
        "Sos un agente de soporte preciso. No inventes datos internos: consultalos con "
        "search_knowledge_base. Si el usuario pide varios datos independientes, buscá uno "
        "por vez con limit=1. Después de observar cada resultado, evaluá si ya podés responder "
        "o si necesitás otra llamada. No hagas llamadas paralelas. Si una búsqueda falla, "
        "reintentá una vez con una consulta más clara; si vuelve a fallar, explicá el problema. "
        "Respondé en español y citá los identificadores de los fragmentos usados."
    )
)


def build_graph(
    model: ToolBindableModel,
    checkpointer: Any,
) -> CompiledStateGraph[AgentState, None, Any, Any]:
    """Construye y compila el ciclo agent -> tools -> agent."""

    # Una tool call por turno: cada dato independiente recorre su propio ciclo ReAct.
    model_with_tools = model.bind_tools(TOOLS, parallel_tool_calls=False)

    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Pide al LLM que responda o que genere una llamada de herramienta."""

        response = await model_with_tools.ainvoke([SYSTEM_PROMPT, *state["messages"]])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
