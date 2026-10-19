"""Agente ReAct de un alumno, con errores a propósito, para la demo multipaso.

Errores sembrados (el sistema tiene que encontrarlos comparando con el módulo 5):
1. El estado redefine `messages` sin reducer: cada nodo pisa el historial.
2. `model.invoke` sincrónico dentro de `async def`: bloquea el event loop.
3. El modelo nunca recibe las herramientas (`bind_tools`).
4. Falta la arista tools -> agent: el ciclo ReAct termina después de la primera herramienta.
5. Se compila sin checkpointer y se invoca sin `thread_id`: no hay memoria entre turnos.
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    messages: list[BaseMessage]


@tool
def search_knowledge_base(query: str) -> str:
    """Busca en la base de conocimiento interna."""
    return f"Resultado para {query}"


TOOLS = [search_knowledge_base]
model = ChatOpenAI(model="gpt-4.1-mini")


async def call_model(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", END)

graph = workflow.compile()


async def main() -> None:
    result = await graph.ainvoke({"messages": [("user", "¿Cuál es la política de reembolsos?")]})
    print(result["messages"][-1].content)
