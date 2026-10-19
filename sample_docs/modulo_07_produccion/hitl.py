"""Nodo Human-in-the-loop: pausa tareas críticas y valida la decisión externa."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.schemas import AnalysisArtifact, HumanDecision


def human_approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """Interrumpe el grafo sin ejecutar efectos antes de la aprobación."""

    analysis = AnalysisArtifact.model_validate(state["analysis_output"])
    if not analysis.critical:
        return {
            "approved": True,
            "approval_decision": {
                "approved": True,
                "reviewer": "automatic-policy",
                "comment": "La tarea fue clasificada como no crítica.",
            },
            "messages": [AIMessage(content="Aprobación automática: riesgo no crítico.")],
        }

    response = interrupt(
        {
            "type": "human_approval_required",
            "question": "¿Aprobás que el agente continúe con esta tarea crítica?",
            "risk_reason": analysis.risk_reason,
            "query": state["query"],
        },
        response_schema=HumanDecision,
    )
    decision = (
        response
        if isinstance(response, HumanDecision)
        else HumanDecision.model_validate(response)
    )
    return {
        "approved": decision.approved,
        "approval_decision": decision.model_dump(mode="json"),
        "messages": [
            AIMessage(
                content=(
                    f"Decisión humana: {'aprobada' if decision.approved else 'rechazada'} "
                    f"por {decision.reviewer}."
                )
            )
        ],
    }
