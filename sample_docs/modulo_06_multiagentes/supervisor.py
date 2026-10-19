"""Supervisor central que delega, controla calidad y decide cuándo finalizar."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import AgentState, Route, SupervisorDecision

SUPERVISOR_PROMPT = SystemMessage(
    content=(
        "Sos el Supervisor de un equipo multi-agente. Coordinás; no investigás, no calculás y no "
        "redactás el informe. Elegí exactamente un próximo paso: researcher, analyst, synthesizer "
        "o FINISH. Rúbrica: researcher debe aportar al menos dos métricas numéricas con períodos, "
        "unidades, source_id y confianza >= 0.6; analyst debe calcular una variación, declarar la "
        "fórmula y confianza >= 0.8; synthesizer debe usar evidencia y análisis, citar fuentes, "
        "recomendar y declarar limitaciones. Si un artefacto es insuficiente, reenviá el trabajo "
        "al especialista correspondiente con quality_feedback concreto. Solo elegí FINISH si el "
        "informe final cumple toda la rúbrica."
    )
)

# Qué artefactos debe tener el estado antes de delegar en cada especialista.
PREREQUISITES: dict[Route, tuple[str, ...]] = {
    "researcher": (),
    "analyst": ("research_output",),
    "synthesizer": ("research_output", "analysis_output"),
    "FINISH": ("research_output", "analysis_output", "final_report"),
}

# Artefactos que quedan desactualizados si se rehace una etapa anterior.
DOWNSTREAM: dict[Route, tuple[str, ...]] = {
    "researcher": ("analysis_output", "final_report"),
    "analyst": ("final_report",),
    "synthesizer": (),
    "FINISH": (),
}


def first_missing_route(state: Mapping[str, Any]) -> Route:
    """Ruteo determinista: la primera etapa cuyo artefacto todavía no existe."""

    if not state.get("research_output"):
        return "researcher"
    if not state.get("analysis_output"):
        return "analyst"
    if not state.get("final_report"):
        return "synthesizer"
    return "FINISH"


def _missing_prerequisites(state: AgentState, route: Route) -> list[str]:
    return [field for field in PREREQUISITES[route] if not state.get(field)]


class SupervisorNode:
    """Nodo de ruteo mediante salida estructurada y una condición de parada defensiva."""

    def __init__(self, model: Any) -> None:
        self._model = model.with_structured_output(SupervisorDecision)

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        current_step = state.get("step_count", 0)
        max_steps = state.get("max_steps", 8)
        if current_step >= max_steps:
            return self._stop_at_limit(state, current_step, max_steps)

        errors: list[str] = []
        snapshot = {
            "request": state["request"],
            "research_output": state.get("research_output"),
            "analysis_output": state.get("analysis_output"),
            "final_report": state.get("final_report"),
            "recent_errors": state.get("errors", [])[-3:],
            "step": current_step + 1,
            "max_steps": max_steps,
        }
        try:
            decision = await self._model.ainvoke(
                [
                    SUPERVISOR_PROMPT,
                    HumanMessage(content=json.dumps(snapshot, ensure_ascii=False, indent=2)),
                ]
            )
            validated = SupervisorDecision.model_validate(decision)
        except (ValueError, *RECOVERABLE_LLM_ERRORS) as exc:
            # ValueError cubre ValidationError de Pydantic y OutputParserException de LangChain.
            errors.append(f"Supervisor: {type(exc).__name__}: {exc}")
            validated = SupervisorDecision(
                next_agent=first_missing_route(state),
                reason="Se aplicó el ruteo defensivo porque falló la decisión del LLM.",
                quality_feedback="Reintentá la tarea respetando el contrato de salida.",
            )

        missing = _missing_prerequisites(state, validated.next_agent)
        if missing:
            # Evita delegar en un especialista sin insumos o terminar sin informe.
            safe_route = first_missing_route(state)
            errors.append(
                f"Supervisor: '{validated.next_agent}' requiere {', '.join(missing)}; "
                f"se delega en '{safe_route}'."
            )
            validated = SupervisorDecision(
                next_agent=safe_route,
                reason=f"Faltan artefactos previos ({', '.join(missing)}).",
                quality_feedback=validated.quality_feedback,
            )

        new_step = current_step + 1
        if new_step >= max_steps and validated.next_agent != "FINISH":
            # La última decisión permitida no puede abrir otra vuelta: se corta acá,
            # dejando constancia, en lugar de que el router la ignore en silencio.
            return self._stop_at_limit(state, new_step, max_steps, errors)

        update: dict[str, Any] = {
            "next_agent": validated.next_agent,
            "supervisor_reason": validated.reason,
            "quality_feedback": validated.quality_feedback,
            "task_completed": validated.next_agent == "FINISH",
            "step_count": new_step,
            "messages": [
                AIMessage(
                    content=f"Próximo agente: {validated.next_agent}. Motivo: {validated.reason}",
                    name="supervisor",
                )
            ],
            "contributions": [
                {
                    "agent": "supervisor",
                    "status": "decision",
                    "next_agent": validated.next_agent,
                    "summary": validated.reason,
                }
            ],
        }
        # Rehacer una etapa invalida lo que dependía de ella (consistencia del estado).
        for field in DOWNSTREAM[validated.next_agent]:
            if state.get(field):
                update[field] = None
        if errors:
            update["errors"] = errors
        return update

    @staticmethod
    def _stop_at_limit(
        state: AgentState,
        step: int,
        max_steps: int,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        reason = f"Se alcanzó el límite defensivo de {max_steps} decisiones del Supervisor."
        return {
            "next_agent": "FINISH",
            "supervisor_reason": reason,
            "quality_feedback": "",
            # Cortar por límite no es completar la tarea, aunque exista un borrador.
            "task_completed": False,
            "step_count": step,
            "errors": [*(errors or []), reason],
            "messages": [AIMessage(content=reason, name="supervisor")],
            "contributions": [
                {
                    "agent": "supervisor",
                    "status": "step_limit",
                    "next_agent": "FINISH",
                    "summary": reason,
                    "had_final_report": bool(state.get("final_report")),
                }
            ],
        }
