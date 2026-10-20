"""Cinco preguntas concurrentes contra la API levantada con Docker Compose.

    python scripts/load_test.py --output docs/evidencia/07_carga_concurrente.json

Mide tres cosas:
- cuánto tarda la API en aceptar cada pregunta (202): no debería depender de la carga;
- la latencia de punta a punta de cada turno y el tiempo total de las cinco;
- la latencia de /health mientras el worker trabaja: si algo bloqueara el event loop, subiría.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

QUESTIONS = [
    "¿Qué diferencia hay entre BM25 y la búsqueda vectorial?",
    "¿Para qué sirve el reducer add_messages en LangGraph?",
    "¿Cómo decide el Supervisor qué especialista trabaja después?",
    "¿Por qué la API devuelve 202 en lugar de esperar la respuesta del agente?",
    "¿Qué es Reciprocal Rank Fusion?",
]
TERMINAL = {"DONE", "FAILED", "WAITING_APPROVAL"}


def p95(values: list[float]) -> float:
    """Percentil nearest-rank: fácil de auditar con pocas muestras."""

    ordered = sorted(values)
    return ordered[max(1, math.ceil(0.95 * len(ordered))) - 1]


async def ask(client: httpx.AsyncClient, user: str, question: str) -> dict[str, Any]:
    headers = {"X-User-Id": user}
    created = await client.post("/conversations", json={}, headers=headers)
    conversation = created.raise_for_status().json()["conversation_id"]

    started = perf_counter()
    response = await client.post(
        f"/conversations/{conversation}/messages", json={"question": question}, headers=headers
    )
    accepted = perf_counter() - started
    status_url = response.raise_for_status().json()["status_url"]

    async with asyncio.timeout(240):
        while True:
            job = (await client.get(status_url, headers=headers)).raise_for_status().json()
            if job["status"] in TERMINAL:
                break
            await asyncio.sleep(0.5)
    return {
        "question": question,
        "status": job["status"],
        "accept_ms": round(accepted * 1000, 1),
        "end_to_end_s": round(perf_counter() - started, 2),
        "agents_used": (job.get("result") or {}).get("agents_used"),
    }


async def probe_health(client: httpx.AsyncClient, stop: asyncio.Event) -> list[float]:
    samples: list[float] = []
    while not stop.is_set():
        started = perf_counter()
        (await client.get("/health")).raise_for_status()
        samples.append((perf_counter() - started) * 1000)
        await asyncio.sleep(0.2)
    return samples


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        stop = asyncio.Event()
        health = asyncio.create_task(probe_health(client, stop))
        started = perf_counter()
        results = await asyncio.gather(
            *(ask(client, f"carga{i}", q) for i, q in enumerate(QUESTIONS, start=1))
        )
        wall = perf_counter() - started
        stop.set()
        health_ms = await health

    end_to_end = [r["end_to_end_s"] for r in results]
    report = {
        "requests": results,
        "accept_p95_ms": p95([r["accept_ms"] for r in results]),
        "end_to_end_p95_s": p95(end_to_end),
        "sum_of_turns_s": round(sum(end_to_end), 2),
        "wall_time_s": round(wall, 2),
        "health_p95_ms_during_load": round(p95(health_ms), 1),
        "health_samples": len(health_ms),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        await asyncio.to_thread(args.output.write_text, text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
