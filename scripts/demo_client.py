"""Cliente de demostración contra la API levantada con Docker Compose.

    python scripts/demo_client.py ask "¿Qué es un checkpointer?" [--conversation ID]
    python scripts/demo_client.py ask "Revisá mi agente" --code-file mi_agente.py
    python scripts/demo_client.py history ID
    python scripts/demo_client.py upload apunte.pdf [--module 4]
    python scripts/demo_client.py review JOB_ID --approve|--reject

Usa httpx async, igual que el resto del proyecto. Imprime JSON para poder guardar evidencia.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"DONE", "FAILED", "WAITING_APPROVAL"}


async def wait_job(client: httpx.AsyncClient, url: str, user: str) -> Any:
    """Polling cada segundo, con un tope total de 3 minutos."""

    async with asyncio.timeout(180):
        while True:
            response = await client.get(url, headers={"X-User-Id": user})
            job = response.raise_for_status().json()
            if job["status"] in TERMINAL:
                return job
            await asyncio.sleep(1)


async def ask(
    client: httpx.AsyncClient,
    user: str,
    question: str,
    conversation: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    headers = {"X-User-Id": user}
    if conversation is None:
        created = await client.post("/conversations", json={}, headers=headers)
        conversation = created.raise_for_status().json()["conversation_id"]
    body = {"question": question, **{k: v for k, v in extra.items() if v is not None}}
    response = await client.post(
        f"/conversations/{conversation}/messages", json=body, headers=headers
    )
    job = await wait_job(client, response.raise_for_status().json()["status_url"], user)
    return {"conversation_id": conversation, **job}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user", default="ana")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--conversation")
    p_ask.add_argument("--level", choices=["principiante", "intermedio", "avanzado"])
    p_ask.add_argument("--code-file", type=Path)
    p_ask.add_argument("--modules", type=int, nargs="*")

    p_hist = sub.add_parser("history")
    p_hist.add_argument("conversation")

    p_up = sub.add_parser("upload")
    p_up.add_argument("path", type=Path)
    p_up.add_argument("--module", type=int)

    p_rev = sub.add_parser("review")
    p_rev.add_argument("job_id")
    group = p_rev.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")
    p_rev.add_argument("--reviewer", default="admin")

    args = parser.parse_args()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        result: Any
        if args.command == "ask":
            code = args.code_file.read_text() if args.code_file else None
            result = await ask(
                client,
                args.user,
                args.question,
                args.conversation,
                level=args.level,
                code=code,
                modules=args.modules,
            )
        elif args.command == "history":
            response = await client.get(
                f"/conversations/{args.conversation}", headers={"X-User-Id": args.user}
            )
            result = response.raise_for_status().json()
        elif args.command == "upload":
            data = {"module": str(args.module)} if args.module is not None else {}
            response = await client.post(
                "/documents",
                files={"file": (args.path.name, args.path.read_bytes())},
                data=data,
                headers={"X-User-Id": args.user},
            )
            result = await wait_job(
                client, response.raise_for_status().json()["status_url"], args.user
            )
        else:
            response = await client.post(
                f"/documents/reviews/{args.job_id}",
                json={"approved": bool(args.approve)},
                headers={"X-User-Id": args.reviewer},
            )
            job = response.raise_for_status().json()
            result = await wait_job(client, f"/jobs/{job['job_id']}", args.reviewer)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    asyncio.run(main())
