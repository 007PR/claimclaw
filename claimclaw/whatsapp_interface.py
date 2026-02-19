from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

from .workflow import run_workflow


def create_whatsapp_app(graph: Any) -> FastAPI:
    app = FastAPI(title="ClaimClaw WhatsApp Interface")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/twilio/whatsapp")
    async def twilio_whatsapp(request: Request) -> Response:
        form = await request.form()
        text = (form.get("Body") or "").strip()
        sender = form.get("From") or "unknown"
        response = MessagingResponse()

        if text.startswith("status "):
            claim_id = text.split(maxsplit=1)[1]
            response.message(f"{claim_id}: send 'run {{json}}' to execute claim workflow.")
        elif text.startswith("run "):
            try:
                payload = json.loads(text.split(maxsplit=1)[1])
                claim_id = payload.get("claim_id", "CLAIM-WA")
                result = run_workflow(graph, payload, claim_id=claim_id)
                stage = result.get("stage", "unknown")
                response.message(f"{sender} -> {claim_id} now at stage: {stage}")
            except Exception as exc:
                response.message(f"Run failed: {exc}")
        else:
            response.message(
                "ClaimClaw commands: 'run {json_state}' or 'status <claim_id>'."
            )

        return Response(content=str(response), media_type="application/xml")

    return app
