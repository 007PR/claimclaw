from __future__ import annotations

import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .agents import EvidenceMatchingAgent, PolicyAnalysisAgent, PortalAutomationAgent
from .config import load_settings
from .evidence_matcher import analyze_documents
from .preflight import run_preflight
from .utils.llm_factory import get_main_llm, get_vision_llm
from .workflow import build_workflow, run_workflow


REQUIRED_DOCUMENTS: dict[str, str] = {
    "policy_document": "Policy Document",
    "rejection_letter": "Rejection Letter",
    "discharge_summary": "Discharge Summary",
    "hospital_bill": "Hospital Bill",
}


class ChatMessageIn(BaseModel):
    session_id: str
    message: str


class ChatStartIn(BaseModel):
    thread_name: str | None = None


class ThreadNoteIn(BaseModel):
    note: str | None = None


class ChatSession(TypedDict, total=False):
    session_id: str
    thread_name: str
    thread_note: str
    created_at: str
    updated_at: str
    work_dir: str
    files: dict[str, str]
    messages: list[dict[str, str]]
    policy_age_years: float
    last_analysis: dict[str, Any]
    legal_analysis: dict[str, Any]
    last_workflow: dict[str, Any]


def _resolve_llms(settings: Any) -> tuple[Any | None, Any | None]:
    forensic_llm = None
    vision_llm = None
    if settings.llm_provider.lower() == "openai":
        forensic_llm = get_main_llm(model_name=settings.openai_chat_model)
        vision_llm = get_vision_llm(model_name=settings.openai_vision_model)
    return forensic_llm, vision_llm


def _forensic_gate(settings: Any, forensic_llm: Any | None) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if forensic_llm is not None:
        return True, warnings
    if settings.strict_llm_mode and not settings.dev_allow_fallback:
        return False, warnings
    warnings.append(
        "Running without forensic LLM (DEV_ALLOW_FALLBACK=true). Results may use heuristic fallback paths."
    )
    return True, warnings


def _clean_claim_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "WEB-CLAIM"


def _clean_thread_name(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "New Claim"
    text = re.sub(r"\s+", " ", text)
    return text[:80]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_thread_note(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    return text[:400]


def _age_days(created_at: str | None) -> int:
    if not created_at:
        return 0
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - created.astimezone(UTC)
    return max(delta.days, 0)


async def _save_upload(upload: UploadFile, target_path: Path) -> None:
    content = await upload.read()
    target_path.write_bytes(content)


def _missing_documents(session: ChatSession) -> list[str]:
    files = session.get("files", {})
    return [doc_type for doc_type in REQUIRED_DOCUMENTS if doc_type not in files]


def _uploaded_documents(session: ChatSession) -> list[str]:
    files = session.get("files", {})
    return [doc_type for doc_type in REQUIRED_DOCUMENTS if doc_type in files]


def _infer_doc_type(filename: str, missing: list[str]) -> str:
    lowered = filename.lower()
    if "policy" in lowered:
        return "policy_document"
    if "rejection" in lowered or "repudiation" in lowered:
        return "rejection_letter"
    if "discharge" in lowered or "doctor" in lowered or "summary" in lowered:
        return "discharge_summary"
    if "bill" in lowered or "invoice" in lowered or "hospital" in lowered:
        return "hospital_bill"
    if missing:
        return missing[0]
    return "policy_document"


def _extract_dates(text: str) -> list[datetime]:
    matches = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b",
        text,
    )
    out: list[datetime] = []
    for token in matches:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                out.append(datetime.strptime(token, fmt))
                break
            except ValueError:
                continue
    return out


def _infer_policy_age_years(policy_excerpt: str) -> float:
    if not policy_excerpt:
        return 3.0
    dates = _extract_dates(policy_excerpt)
    if not dates:
        return 3.0
    start = min(dates)
    now = datetime.now(UTC).replace(tzinfo=None)
    delta_days = max((now - start).days, 0)
    return round(delta_days / 365.25, 2)


def _derive_legal_analysis(policy_age_years: float, rejection_reason: str) -> dict[str, Any]:
    policy_agent = PolicyAnalysisAgent(retriever=None, llm=None)
    return policy_agent.run(
        policy_age_years=policy_age_years,
        rejection_reason=rejection_reason,
        alleged_fraud=False,
    )


def _analysis_summary_text(report: dict[str, Any], legal: dict[str, Any]) -> str:
    contestable = bool(report.get("contestable"))
    flags = report.get("flags", [])
    top_flags = "\n".join(f"- {flag}" for flag in flags[:3]) if flags else "- No contradictions extracted."
    clause_id = (
        report.get("forensic_validation", {}).get("clause_id")
        or report.get("clause_extraction", {}).get("clause_id")
        or "Not extracted"
    )
    recommended_action = legal.get("recommended_action", "Proceed with escalation if insurer grounds are weak.")
    return (
        f"Contestable: {'Yes' if contestable else 'No'}\n"
        f"Clause Anchor: {clause_id}\n"
        f"Top Findings:\n{top_flags}\n"
        f"Recommended Next Step: {recommended_action}"
    )


def _run_analysis_for_session(session: ChatSession, settings: Any) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    missing = _missing_documents(session)
    if missing:
        raise ValueError(f"Missing required documents: {', '.join(REQUIRED_DOCUMENTS[d] for d in missing)}")

    forensic_llm, vision_llm = _resolve_llms(settings)
    allowed, warnings = _forensic_gate(settings, forensic_llm)
    if not allowed:
        raise ValueError(
            "LLM-based forensic extraction is required in strict mode. Set OPENAI_API_KEY or enable DEV_ALLOW_FALLBACK=true."
        )

    files = session.get("files", {})
    report = analyze_documents(
        policy_document_path=files["policy_document"],
        rejection_letter_path=files["rejection_letter"],
        discharge_summary_path=files["discharge_summary"],
        hospital_bill_path=files["hospital_bill"],
        vision_llm=vision_llm,
        forensic_llm=forensic_llm,
    )
    policy_age_years = _infer_policy_age_years(str(report.get("policy_excerpt", "")))
    legal = _derive_legal_analysis(
        policy_age_years=policy_age_years,
        rejection_reason=str(report.get("rejection_reason", "")),
    )

    session["policy_age_years"] = policy_age_years
    session["last_analysis"] = report
    session["legal_analysis"] = legal
    if warnings:
        report["runtime_warnings"] = warnings
    return report, legal, warnings


def _run_workflow_for_session(session: ChatSession, settings: Any) -> tuple[dict[str, Any], list[str]]:
    missing = _missing_documents(session)
    if missing:
        raise ValueError(f"Missing required documents: {', '.join(REQUIRED_DOCUMENTS[d] for d in missing)}")

    forensic_llm, vision_llm = _resolve_llms(settings)
    allowed, warnings = _forensic_gate(settings, forensic_llm)
    if not allowed:
        raise ValueError(
            "LLM-based forensic extraction is required in strict mode. Set OPENAI_API_KEY or enable DEV_ALLOW_FALLBACK=true."
        )

    if not session.get("last_analysis"):
        _run_analysis_for_session(session, settings)

    clean_claim_id = _clean_claim_id(session.get("session_id", "WEB-CLAIM"))
    policy_agent = PolicyAnalysisAgent(retriever=None, llm=None)
    evidence_agent = EvidenceMatchingAgent(vision_llm=vision_llm, forensic_llm=forensic_llm)
    portal_agent = PortalAutomationAgent()

    checkpoint_db = Path("/tmp") / f"claimclaw_web_{clean_claim_id}.sqlite"
    graph = build_workflow(
        policy_agent=policy_agent,
        evidence_agent=evidence_agent,
        portal_agent=portal_agent,
        checkpoint_db=checkpoint_db,
    )

    files = session.get("files", {})
    policy_age = float(session.get("policy_age_years", 3.0))
    initial_state = {
        "claim_id": clean_claim_id,
        "policy_age_years": policy_age,
        "policy_document_path": files["policy_document"],
        "rejection_letter_path": files["rejection_letter"],
        "discharge_summary_path": files["discharge_summary"],
        "hospital_bill_path": files["hospital_bill"],
        "insurer_name": "Unknown Insurer",
        "policy_number": "UNKNOWN-POLICY",
        "complainant_name": "Policyholder",
        "mobile": "9999999999",
        "email": "policyholder@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph=graph, initial_state=initial_state, claim_id=clean_claim_id)
    session["last_workflow"] = result
    return result, warnings


def _get_session(app: FastAPI, session_id: str) -> ChatSession:
    sessions: dict[str, ChatSession] = app.state.chat_sessions
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_session_not_found")
    return session


def _touch_session(session: ChatSession) -> None:
    session["updated_at"] = _now_iso()


def _session_snapshot(session: ChatSession) -> dict[str, Any]:
    created_at = session.get("created_at")
    return {
        "session_id": session["session_id"],
        "thread_name": session.get("thread_name", "New Claim"),
        "thread_note": session.get("thread_note", ""),
        "created_at": created_at,
        "updated_at": session.get("updated_at"),
        "age_days": _age_days(created_at),
        "uploaded_documents": _uploaded_documents(session),
        "missing_documents": _missing_documents(session),
        "message_count": len(session.get("messages", [])),
    }


def create_web_app() -> FastAPI:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    settings = load_settings()

    app = FastAPI(title="ClaimClaw Web Interface")
    app.state.chat_sessions = {}
    web_root = Path(__file__).resolve().parent / "web"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web_root / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/doctor")
    async def api_doctor(strict: bool = False) -> JSONResponse:
        report = run_preflight(project_root=project_root)
        if strict and report.get("summary", {}).get("warnings", 0) > 0 and report["status"] != "fail":
            report["status"] = "fail"
            report["strict_override"] = "warnings_promoted_to_failures"
        return JSONResponse(report)

    @app.post("/api/chat/start")
    async def chat_start(body: ChatStartIn | None = None, thread_name: str | None = None) -> JSONResponse:
        session_id = uuid.uuid4().hex[:12]
        work_dir = tempfile.mkdtemp(prefix=f"claimclaw-chat-{session_id}-")
        now_iso = _now_iso()
        resolved_thread_name = _clean_thread_name(
            body.thread_name if body and body.thread_name else thread_name
        )
        session: ChatSession = {
            "session_id": session_id,
            "thread_name": resolved_thread_name,
            "thread_note": "",
            "created_at": now_iso,
            "updated_at": now_iso,
            "work_dir": work_dir,
            "files": {},
            "messages": [],
            "last_analysis": {},
            "legal_analysis": {},
            "last_workflow": {},
        }
        app.state.chat_sessions[session_id] = session

        assistant_message = (
            "I am ClaimClaw. Upload these documents directly in chat: Policy Document, Rejection Letter, "
            "Discharge Summary, and Hospital Bill. Once all are uploaded, I will automatically run analysis and "
            "suggest the strongest next legal steps."
        )
        session["messages"].append({"role": "assistant", "content": assistant_message})

        return JSONResponse(
            {
                "session_id": session_id,
                "thread_name": resolved_thread_name,
                "assistant_message": assistant_message,
                "uploaded_documents": _uploaded_documents(session),
                "missing_documents": _missing_documents(session),
            }
        )

    @app.get("/api/chat/threads")
    async def chat_threads() -> JSONResponse:
        sessions: dict[str, ChatSession] = app.state.chat_sessions
        items = [_session_snapshot(s) for s in sessions.values()]
        items.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
        return JSONResponse({"threads": items})

    @app.get("/api/chat/thread/{session_id}")
    async def chat_thread(session_id: str) -> JSONResponse:
        session = _get_session(app, session_id)
        payload = _session_snapshot(session)
        payload.update(
            {
                "messages": session.get("messages", []),
                "last_analysis": session.get("last_analysis", {}),
                "legal_analysis": session.get("legal_analysis", {}),
                "last_workflow": session.get("last_workflow", {}),
            }
        )
        return JSONResponse(payload)

    @app.post("/api/chat/thread/{session_id}/note")
    async def chat_thread_note(session_id: str, body: ThreadNoteIn) -> JSONResponse:
        session = _get_session(app, session_id)
        session["thread_note"] = _clean_thread_note(body.note)
        _touch_session(session)
        return JSONResponse(_session_snapshot(session))

    @app.post("/api/chat/upload")
    async def chat_upload(
        session_id: str = Form(...),
        file: UploadFile = File(...),
        doc_type: str | None = Form(default=None),
    ) -> JSONResponse:
        session = _get_session(app, session_id)
        missing_before = _missing_documents(session)

        if doc_type and doc_type not in REQUIRED_DOCUMENTS:
            raise HTTPException(status_code=400, detail=f"Unsupported doc_type: {doc_type}")
        resolved_doc_type = doc_type or _infer_doc_type(file.filename or "document.pdf", missing_before)

        target_path = Path(session["work_dir"]) / f"{resolved_doc_type}.pdf"
        await _save_upload(file, target_path)
        session["files"][resolved_doc_type] = str(target_path)

        missing_after = _missing_documents(session)
        uploaded = _uploaded_documents(session)

        assistant_message = f"Received {REQUIRED_DOCUMENTS[resolved_doc_type]}."
        payload: dict[str, Any] = {
            "session_id": session_id,
            "resolved_doc_type": resolved_doc_type,
            "uploaded_documents": uploaded,
            "missing_documents": missing_after,
        }

        if missing_after:
            assistant_message += (
                " Please upload: "
                + ", ".join(REQUIRED_DOCUMENTS[doc] for doc in missing_after)
                + "."
            )
        else:
            assistant_message += " All required documents received. Running analysis now..."
            try:
                report, legal, _warnings = _run_analysis_for_session(session, settings)
                summary = _analysis_summary_text(report, legal)
                assistant_message += f"\n\nAnalysis complete.\n{summary}"
                payload["analysis"] = report
                payload["legal_analysis"] = legal
                payload["analysis_summary"] = summary
            except Exception as exc:
                assistant_message += f"\n\nAnalysis failed: {exc}"
                payload["analysis_error"] = str(exc)

        session["messages"].append({"role": "assistant", "content": assistant_message})
        _touch_session(session)
        payload["assistant_message"] = assistant_message
        return JSONResponse(payload)

    @app.post("/api/chat/message")
    async def chat_message(body: ChatMessageIn) -> JSONResponse:
        session = _get_session(app, body.session_id)
        text = body.message.strip()
        if not text:
            _touch_session(session)
            return JSONResponse(
                {
                    "session_id": body.session_id,
                    "assistant_message": "Send a message or upload documents to continue.",
                    "uploaded_documents": _uploaded_documents(session),
                    "missing_documents": _missing_documents(session),
                }
            )

        session["messages"].append({"role": "user", "content": text})
        lowered = text.lower()
        missing = _missing_documents(session)

        if missing:
            assistant_message = (
                "Upload pending documents first: "
                + ", ".join(REQUIRED_DOCUMENTS[doc] for doc in missing)
                + ". I will run analysis automatically once all are uploaded."
            )
            session["messages"].append({"role": "assistant", "content": assistant_message})
            _touch_session(session)
            return JSONResponse(
                {
                    "session_id": body.session_id,
                    "assistant_message": assistant_message,
                    "uploaded_documents": _uploaded_documents(session),
                    "missing_documents": missing,
                }
            )

        wants_analysis = any(k in lowered for k in ("analyze", "analysis", "review", "check claim"))
        wants_next = any(k in lowered for k in ("next step", "what can", "what should", "what now"))
        wants_escalation = any(k in lowered for k in ("escalate", "ombudsman", "bima", "file complaint"))
        wants_status = any(k in lowered for k in ("status", "progress", "stage"))

        payload: dict[str, Any] = {
            "session_id": body.session_id,
            "uploaded_documents": _uploaded_documents(session),
            "missing_documents": [],
        }

        try:
            if wants_analysis or not session.get("last_analysis"):
                report, legal, _warnings = _run_analysis_for_session(session, settings)
                summary = _analysis_summary_text(report, legal)
                assistant_message = f"Analysis complete.\n{summary}\n\nSay 'escalate now' if you want me to trigger Bima Bharosa workflow."
                payload["analysis"] = report
                payload["legal_analysis"] = legal
                payload["analysis_summary"] = summary
            elif wants_escalation:
                result, warnings = _run_workflow_for_session(session, settings)
                assistant_message = (
                    f"Escalation workflow executed. Current stage: {result.get('stage', 'unknown')}."
                )
                payload["workflow"] = result
                if warnings:
                    payload["runtime_warnings"] = warnings
            elif wants_next:
                legal = session.get("legal_analysis", {})
                assistant_message = (
                    "Recommended next step:\n"
                    f"{legal.get('recommended_action', 'Proceed to grievance escalation on weak repudiation grounds.')}\n\n"
                    "If you want immediate escalation filing flow, say: escalate now."
                )
                payload["legal_analysis"] = legal
            elif wants_status:
                stage = session.get("last_workflow", {}).get("stage", "analysis_pending")
                has_analysis = bool(session.get("last_analysis"))
                assistant_message = (
                    f"Status: stage={stage}, analysis={'ready' if has_analysis else 'not_run'}. "
                    f"Uploaded {len(_uploaded_documents(session))}/4 required documents."
                )
            else:
                legal = session.get("legal_analysis", {})
                assistant_message = (
                    "I have your documents and can proceed. Say 'analyze now' for fresh analysis, or 'escalate now' "
                    "to run the escalation workflow.\n\n"
                    f"Current recommended step: {legal.get('recommended_action', 'Run analysis first.')}"
                )
        except Exception as exc:
            assistant_message = f"Operation failed: {exc}"
            payload["error"] = str(exc)

        session["messages"].append({"role": "assistant", "content": assistant_message})
        _touch_session(session)
        payload["assistant_message"] = assistant_message
        return JSONResponse(payload)

    return app
