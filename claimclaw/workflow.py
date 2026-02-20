from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from .agents import EvidenceMatchingAgent, PolicyAnalysisAgent, PortalAutomationAgent
from .schemas import ComplaintPayload, PortalCredentials

MORATORIUM_CITATION_ID = "IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1"


class ClaimWorkflowState(TypedDict, total=False):
    claim_id: str
    policy_age_years: float
    policy_document_path: str
    rejection_letter_path: str
    discharge_summary_path: str
    hospital_bill_path: str
    insurer_name: str
    policy_number: str
    complainant_name: str
    mobile: str
    email: str
    insurer_reply_received: bool
    dry_run_portal: bool

    evidence_report: dict[str, Any]
    legal_analysis: dict[str, Any]
    rebuttal_email: str
    portal_result: dict[str, str]
    stage: str
    timeline: list[str]


def _append_timeline(state: ClaimWorkflowState, event: str) -> list[str]:
    existing = list(state.get("timeline", []))
    timestamp = datetime.now(UTC).isoformat()
    existing.append(f"{timestamp} - {event}")
    return existing


def _draft_rebuttal_email_body(state: ClaimWorkflowState) -> str:
    legal = state["legal_analysis"]
    evidence = state["evidence_report"]

    flags = evidence.get("flags", [])
    bullet_flags = "\n".join(f"- {flag}" for flag in flags) if flags else "- Evidence under review."

    legal_basis = legal.get("legal_basis", "")
    recommended_action = legal.get("recommended_action", "")
    rejection_reason = evidence.get("rejection_reason", "Not specified")
    constructive_note = str(evidence.get("constructive_knowledge_note", "")).strip()
    constructive_section = ""
    if constructive_note:
        constructive_section = f"Constructive knowledge position:\n{constructive_note}\n\n"

    return (
        f"Subject: Immediate Reconsideration of Wrongful Claim Rejection - Policy {state['policy_number']}\n\n"
        f"To: Grievance Officer, {state['insurer_name']}\n\n"
        f"This is a formal challenge to your rejection based on: {rejection_reason}.\n\n"
        "Key contradictions in your repudiation:\n"
        f"{bullet_flags}\n\n"
        f"{constructive_section}"
        "Regulatory basis:\n"
        f"{legal_basis}\n\n"
        "Required action:\n"
        f"{recommended_action}\n\n"
        "You are directed to reverse this rejection and process admissible claim amounts immediately. "
        "Failing resolution, I will file and pursue escalation on Bima Bharosa and before the Insurance Ombudsman without further notice.\n\n"
        f"Regards,\n{state['complainant_name']}"
    )


def build_workflow(
    policy_agent: PolicyAnalysisAgent,
    evidence_agent: EvidenceMatchingAgent,
    portal_agent: PortalAutomationAgent,
    checkpoint_db: str | Path,
) -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    def analyze_documents_node(state: ClaimWorkflowState) -> ClaimWorkflowState:
        evidence_report = evidence_agent.run(
            policy_document_path=state["policy_document_path"],
            rejection_letter_path=state["rejection_letter_path"],
            discharge_summary_path=state["discharge_summary_path"],
            hospital_bill_path=state["hospital_bill_path"],
        )
        legal_analysis = policy_agent.run(
            policy_age_years=state["policy_age_years"],
            rejection_reason=evidence_report.get("rejection_reason", ""),
            alleged_fraud=False,
        )

        forensic = evidence_report.get("forensic_validation", {})
        global_reasoning = evidence_report.get("global_reasoning", {})
        raw_clause_id = forensic.get("clause_id") or forensic.get("clause_identifier") or "UNKNOWN"
        clause_id = str(raw_clause_id).strip()
        lower = clause_id.lower()
        for prefix in ("clause ", "section ", "clause:", "section:"):
            if lower.startswith(prefix):
                clause_id = clause_id[len(prefix) :].strip()
                break
        conflict_statement = (
            "Conflict Detected: Policy Clause "
            f"{clause_id} cited by the insurer is now VOID as per IRDAI Master Circular Ref: "
            "IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1."
        )
        legal_basis = str(legal_analysis.get("legal_basis", "")).strip()

        clinical_invalid = bool(global_reasoning.get("clinical_distinction_invalid"))
        if clinical_invalid:
            clinical_reason = str(global_reasoning.get("clinical_distinction_reason", "")).strip()
            clinical_statement = (
                f"Clinical Distinction: {clinical_reason}"
                if clinical_reason
                else (
                    "Clinical Distinction: chronic/pre-existing rejection basis does not match acute "
                    "clinical presentation in discharge records."
                )
            )
            if clinical_statement not in legal_basis:
                legal_basis = f"{legal_basis}\n\n{clinical_statement}".strip() if legal_basis else clinical_statement

        constructive_knowledge = bool(global_reasoning.get("constructive_knowledge"))
        if constructive_knowledge:
            constructive_reason = str(global_reasoning.get("constructive_knowledge_reason", "")).strip()
            constructive_statement = (
                f"Constructive Knowledge: {constructive_reason}"
                if constructive_reason
                else (
                    "Constructive Knowledge: disclosed maintenance medication established underwriting "
                    "knowledge of the treated condition."
                )
            )
            if constructive_statement not in legal_basis:
                legal_basis = (
                    f"{legal_basis}\n\n{constructive_statement}".strip()
                    if legal_basis
                    else constructive_statement
                )

        statutory_primacy_override = bool(global_reasoning.get("statutory_primacy_override")) or bool(
            forensic.get("irdai_override_applies")
        )
        if legal_analysis.get("illegal_rejection") or statutory_primacy_override:
            legal_analysis["citation_id"] = MORATORIUM_CITATION_ID
            legal_analysis["legal_basis"] = legal_basis
            if conflict_statement not in legal_basis:
                legal_analysis["legal_basis"] = (
                    f"{legal_basis}\n\n{conflict_statement}".strip() if legal_basis else conflict_statement
                )
            statutory_reason = str(global_reasoning.get("statutory_primacy_reason", "")).strip()
            if statutory_reason and statutory_reason not in legal_analysis["legal_basis"]:
                legal_analysis["legal_basis"] = (
                    f"{legal_analysis['legal_basis']}\n\nStatutory Primacy: {statutory_reason}".strip()
                )
        else:
            legal_analysis["legal_basis"] = legal_basis

        has_clause_anchor = bool(
            str(forensic.get("clause_id") or forensic.get("clause_identifier") or "").strip()
        )
        if forensic.get("ghost_rejection") or (
            has_clause_anchor and forensic.get("is_clause_present_in_policy") is False
        ):
            legal_analysis["recommended_action"] = (
                'Immediate Escalation: The insurer is citing a non-existent policy clause. '
                'File a "Bad Faith" grievance on Bima Bharosa immediately.'
            )
        else:
            recommended_action = str(legal_analysis.get("recommended_action", "")).strip()
            if clinical_invalid:
                clinical_action = (
                    "Challenge medical mismatch: acute onset condition cannot be denied under chronic/pre-existing "
                    "disease exclusion."
                )
                if clinical_action not in recommended_action:
                    recommended_action = (
                        f"{recommended_action} {clinical_action}".strip()
                        if recommended_action
                        else clinical_action
                    )
            if constructive_knowledge:
                constructive_action = (
                    "Invoke constructive knowledge: disclosed maintenance medication put insurer on notice, "
                    "waiving non-disclosure defense."
                )
                if constructive_action not in recommended_action:
                    recommended_action = (
                        f"{recommended_action} {constructive_action}".strip()
                        if recommended_action
                        else constructive_action
                    )
            legal_analysis["recommended_action"] = recommended_action

        return {
            "evidence_report": evidence_report,
            "legal_analysis": legal_analysis,
            "stage": "analysis_complete",
            "timeline": _append_timeline(state, "Analysis complete"),
        }

    def draft_rebuttal_node(state: ClaimWorkflowState) -> ClaimWorkflowState:
        rebuttal = _draft_rebuttal_email_body(state)
        return {
            "rebuttal_email": rebuttal,
            "stage": "rebuttal_drafted",
            "timeline": _append_timeline(state, "Rebuttal drafted"),
        }

    def wait_reply_node(state: ClaimWorkflowState) -> ClaimWorkflowState:
        # Simulation node: in production, this can be scheduled on an async queue with real 24h wait.
        no_reply = not state.get("insurer_reply_received", False)
        stage = "escalation_required" if no_reply else "resolved_by_insurer"
        event = "No insurer reply in 24h, escalating" if no_reply else "Insurer replied in time"
        return {
            "stage": stage,
            "timeline": _append_timeline(state, event),
        }

    def file_complaint_node(state: ClaimWorkflowState) -> ClaimWorkflowState:
        payload = ComplaintPayload(
            complainant_name=state["complainant_name"],
            insurer_name=state["insurer_name"],
            policy_number=state["policy_number"],
            mobile=state["mobile"],
            email=state["email"],
            grievance_category="Health Insurance Claim Rejection",
            issue_summary=state["rebuttal_email"],
            relief_sought=(
                "Reverse repudiation, settle admissible amount with interest, and issue written compliance response."
            ),
            attachments=[
                state["policy_document_path"],
                state["rejection_letter_path"],
                state["discharge_summary_path"],
                state["hospital_bill_path"],
            ],
        )
        credentials = PortalCredentials(username=state["email"], password="CHANGE_ME")
        portal_result = portal_agent.run(
            payload=payload,
            credentials=credentials,
            dry_run=state.get("dry_run_portal", True),
            headless=False,
        )
        return {
            "portal_result": portal_result,
            "stage": "bima_bharosa_filed",
            "timeline": _append_timeline(state, "Bima Bharosa filing executed"),
        }

    def route_after_wait(state: ClaimWorkflowState) -> Literal["escalate", "end"]:
        if state.get("stage") == "escalation_required":
            return "escalate"
        return "end"

    graph = StateGraph(ClaimWorkflowState)
    graph.add_node("analyze_documents", analyze_documents_node)
    graph.add_node("draft_rebuttal", draft_rebuttal_node)
    graph.add_node("wait_for_reply", wait_reply_node)
    graph.add_node("file_bima_bharosa", file_complaint_node)

    graph.add_edge(START, "analyze_documents")
    graph.add_edge("analyze_documents", "draft_rebuttal")
    graph.add_edge("draft_rebuttal", "wait_for_reply")
    graph.add_conditional_edges(
        "wait_for_reply",
        route_after_wait,
        {"escalate": "file_bima_bharosa", "end": END},
    )
    graph.add_edge("file_bima_bharosa", END)

    checkpoint_db = Path(checkpoint_db)

    def _build_checkpointer(db_path: Path) -> SqliteSaver:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    try:
        checkpointer = _build_checkpointer(checkpoint_db)
    except sqlite3.OperationalError:
        fallback_db = Path("/tmp/claimclaw_state.sqlite")
        checkpointer = _build_checkpointer(fallback_db)

    return graph.compile(checkpointer=checkpointer)


def run_workflow(graph: Any, initial_state: ClaimWorkflowState, claim_id: str) -> ClaimWorkflowState:
    config = {"configurable": {"thread_id": claim_id}}
    try:
        snapshot = graph.get_state(config=config)
        if snapshot and snapshot.values:
            existing = dict(snapshot.values)
            existing_stage = existing.get("stage", "")
            # Resume-safe behavior: if a final stage is already reached, avoid duplicate re-execution.
            if existing_stage in {"bima_bharosa_filed", "resolved_by_insurer"}:
                return existing
    except Exception:
        # If checkpoint read fails, continue with normal invocation path.
        pass
    final_state = graph.invoke(initial_state, config=config)
    return final_state
