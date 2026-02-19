from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .agents import EvidenceMatchingAgent, PolicyAnalysisAgent, PortalAutomationAgent
from .config import load_settings
from .evidence_matcher import analyze_documents
from .legalbrain.rules import (
    check_moratorium_eligibility,
    should_override_nondisclosure_rejection,
)
from .legal_rag import (
    answer_legal_question,
    default_hidden_bp_counter,
    extract_moratorium_clause_hint,
    ingest_legal_corpus,
    load_legal_retriever,
    moratorium_rule_check,
)
from .workflow import build_workflow, run_workflow


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _build_openai_chat_llm(model_name: str) -> Any | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    return ChatOpenAI(model=model_name, temperature=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimclaw", description="ClaimClaw CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest-legal", help="Ingest IRDAI legal corpus into vector DB")
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--persist-dir", required=True)
    ingest.add_argument("--store", choices=["chroma", "faiss"], default="chroma")

    ask = sub.add_parser("ask-legal", help="Query ingested legal corpus")
    ask.add_argument("--persist-dir", required=True)
    ask.add_argument("--store", choices=["chroma", "faiss"], default="chroma")
    ask.add_argument("--question", required=True)

    validate = sub.add_parser("validate-moratorium", help="Run 5-year moratorium validation")
    validate.add_argument("--policy-age", type=float, required=True)
    validate.add_argument("--rejection-reason", required=True)
    validate.add_argument("--alleged-fraud", action="store_true")

    date_validate = sub.add_parser(
        "check-moratorium-date",
        help="Date-aware moratorium check (5-year vs 8-year transition logic)",
    )
    date_validate.add_argument("--policy-start-date", required=True, help="YYYY-MM-DD")
    date_validate.add_argument("--as-of-date", default=None, help="YYYY-MM-DD")
    date_validate.add_argument("--rejection-reason", default="Non-disclosure")
    date_validate.add_argument("--alleged-fraud", action="store_true")

    live = sub.add_parser(
        "live-legal-brain",
        help="Live scrape IRDAI + Ombudsman and embed into in-memory ChromaDB",
    )
    live.add_argument("--include-2025-updates", default="true")
    live.add_argument("--max-ombudsman-pages", type=int, default=20)
    live.add_argument("--embedding-model", default="text-embedding-3-large")
    live.add_argument("--question", default="")

    docs = sub.add_parser("analyze-docs", help="Analyze claim PDFs for contestability")
    docs.add_argument("--policy-document", required=True)
    docs.add_argument("--rejection-letter", required=True)
    docs.add_argument("--discharge-summary", required=True)
    docs.add_argument("--hospital-bill", required=True)

    workflow = sub.add_parser("run-workflow", help="Run full LangGraph claim workflow")
    workflow.add_argument("--claim-id", required=True)
    workflow.add_argument("--policy-age", type=float, required=True)
    workflow.add_argument("--policy-document", required=True)
    workflow.add_argument("--rejection-letter", required=True)
    workflow.add_argument("--discharge-summary", required=True)
    workflow.add_argument("--hospital-bill", required=True)
    workflow.add_argument("--insurer-name", default="Unknown Insurer")
    workflow.add_argument("--policy-number", default="UNKNOWN-POLICY")
    workflow.add_argument("--complainant-name", default="Policyholder")
    workflow.add_argument("--mobile", default="9999999999")
    workflow.add_argument("--email", default="policyholder@example.com")
    workflow.add_argument("--insurer-reply-received", action="store_true")
    workflow.add_argument("--checkpoint-db", default="storage/claimclaw_state.sqlite")
    workflow.add_argument("--dry-run-portal", default="true")

    return parser


def main() -> None:
    load_dotenv()
    settings = load_settings()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest-legal":
        result = ingest_legal_corpus(
            source_dir=args.source_dir,
            persist_dir=args.persist_dir,
            store=args.store,
        )
        _json_print(result)
        return

    if args.command == "ask-legal":
        retriever = load_legal_retriever(
            persist_dir=args.persist_dir,
            store=args.store,
            k=5,
        )
        result = answer_legal_question(
            question=args.question,
            retriever=retriever,
            llm=None,
        )
        result["moratorium_clause_hint"] = extract_moratorium_clause_hint(retriever)
        _json_print(result)
        return

    if args.command == "validate-moratorium":
        legal_position = moratorium_rule_check(
            policy_age_years=args.policy_age,
            rejection_reason=args.rejection_reason,
            alleged_fraud=args.alleged_fraud,
        )
        payload = {
            "moratorium_applies": legal_position.moratorium_applies,
            "illegal_rejection": legal_position.illegal_rejection,
            "legal_basis": legal_position.legal_basis,
            "recommended_action": legal_position.recommended_action,
        }
        if args.policy_age >= 6 and "bp" in args.rejection_reason.lower():
            payload["validation_case"] = default_hidden_bp_counter(args.policy_age)
        _json_print(payload)
        return

    if args.command == "check-moratorium-date":
        base = check_moratorium_eligibility(
            policy_start_date=args.policy_start_date,
            as_of_date=args.as_of_date,
        )
        decision = should_override_nondisclosure_rejection(
            policy_start_date=args.policy_start_date,
            rejection_reason=args.rejection_reason,
            as_of_date=args.as_of_date,
            alleged_fraud=args.alleged_fraud,
        )
        _json_print(
            {
                "moratorium": base.to_dict(),
                "override_decision": decision.to_dict(),
            }
        )
        return

    if args.command == "live-legal-brain":
        try:
            from .legalbrain.ombudsman_kb import (
                build_ombudsman_few_shots,
                build_session_vectorstore,
                scrape_ombudsman_awards,
            )
            from .legalbrain.scraper import get_latest_irdai_circulars
        except Exception as exc:
            _json_print(
                {
                    "error": "live_legal_brain_dependencies_missing",
                    "detail": str(exc),
                }
            )
            return

        include_2025 = str(args.include_2025_updates).lower() in {"1", "true", "yes", "y"}
        irdai_circulars: list[dict[str, Any]] = []
        ombudsman_findings: list[dict[str, Any]] = []
        scrape_errors: dict[str, str] = {}

        try:
            irdai_circulars = get_latest_irdai_circulars(include_2025_updates=include_2025)
        except Exception as exc:
            scrape_errors["irdai_scrape_error"] = str(exc)

        try:
            ombudsman_findings = scrape_ombudsman_awards(max_pages=args.max_ombudsman_pages)
        except Exception as exc:
            scrape_errors["ombudsman_scrape_error"] = str(exc)

        few_shots = build_ombudsman_few_shots(ombudsman_findings)

        payload: dict[str, Any] = {
            "irdai_circulars_found": len(irdai_circulars),
            "ombudsman_cases_found": len(ombudsman_findings),
            "few_shot_examples": {k: len(v) for k, v in few_shots.items()},
            "irdai_titles": [item.get("title", "") for item in irdai_circulars[:5]],
            "ombudsman_titles": [item.get("title", "") for item in ombudsman_findings[:5]],
        }
        payload.update(scrape_errors)

        if not os.getenv("OPENAI_API_KEY"):
            payload["vectorstore_status"] = "skipped_missing_openai_api_key"
        else:
            try:
                vectorstore = build_session_vectorstore(
                    irdai_circulars=irdai_circulars,
                    ombudsman_findings=ombudsman_findings,
                    embedding_model=args.embedding_model,
                )
                payload["vectorstore_status"] = "ready_in_memory"
                if args.question:
                    docs = vectorstore.similarity_search(args.question, k=5)
                    payload["question"] = args.question
                    payload["top_hits"] = [
                        {
                            "source_type": doc.metadata.get("source_type", "unknown"),
                            "title": doc.metadata.get("title", ""),
                            "source": doc.metadata.get("source", ""),
                            "snippet": doc.page_content[:360],
                        }
                        for doc in docs
                    ]
            except Exception as exc:
                payload["vectorstore_status"] = "failed"
                payload["vectorstore_error"] = str(exc)

        _json_print(payload)
        return

    if args.command == "analyze-docs":
        forensic_llm = None
        vision_llm = None
        if settings.llm_provider.lower() == "openai":
            forensic_llm = _build_openai_chat_llm(settings.openai_chat_model)
            vision_llm = _build_openai_chat_llm(settings.openai_vision_model)
        if forensic_llm is None:
            _json_print(
                {
                    "error": "forensic_llm_unavailable",
                    "detail": (
                        "LLM-based forensic extraction is required in production mode. "
                        "Set OPENAI_API_KEY and OPENAI_CHAT_MODEL."
                    ),
                }
            )
            return

        report = analyze_documents(
            policy_document_path=args.policy_document,
            rejection_letter_path=args.rejection_letter,
            discharge_summary_path=args.discharge_summary,
            hospital_bill_path=args.hospital_bill,
            vision_llm=vision_llm,
            forensic_llm=forensic_llm,
        )
        _json_print(report)
        return

    if args.command == "run-workflow":
        forensic_llm = None
        vision_llm = None
        if settings.llm_provider.lower() == "openai":
            forensic_llm = _build_openai_chat_llm(settings.openai_chat_model)
            vision_llm = _build_openai_chat_llm(settings.openai_vision_model)
        if forensic_llm is None:
            _json_print(
                {
                    "error": "forensic_llm_unavailable",
                    "detail": (
                        "LLM-based forensic extraction is required in production mode. "
                        "Set OPENAI_API_KEY and OPENAI_CHAT_MODEL."
                    ),
                }
            )
            return

        policy_agent = PolicyAnalysisAgent(retriever=None, llm=None)
        evidence_agent = EvidenceMatchingAgent(vision_llm=vision_llm, forensic_llm=forensic_llm)
        portal_agent = PortalAutomationAgent()

        graph = build_workflow(
            policy_agent=policy_agent,
            evidence_agent=evidence_agent,
            portal_agent=portal_agent,
            checkpoint_db=args.checkpoint_db or settings.checkpoint_db,
        )
        dry_run = str(args.dry_run_portal).lower() in {"1", "true", "yes", "y"}
        initial_state = {
            "claim_id": args.claim_id,
            "policy_age_years": args.policy_age,
            "policy_document_path": str(Path(args.policy_document)),
            "rejection_letter_path": str(Path(args.rejection_letter)),
            "discharge_summary_path": str(Path(args.discharge_summary)),
            "hospital_bill_path": str(Path(args.hospital_bill)),
            "insurer_name": args.insurer_name,
            "policy_number": args.policy_number,
            "complainant_name": args.complainant_name,
            "mobile": args.mobile,
            "email": args.email,
            "insurer_reply_received": bool(args.insurer_reply_received),
            "dry_run_portal": dry_run,
        }
        result = run_workflow(graph=graph, initial_state=initial_state, claim_id=args.claim_id)
        _json_print(result)
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
