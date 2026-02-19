from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .evidence_matcher import analyze_documents
from .legal_rag import default_hidden_bp_counter, moratorium_rule_check
from .portal_automation import file_bima_bharosa_complaint
from .schemas import ComplaintPayload, PortalCredentials


class PolicyAnalysisAgent:
    def __init__(self, retriever: Any | None = None, llm: Any | None = None) -> None:
        self.retriever = retriever
        self.llm = llm

    def run(
        self,
        policy_age_years: float,
        rejection_reason: str,
        alleged_fraud: bool = False,
    ) -> dict[str, Any]:
        position = moratorium_rule_check(
            policy_age_years=policy_age_years,
            rejection_reason=rejection_reason,
            alleged_fraud=alleged_fraud,
        )

        data = asdict(position)
        if policy_age_years >= 6 and "bp" in rejection_reason.lower():
            data["validation_example"] = default_hidden_bp_counter(policy_age_years=policy_age_years)
        return data


class EvidenceMatchingAgent:
    def __init__(
        self,
        vision_llm: Any | None = None,
        forensic_llm: Any | None = None,
    ) -> None:
        self.vision_llm = vision_llm
        self.forensic_llm = forensic_llm

    def run(
        self,
        policy_document_path: str,
        rejection_letter_path: str,
        discharge_summary_path: str,
        hospital_bill_path: str,
    ) -> dict[str, Any]:
        return analyze_documents(
            policy_document_path=policy_document_path,
            rejection_letter_path=rejection_letter_path,
            discharge_summary_path=discharge_summary_path,
            hospital_bill_path=hospital_bill_path,
            vision_llm=self.vision_llm,
            forensic_llm=self.forensic_llm,
        )


class PortalAutomationAgent:
    def run(
        self,
        payload: ComplaintPayload,
        credentials: PortalCredentials,
        dry_run: bool = True,
        headless: bool = False,
    ) -> dict[str, str]:
        return file_bima_bharosa_complaint(
            payload=payload,
            credentials=credentials,
            dry_run=dry_run,
            headless=headless,
        )
