from __future__ import annotations

from pathlib import Path

from claimclaw.workflow import build_workflow, run_workflow


class _StubPolicyAgent:
    def run(
        self,
        policy_age_years: float,
        rejection_reason: str,
        alleged_fraud: bool = False,
    ) -> dict[str, object]:
        return {
            "illegal_rejection": True,
            "moratorium_applies": True,
            "legal_basis": "Test legal basis",
            "recommended_action": "Test action",
        }


class _StubEvidenceAgent:
    def run(
        self,
        policy_document_path: str,
        rejection_letter_path: str,
        discharge_summary_path: str,
        hospital_bill_path: str,
    ) -> dict[str, object]:
        return {
            "rejection_reason": "Non-disclosure of PED",
            "flags": ["Test contradiction"],
        }


class _StubPortalAgent:
    def run(
        self,
        payload: object,
        credentials: object,
        dry_run: bool = True,
        headless: bool = False,
    ) -> dict[str, str]:
        return {"status": "dry_run", "message": "stub"}


def test_workflow_includes_hardcoded_citation_when_moratorium_violation(
    tmp_path: Path,
) -> None:
    graph = build_workflow(
        policy_agent=_StubPolicyAgent(),
        evidence_agent=_StubEvidenceAgent(),
        portal_agent=_StubPortalAgent(),
        checkpoint_db=tmp_path / "state.sqlite",
    )
    initial_state = {
        "claim_id": "WF-CIT-1",
        "policy_age_years": 6.0,
        "policy_document_path": "Policy.pdf",
        "rejection_letter_path": "Reject.pdf",
        "discharge_summary_path": "Discharge.pdf",
        "hospital_bill_path": "Bill.pdf",
        "insurer_name": "Test Insurer",
        "policy_number": "P-12345",
        "complainant_name": "Test User",
        "mobile": "9999999999",
        "email": "test@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph, initial_state, claim_id="WF-CIT-1")
    assert result["legal_analysis"]["citation_id"] == "IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1"


def test_workflow_adds_conflict_statement_to_legal_basis_when_moratorium_violation(
    tmp_path: Path,
) -> None:
    class _ConflictEvidenceAgent:
        def run(
            self,
            policy_document_path: str,
            rejection_letter_path: str,
            discharge_summary_path: str,
            hospital_bill_path: str,
        ) -> dict[str, object]:
            return {
                "rejection_reason": "Non-disclosure of PED",
                "flags": ["Test contradiction"],
                "forensic_validation": {
                    "clause_id": "4.1",
                    "is_clause_present_in_policy": True,
                    "ghost_rejection": False,
                },
            }

    graph = build_workflow(
        policy_agent=_StubPolicyAgent(),
        evidence_agent=_ConflictEvidenceAgent(),
        portal_agent=_StubPortalAgent(),
        checkpoint_db=tmp_path / "state_conflict.sqlite",
    )
    initial_state = {
        "claim_id": "WF-CIT-2",
        "policy_age_years": 6.0,
        "policy_document_path": "Policy.pdf",
        "rejection_letter_path": "Reject.pdf",
        "discharge_summary_path": "Discharge.pdf",
        "hospital_bill_path": "Bill.pdf",
        "insurer_name": "Test Insurer",
        "policy_number": "P-12345",
        "complainant_name": "Test User",
        "mobile": "9999999999",
        "email": "test@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph, initial_state, claim_id="WF-CIT-2")
    assert (
        "Conflict Detected: Policy Clause 4.1 cited by the insurer is now VOID as per "
        "IRDAI Master Circular Ref: IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1."
    ) in result["legal_analysis"]["legal_basis"]


def test_workflow_ghost_rejection_overrides_recommended_action(tmp_path: Path) -> None:
    class _GhostEvidenceAgent:
        def run(
            self,
            policy_document_path: str,
            rejection_letter_path: str,
            discharge_summary_path: str,
            hospital_bill_path: str,
        ) -> dict[str, object]:
            return {
                "rejection_reason": "Repudiation under Clause 9.9",
                "flags": ["Ghost Rejection"],
                "forensic_validation": {
                    "clause_id": "9.9",
                    "is_clause_present_in_policy": False,
                    "ghost_rejection": True,
                },
            }

    class _NonIllegalPolicyAgent:
        def run(
            self,
            policy_age_years: float,
            rejection_reason: str,
            alleged_fraud: bool = False,
        ) -> dict[str, object]:
            return {
                "illegal_rejection": False,
                "moratorium_applies": False,
                "legal_basis": "Test legal basis",
                "recommended_action": "Generic action",
            }

    graph = build_workflow(
        policy_agent=_NonIllegalPolicyAgent(),
        evidence_agent=_GhostEvidenceAgent(),
        portal_agent=_StubPortalAgent(),
        checkpoint_db=tmp_path / "state_ghost.sqlite",
    )
    initial_state = {
        "claim_id": "WF-CIT-3",
        "policy_age_years": 2.0,
        "policy_document_path": "Policy.pdf",
        "rejection_letter_path": "Reject.pdf",
        "discharge_summary_path": "Discharge.pdf",
        "hospital_bill_path": "Bill.pdf",
        "insurer_name": "Test Insurer",
        "policy_number": "P-12345",
        "complainant_name": "Test User",
        "mobile": "9999999999",
        "email": "test@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph, initial_state, claim_id="WF-CIT-3")
    assert (
        result["legal_analysis"]["recommended_action"]
        == 'Immediate Escalation: The insurer is citing a non-existent policy clause. '
        'File a "Bad Faith" grievance on Bima Bharosa immediately.'
    )


def test_workflow_rebuttal_includes_constructive_knowledge_note(tmp_path: Path) -> None:
    class _ConstructiveEvidenceAgent:
        def run(
            self,
            policy_document_path: str,
            rejection_letter_path: str,
            discharge_summary_path: str,
            hospital_bill_path: str,
        ) -> dict[str, object]:
            return {
                "rejection_reason": "Non-disclosure of Hypertension",
                "flags": ["Constructive knowledge check: Telmisartan indicates Hypertension disclosure."],
                "constructive_knowledge_note": (
                    "The insurer had constructive knowledge of the condition as the specific maintenance "
                    "medication (Telmisartan 40mg) was explicitly disclosed in the proposal form. Failure "
                    "to seek further clarification at the time of underwriting waives the right to reject now."
                ),
                "forensic_validation": {
                    "clause_id": "Clause 6.3",
                    "is_clause_present_in_policy": True,
                    "ghost_rejection": False,
                },
            }

    graph = build_workflow(
        policy_agent=_StubPolicyAgent(),
        evidence_agent=_ConstructiveEvidenceAgent(),
        portal_agent=_StubPortalAgent(),
        checkpoint_db=tmp_path / "state_constructive.sqlite",
    )
    initial_state = {
        "claim_id": "WF-CIT-4",
        "policy_age_years": 6.0,
        "policy_document_path": "Policy.pdf",
        "rejection_letter_path": "Reject.pdf",
        "discharge_summary_path": "Discharge.pdf",
        "hospital_bill_path": "Bill.pdf",
        "insurer_name": "Test Insurer",
        "policy_number": "P-12345",
        "complainant_name": "Test User",
        "mobile": "9999999999",
        "email": "test@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph, initial_state, claim_id="WF-CIT-4")
    assert (
        "The insurer had constructive knowledge of the condition as the specific maintenance medication "
        "(Telmisartan 40mg) was explicitly disclosed in the proposal form."
    ) in result["rebuttal_email"]


def test_workflow_applies_global_brain_layers_to_legal_basis_and_action(tmp_path: Path) -> None:
    class _GlobalEvidenceAgent:
        def run(
            self,
            policy_document_path: str,
            rejection_letter_path: str,
            discharge_summary_path: str,
            hospital_bill_path: str,
        ) -> dict[str, object]:
            return {
                "rejection_reason": "Non-disclosure of CKD under Clause 4.2",
                "flags": ["Clinical distinction: AKI is acute and not CKD."],
                "forensic_validation": {
                    "clause_id": "Clause 4.2",
                    "is_clause_present_in_policy": True,
                    "ghost_rejection": False,
                    "irdai_override_applies": True,
                },
                "global_reasoning": {
                    "clinical_distinction_invalid": True,
                    "clinical_distinction_reason": "AKI acute episode cannot be denied under CKD chronic exclusion.",
                    "constructive_knowledge": True,
                    "constructive_knowledge_reason": "Telmisartan disclosure established underwriting knowledge.",
                    "statutory_primacy_override": True,
                    "statutory_primacy_reason": "Private 8-year clause conflicts with IRDAI 2024 5-year standard.",
                },
            }

    class _NonIllegalPolicyAgent:
        def run(
            self,
            policy_age_years: float,
            rejection_reason: str,
            alleged_fraud: bool = False,
        ) -> dict[str, object]:
            return {
                "illegal_rejection": False,
                "moratorium_applies": False,
                "legal_basis": "Base legal basis.",
                "recommended_action": "Base action.",
            }

    graph = build_workflow(
        policy_agent=_NonIllegalPolicyAgent(),
        evidence_agent=_GlobalEvidenceAgent(),
        portal_agent=_StubPortalAgent(),
        checkpoint_db=tmp_path / "state_global_layers.sqlite",
    )
    initial_state = {
        "claim_id": "WF-CIT-5",
        "policy_age_years": 6.0,
        "policy_document_path": "Policy.pdf",
        "rejection_letter_path": "Reject.pdf",
        "discharge_summary_path": "Discharge.pdf",
        "hospital_bill_path": "Bill.pdf",
        "insurer_name": "Test Insurer",
        "policy_number": "P-12345",
        "complainant_name": "Test User",
        "mobile": "9999999999",
        "email": "test@example.com",
        "insurer_reply_received": False,
        "dry_run_portal": True,
    }
    result = run_workflow(graph, initial_state, claim_id="WF-CIT-5")
    assert "Clinical Distinction:" in result["legal_analysis"]["legal_basis"]
    assert "Constructive Knowledge:" in result["legal_analysis"]["legal_basis"]
    assert "Statutory Primacy:" in result["legal_analysis"]["legal_basis"]
    assert result["legal_analysis"]["citation_id"] == "IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1"
    assert "Challenge medical mismatch" in result["legal_analysis"]["recommended_action"]
    assert "Invoke constructive knowledge" in result["legal_analysis"]["recommended_action"]
