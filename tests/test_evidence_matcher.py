from claimclaw.evidence_matcher import (
    analyze_documents,
    check_irdai_2017_response_time_compliance,
    evaluate_contestability,
)


def test_non_medical_vs_surgery_consumable_is_flagged() -> None:
    finding = evaluate_contestability(
        rejection_reason="Reason for rejection: Non-medical expense",
        diagnosis_summary="Doctor diagnosis: acute appendicitis surgery",
        bill_items=[
            {
                "description": "Surgery consumables kit",
                "category": "Surgery Consumables",
                "amount": 5400,
                "medically_necessary": True,
            }
        ],
    )
    assert finding.contestable is True
    assert any("non-medical expense" in flag.lower() for flag in finding.flags)


def test_irdai_2017_response_time_violation_when_over_15_days() -> None:
    compliance = check_irdai_2017_response_time_compliance(
        claim_submission_date="2026-01-01",
        rejection_date="2026-01-20",
    )
    assert compliance["days_to_rejection"] == 19
    assert compliance["violation_15_day_rule"] is True
    assert compliance["compliance_status"] == "violation"


def test_forensic_validation_flags_ghost_rejection(monkeypatch) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": (
                "Health Policy Contract Clause 4.1: PED review window 8 years from policy inception."
            ),
            "Rejection.pdf": "Reason for rejection: Non-disclosure of PED under Clause 9.9.",
            "Discharge.pdf": "Diagnosis: Acute appendicitis requiring surgery.",
            "Bill.pdf": "Surgery Consumables - INR 5500",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Clause 9.9",
            "clause_text": "Non-disclosure of PED under Clause 9.9.",
            "policy_clause_excerpt": "",
            "is_clause_present_in_policy": False,
            "ghost_rejection": True,
            "rejection_reason": "Non-disclosure of PED under Clause 9.9.",
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
    )

    forensic = report["forensic_validation"]
    extraction = report["clause_extraction"]
    assert set(extraction.keys()) == {"clause_id", "clause_text", "ghost_rejection"}
    assert forensic["cited_clause_or_exclusion"] == "Clause 9.9"
    assert forensic["clause_found_in_policy"] is False
    assert forensic["is_clause_present_in_policy"] is False
    assert forensic["ghost_rejection"] is True
    assert extraction["clause_id"] == "Clause 9.9"
    assert extraction["ghost_rejection"] is True
    assert any("Ghost Rejection" in flag for flag in report["flags"])
    assert report["contestable"] is True


def test_forensic_validation_prioritizes_irdai_override(monkeypatch) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": (
                "Health Policy Contract Clause 4.1 (Non-disclosure): "
                "Insurer may repudiate claims for non-disclosure within 8 years from inception."
            ),
            "Rejection.pdf": "Reason for rejection: Non-disclosure of PED under Clause 4.1.",
            "Discharge.pdf": "Diagnosis: Acute appendicitis requiring surgery.",
            "Bill.pdf": "Surgery Consumables - INR 5500",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Clause 4.1",
            "clause_text": "Non-disclosure of PED under Clause 4.1.",
            "policy_clause_excerpt": "Clause 4.1: repudiation allowed within 8 years for non-disclosure.",
            "is_clause_present_in_policy": True,
            "ghost_rejection": False,
            "rejection_reason": "Non-disclosure of PED under Clause 4.1.",
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
    )

    forensic = report["forensic_validation"]
    extraction = report["clause_extraction"]
    assert forensic["cited_clause_or_exclusion"] == "Clause 4.1"
    assert forensic["clause_found_in_policy"] is True
    assert forensic["is_clause_present_in_policy"] is True
    assert forensic["irdai_override_applies"] is True
    assert extraction["clause_id"] == "Clause 4.1"
    assert extraction["ghost_rejection"] is False
    assert (
        forensic["conflict_note"]
        == "Policy Clause 4.1 is now void as per IRDAI Ref: IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1."
    )
    assert any("Policy Clause 4.1 is now void as per IRDAI Ref" in flag for flag in report["flags"])
    assert report["contestable"] is True


def test_forensic_validation_prefers_clause_when_exclusion_and_clause_both_present(
    monkeypatch,
) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": (
                "Health Policy Contract Clause 4.1 (Non-disclosure): "
                "Insurer may repudiate claims for non-disclosure within 8 years from inception."
            ),
            "Rejection.pdf": (
                "Exclusion Code: EX-41. Repudiation Clause: 4.1. "
                "Reason for Rejection: Non-disclosure of PED."
            ),
            "Discharge.pdf": "Diagnosis: Acute appendicitis requiring surgery.",
            "Bill.pdf": "Surgery Consumables - INR 5500",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Clause 4.1",
            "clause_text": "Repudiation Clause: 4.1",
            "policy_clause_excerpt": "Clause 4.1 in policy text.",
            "is_clause_present_in_policy": True,
            "ghost_rejection": False,
            "rejection_reason": "Non-disclosure of PED.",
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
    )

    forensic = report["forensic_validation"]
    assert forensic["cited_clause_or_exclusion"] == "Clause 4.1"
    assert forensic["clause_identifier"] == "Clause 4.1"
    assert forensic["clause_id"] == "Clause 4.1"


def test_forensic_llm_structural_extraction_is_preferred(monkeypatch) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": (
                "Health Policy Contract Section 4.2 (Non-disclosure): "
                "Insurer may repudiate claims for non-disclosure within 8 years."
            ),
            "Rejection.pdf": (
                "Repudiation note. Grounds of repudiation: suppression of PED under Section 4.2."
            ),
            "Discharge.pdf": "Diagnosis: Acute appendicitis requiring surgery.",
            "Bill.pdf": "Surgery Consumables - INR 5500",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Section 4.2",
            "clause_text": "suppression of PED under Section 4.2",
            "policy_clause_excerpt": "Section 4.2 allows repudiation within 8 years.",
            "is_clause_present_in_policy": True,
            "ghost_rejection": False,
            "rejection_reason": "suppression of PED under Section 4.2",
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
        forensic_llm=object(),
    )

    forensic = report["forensic_validation"]
    assert report["rejection_reason"] == "suppression of PED under Section 4.2"
    assert forensic["cited_clause_or_exclusion"] == "Section 4.2"
    assert forensic["clause_id"] == "Section 4.2"
    assert forensic["clause_text"] == "suppression of PED under Section 4.2"
    assert report["clause_extraction"] == {
        "clause_id": "Section 4.2",
        "clause_text": "suppression of PED under Section 4.2",
        "ghost_rejection": False,
    }


def test_constructive_knowledge_from_disclosed_medication_marks_contestable(monkeypatch) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": (
                "Proposal Form: Hypertension: No. Current medications: Daily Telmisartan 40mg."
            ),
            "Rejection.pdf": "Reason for Rejection: Non-disclosure of Hypertension for bypass claim.",
            "Discharge.pdf": "Procedure: Heart Bypass (CABG).",
            "Bill.pdf": "Surgery Consumables - INR 5500",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Clause 6.3",
            "clause_text": "Non-disclosure of Hypertension for bypass claim.",
            "policy_clause_excerpt": "Clause 6.3 non-disclosure clause.",
            "is_clause_present_in_policy": True,
            "ghost_rejection": False,
            "rejection_reason": "Non-disclosure of Hypertension for bypass claim.",
        },
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_medication_condition_map_with_llm",
        lambda policy_text, rejection_text, discharge_text, forensic_llm: {
            "rejected_condition": "Hypertension",
            "disclosed_medications": [{"name": "Telmisartan", "dosage": "40mg"}],
            "medication_indications": [
                {"medication": "Telmisartan", "primary_indication": "Hypertension"}
            ],
            "constructive_knowledge": True,
            "reason": (
                "Telmisartan is a maintenance antihypertensive and was disclosed in proposal form."
            ),
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
        forensic_llm=object(),
    )

    assert report["contestable"] is True
    assert report["medication_cross_reference"]["constructive_knowledge"] is True
    assert (
        report["constructive_knowledge_note"]
        == "The insurer had constructive knowledge of the condition as the specific maintenance medication (Telmisartan 40mg) was explicitly disclosed in the proposal form. Failure to seek further clarification at the time of underwriting waives the right to reject now."
    )
    assert any("constructive knowledge check:" in flag.lower() for flag in report["flags"])


def test_global_brain_clinical_distinction_marks_rejection_invalid(monkeypatch) -> None:
    def fake_extract_text(pdf_path: str) -> str:
        mapping = {
            "Policy.pdf": "Clause 4.2: Chronic Kidney Disease waiting period 4 years.",
            "Rejection.pdf": "Reason for Rejection: Chronic Kidney Disease waiting period under Clause 4.2.",
            "Discharge.pdf": "Doctor diagnosis: Acute Kidney Injury due to severe infection.",
            "Bill.pdf": "ICU Charges - INR 12000",
        }
        return mapping[pdf_path]

    monkeypatch.setattr("claimclaw.evidence_matcher.extract_text_pymupdf", fake_extract_text)
    monkeypatch.setattr(
        "claimclaw.evidence_matcher.parse_itemized_bill_with_vision",
        lambda bill_pdf_path, vision_llm: {"items": [], "notes": "mock"},
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_rejection_structure_with_llm",
        lambda rejection_text, policy_text, forensic_llm: {
            "clause_id": "Clause 4.2",
            "clause_text": "CKD waiting period under Clause 4.2",
            "policy_clause_excerpt": "Clause 4.2 chronic kidney disease wait 4 years.",
            "is_clause_present_in_policy": True,
            "ghost_rejection": False,
            "rejection_reason": "Chronic Kidney Disease waiting period under Clause 4.2",
        },
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._extract_medication_condition_map_with_llm",
        lambda policy_text, rejection_text, discharge_text, forensic_llm: {
            "rejected_condition": "Chronic Kidney Disease",
            "disclosed_medications": [],
            "medication_indications": [],
            "constructive_knowledge": False,
            "reason": "",
        },
    )
    monkeypatch.setattr(
        "claimclaw.evidence_matcher._run_global_brain_reasoning_with_llm",
        lambda **kwargs: {
            "clinical_distinction_invalid": True,
            "clinical_distinction_reason": (
                "Rejection cites chronic CKD exclusion, but discharge confirms acute AKI onset."
            ),
            "constructive_knowledge": False,
            "constructive_knowledge_reason": "",
            "statutory_primacy_override": False,
            "statutory_primacy_reason": "",
        },
    )

    report = analyze_documents(
        policy_document_path="Policy.pdf",
        rejection_letter_path="Rejection.pdf",
        discharge_summary_path="Discharge.pdf",
        hospital_bill_path="Bill.pdf",
        vision_llm=None,
        forensic_llm=object(),
    )

    assert report["contestable"] is True
    assert report["global_reasoning"]["clinical_distinction_invalid"] is True
    assert any("clinical distinction:" in flag.lower() for flag in report["flags"])
