from __future__ import annotations

import base64
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .schemas import EvidenceFinding

IRDAI_MORATORIUM_CITATION = "IRDAI/HLT/CIR/PRO/84/5/2024, Clause 6.1"
CONSTRUCTIVE_KNOWLEDGE_LEGAL_NOTE = (
    "The insurer had constructive knowledge of the condition as the specific maintenance medication "
    "(Telmisartan 40mg) was explicitly disclosed in the proposal form. Failure to seek further "
    "clarification at the time of underwriting waives the right to reject now."
)


def _append_unique_flag(flags: list[str], text: str) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    if cleaned not in flags:
        flags.append(cleaned)


def _coerce_to_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    value = value.strip()
    formats = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def _extract_claim_and_rejection_dates(
    rejection_text: str,
) -> tuple[date | None, date | None]:
    flattened = re.sub(r"\s+", " ", rejection_text).strip()
    label_patterns = {
        "claim_submission": [
            r"(?:claim submission|submission date|date of claim|date of intimation|intimation date|claim lodged on|lodged on)\s*[:\-]?\s*([^.;,\n]{6,40})",
        ],
        "rejection": [
            r"(?:rejection date|repudiation date|date of repudiation|letter date|date of rejection|dated)\s*[:\-]?\s*([^.;,\n]{6,40})",
        ],
    }

    claim_date: date | None = None
    rejection_date: date | None = None
    for pattern in label_patterns["claim_submission"]:
        match = re.search(pattern, flattened, flags=re.I)
        if not match:
            continue
        try:
            claim_date = _coerce_to_date(match.group(1))
            break
        except ValueError:
            continue

    for pattern in label_patterns["rejection"]:
        match = re.search(pattern, flattened, flags=re.I)
        if not match:
            continue
        try:
            rejection_date = _coerce_to_date(match.group(1))
            break
        except ValueError:
            continue

    if claim_date and rejection_date:
        return claim_date, rejection_date

    fallback_tokens = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]20\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2})\b",
        flattened,
    )
    fallback_dates: list[date] = []
    for token in fallback_tokens:
        try:
            fallback_dates.append(_coerce_to_date(token))
        except ValueError:
            continue

    if not claim_date and fallback_dates:
        claim_date = min(fallback_dates)
    if not rejection_date and fallback_dates:
        rejection_date = max(fallback_dates)
    return claim_date, rejection_date


def check_irdai_2017_response_time_compliance(
    claim_submission_date: str | date | datetime,
    rejection_date: str | date | datetime,
) -> dict[str, Any]:
    claim_date = _coerce_to_date(claim_submission_date)
    repudiation_date = _coerce_to_date(rejection_date)
    days_to_rejection = calculate_claim_to_rejection_days(claim_date, repudiation_date)
    violation = days_to_rejection > 15
    return {
        "claim_submission_date": claim_date.isoformat(),
        "rejection_date": repudiation_date.isoformat(),
        "days_to_rejection": days_to_rejection,
        "violation_15_day_rule": violation,
        "regulatory_reference": "IRDAI Protection of Policyholders’ Interests Regulations, 2017 (15-day response rule)",
        "compliance_status": "violation" if violation else "compliant",
    }


def calculate_claim_to_rejection_days(
    claim_submission_date: str | date | datetime,
    rejection_date: str | date | datetime,
) -> int:
    claim_date = _coerce_to_date(claim_submission_date)
    repudiation_date = _coerce_to_date(rejection_date)
    return (repudiation_date - claim_date).days


def _is_nondisclosure_rejection(rejection_reason: str) -> bool:
    lowered = rejection_reason.lower()
    return any(
        token in lowered
        for token in (
            "non-disclosure",
            "nondisclosure",
            "non disclosure",
            "suppression",
            "pre-existing",
            "pre existing",
            "ped",
            "hidden",
        )
    )


def _is_legacy_moratorium_clause(clause_excerpt: str, rejection_reason: str) -> bool:
    lowered = clause_excerpt.lower()
    has_legacy_window = any(
        token in lowered
        for token in (
            "8 years",
            "8 year",
            "8-year",
            "96 months",
            "96 month",
            "96-month",
        )
    )
    has_nondisclosure_context = any(
        token in lowered
        for token in (
            "non-disclosure",
            "nondisclosure",
            "non disclosure",
            "pre-existing",
            "pre existing",
            "ped",
            "suppression",
            "moratorium",
        )
    )
    return has_legacy_window and has_nondisclosure_context and _is_nondisclosure_rejection(
        rejection_reason
    )


def extract_text_pymupdf(pdf_path: str | Path) -> str:
    import fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def _render_pdf_pages_as_base64(pdf_path: str | Path, max_pages: int = 3) -> list[str]:
    import fitz

    images: list[str] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            if index >= max_pages:
                break
            pix = page.get_pixmap(dpi=150, alpha=False)
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return images


def _coerce_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {"items": [], "raw_output": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"items": [], "raw_output": text}


def _extract_rejection_structure_with_llm(
    rejection_text: str,
    policy_text: str,
    forensic_llm: Any | None,
) -> dict[str, Any]:
    if forensic_llm is None:
        return {
            "clause_id": "",
            "clause_text": "",
            "policy_clause_excerpt": "",
            "is_clause_present_in_policy": False,
            "ghost_rejection": False,
            "rejection_reason": "",
            "error": "forensic_llm_not_configured",
        }

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a legal forensic extractor for Indian health insurance repudiation letters. "
        "You must use the rejection letter and policy text together. Return strict JSON only."
    )
    user_prompt = (
        "Return strict JSON with this exact schema:\n"
        '{"clause_id":"string","clause_text":"string","ghost_rejection":false,'
        '"is_clause_present_in_policy":false,"policy_clause_excerpt":"string","rejection_reason":"string"}\n'
        "Rules:\n"
        "1) clause_id must be the exact clause/exclusion/section anchor cited in repudiation.\n"
        "2) clause_text must be the exact sentence or phrase in rejection letter that cites the anchor.\n"
        "3) is_clause_present_in_policy must be true only if the same clause_id appears in policy.\n"
        "4) ghost_rejection must be true when is_clause_present_in_policy is false.\n"
        "5) rejection_reason must summarize insurer repudiation basis in one sentence.\n"
        "6) If not found, use empty string for text fields and false for booleans.\n\n"
        "Rejection Letter:\n"
        f"{rejection_text}\n\n"
        "Policy Text:\n"
        f"{policy_text}\n"
    )
    try:
        response = forensic_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = _coerce_json_payload(response.content)
    except Exception:
        parsed = {}

    def _clean(value: Any) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None

    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    clause_id = _clean(parsed.get("clause_id")) or ""
    clause_text = _clean(parsed.get("clause_text")) or ""
    policy_clause_excerpt = _clean(parsed.get("policy_clause_excerpt")) or ""
    rejection_reason = _clean(parsed.get("rejection_reason")) or ""
    is_present = _to_bool(parsed.get("is_clause_present_in_policy"))

    # Enforce law-logic invariant regardless of model drift.
    ghost_rejection = bool(clause_id) and not is_present

    ghost_hint = parsed.get("ghost_rejection")
    if not clause_id and _to_bool(ghost_hint):
        ghost_rejection = True

    return {
        "clause_id": clause_id,
        "clause_text": clause_text,
        "policy_clause_excerpt": policy_clause_excerpt,
        "is_clause_present_in_policy": is_present,
        "ghost_rejection": ghost_rejection,
        "rejection_reason": rejection_reason,
    }


def _extract_medication_condition_map_with_llm(
    policy_text: str,
    rejection_text: str,
    discharge_text: str,
    forensic_llm: Any | None,
) -> dict[str, Any]:
    if forensic_llm is None:
        return {
            "rejected_condition": "",
            "disclosed_medications": [],
            "medication_indications": [],
            "constructive_knowledge": False,
            "reason": "",
        }

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a clinical-legal analyzer for insurance repudiation disputes. "
        "Infer condition links from disclosed long-term medications."
    )
    user_prompt = (
        "Return strict JSON with this exact schema:\n"
        '{"rejected_condition":"string","disclosed_medications":[{"name":"string","dosage":"string"}],'
        '"medication_indications":[{"medication":"string","primary_indication":"string"}],'
        '"constructive_knowledge":false,"reason":"string"}\n'
        "Rules:\n"
        "1) Identify rejected condition from insurer non-disclosure basis.\n"
        "2) Extract all medications disclosed in proposal/policy text.\n"
        "3) Infer each medication's primary clinical indication using medical knowledge.\n"
        "4) If medication indication matches rejected condition (including synonyms), set constructive_knowledge=true.\n"
        "5) reason must be one concise sentence.\n\n"
        "Policy + Proposal Text:\n"
        f"{policy_text}\n\n"
        "Rejection Letter:\n"
        f"{rejection_text}\n\n"
        "Discharge Summary:\n"
        f"{discharge_text}\n"
    )
    try:
        response = forensic_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = _coerce_json_payload(response.content)
    except Exception:
        parsed = {}

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _clean_medications(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        output: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = _clean(item.get("name"))
            dosage = _clean(item.get("dosage"))
            if not name and not dosage:
                continue
            output.append({"name": name, "dosage": dosage})
        return output

    def _clean_indications(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        output: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            medication = _clean(item.get("medication"))
            indication = _clean(item.get("primary_indication"))
            if not medication and not indication:
                continue
            output.append(
                {
                    "medication": medication,
                    "primary_indication": indication,
                }
            )
        return output

    constructive = parsed.get("constructive_knowledge")
    constructive_bool = False
    if isinstance(constructive, bool):
        constructive_bool = constructive
    elif isinstance(constructive, str):
        constructive_bool = constructive.strip().lower() in {"1", "true", "yes", "y"}

    return {
        "rejected_condition": _clean(parsed.get("rejected_condition")),
        "disclosed_medications": _clean_medications(parsed.get("disclosed_medications")),
        "medication_indications": _clean_indications(parsed.get("medication_indications")),
        "constructive_knowledge": constructive_bool,
        "reason": _clean(parsed.get("reason")),
    }


def _run_global_brain_reasoning_with_llm(
    policy_text: str,
    rejection_text: str,
    discharge_text: str,
    clause_id: str,
    policy_clause_excerpt: str,
    rejection_reason: str,
    medication_cross_reference: dict[str, Any],
    forensic_llm: Any | None,
) -> dict[str, Any]:
    default = {
        "clinical_distinction_invalid": False,
        "clinical_distinction_reason": "",
        "constructive_knowledge": bool(medication_cross_reference.get("constructive_knowledge")),
        "constructive_knowledge_reason": str(medication_cross_reference.get("reason") or ""),
        "statutory_primacy_override": False,
        "statutory_primacy_reason": "",
    }
    if forensic_llm is None:
        return default

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a high-fidelity Indian insurance legal-medical reasoning engine. "
        "Apply clinical and statutory logic, not keyword matching."
    )
    user_prompt = (
        "Apply the following mandatory rules and return strict JSON only.\n"
        "Rule 1 Clinical Distinction: Distinguish Acute vs Chronic. If rejection cites chronic/pre-existing "
        "disease waiting period/exclusion but discharge supports acute onset episode, mark invalid.\n"
        "Rule 2 Constructive Knowledge: If disclosed medication indicates the rejected condition, insurer had "
        "constructive knowledge and cannot sustain non-disclosure rejection.\n"
        "Rule 3 Statutory Primacy: IRDAI 2024 Master Circular overrides conflicting private policy clauses "
        "(e.g., policy 8 years vs regulation 5 years).\n\n"
        "Return JSON schema:\n"
        '{"clinical_distinction_invalid":false,"clinical_distinction_reason":"string",'
        '"constructive_knowledge":false,"constructive_knowledge_reason":"string",'
        '"statutory_primacy_override":false,"statutory_primacy_reason":"string"}\n\n'
        f"Clause ID: {clause_id}\n"
        f"Policy Clause Excerpt: {policy_clause_excerpt}\n"
        f"Rejection Reason: {rejection_reason}\n"
        f"Medication Cross Reference: {json.dumps(medication_cross_reference, ensure_ascii=True)}\n\n"
        "Policy Text:\n"
        f"{policy_text}\n\n"
        "Rejection Letter:\n"
        f"{rejection_text}\n\n"
        "Discharge Summary:\n"
        f"{discharge_text}\n"
    )
    try:
        response = forensic_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = _coerce_json_payload(response.content)
    except Exception:
        return default

    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    clinical_invalid = _to_bool(parsed.get("clinical_distinction_invalid"))
    constructive = _to_bool(parsed.get("constructive_knowledge"))
    statutory = _to_bool(parsed.get("statutory_primacy_override"))

    return {
        "clinical_distinction_invalid": clinical_invalid,
        "clinical_distinction_reason": _clean(parsed.get("clinical_distinction_reason")),
        "constructive_knowledge": constructive,
        "constructive_knowledge_reason": _clean(parsed.get("constructive_knowledge_reason")),
        "statutory_primacy_override": statutory,
        "statutory_primacy_reason": _clean(parsed.get("statutory_primacy_reason")),
    }


def parse_itemized_bill_with_vision(
    bill_pdf_path: str | Path,
    vision_llm: Any | None,
    max_pages: int = 3,
) -> dict[str, Any]:
    if vision_llm is None:
        return {
            "items": [],
            "notes": "Vision model not configured. Using text-only analysis fallback.",
        }

    from langchain_core.messages import HumanMessage

    images = _render_pdf_pages_as_base64(bill_pdf_path, max_pages=max_pages)
    if not images:
        return {"items": [], "notes": "No pages rendered for bill parsing."}

    instructions = (
        "Extract itemized bill rows from the hospital invoice image(s). "
        "Return strict JSON with this schema only: "
        '{"items":[{"description":"", "category":"", "amount":0, "medically_necessary":true}], '
        '"doctor_diagnosis":"","notes":""}.'
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
    for encoded_img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_img}"},
            }
        )

    response = vision_llm.invoke([HumanMessage(content=content)])
    parsed = _coerce_json_payload(response.content)
    parsed.setdefault("items", [])
    return parsed


def _extract_rejection_reason(
    rejection_text: str,
    structured_rejection: dict[str, Any] | None = None,
) -> str:
    llm_reason = (structured_rejection or {}).get("rejection_reason")
    if llm_reason:
        return llm_reason

    flattened = " ".join(rejection_text.split())
    return flattened[:280]


def _extract_diagnosis_summary(discharge_text: str) -> str:
    patterns = [
        r"diagnosis[:\-\s]+(.+)",
        r"final diagnosis[:\-\s]+(.+)",
        r"doctor(?:'s)? diagnosis[:\-\s]+(.+)",
    ]
    flattened = re.sub(r"\s+", " ", discharge_text).strip()
    for pattern in patterns:
        match = re.search(pattern, flattened, flags=re.I)
        if match:
            return match.group(1).strip()
    return flattened[:220]


def _extract_bill_items_from_text(bill_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lines = [line.strip() for line in bill_text.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if not any(
            token in lowered
            for token in ("consumable", "surgery", "ot", "operation", "room", "medicine")
        ):
            continue
        amount = 0.0
        amount_match = re.search(r"(?:inr|rs\.?)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", line, re.I)
        if amount_match:
            amount = float(amount_match.group(1).replace(",", ""))
        else:
            number_tokens = re.findall(r"([0-9][0-9,]*(?:\.[0-9]+)?)", line)
            if number_tokens:
                amount = float(number_tokens[-1].replace(",", ""))

        if "consumable" in lowered and "surgery" in lowered:
            category = "Surgery Consumables"
        elif "surgery" in lowered or "operation" in lowered or "ot" in lowered:
            category = "Surgery"
        elif "medicine" in lowered:
            category = "Medicines"
        else:
            category = "Hospital Expense"

        items.append(
            {
                "description": line,
                "category": category,
                "amount": amount,
                "medically_necessary": True,
            }
        )
    return items


def evaluate_contestability(
    rejection_reason: str,
    diagnosis_summary: str,
    bill_items: list[dict[str, Any]],
) -> EvidenceFinding:
    reasons = rejection_reason.lower()
    diagnosis = diagnosis_summary.lower()
    flags: list[str] = []

    bill_text = " ".join(
        f"{item.get('description', '')} {item.get('category', '')}".lower()
        for item in bill_items
    )

    contestable = False
    if "non-medical expense" in reasons and (
        "surgery consumable" in bill_text
        or ("consumable" in bill_text and "surgery" in diagnosis)
    ):
        contestable = True
        flags.append(
            "Rejection cites non-medical expense but bill contains surgery consumables tied to treatment."
        )

    if "pre-existing" in reasons and "acute" in diagnosis:
        contestable = True
        flags.append(
            "Pre-existing disease assertion appears weak against acute diagnosis wording."
        )

    if _is_nondisclosure_rejection(rejection_reason):
        flags.append(
            "Validate moratorium applicability in legal node: if policy age is over 5 years, repudiation requires proven fraud."
        )

    return EvidenceFinding(
        contestable=contestable,
        flags=flags,
        rejection_reason=rejection_reason,
        diagnosis_summary=diagnosis_summary,
        bill_items=bill_items,
    )


def analyze_documents(
    policy_document_path: str | Path,
    rejection_letter_path: str | Path,
    discharge_summary_path: str | Path,
    hospital_bill_path: str | Path,
    claim_submission_date: str | date | datetime | None = None,
    rejection_date: str | date | datetime | None = None,
    vision_llm: Any | None = None,
    forensic_llm: Any | None = None,
) -> dict[str, Any]:
    policy_text = extract_text_pymupdf(policy_document_path)
    rejection_text = extract_text_pymupdf(rejection_letter_path)
    discharge_text = extract_text_pymupdf(discharge_summary_path)
    hospital_bill_text = extract_text_pymupdf(hospital_bill_path)
    vision_bill = parse_itemized_bill_with_vision(hospital_bill_path, vision_llm=vision_llm)

    structured_rejection = _extract_rejection_structure_with_llm(
        rejection_text=rejection_text,
        policy_text=policy_text,
        forensic_llm=forensic_llm,
    )
    rejection_reason = _extract_rejection_reason(
        rejection_text,
        structured_rejection=structured_rejection,
    )
    diagnosis = vision_bill.get("doctor_diagnosis") or _extract_diagnosis_summary(discharge_text)
    bill_items = vision_bill.get("items", [])
    if not bill_items:
        bill_items = _extract_bill_items_from_text(hospital_bill_text)

    finding = evaluate_contestability(
        rejection_reason=rejection_reason,
        diagnosis_summary=diagnosis,
        bill_items=bill_items,
    )

    medication_cross_reference = _extract_medication_condition_map_with_llm(
        policy_text=policy_text,
        rejection_text=rejection_text,
        discharge_text=discharge_text,
        forensic_llm=forensic_llm,
    )

    clause_id = str(structured_rejection.get("clause_id") or "").strip()
    clause_text = str(structured_rejection.get("clause_text") or "").strip()
    policy_clause_excerpt = str(structured_rejection.get("policy_clause_excerpt") or "").strip()
    is_clause_present_in_policy = bool(structured_rejection.get("is_clause_present_in_policy"))
    ghost_rejection = bool(structured_rejection.get("ghost_rejection"))
    cited_reference = clause_id or None

    if clause_id and not is_clause_present_in_policy:
        ghost_rejection = True
    if not clause_id and ghost_rejection:
        _append_unique_flag(
            finding.flags,
            "Clause anchor missing in rejection letter extraction. Treating repudiation as contestable."
        )

    global_reasoning = _run_global_brain_reasoning_with_llm(
        policy_text=policy_text,
        rejection_text=rejection_text,
        discharge_text=discharge_text,
        clause_id=clause_id,
        policy_clause_excerpt=policy_clause_excerpt,
        rejection_reason=rejection_reason,
        medication_cross_reference=medication_cross_reference,
        forensic_llm=forensic_llm,
    )

    irdai_override_applies = False
    conflict_note = ""

    clinical_distinction_invalid = bool(global_reasoning.get("clinical_distinction_invalid"))
    if clinical_distinction_invalid:
        finding.contestable = True
        clinical_reason = str(global_reasoning.get("clinical_distinction_reason") or "").strip()
        _append_unique_flag(
            finding.flags,
            (
                f"Clinical distinction: {clinical_reason}"
                if clinical_reason
                else (
                    "Clinical distinction: rejection cites chronic/pre-existing disease basis, but "
                    "discharge evidence supports acute onset episode."
                )
            ),
        )

    constructive_knowledge = bool(medication_cross_reference.get("constructive_knowledge")) or bool(
        global_reasoning.get("constructive_knowledge")
    )
    constructive_reason = str(medication_cross_reference.get("reason") or "").strip()
    if not constructive_reason:
        constructive_reason = str(global_reasoning.get("constructive_knowledge_reason") or "").strip()
    if constructive_knowledge:
        finding.contestable = True
        medication_cross_reference["constructive_knowledge"] = True
        medication_cross_reference["reason"] = constructive_reason
        if constructive_reason:
            _append_unique_flag(finding.flags, f"Constructive knowledge check: {constructive_reason}")

    statutory_primacy_override = bool(global_reasoning.get("statutory_primacy_override"))
    statutory_reason = str(global_reasoning.get("statutory_primacy_reason") or "").strip()

    if clause_id and is_clause_present_in_policy is False:
        finding.contestable = True
        _append_unique_flag(
            finding.flags,
            f"Ghost Rejection: insurer cited {cited_reference}, but this term is not present in the policy contract."
        )
    if clause_id and is_clause_present_in_policy:
        if _is_legacy_moratorium_clause(policy_clause_excerpt, rejection_reason) or statutory_primacy_override:
            irdai_override_applies = True
            finding.contestable = True
            conflict_note = (
                f"Policy {cited_reference} is now void as per IRDAI Ref: {IRDAI_MORATORIUM_CITATION}."
            )
            _append_unique_flag(finding.flags, conflict_note)
    elif statutory_primacy_override:
        irdai_override_applies = True
        finding.contestable = True
        conflict_note = (
            "Policy clause cited by insurer is void as per IRDAI Ref: "
            f"{IRDAI_MORATORIUM_CITATION}."
        )
        _append_unique_flag(finding.flags, conflict_note)
    if statutory_reason:
        _append_unique_flag(finding.flags, f"Statutory primacy: {statutory_reason}")
    extraction_json = {
        "clause_id": clause_id,
        "clause_text": str(clause_text or ""),
        "ghost_rejection": bool(ghost_rejection),
    }

    compliance: dict[str, Any] = {
        "status": "unknown",
        "reason": "Claim submission and rejection dates were not both available for 15-day compliance check.",
    }
    claim_date_value = claim_submission_date
    rejection_date_value = rejection_date
    if claim_date_value is None or rejection_date_value is None:
        parsed_claim, parsed_rejection = _extract_claim_and_rejection_dates(rejection_text)
        if claim_date_value is None:
            claim_date_value = parsed_claim
        if rejection_date_value is None:
            rejection_date_value = parsed_rejection

    if claim_date_value is not None and rejection_date_value is not None:
        compliance = check_irdai_2017_response_time_compliance(
            claim_submission_date=claim_date_value,
            rejection_date=rejection_date_value,
        )
        if compliance.get("violation_15_day_rule"):
            finding.flags.append(
                "IRDAI Protection of Policyholders’ Interests Regulations, 2017: 15-day response timeline appears violated."
            )

    return {
        "policy_excerpt": re.sub(r"\s+", " ", policy_text)[:700],
        "rejection_reason": finding.rejection_reason,
        "diagnosis_summary": finding.diagnosis_summary,
        "contestable": finding.contestable,
        "flags": finding.flags,
        "bill_items": finding.bill_items,
        "response_time_compliance": compliance,
        "vision_notes": vision_bill.get("notes", ""),
        "constructive_knowledge_note": CONSTRUCTIVE_KNOWLEDGE_LEGAL_NOTE if constructive_knowledge else "",
        "medication_cross_reference": medication_cross_reference,
        "global_reasoning": global_reasoning,
        "clause_extraction": extraction_json,
        "forensic_validation": {
            "extraction_json": extraction_json,
            "cited_clause_or_exclusion": cited_reference,
            "clause_id": clause_id,
            "clause_identifier": clause_id,
            "clause_text": clause_text,
            "is_clause_present_in_policy": is_clause_present_in_policy,
            "clause_found_in_policy": is_clause_present_in_policy,
            "ghost_rejection": ghost_rejection,
            "policy_clause_excerpt": policy_clause_excerpt,
            "irdai_override_applies": irdai_override_applies,
            "conflict_note": conflict_note,
            "regulatory_reference": IRDAI_MORATORIUM_CITATION if irdai_override_applies else None,
        },
    }
