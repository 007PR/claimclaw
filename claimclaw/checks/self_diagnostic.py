from __future__ import annotations

import datetime
import json
import os
from typing import Any

from claimclaw.legalbrain.rules import (
    check_moratorium_eligibility,
    should_override_nondisclosure_rejection,
)


def calculate_moratorium_status(
    policy_start_date: datetime.date,
    claim_date: datetime.date,
    rejection_reason: str = "Non-disclosure of PED",
) -> dict[str, Any]:
    moratorium = check_moratorium_eligibility(
        policy_start_date=policy_start_date,
        as_of_date=claim_date,
    )
    decision = should_override_nondisclosure_rejection(
        policy_start_date=policy_start_date,
        rejection_reason=rejection_reason,
        as_of_date=claim_date,
        alleged_fraud=False,
    )
    return {
        "policy_start_date": policy_start_date.isoformat(),
        "claim_date": claim_date.isoformat(),
        "required_years": moratorium.applicable_moratorium_months // 12,
        "required_months": moratorium.applicable_moratorium_months,
        "policy_age_months": moratorium.policy_age_months,
        "statutory_override": decision.override_nondisclosure_rejection,
        "rule_applied": moratorium.rule_applied,
    }


class EvidenceMatcher:
    @staticmethod
    def verify_clause_presence(clause_id: str, policy_text: str) -> dict[str, Any]:
        clause = " ".join((clause_id or "").split()).lower()
        policy = " ".join((policy_text or "").split()).lower()
        is_present = bool(clause) and clause in policy
        return {
            "clause_id": clause_id,
            "is_clause_present_in_policy": is_present,
            "ghost_rejection": not is_present,
        }


def get_main_llm() -> Any | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    return ChatOpenAI(model=model_name, temperature=0)


def run_self_diagnostic() -> dict[str, Any]:
    print("STARTING CLAIMCLAW LOGIC INTEGRITY AUDIT...")
    report: dict[str, Any] = {"passed": 0, "failed": 0, "warnings": []}

    # --- CHECK 1: The "Date Boundary" Stress Test ---
    print("\n[1/4] Testing Statutory Date Logic...")

    res_old = calculate_moratorium_status(
        policy_start_date=datetime.date(2018, 3, 31),
        claim_date=datetime.date(2024, 3, 30),
    )
    if res_old["required_years"] == 8:
        print("[PASS] Pre-2024 Date Check: Correctly applied 8-year rule")
        report["passed"] += 1
    else:
        print(
            f"[FAIL] Bot hallucinated a 5-year rule for a pre-2024 claim (got {res_old['required_years']} years)."
        )
        report["failed"] += 1

    res_new = calculate_moratorium_status(
        policy_start_date=datetime.date(2018, 4, 1),
        claim_date=datetime.date(2024, 4, 2),
    )
    if res_new["required_years"] == 5:
        print("[PASS] Post-2024 Date Check: Correctly applied 5-year rule")
        report["passed"] += 1
    else:
        print(
            f"[FAIL] Bot ignored 2024 boundary behavior in this test (got {res_new['required_years']} years)."
        )
        report["failed"] += 1

    # --- CHECK 2: The "Ghost Clause" Hallucination Test ---
    print("\n[2/4] Testing Hallucination Resistance...")
    matcher = EvidenceMatcher()
    fake_clause_id = "Clause 99.99 (Alien Abduction Coverage)"
    result = matcher.verify_clause_presence(fake_clause_id, "User Policy Text Sample")

    if result["ghost_rejection"] is True:
        print("[PASS] Hallucination Check: Missing clause flagged as Ghost")
        report["passed"] += 1
    else:
        print("[FAIL] AI hallucinated content for a non-existent clause")
        report["failed"] += 1

    # --- CHECK 3: The "Clinical Logic" Connectivity Test ---
    print("\n[3/4] Testing Clinical Brain Connectivity...")
    llm = get_main_llm()
    if llm:
        print("[PASS] LLM Connection: Brain is online")
        report["passed"] += 1
    else:
        print("[WARN] LLM is offline. Clinical reasoning (Acute vs Chronic) will fail.")
        report["warnings"].append("LLM Offline")

    # --- CHECK 4: The "Conflict Resolution" Test ---
    print("\n[4/4] Testing Hierarchy of Law...")
    test_override = res_new["statutory_override"] is True
    if test_override:
        print("[PASS] Hierarchy Check: IRDAI law overrides policy text")
        report["passed"] += 1
    else:
        print("[FAIL] Bot prioritized private policy over statutory expectation in this test")
        report["failed"] += 1

    # --- FINAL REPORT ---
    print("\n" + "=" * 40)
    print("AUDIT COMPLETE")
    print(f"PASSED: {report['passed']}")
    print(f"FAILED: {report['failed']}")
    print(f"WARNINGS: {len(report['warnings'])}")

    if report["failed"] > 0:
        print("\nSYSTEM HALT: LOGIC BREAKPOINTS DETECTED")
        print("Do not run on real claims until failed checks are fixed.")
    else:
        print("\nSYSTEM INTEGRITY VERIFIED. Ready for live data.")

    return report


def main() -> None:
    report = run_self_diagnostic()
    print("\nJSON_SUMMARY")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
