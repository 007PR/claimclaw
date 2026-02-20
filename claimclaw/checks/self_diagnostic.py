from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from claimclaw.evidence_matcher import EvidenceMatcher
from claimclaw.rules import calculate_moratorium_status
from claimclaw.utils.llm_factory import get_main_llm


def run_self_diagnostic() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)

    print("STARTING CLAIMCLAW LOGIC INTEGRITY AUDIT...")
    report: dict[str, Any] = {"passed": 0, "failed": 0, "warnings": []}

    print("\n[1/4] Testing Statutory Date Logic...")

    # One day before switch: old 8-year rule.
    res_old = calculate_moratorium_status(
        policy_start_date=datetime.date(2024, 3, 31),
        claim_date=datetime.date(2026, 3, 31),
    )
    if res_old["required_years"] == 8:
        print("[PASS] Pre-switch Date Check: Correctly applied 8-year rule")
        report["passed"] += 1
    else:
        print(
            f"[FAIL] Incorrect pre-switch rule (got {res_old['required_years']} years, expected 8)."
        )
        report["failed"] += 1

    # On/after switch: new 5-year rule.
    res_new = calculate_moratorium_status(
        policy_start_date=datetime.date(2024, 4, 1),
        claim_date=datetime.date(2026, 4, 2),
    )
    if res_new["required_years"] == 5:
        print("[PASS] Post-switch Date Check: Correctly applied 5-year rule")
        report["passed"] += 1
    else:
        print(
            f"[FAIL] Incorrect post-switch rule (got {res_new['required_years']} years, expected 5)."
        )
        report["failed"] += 1

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

    print("\n[3/4] Testing Clinical Brain Connectivity...")
    llm = get_main_llm()
    if llm:
        print("[PASS] LLM Connection: Brain is online")
        report["passed"] += 1
    else:
        print("[WARN] LLM is offline. Clinical reasoning (Acute vs Chronic) will fail.")
        report["warnings"].append("LLM Offline")

    print("\n[4/4] Testing Hierarchy of Law...")
    res_override = calculate_moratorium_status(
        policy_start_date=datetime.date(2015, 1, 1),
        claim_date=datetime.date(2026, 2, 19),
        rejection_reason="Non-disclosure of Diabetes",
        alleged_fraud=False,
    )
    if res_override["statutory_override"] is True:
        print("[PASS] Hierarchy Check: IRDAI law overrides policy text after moratorium")
        report["passed"] += 1
    else:
        print("[FAIL] Statutory override did not trigger when expected.")
        report["failed"] += 1

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
