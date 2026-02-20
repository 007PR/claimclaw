from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .legalbrain.rules import (
    check_moratorium_eligibility,
    should_override_nondisclosure_rejection,
)


def calculate_moratorium_status(
    policy_start_date: date | datetime | str,
    claim_date: date | datetime | str,
    rejection_reason: str = "Non-disclosure of PED",
    alleged_fraud: bool = False,
) -> dict[str, Any]:
    moratorium = check_moratorium_eligibility(
        policy_start_date=policy_start_date,
        as_of_date=claim_date,
    )
    decision = should_override_nondisclosure_rejection(
        policy_start_date=policy_start_date,
        rejection_reason=rejection_reason,
        as_of_date=claim_date,
        alleged_fraud=alleged_fraud,
    )
    return {
        "policy_start_date": moratorium.policy_start_date,
        "claim_date": moratorium.as_of_date,
        "required_years": moratorium.applicable_moratorium_months // 12,
        "required_months": moratorium.applicable_moratorium_months,
        "policy_age_months": moratorium.policy_age_months,
        "eligible_for_moratorium_protection": moratorium.eligible_for_moratorium_protection,
        "statutory_override": decision.override_nondisclosure_rejection,
        "rule_applied": moratorium.rule_applied,
        "decision_reason": decision.decision_reason,
    }
