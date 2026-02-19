from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any

MORATORIUM_SWITCH_DATE = date(2024, 4, 1)
POST_SWITCH_MORATORIUM_MONTHS = 60
PRE_SWITCH_MORATORIUM_MONTHS = 96


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def _months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(0, months)


@dataclass(slots=True)
class MoratoriumCheck:
    policy_start_date: str
    as_of_date: str
    policy_age_months: int
    applicable_moratorium_months: int
    eligible_for_moratorium_protection: bool
    rule_applied: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_moratorium_eligibility(
    policy_start_date: str | date | datetime,
    as_of_date: str | date | datetime | None = None,
) -> MoratoriumCheck:
    policy_start = _to_date(policy_start_date)
    current = _to_date(as_of_date) if as_of_date else date.today()

    if policy_start >= MORATORIUM_SWITCH_DATE:
        cap = POST_SWITCH_MORATORIUM_MONTHS
        rule_applied = "post_april_2024_5_year_rule"
    else:
        cap = PRE_SWITCH_MORATORIUM_MONTHS
        rule_applied = "pre_april_2024_8_year_rule"

    age_months = _months_between(policy_start, current)
    eligible = age_months >= cap
    return MoratoriumCheck(
        policy_start_date=policy_start.isoformat(),
        as_of_date=current.isoformat(),
        policy_age_months=age_months,
        applicable_moratorium_months=cap,
        eligible_for_moratorium_protection=eligible,
        rule_applied=rule_applied,
    )


@dataclass(slots=True)
class RejectionOverrideDecision:
    moratorium: MoratoriumCheck
    rejection_reason: str
    alleged_fraud: bool
    is_nondisclosure_rejection: bool
    override_nondisclosure_rejection: bool
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["moratorium"] = self.moratorium.to_dict()
        return payload


def should_override_nondisclosure_rejection(
    policy_start_date: str | date | datetime,
    rejection_reason: str,
    as_of_date: str | date | datetime | None = None,
    alleged_fraud: bool = False,
) -> RejectionOverrideDecision:
    moratorium = check_moratorium_eligibility(
        policy_start_date=policy_start_date,
        as_of_date=as_of_date,
    )
    lowered = rejection_reason.lower()
    nondisclosure_tokens = (
        "non-disclosure",
        "nondisclosure",
        "non disclosure",
        "suppression",
        "pre-existing",
        "pre existing",
        "ped",
        "hidden",
    )
    is_nondisclosure = any(token in lowered for token in nondisclosure_tokens)
    override = (
        moratorium.eligible_for_moratorium_protection
        and is_nondisclosure
        and not alleged_fraud
    )

    if override:
        decision = (
            "Override rejection: moratorium window elapsed, non-disclosure basis cannot be used "
            "without proven fraud."
        )
    elif alleged_fraud and is_nondisclosure:
        decision = "Fraud allegation present: override not automatic, insurer proof must be examined."
    elif not is_nondisclosure:
        decision = "Rejection reason is not a non-disclosure class dispute."
    else:
        decision = "Moratorium period not yet elapsed."

    return RejectionOverrideDecision(
        moratorium=moratorium,
        rejection_reason=rejection_reason,
        alleged_fraud=alleged_fraud,
        is_nondisclosure_rejection=is_nondisclosure,
        override_nondisclosure_rejection=override,
        decision_reason=decision,
    )
