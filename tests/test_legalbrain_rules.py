from claimclaw.legalbrain.rules import (
    POST_SWITCH_MORATORIUM_MONTHS,
    PRE_SWITCH_MORATORIUM_MONTHS,
    check_moratorium_eligibility,
    should_override_nondisclosure_rejection,
)


def test_pre_april_2024_policy_uses_96_month_rule() -> None:
    result = check_moratorium_eligibility(
        policy_start_date="2020-01-01",
        as_of_date="2026-02-17",
    )
    assert result.applicable_moratorium_months == PRE_SWITCH_MORATORIUM_MONTHS
    assert result.eligible_for_moratorium_protection is False


def test_post_april_2024_policy_uses_60_month_rule() -> None:
    result = check_moratorium_eligibility(
        policy_start_date="2024-04-02",
        as_of_date="2029-05-02",
    )
    assert result.applicable_moratorium_months == POST_SWITCH_MORATORIUM_MONTHS
    assert result.eligible_for_moratorium_protection is True


def test_april_1_2024_boundary_uses_60_month_rule() -> None:
    result = check_moratorium_eligibility(
        policy_start_date="2024-04-01",
        as_of_date="2029-04-01",
    )
    assert result.applicable_moratorium_months == POST_SWITCH_MORATORIUM_MONTHS
    assert result.rule_applied == "post_april_2024_5_year_rule"
    assert result.eligible_for_moratorium_protection is True


def test_override_applies_to_nondisclosure_after_moratorium() -> None:
    decision = should_override_nondisclosure_rejection(
        policy_start_date="2015-01-10",
        as_of_date="2026-02-17",
        rejection_reason="Repudiation for non-disclosure / hidden BP condition",
        alleged_fraud=False,
    )
    assert decision.is_nondisclosure_rejection is True
    assert decision.override_nondisclosure_rejection is True


def test_no_override_when_fraud_alleged() -> None:
    decision = should_override_nondisclosure_rejection(
        policy_start_date="2015-01-10",
        as_of_date="2026-02-17",
        rejection_reason="Repudiation for non-disclosure",
        alleged_fraud=True,
    )
    assert decision.override_nondisclosure_rejection is False
