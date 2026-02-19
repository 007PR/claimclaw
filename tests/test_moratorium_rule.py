from claimclaw.legal_rag import MORATORIUM_YEARS, moratorium_rule_check


def test_policy_over_five_years_hidden_bp_is_illegal_without_fraud() -> None:
    result = moratorium_rule_check(
        policy_age_years=6.0,
        rejection_reason="Claim rejected due to hidden BP / non-disclosure",
        alleged_fraud=False,
    )
    assert MORATORIUM_YEARS == 5
    assert result.moratorium_applies is True
    assert result.illegal_rejection is True


def test_fraud_exception_keeps_rejection_open_for_review() -> None:
    result = moratorium_rule_check(
        policy_age_years=6.0,
        rejection_reason="Suppression of pre-existing disease details",
        alleged_fraud=True,
    )
    assert result.moratorium_applies is True
    assert result.illegal_rejection is False
