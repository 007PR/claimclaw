from claimclaw.legalbrain.ombudsman_kb import (
    _extract_book_number,
    _extract_case_findings_from_text,
)


def test_extract_book_number_from_url() -> None:
    value = _extract_book_number(
        "https://www.cioins.co.in/GIC/mediclaim/Mediclaim-Book20.pdf"
    )
    assert value == 20


def test_extract_case_findings_from_text_labels_ped_and_non_medical() -> None:
    text = (
        "Case No. X Award dated 1 Jan 2013. Repudiation of claim on pre-existing disease ground. "
        "The insurer also treated surgery consumable lines as non-medical expense."
    )
    findings = _extract_case_findings_from_text(
        text=text,
        source_title="Unit test source",
        source_url="https://example.com/award.pdf",
        date_hint="2013",
        max_hits_per_label=4,
    )
    assert findings
    labels = {label for item in findings for label in item["labels"]}
    assert "ped_repudiation" in labels
    assert "non_medical_expense" in labels
