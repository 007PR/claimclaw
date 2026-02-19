from __future__ import annotations

import re
from typing import Any
from typing import Callable

from .schemas import ComplaintPayload, PortalCredentials

BIMA_BHAROSA_URL = "https://bimabharosa.irdai.gov.in/"


def _first_success(actions: list[Callable[[], None]]) -> bool:
    for action in actions:
        try:
            action()
            return True
        except Exception:
            continue
    return False


def wait_for_human_captcha() -> None:
    print(
        "\n[Human Step Required] Solve the CAPTCHA in the browser window, "
        "then press Enter here to continue."
    )
    input()


def _login(page: Any, credentials: PortalCredentials) -> None:
    _first_success(
        [
            lambda: page.get_by_role("link", name=re.compile("login", re.I)).click(),
            lambda: page.get_by_role("button", name=re.compile("login", re.I)).click(),
        ]
    )

    _first_success(
        [
            lambda: page.get_by_label(re.compile("username|email|mobile", re.I)).fill(
                credentials.username
            ),
            lambda: page.locator("input[type='text']").first.fill(credentials.username),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("password", re.I)).fill(
                credentials.password
            ),
            lambda: page.locator("input[type='password']").first.fill(credentials.password),
        ]
    )

    wait_for_human_captcha()
    _first_success(
        [
            lambda: page.get_by_role("button", name=re.compile("sign in|login", re.I)).click(),
            lambda: page.get_by_text(re.compile("submit", re.I)).click(),
        ]
    )


def _open_complaint_form(page: Any) -> None:
    clicked = _first_success(
        [
            lambda: page.get_by_role(
                "link", name=re.compile("file a complaint|register complaint", re.I)
            ).click(),
            lambda: page.get_by_role(
                "button", name=re.compile("file a complaint|register complaint", re.I)
            ).click(),
        ]
    )
    if not clicked:
        raise RuntimeError(
            "Could not locate complaint filing entry point. Update selectors for the current portal."
        )


def _fill_complaint_form(page: Any, payload: ComplaintPayload) -> None:
    _first_success(
        [
            lambda: page.get_by_label(re.compile("name", re.I)).fill(payload.complainant_name),
            lambda: page.locator("input[name*='name']").first.fill(payload.complainant_name),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("insurer", re.I)).fill(payload.insurer_name),
            lambda: page.locator("input[name*='insurer']").first.fill(payload.insurer_name),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("policy", re.I)).fill(payload.policy_number),
            lambda: page.locator("input[name*='policy']").first.fill(payload.policy_number),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("mobile|phone", re.I)).fill(payload.mobile),
            lambda: page.locator("input[type='tel']").first.fill(payload.mobile),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("email", re.I)).fill(payload.email),
            lambda: page.locator("input[type='email']").first.fill(payload.email),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("category|grievance", re.I)).fill(
                payload.grievance_category
            ),
            lambda: page.locator("input[name*='category']").first.fill(payload.grievance_category),
        ]
    )
    _first_success(
        [
            lambda: page.get_by_label(re.compile("summary|details|complaint", re.I)).fill(
                payload.issue_summary
            ),
            lambda: page.locator("textarea").first.fill(payload.issue_summary),
        ]
    )

    _first_success(
        [
            lambda: page.get_by_label(re.compile("relief|resolution|expect", re.I)).fill(
                payload.relief_sought
            ),
            lambda: page.locator("textarea").nth(1).fill(payload.relief_sought),
        ]
    )

    if payload.attachments:
        for path in payload.attachments:
            _first_success(
                [
                    lambda p=path: page
                    .locator("input[type='file']")
                    .first.set_input_files(p),
                ]
            )


def file_bima_bharosa_complaint(
    payload: ComplaintPayload,
    credentials: PortalCredentials,
    headless: bool = False,
    dry_run: bool = True,
) -> dict[str, str]:
    if dry_run:
        return {
            "status": "dry_run",
            "message": "Portal run skipped. Set dry_run=False to execute browser automation.",
            "portal_url": BIMA_BHAROSA_URL,
        }

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=150)
        page = browser.new_page()
        page.goto(BIMA_BHAROSA_URL, wait_until="domcontentloaded", timeout=60000)
        _login(page, credentials)
        _open_complaint_form(page)
        _fill_complaint_form(page, payload)

        wait_for_human_captcha()
        _first_success(
            [
                lambda: page.get_by_role("button", name=re.compile("submit", re.I)).click(),
                lambda: page.get_by_text(re.compile("submit", re.I)).click(),
            ]
        )

        confirmation_text = page.locator("body").inner_text(timeout=15000)[:600]
        browser.close()
        return {
            "status": "submitted",
            "message": "Complaint form submitted (verify on portal dashboard).",
            "confirmation_excerpt": confirmation_text,
        }
