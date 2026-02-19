from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import fitz
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

IRDAI_HEALTH_DEPARTMENT_URL = "https://irdai.gov.in/department/health"
MASTER_CIRCULAR_HINT = "master circular on health insurance business"
PLAYWRIGHT_LAUNCH_ARGS = ["--no-sandbox", "--disable-gpu"]
LEGAL_BACKUP_DIR = Path(__file__).resolve().parents[2] / "data" / "legal"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 ClaimClaw/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_links_from_page_html(page_url: str, html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        text = anchor.get_text(" ", strip=True)
        if not href:
            continue
        links.append(
            {
                "title": text,
                "url": urljoin(page_url, href),
            }
        )
    return links


def _is_master_circular_candidate(link: dict[str, str]) -> bool:
    title = link.get("title", "").lower()
    url = link.get("url", "").lower()
    return (
        "master circular" in title
        or MASTER_CIRCULAR_HINT in title
        or MASTER_CIRCULAR_HINT in url
    )


def _extract_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    if not match:
        return None
    return int(match.group(1))


def _extract_date_hint(text: str) -> str | None:
    for pattern in [
        r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
        r"([A-Za-z]+\s+\d{1,2},\s*20\d{2})",
        r"(\d{1,2}[-/]\d{1,2}[-/]20\d{2})",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _fetch_pdf_text(pdf_url: str, timeout_seconds: int = 45) -> str:
    response = requests.get(pdf_url, timeout=timeout_seconds, headers=REQUEST_HEADERS)
    response.raise_for_status()

    # Keep the PDF in-memory only; no file-system write.
    memory_buffer = BytesIO(response.content)
    with fitz.open(stream=memory_buffer.read(), filetype="pdf") as pdf_doc:
        return "\n".join(page.get_text("text") for page in pdf_doc)


def _find_pdf_links_for_master_circular(url: str) -> list[str]:
    response = requests.get(url, timeout=30, headers=REQUEST_HEADERS)
    response.raise_for_status()
    links = _extract_links_from_page_html(url, response.text)
    output: list[str] = []
    for link in links:
        candidate_url = link["url"]
        lowered = candidate_url.lower()
        path = urlparse(candidate_url).path.lower()
        if ".pdf" in path or "download=true" in lowered or "/documents/" in lowered:
            output.append(candidate_url)
    return output


def _read_local_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with fitz.open(str(path)) as pdf_doc:
            return "\n".join(page.get_text("text") for page in pdf_doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_backup_circulars_from_local_legal(
    include_2025_updates: bool,
    warning: str | None = None,
) -> list[dict[str, Any]]:
    if not LEGAL_BACKUP_DIR.exists():
        return []

    backup_items: list[dict[str, Any]] = []
    for path in sorted(LEGAL_BACKUP_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower().startswith("readme"):
            continue
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            text = _read_local_document_text(path)
        except Exception:
            continue

        preview = f"{path.name} {text[:2000]}".lower()
        if "irdai" not in preview and "master circular" not in preview and "health insurance" not in preview:
            continue

        year = _extract_year(f"{path.name} {text[:8000]}")
        if year is None:
            year = 2024
        if year != 2024 and not (include_2025_updates and year >= 2025):
            continue

        payload: dict[str, Any] = {
            "title": path.name,
            "url": str(path),
            "year": year,
            "date_hint": _extract_date_hint(f"{path.name} {text[:1000]}"),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "text": text,
            "source": "local_backup_data_legal",
            "is_may_29_2024": "29 may 2024" in preview or "may 29, 2024" in preview,
        }
        if warning:
            payload["warning"] = warning
        backup_items.append(payload)
    return backup_items


def get_latest_irdai_circulars(
    include_2025_updates: bool = True,
    must_include_may_29_2024: bool = True,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, str]] = []

    html = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with sync_playwright() as playwright:
                executable_path = playwright.chromium.executable_path
                launch_kwargs: dict[str, Any] = {
                    "headless": True,
                    "args": PLAYWRIGHT_LAUNCH_ARGS,
                }
                if executable_path and Path(executable_path).exists():
                    launch_kwargs["executable_path"] = executable_path
                browser = playwright.chromium.launch(
                    **launch_kwargs,
                )
                page = browser.new_page(user_agent=REQUEST_HEADERS["User-Agent"])
                page.goto(
                    IRDAI_HEALTH_DEPARTMENT_URL,
                    wait_until="domcontentloaded",
                    timeout=70000 + attempt * 20000,
                )
                page.wait_for_timeout(3000 + attempt * 1000)
                html = page.content()
                browser.close()
            if html:
                break
        except Exception as exc:  # pragma: no cover - network/runtime variability
            last_error = exc
            time.sleep(1 + attempt)

    if not html:
        # Fallback to direct HTTP fetch when browser load fails due transient connectivity.
        try:
            response = requests.get(
                IRDAI_HEALTH_DEPARTMENT_URL,
                timeout=45,
                headers=REQUEST_HEADERS,
            )
            response.raise_for_status()
            html = response.text
        except Exception:
            warning = (
                f"Live IRDAI scrape failed. Using /data/legal backup. Error: {last_error}"
                if last_error
                else "Live IRDAI scrape failed. Using /data/legal backup."
            )
            return _load_backup_circulars_from_local_legal(
                include_2025_updates=include_2025_updates,
                warning=warning,
            )

    links = _extract_links_from_page_html(IRDAI_HEALTH_DEPARTMENT_URL, html)
    for link in links:
        if _is_master_circular_candidate(link):
            candidates.append(link)

    fallback_links = [
        link
        for link in links
        if "health insurance" in link["title"].lower() and ".pdf" in link["url"].lower()
    ]
    if not candidates:
        candidates = fallback_links

    expanded_pdf_links: list[dict[str, str]] = []
    for link in candidates:
        title = link["title"]
        url = link["url"]
        if url.lower().endswith(".pdf"):
            expanded_pdf_links.append({"title": title, "url": url})
            continue

        try:
            for pdf_url in _find_pdf_links_for_master_circular(url):
                expanded_pdf_links.append(
                    {
                        "title": title,
                        "url": pdf_url,
                    }
                )
        except Exception:
            continue

    seen: set[str] = set()
    filtered: list[dict[str, str]] = []
    for item in expanded_pdf_links:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        metadata_text = f"{item['title']} {url}"
        decoded = unquote(metadata_text).lower()

        # Keep only relevant master-circular health-insurance artifacts.
        if "master circular" not in decoded or "health insurance" not in decoded:
            continue

        year = _extract_year(decoded)
        if year is None:
            continue
        if year == 2024 or (include_2025_updates and year >= 2025):
            filtered.append(item)

    circulars: list[dict[str, Any]] = []
    for item in filtered:
        metadata_text = f"{item['title']} {item['url']}"
        decoded = unquote(metadata_text)
        date_hint = _extract_date_hint(decoded)
        year = _extract_year(decoded)
        try:
            text = _fetch_pdf_text(item["url"])
        except Exception as exc:
            circulars.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "year": year,
                    "date_hint": date_hint,
                    "error": str(exc),
                }
            )
            continue

        circulars.append(
            {
                "title": item["title"],
                "url": item["url"],
                "year": year,
                "date_hint": date_hint,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "text": text,
                "is_may_29_2024": "29 may 2024" in decoded.lower()
                or "may 29, 2024" in decoded.lower()
                or "29052024" in decoded.lower(),
            }
        )

    circulars.sort(key=lambda x: (x.get("year") or 0, x.get("date_hint") or ""), reverse=True)
    if not circulars:
        circulars = _load_backup_circulars_from_local_legal(
            include_2025_updates=include_2025_updates,
            warning="No live circulars discovered. Loaded backup from /data/legal.",
        )
    if must_include_may_29_2024:
        has_may_2024 = any(item.get("is_may_29_2024") for item in circulars)
        if not has_may_2024:
            circulars.append(
                {
                    "title": "May 29, 2024 master circular candidate not explicitly discovered",
                    "url": "",
                    "year": 2024,
                    "date_hint": "29 May 2024",
                    "text": "",
                    "warning": (
                        "The scraper did not identify an explicit May 29, 2024 item by text label. "
                        "Review the IRDAI page structure/selectors."
                    ),
                }
            )

    return circulars
