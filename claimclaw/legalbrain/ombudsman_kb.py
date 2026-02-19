from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import fitz
import requests
from bs4 import BeautifulSoup
from chromadb import Client as ChromaClient
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

OMBUDSMAN_BASE_URL = "https://www.cioins.co.in/"
OMBUDSMAN_ANNUAL_REPORTS_URL = "https://www.cioins.co.in/AnnualReports"
OMBUDSMAN_AWARDS_ARCHIVE_URL = "https://www.cioins.co.in/Awards/Archive"
MEDICLAIM_BOOK_URL_TEMPLATE = "https://www.cioins.co.in/GIC/mediclaim/Mediclaim-Book{book_no}.pdf"
LEGAL_BACKUP_DIR = Path(__file__).resolve().parents[2] / "data" / "legal"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 ClaimClaw/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PED_KEYWORDS = (
    "pre-existing disease",
    "pre existing disease",
    "ped",
    "repudiation",
    "non-disclosure",
)

NON_MEDICAL_KEYWORDS = (
    "non-medical",
    "non medical",
    "consumable",
    "hospital expense",
)


def _is_same_domain(url: str) -> bool:
    base_host = urlparse(OMBUDSMAN_BASE_URL).netloc
    return urlparse(url).netloc == base_host or url.startswith("/")


def _http_get(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS, **kwargs: Any) -> requests.Response:
    headers = dict(REQUEST_HEADERS)
    extra_headers = kwargs.pop("headers", None) or {}
    headers.update(extra_headers)
    response = requests.get(url, timeout=timeout, headers=headers, **kwargs)
    response.raise_for_status()
    return response


def _http_head(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS, **kwargs: Any) -> requests.Response:
    headers = dict(REQUEST_HEADERS)
    extra_headers = kwargs.pop("headers", None) or {}
    headers.update(extra_headers)
    response = requests.head(url, timeout=timeout, headers=headers, **kwargs)
    return response


def _fetch_html(url: str) -> str:
    response = _http_get(url)
    return response.text


def _extract_links(page_url: str, html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        text = anchor.get_text(" ", strip=True)
        if not href:
            continue
        absolute_url = urljoin(page_url, href)
        if not _is_same_domain(absolute_url):
            continue
        links.append({"title": text, "url": absolute_url})
    return links


def _extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def _extract_date_hint(text: str) -> str | None:
    patterns = (
        r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
        r"([A-Za-z]+\s+\d{1,2},\s*20\d{2})",
        r"(\d{1,2}[-/]\d{1,2}[-/]20\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _classify_case(text: str) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []
    has_claim_dispute = any(
        token in lowered for token in ("repudiation", "rejected", "denied", "claim")
    )
    if any(token in lowered for token in PED_KEYWORDS) and has_claim_dispute:
        labels.append("ped_repudiation")
    if any(token in lowered for token in NON_MEDICAL_KEYWORDS) and has_claim_dispute:
        labels.append("non_medical_expense")
    return labels


def _summarize_case_text(text: str, max_chars: int = 700) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars]


def _extract_year_hint(text: str) -> str | None:
    match = re.search(r"(20\d{2})(?:\D+(20\d{2}))?", text)
    if not match:
        return None
    if match.group(2):
        return f"{match.group(1)}-{match.group(2)}"
    return match.group(1)


def _fetch_pdf_text(pdf_url: str) -> str:
    response = _http_get(pdf_url)
    with fitz.open(stream=BytesIO(response.content).read(), filetype="pdf") as pdf_doc:
        return "\n".join(page.get_text("text") for page in pdf_doc)


def _read_local_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with fitz.open(str(path)) as pdf_doc:
            return "\n".join(page.get_text("text") for page in pdf_doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_ombudsman_backup_from_local_legal(max_items: int = 20) -> list[dict[str, Any]]:
    if not LEGAL_BACKUP_DIR.exists():
        return []

    findings: list[dict[str, Any]] = []
    for path in sorted(LEGAL_BACKUP_DIR.rglob("*")):
        if len(findings) >= max_items:
            break
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

        preview = f"{path.name} {text[:4000]}".lower()
        if "ombudsman" not in preview and "award" not in preview and "repudiation" not in preview:
            continue

        findings.extend(
            _extract_case_findings_from_text(
                text=text,
                source_title=f"{path.name} (local backup)",
                source_url=str(path),
                date_hint=_extract_year_hint(preview),
                max_hits_per_label=3,
            )
        )
    return findings[:max_items]


def _extract_book_number(text: str) -> int | None:
    match = re.search(r"mediclaim-book\s*([0-9]+)\.pdf", text, flags=re.I)
    if not match:
        return None
    return int(match.group(1))


def _find_keyword_windows(
    text: str, keywords: tuple[str, ...], max_hits: int = 6, radius: int = 260
) -> list[str]:
    lowered = text.lower()
    snippets: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        start = 0
        while len(snippets) < max_hits:
            idx = lowered.find(keyword, start)
            if idx < 0:
                break
            left = max(0, idx - radius)
            right = min(len(text), idx + len(keyword) + radius)
            snippet = re.sub(r"\s+", " ", text[left:right]).strip()
            signature = snippet.lower()
            if len(snippet) > 90 and signature not in seen:
                seen.add(signature)
                snippets.append(snippet)
            start = idx + len(keyword)
    return snippets


def _discover_mediclaim_book_urls(max_book_index: int = 30) -> list[tuple[int, str]]:
    # Try parsing links from awards archive page first.
    discovered: dict[int, str] = {}
    try:
        html = _fetch_html(OMBUDSMAN_AWARDS_ARCHIVE_URL)
        for link in _extract_links(OMBUDSMAN_AWARDS_ARCHIVE_URL, html):
            candidate = link["url"]
            book_no = _extract_book_number(candidate)
            if book_no is None:
                continue
            discovered[book_no] = candidate
    except Exception:
        pass

    # Fallback probe because archive pages do not always expose direct links in HTML.
    for book_no in range(2, max_book_index + 1):
        if book_no in discovered:
            continue
        candidate = MEDICLAIM_BOOK_URL_TEMPLATE.format(book_no=book_no)
        try:
            probe = _http_head(candidate, allow_redirects=True)
            if probe.status_code == 200:
                discovered[book_no] = candidate
                continue
            if probe.status_code in {403, 405}:
                # Some servers deny HEAD but allow GET.
                probe_get = _http_get(candidate, stream=True)
                if probe_get.status_code == 200:
                    discovered[book_no] = candidate
                probe_get.close()
        except Exception:
            continue

    return sorted(discovered.items(), key=lambda item: item[0], reverse=True)


def _extract_case_findings_from_text(
    text: str,
    source_title: str,
    source_url: str,
    date_hint: str | None = None,
    max_hits_per_label: int = 6,
) -> list[dict[str, Any]]:
    ped_snippets = _find_keyword_windows(
        text,
        PED_KEYWORDS + ("repudiation", "non-disclosure"),
        max_hits=max_hits_per_label,
    )
    non_medical_snippets = _find_keyword_windows(
        text,
        NON_MEDICAL_KEYWORDS + ("repudiation",),
        max_hits=max_hits_per_label,
    )
    findings: list[dict[str, Any]] = []
    seen_summary: set[str] = set()

    for snippet in ped_snippets + non_medical_snippets:
        labels = _classify_case(snippet)
        if not labels:
            continue
        summary = _summarize_case_text(snippet)
        signature = summary.lower()
        if signature in seen_summary:
            continue
        seen_summary.add(signature)
        findings.append(
            {
                "title": source_title,
                "url": source_url,
                "date_hint": date_hint,
                "labels": labels,
                "summary": summary,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
        )
    return findings


def _collect_award_summaries_from_mediclaim_books(
    max_books: int = 8,
    max_hits_per_label: int = 6,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    discovered_books = _discover_mediclaim_book_urls(max_book_index=30)
    for book_no, url in discovered_books[:max_books]:
        try:
            text = _fetch_pdf_text(url)
        except Exception:
            continue
        book_title = f"Mediclaim Award Book {book_no}"
        # Reports typically include a year span in the opening pages.
        leading_chunk = re.sub(r"\s+", " ", text)[:5000]
        date_hint = _extract_year_hint(leading_chunk) or str(book_no)
        findings.extend(
            _extract_case_findings_from_text(
                text=text,
                source_title=book_title,
                source_url=url,
                date_hint=date_hint,
                max_hits_per_label=max_hits_per_label,
            )
        )
    return findings


def _collect_award_summaries_from_annual_reports(max_reports: int = 5) -> list[dict[str, Any]]:
    html = _fetch_html(OMBUDSMAN_ANNUAL_REPORTS_URL)
    links = _extract_links(OMBUDSMAN_ANNUAL_REPORTS_URL, html)
    pdf_links = [
        link
        for link in links
        if ".pdf" in link["url"].lower() and "annual" in f"{link['title']} {link['url']}".lower()
    ]
    findings: list[dict[str, Any]] = []
    for link in pdf_links[:max_reports]:
        try:
            report_text = _fetch_pdf_text(link["url"])
        except Exception:
            continue
        findings.extend(
            _extract_case_findings_from_text(
                text=report_text,
                source_title=f"{link['title']} - award summary extract",
                source_url=link["url"],
                date_hint=_extract_year_hint(f"{link['title']} {link['url']}"),
                max_hits_per_label=4,
            )
        )
    return findings


def scrape_ombudsman_awards(max_pages: int = 20) -> list[dict[str, Any]]:
    try:
        root_html = _fetch_html(OMBUDSMAN_BASE_URL)
    except Exception:
        return _load_ombudsman_backup_from_local_legal(max_items=max_pages)
    links = _extract_links(OMBUDSMAN_BASE_URL, root_html)
    queue = [
        link
        for link in links
        if any(
            token in f"{link['title']} {link['url']}".lower()
            for token in ("award", "ombudsman", "case", "order")
        )
    ]

    visited: set[str] = set()
    findings: list[dict[str, Any]] = []

    # Primary source: public Mediclaim award-book archives with synopsis-level case entries.
    findings.extend(_collect_award_summaries_from_mediclaim_books(max_books=8))

    # Secondary source for additional context.
    findings.extend(_collect_award_summaries_from_annual_reports(max_reports=5))

    # Tertiary source: crawl candidate pages and classify text blocks.
    for link in queue[:max_pages]:
        url = link["url"]
        if url in visited:
            continue
        visited.add(url)
        try:
            html = _fetch_html(url)
        except Exception:
            continue
        text = _extract_page_text(html)
        findings.extend(
            _extract_case_findings_from_text(
                text=text,
                source_title=link["title"] or "Ombudsman case summary",
                source_url=url,
                date_hint=_extract_date_hint(text),
                max_hits_per_label=2,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in findings:
        key = (item.get("url", ""), item.get("summary", "")[:220].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if not deduped:
        deduped = _load_ombudsman_backup_from_local_legal(max_items=max_pages)

    deduped.sort(key=lambda x: x.get("date_hint") or "", reverse=True)
    return deduped


def build_ombudsman_few_shots(
    ombudsman_findings: list[dict[str, Any]],
    max_examples_per_label: int = 3,
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {
        "ped_repudiation": [],
        "non_medical_expense": [],
    }
    for finding in ombudsman_findings:
        for label in finding.get("labels", []):
            if label not in output:
                continue
            if len(output[label]) >= max_examples_per_label:
                continue
            output[label].append(
                {
                    "input": (
                        f"Case summary ({finding.get('date_hint', 'undated')}): "
                        f"{finding.get('summary', '')}"
                    ),
                    "output": (
                        "Draft a firm rebuttal citing IRDAI policyholder protections and "
                        "demand reversal with escalation timeline."
                    ),
                }
            )
    return output


def build_session_vectorstore(
    irdai_circulars: list[dict[str, Any]],
    ombudsman_findings: list[dict[str, Any]],
    collection_name: str = "claimclaw_live_legal",
    embedding_model: str = "text-embedding-3-large",
) -> Chroma:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required to embed live scraped text into the in-memory ChromaDB."
        )

    embeddings = OpenAIEmbeddings(model=embedding_model)
    client = ChromaClient(
        ChromaSettings(
            is_persistent=False,
            anonymized_telemetry=False,
        )
    )
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    docs: list[Document] = []
    for circular in irdai_circulars:
        text = circular.get("text", "")
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_type": "irdai_circular",
                    "title": circular.get("title", ""),
                    "source": circular.get("url", ""),
                    "year": circular.get("year"),
                    "date_hint": circular.get("date_hint"),
                },
            )
        )

    for case in ombudsman_findings:
        summary = case.get("summary", "")
        if not summary:
            continue
        docs.append(
            Document(
                page_content=summary,
                metadata={
                    "source_type": "ombudsman_award",
                    "title": case.get("title", ""),
                    "source": case.get("url", ""),
                    "labels": ",".join(case.get("labels", [])),
                    "date_hint": case.get("date_hint"),
                },
            )
        )

    if docs:
        vectorstore.add_documents(docs)
    return vectorstore
