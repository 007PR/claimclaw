from __future__ import annotations

import socket
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import load_settings


def _check_dns(host: str) -> tuple[bool, str]:
    try:
        socket.gethostbyname(host)
        return True, "resolved"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def run_preflight(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    settings = load_settings()

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, severity: str, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "severity": severity, "detail": detail})

    add("project_root", root.exists(), "fail", str(root))
    add("claims_dir", (root / "data" / "claims").exists(), "fail", "data/claims")
    add("venv", (root / ".venv").exists(), "warn", str(root / ".venv"))

    if settings.strict_llm_mode and not settings.dev_allow_fallback:
        add(
            "openai_api_key",
            bool(settings.openai_api_key),
            "fail",
            "Required in strict LLM mode",
        )
    else:
        add(
            "openai_api_key",
            bool(settings.openai_api_key),
            "warn",
            "Optional when DEV_ALLOW_FALLBACK=true",
        )

    for host in ("api.openai.com", "openaipublic.blob.core.windows.net", "huggingface.co"):
        ok, detail = _check_dns(host)
        add(f"dns:{host}", ok, "warn", detail)

    checkpoint_path = (root / settings.checkpoint_db).resolve()
    checkpoint_parent = checkpoint_path.parent
    checkpoint_ok = True
    checkpoint_detail = str(checkpoint_path)
    checkpoint_severity = "fail"

    try:
        checkpoint_parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_path))
        conn.execute("CREATE TABLE IF NOT EXISTS preflight_ping(id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
    except Exception as primary_exc:  # pragma: no cover
        # Sandbox and restrictive filesystems can block project-local sqlite writes.
        fallback_path = Path("/tmp/claimclaw_preflight.sqlite")
        try:
            conn = sqlite3.connect(str(fallback_path))
            conn.execute("CREATE TABLE IF NOT EXISTS preflight_ping(id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()
            checkpoint_ok = True
            checkpoint_severity = "warn"
            checkpoint_detail = (
                f"Primary path unavailable ({primary_exc}); fallback writable at {fallback_path}"
            )
        except Exception as fallback_exc:
            checkpoint_ok = False
            checkpoint_severity = "fail"
            checkpoint_detail = f"Primary error: {primary_exc}; fallback error: {fallback_exc}"

    add("checkpoint_db", checkpoint_ok, checkpoint_severity, checkpoint_detail)

    try:
        import playwright  # noqa: F401

        add("playwright_import", True, "warn", "import ok")
    except Exception as exc:  # pragma: no cover
        add("playwright_import", False, "warn", str(exc))

    failed = [c for c in checks if not c["ok"] and c["severity"] == "fail"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warn"]

    status = "ok"
    if failed:
        status = "fail"
    elif warnings:
        status = "warn"

    return {
        "status": status,
        "settings": {
            "llm_provider": settings.llm_provider,
            "strict_llm_mode": settings.strict_llm_mode,
            "dev_allow_fallback": settings.dev_allow_fallback,
            "checkpoint_db": str(settings.checkpoint_db),
        },
        "summary": {
            "passed": len([c for c in checks if c["ok"]]),
            "failed": len(failed),
            "warnings": len(warnings),
        },
        "checks": checks,
    }
