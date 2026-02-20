from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_project_env() -> None:
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)


def _build_openai_chat_llm(model_name: str, temperature: float = 0) -> Any | None:
    _load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    return ChatOpenAI(model=model_name, temperature=temperature)


def get_main_llm(model_name: str | None = None, temperature: float = 0) -> Any | None:
    _load_project_env()
    resolved = model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    return _build_openai_chat_llm(resolved, temperature=temperature)


def get_vision_llm(model_name: str | None = None, temperature: float = 0) -> Any | None:
    _load_project_env()
    resolved = model_name or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    return _build_openai_chat_llm(resolved, temperature=temperature)
