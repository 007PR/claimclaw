from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class Settings:
    llm_provider: str
    openai_chat_model: str
    openai_vision_model: str
    openai_api_key: str | None

    embeddings_provider: str
    hf_embeddings_model: str
    openai_embeddings_model: str

    vector_store: str
    legal_index_dir: Path
    checkpoint_db: Path
    twilio_auth_token: str | None

    strict_llm_mode: bool
    dev_allow_fallback: bool


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "local"),
        hf_embeddings_model=os.getenv(
            "HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"
        ),
        vector_store=os.getenv("VECTOR_STORE", "chroma"),
        legal_index_dir=Path(os.getenv("LEGAL_INDEX_DIR", "storage/legal_index")),
        checkpoint_db=Path(os.getenv("CHECKPOINT_DB", "storage/claimclaw_state.sqlite")),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        strict_llm_mode=_env_bool("STRICT_LLM_MODE", True),
        dev_allow_fallback=_env_bool("DEV_ALLOW_FALLBACK", False),
    )
