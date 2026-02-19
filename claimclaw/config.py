from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local")
    hf_embeddings_model: str = os.getenv(
        "HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_embeddings_model: str = os.getenv(
        "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"
    )

    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    legal_index_dir: Path = Path(os.getenv("LEGAL_INDEX_DIR", "storage/legal_index"))
    checkpoint_db: Path = Path(os.getenv("CHECKPOINT_DB", "storage/claimclaw_state.sqlite"))
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")


def load_settings() -> Settings:
    return Settings()
