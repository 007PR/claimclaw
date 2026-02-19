from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from .schemas import LegalPosition

MORATORIUM_EFFECTIVE_DATE = date(2024, 4, 1)
MORATORIUM_YEARS = 5


def moratorium_rule_check(
    policy_age_years: float, rejection_reason: str, alleged_fraud: bool = False
) -> LegalPosition:
    reason = rejection_reason.lower()
    nondisclosure_keywords = [
        "non-disclosure",
        "nondisclosure",
        "suppression",
        "pre-existing",
        "pre existing",
        "ped",
        "hidden",
    ]
    nondisclosure_dispute = any(token in reason for token in nondisclosure_keywords)
    moratorium_applies = policy_age_years > MORATORIUM_YEARS
    illegal_rejection = moratorium_applies and nondisclosure_dispute and not alleged_fraud

    legal_basis = (
        "IRDAI health insurance moratorium standard (updated April 2024): "
        "after 5 years of continuous coverage, non-disclosure disputes cannot be used "
        "to repudiate claims unless fraud is proven."
    )

    if illegal_rejection:
        recommended_action = (
            "Demand immediate reversal, cite 5-year moratorium protection, and escalate "
            "to Bima Bharosa and Insurance Ombudsman if insurer does not comply."
        )
    else:
        recommended_action = (
            "Request full repudiation basis with clause citations and evidence, then "
            "proceed to escalation if grounds remain weak or non-compliant."
        )

    return LegalPosition(
        moratorium_applies=moratorium_applies,
        illegal_rejection=illegal_rejection,
        legal_basis=legal_basis,
        recommended_action=recommended_action,
    )


def _build_embeddings() -> Any:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        return OpenAIEmbeddings(model=model)

    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = os.getenv(
        "HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)


def _load_legal_documents(source_dir: Path) -> list[Any]:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs: list[Any] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def ingest_legal_corpus(
    source_dir: str | Path, persist_dir: str | Path, store: str = "chroma"
) -> dict[str, Any]:
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    source_dir = Path(source_dir)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_legal_documents(source_dir)
    if not docs:
        raise ValueError(
            f"No legal documents found in {source_dir}. Add IRDAI circular PDFs first."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180)
    chunks = splitter.split_documents(docs)
    embeddings = _build_embeddings()

    store_normalized = store.lower()
    if store_normalized == "faiss":
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(persist_dir))
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        vectorstore.persist()

    return {
        "documents_ingested": len(docs),
        "chunks_ingested": len(chunks),
        "store": store_normalized,
        "persist_dir": str(persist_dir),
    }


def load_legal_retriever(
    persist_dir: str | Path, store: str = "chroma", k: int = 5
) -> Any:
    from langchain_community.vectorstores import FAISS, Chroma

    persist_dir = Path(persist_dir)
    embeddings = _build_embeddings()
    store_normalized = store.lower()

    if store_normalized == "faiss":
        vectorstore = FAISS.load_local(
            str(persist_dir), embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def extract_moratorium_clause_hint(retriever: Any) -> str:
    docs = retriever.invoke(
        "Find IRDAI clause language on 5-year moratorium for non-disclosure and fraud exception."
    )
    for doc in docs:
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        lowered = text.lower()
        if "moratorium" in lowered and "5" in lowered and "fraud" in lowered:
            return text[:450]
    return "No explicit moratorium snippet was found in top retrieved chunks."


def answer_legal_question(question: str, retriever: Any, llm: Any | None = None) -> dict[str, Any]:
    docs = retriever.invoke(question)
    snippets: list[str] = []
    sources: list[str] = []
    for doc in docs:
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        snippets.append(text[:420])
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)

    if llm is None:
        answer = snippets[0] if snippets else "No relevant legal context found."
    else:
        from langchain_core.messages import HumanMessage, SystemMessage

        context = "\n\n".join(snippets)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an Indian insurance legal analysis assistant. "
                        "Answer with direct clause-level precision and include fraud exception logic."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        "Return a concise legal answer."
                    )
                ),
            ]
        )
        answer = str(response.content)

    return {"question": question, "answer": answer, "sources": sources, "snippets": snippets}


def default_hidden_bp_counter(policy_age_years: float = 6.0) -> dict[str, Any]:
    position = moratorium_rule_check(
        policy_age_years=policy_age_years,
        rejection_reason="hidden BP issue / non-disclosure",
        alleged_fraud=False,
    )
    return {
        "question": "A 6-year-old policy was rejected for hidden BP. What legal clause counters this?",
        "answer": (
            "Counter with the IRDAI 5-year moratorium protection (effective April 2024): "
            "after 5 years of continuous coverage, repudiation for non-disclosure/PED is not valid "
            "unless the insurer proves fraud."
        ),
        "moratorium_check": {
            "moratorium_applies": position.moratorium_applies,
            "illegal_rejection": position.illegal_rejection,
            "legal_basis": position.legal_basis,
        },
    }
