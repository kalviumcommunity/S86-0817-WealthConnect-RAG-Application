"""
GY3.44, GY3.45, GY3.47, GY3.48 — Unified Backend API for the RAG Service
WealthConnect RAG Application

Unified enterprise API supporting:
1. POST /query         — Grounded question answering with citations and caching.
2. POST /query/stream  — Server-Sent Events (SSE) streaming with early citations.
3. POST /upload        — Runtime document intake, chunking, and live indexing.
4. GET  /health        — Service liveness and collection diagnostics.
5. GET  /metrics       — Real-time operational telemetry (cache hits, latency, tokens).

Run locally:
    uvicorn rag_api:app --reload --port 8000
"""

import os
import re
import time
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from query_cache import QueryCache
from monitoring import metrics, log_audit_event
from text_cleaner import clean_text
from chunking_strategies import chunk_by_paragraph
from chunk_metadata_tracker import tag_chunks
from src.document_loader import load_text

load_dotenv()

# ---------------------------------------------------------------------------
# Config — all values loaded from environment (never hardcoded)
# ---------------------------------------------------------------------------

EMBED_MODEL           = os.getenv("EMBED_MODEL",           "text-embedding-3-small")
CHAT_MODEL            = os.getenv("CHAT_MODEL",            "gpt-4o-mini")
COLLECTION_NAME       = os.getenv("COLLECTION_NAME",       "wealthconnect_chunks")
MIN_TOP_SCORE         = float(os.getenv("MIN_TOP_SCORE",         "0.06"))
MIN_SUPPORTING_CHUNKS = int(os.getenv("MIN_SUPPORTING_CHUNKS",   "1"))
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K",             "5"))

REFUSAL_MESSAGE = (
    "I don't have enough reliable context in my knowledge base to answer that question. "
    "Please rephrase your question or contact a WealthConnect advisor directly."
)

query_cache = QueryCache(max_size=512, ttl_seconds=3600.0)

# ---------------------------------------------------------------------------
# In-memory Vector Collection & Topic Embeddings
# ---------------------------------------------------------------------------

_TOPIC_BASES: dict[str, np.ndarray] = {}
DIM = 1536

_TOPIC_MAP = {
    "password": [
        "How can I reset my WealthConnect password?",
        "To reset your WealthConnect password, visit the login page and select Forgot Password.",
        "Account recovery requires identity verification via registered email or phone number.",
        "Two-factor authentication is mandatory for all premium account holders.",
    ],
    "portfolio": [
        "What was the Q4 portfolio return?",
        "How did the growth fund perform last quarter?",
        "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
        "WealthConnect Q4 Portfolio Analysis and Earnings Overview.",
        "The tech sector allocation increased 8 percentage points in Q4.",
    ],
    "refund": [
        "What is the refund policy?",
        "How do I request a refund?",
        "All refund requests must be submitted within 30 days of the original transaction date.",
        "Refunds are processed within 5–7 business days after approval.",
    ],
    "platform": [
        "What is WealthConnect?",
        "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing.",
        "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents.",
        "We integrate with major brokerages to give users a single, holistic view of their portfolio.",
        "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant.",
        "The assistant resolves 65% of Tier 1 queries without human intervention.",
    ],
    "tax": [
        "What is the tax exemption limit for capital gains?",
        "What are the tax rules for municipal bond investments?",
        "Capital gains exemptions and wealth surtax rules for retail clients.",
        "Long-term capital gains up to $50,000 on approved clean energy and municipal bond investments are exempt from wealth surtax.",
        "Tax loss harvesting can offset ordinary income up to $3,000 annually.",
    ],
    "policy": [
        "What is the investment policy for aggressive portfolios?",
        "Moderate Growth Fund asset allocation constraints and rebalancing rules.",
        "The Moderate Growth Fund requires a minimum initial investment of $10,000 and maintains a 60/40 equity to fixed income split.",
        "Client portfolios must not exceed 25% single-sector concentration without senior compliance approval.",
    ],
    "unrelated": [
        "The cafeteria menu has pasta and salad today.",
        "Tomorrow's weather forecast shows heavy rain.",
        "What is the capital of France?",
    ],
}

CORPUS = [
    {"source": "policies.md",            "section": "Password Reset",   "chunk_index": 0,
     "text": "To reset your WealthConnect password, visit the login page and select Forgot Password."},
    {"source": "policies.md",            "section": "Account Recovery", "chunk_index": 1,
     "text": "Account recovery requires identity verification via registered email or phone number."},
    {"source": "policies.md",            "section": "Security",         "chunk_index": 2,
     "text": "Two-factor authentication is mandatory for all premium account holders."},
    {"source": "policies.md",            "section": "Refunds",          "chunk_index": 3,
     "text": "All refund requests must be submitted within 30 days of the original transaction date."},
    {"source": "policies.md",            "section": "Refunds",          "chunk_index": 4,
     "text": "Refunds are processed within 5–7 business days after approval."},
    {"source": "q4_earnings_report.pdf", "section": "Overview",         "chunk_index": 0,
     "text": "WealthConnect Q4 Portfolio Analysis and Earnings Overview."},
    {"source": "q4_earnings_report.pdf", "section": "Performance",      "chunk_index": 1,
     "text": "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy."},
    {"source": "q4_earnings_report.pdf", "section": "Outlook",          "chunk_index": 2,
     "text": "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds."},
    {"source": "q4_earnings_report.pdf", "section": "Operations",       "chunk_index": 3,
     "text": "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant."},
    {"source": "q4_earnings_report.pdf", "section": "Allocation",       "chunk_index": 4,
     "text": "The tech sector allocation increased 8 percentage points in Q4."},
    {"source": "sample_tax_rules.md",    "section": "Exemptions",       "chunk_index": 0,
     "text": "Long-term capital gains up to $50,000 on approved clean energy and municipal bond investments are exempt from wealth surtax."},
    {"source": "sample_investment_policy.txt", "section": "Allocation", "chunk_index": 0,
     "text": "The Moderate Growth Fund requires a minimum initial investment of $10,000 and maintains a 60/40 equity to fixed income split."},
    {"source": "sample_eligibility_guidelines.txt", "section": "Eligibility", "chunk_index": 0,
     "text": "Premier Wealth Advisory eligibility requires a minimum portfolio valuation of $250,000 in qualifying liquid assets."},
]


def _get_topic_base(topic: str) -> np.ndarray:
    if topic not in _TOPIC_BASES:
        seed = int.from_bytes(topic.encode("utf-8"), "big") % (2**31 - 1)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(DIM)
        _TOPIC_BASES[topic] = v / np.linalg.norm(v)
    return _TOPIC_BASES[topic]


def _classify_topic(text: str) -> str:
    lower = text.lower()
    for topic, phrases in _TOPIC_MAP.items():
        if any(p.lower() in lower or lower in p.lower() for p in phrases):
            return topic
    # Keyword fallback
    if any(k in lower for k in ["password", "reset", "login", "auth"]):
        return "password"
    if any(k in lower for k in ["portfolio", "return", "fund", "growth", "yield", "q4"]):
        return "portfolio"
    if any(k in lower for k in ["refund", "return", "money back"]):
        return "refund"
    if any(k in lower for k in ["platform", "wealthconnect", "assistant", "ai"]):
        return "platform"
    if any(k in lower for k in ["tax", "capital gains", "deduction", "exemption"]):
        return "tax"
    if any(k in lower for k in ["policy", "eligibility", "allocation", "minimum"]):
        return "policy"
    return "unrelated"


def _mock_embed(text: str) -> list[float]:
    topic = _classify_topic(text)
    base = _get_topic_base(topic).copy()
    seed = int.from_bytes(text[:16].encode("utf-8"), "big") % (2**31 - 1)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(DIM) * 0.05
    vec = base + noise
    return (vec / np.linalg.norm(vec)).tolist()


_embedding_client = None
_embedding_api_available = True

def _get_embedding_client():
    global _embedding_client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and _embedding_client is None:
        try:
            from openai import OpenAI
            _embedding_client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                max_retries=0,
            )
        except Exception:
            _embedding_client = None
    return _embedding_client


def _embed(text: str) -> list[float]:
    global _embedding_api_available
    if _embedding_api_available:
        client = _get_embedding_client()
        if client:
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=text)
                return resp.data[0].embedding
            except Exception:
                # Quota exhausted or network unavailable: instantly fall back to mock embed
                _embedding_api_available = False
    return _mock_embed(text)


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


_DOCUMENTS_REGISTRY: dict[str, dict] = {
    "sample_investment_policy.txt": {
        "name": "sample_investment_policy.txt",
        "title": "Moderate Growth Asset Allocation Rules",
        "category": "Investment Policy",
        "product": "Moderate Growth Fund",
        "version": "v3.2",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "sample_tax_rules.md": {
        "name": "sample_tax_rules.md",
        "title": "Capital Gains & Wealth Surtax Rules",
        "category": "Tax Rules & Circulars",
        "product": "Wealth Advisory Tax",
        "version": "v2.1",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "sample_product_brochure.html": {
        "name": "sample_product_brochure.html",
        "title": "Global Wealth Portfolio Brochure",
        "category": "Product Brochure",
        "product": "Global Wealth Portfolio",
        "version": "v1.4",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "sample_eligibility_guidelines.txt": {
        "name": "sample_eligibility_guidelines.txt",
        "title": "Premier Wealth Advisory Guidelines",
        "category": "Eligibility Guidelines",
        "product": "Premier Wealth Advisory",
        "version": "v2.0",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "q4_earnings_report.pdf": {
        "name": "q4_earnings_report.pdf",
        "title": "WealthConnect Q4 Portfolio Analysis",
        "category": "Corporate Report",
        "product": "WealthConnect Q4 Portfolio",
        "version": "v1.0",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "policies.md": {
        "name": "policies.md",
        "title": "Account Recovery & Password Policies",
        "category": "Compliance Policy",
        "product": "Platform Security",
        "version": "v1.2",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
    "Investment_Policy_IP-001.pdf": {
        "name": "Investment_Policy_IP-001.pdf",
        "title": "High Net Worth Mandate IP-001",
        "category": "Investment Policy",
        "product": "HNW Growth Strategy",
        "version": "v2.5",
        "approval_status": "approved",
        "uploaded_at": 1718000000.0,
        "chunks_count": 0,
    },
}


class _VectorCollection:
    def __init__(self):
        self.records: list[dict] = []

    def add(self, text: str, vector: list[float], metadata: dict):
        self.records.append({"text": text, "vector": vector, "metadata": metadata})

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
        exclude_unapproved: bool = True,
    ) -> list[dict]:
        scored = []
        for r in self.records:
            meta = r.get("metadata", {})
            if exclude_unapproved:
                # FR-04: Ground only in current approved documents (exclude superseded & draft)
                status = meta.get("approval_status", "approved")
                if status != "approved":
                    continue
            if metadata_filter:
                match = all(meta.get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue
            scored.append({
                "score": _cosine(vector, r["vector"]),
                "text": r["text"],
                "metadata": meta,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def _build_collection() -> _VectorCollection:
    col = _VectorCollection()
    # 1. Base demo corpus
    for c in CORPUS:
        src = c["source"]
        if src not in _DOCUMENTS_REGISTRY:
            _DOCUMENTS_REGISTRY[src] = {
                "name": src,
                "title": src.replace("_", " ").replace("-", " ").title(),
                "category": "General Policy",
                "product": "Wealth Advisory",
                "version": "v1.0",
                "approval_status": "approved",
                "uploaded_at": 1718000000.0,
                "chunks_count": 0,
            }
        _DOCUMENTS_REGISTRY[src]["chunks_count"] += 1
        d_meta = _DOCUMENTS_REGISTRY[src]

        vec = _embed(c["text"])
        col.add(
            text=c["text"],
            vector=vec,
            metadata={
                "source": c["source"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "approval_status": d_meta.get("approval_status", "approved"),
                "category": d_meta.get("category", "General"),
                "product": d_meta.get("product", "Wealth Advisory"),
                "version": d_meta.get("version", "v1.0"),
            },
        )

    # 2. Ingest approved documents from data/ directory if present
    data_dir = Path("data")
    if data_dir.exists():
        for file_path in data_dir.iterdir():
            if file_path.suffix.lower() in [".txt", ".md", ".pdf", ".html", ".htm"]:
                try:
                    raw_text = load_text(file_path)
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        chunks = chunk_by_paragraph(cleaned)
                        src = file_path.name
                        if src not in _DOCUMENTS_REGISTRY:
                            _DOCUMENTS_REGISTRY[src] = {
                                "name": src,
                                "title": src.replace("_", " ").replace("-", " ").title(),
                                "category": "General Policy",
                                "product": "Wealth Advisory",
                                "version": "v1.0",
                                "approval_status": "approved",
                                "uploaded_at": time.time(),
                                "chunks_count": 0,
                            }
                        d_meta = _DOCUMENTS_REGISTRY[src]
                        tagged = tag_chunks(
                            source=src,
                            chunks=chunks,
                            extra_meta={
                                "document_type": file_path.suffix.lstrip("."),
                                "approval_status": d_meta.get("approval_status", "approved"),
                                "category": d_meta.get("category", "General"),
                                "product": d_meta.get("product", "Wealth Advisory"),
                                "version": d_meta.get("version", "v1.0"),
                            },
                        )
                        d_meta["chunks_count"] += len(tagged)
                        for item in tagged:
                            col.add(
                                text=item["text"],
                                vector=_embed(item["text"]),
                                metadata=item["metadata"],
                            )
                except Exception:
                    pass

    return col


_collection: _VectorCollection = _build_collection()


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

def _retrieve(question: str, col: _VectorCollection, exclude_unapproved: bool = True) -> list[dict]:
    qv = _embed(question)
    return [
        {"score": round(r["score"], 6), "text": r["text"], "metadata": r["metadata"]}
        for r in col.search(vector=qv, top_k=RETRIEVAL_K, exclude_unapproved=exclude_unapproved)
    ]


def _retrieval_is_strong(chunks: list[dict]) -> tuple[bool, str]:
    if not chunks:
        return False, "no_chunks_retrieved"
    strong = [c for c in chunks if c["score"] >= MIN_TOP_SCORE]
    if len(strong) < MIN_SUPPORTING_CHUNKS:
        return False, f"only {len(strong)} chunk(s) above threshold {MIN_TOP_SCORE}"
    return True, "sufficient_evidence"


def _on_topic(chunks: list[dict], query: str) -> bool:
    STOPWORDS = {"the","a","an","is","are","was","were","of","in","to","and",
                 "or","for","my","i","how","what","when","where","why","do","did"}
    def tokens(t: str) -> set[str]:
        return set(re.sub(r"[^\w]", " ", t.lower()).split()) - STOPWORDS
    qtoks = tokens(query)
    for chunk in chunks[:3]:
        if qtoks & tokens(chunk["text"]):
            return True
    return False


def _grounded_answer(question: str, chunks: list[dict]) -> dict:
    context_parts = []
    for i, c in enumerate(chunks[:5], start=1):
        m = c["metadata"]
        sec = m.get("section", "General")
        context_parts.append(
            f"[{i}] {m.get('source', 'unknown')}#{m.get('chunk_index', 0)} ({sec})\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    sources = []
    for c in chunks[:5]:
        m = c["metadata"]
        src = m.get("source", "unknown")
        reg_info = _DOCUMENTS_REGISTRY.get(src, {})
        sources.append({
            "source":      src,
            "chunk_id":    f"{src}:{m.get('chunk_index', 0)}",
            "score":       c["score"],
            "section":     m.get("section", "General"),
            "text":        c["text"],
            "version":     m.get("version") or reg_info.get("version", "v1.0"),
            "category":    m.get("category") or reg_info.get("category", "General"),
            "product":     m.get("product") or reg_info.get("product", "Wealth Advisory"),
        })

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                max_retries=0,
            )
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded financial assistant for WealthConnect. "
                            "Answer the question using ONLY the provided context. "
                            "If the answer is not in the context, say: "
                            "'I don't have enough information in the provided context.' "
                            "Cite sources using markers like [1] or [2]."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )
            answer = completion.choices[0].message.content.strip()
            return {"answer": answer, "sources": sources}
        except Exception:
            pass

    # Deterministic mock answer
    top = chunks[0]
    sec = top["metadata"].get("section", "General")
    src = top["metadata"].get("source", "unknown")
    answer = (
        f"Based on the provided context: {top['text']} "
        f"[1] See {src} ({sec}) for details."
    )
    return {"answer": answer, "sources": sources}


def guarded_answer(question: str) -> dict:
    """Execute full guarded RAG logic."""
    chunks = _retrieve(question, _collection)

    strong, reason = _retrieval_is_strong(chunks)
    if not strong:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "status": "refused_weak_context",
            "diagnostics": {"reason": reason},
        }

    if not _on_topic(chunks, question):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "status": "refused_off_topic",
            "diagnostics": {"reason": "no vocabulary overlap with top-3 chunks"},
        }

    result = _grounded_answer(question, chunks)
    return {
        **result,
        "status": "answered",
        "diagnostics": {"reason": "sufficient_evidence"},
    }


async def rag_pipeline_stream(question: str):
    """
    Async generator that streams citations first, then token by token.
    """
    chunks = _retrieve(question, _collection)
    strong, reason = _retrieval_is_strong(chunks)

    if not strong:
        for word in REFUSAL_MESSAGE.split(" "):
            yield {"type": "token", "text": word + " "}
            await asyncio.sleep(0.01)
        yield {"type": "done", "status": "refused_weak_context"}
        return

    if not _on_topic(chunks, question):
        for word in REFUSAL_MESSAGE.split(" "):
            yield {"type": "token", "text": word + " "}
            await asyncio.sleep(0.01)
        yield {"type": "done", "status": "refused_off_topic"}
        return

    # Early citation dispatch
    sources = [
        {
            "label": f"[{i+1}]",
            "source": c["metadata"].get("source", "unknown"),
            "document": c["metadata"].get("source", "unknown"),
            "chunk_id": f"{c['metadata'].get('source', 'unknown')}:{c['metadata'].get('chunk_index', 0)}",
            "section": c["metadata"].get("section", "General"),
            "score": c["score"],
            "text": c["text"],
            "version": c["metadata"].get("version") or _DOCUMENTS_REGISTRY.get(c["metadata"].get("source", ""), {}).get("version", "v1.0"),
            "category": c["metadata"].get("category") or _DOCUMENTS_REGISTRY.get(c["metadata"].get("source", ""), {}).get("category", "General"),
            "product": c["metadata"].get("product") or _DOCUMENTS_REGISTRY.get(c["metadata"].get("source", ""), {}).get("product", "Wealth Advisory"),
        }
        for i, c in enumerate(chunks[:5])
    ]
    yield {"type": "citations", "sources": sources}

    context_parts = [
        f"[{i+1}] {s['document']}#{s['chunk_id'].split(':')[-1]} ({s['section']})\n{s['text']}"
        for i, s in enumerate(sources)
    ]
    context = "\n\n---\n\n".join(context_parts)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                max_retries=0,
            )
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded financial assistant for WealthConnect. "
                            "Answer using ONLY the provided context. Cite sources using [1] or [2]."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.0,
                max_tokens=512,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "text": delta.content}
                    await asyncio.sleep(0)
            yield {"type": "done", "status": "answered"}
            return
        except Exception:
            pass

    # Mock token streaming
    top = chunks[0]
    sec = top["metadata"].get("section", "General")
    src = top["metadata"].get("source", "unknown")
    mock_answer = (
        f"Based on the provided context, {top['text']} "
        f"[1] For more details, refer to {src} ({sec})."
    )
    for word in mock_answer.split(" "):
        yield {"type": "token", "text": word + " "}
        await asyncio.sleep(0.02)

    yield {"type": "done", "status": "answered"}


# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WealthConnect Grounded RAG API",
    description="Enterprise Knowledge Assistant for Retail Bank Wealth Advisors.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The advisory question to answer.",
        examples=["What was the Q4 portfolio return?"],
    )


class Source(BaseModel):
    source:   str
    chunk_id: str | None   = None
    score:    float | None = None
    section:  str | None   = None
    text:     str | None   = None
    version:  str | None   = None
    category: str | None   = None
    product:  str | None   = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str | None = None
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    comment: str | None = None


class DocumentStatusUpdate(BaseModel):
    approval_status: str = Field(..., description="'approved', 'draft', or 'superseded'")


# ---------------------------------------------------------------------------
# Feedback & Governance Store
# ---------------------------------------------------------------------------

FEEDBACK_LOG_FILE = Path("logs") / "feedback.jsonl"
_feedback_store: list[dict] = []


def _load_feedback():
    if FEEDBACK_LOG_FILE.exists():
        try:
            with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _feedback_store.append(json.loads(line))
        except Exception:
            pass


_load_feedback()


# ---------------------------------------------------------------------------
# Authentication Store & Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    username:   str
    full_name:  str
    role:       str  # 'relationship_manager' or 'wealth_admin'
    department: str
    title:      str


class LoginResponse(BaseModel):
    status: str
    token:  str
    user:   UserInfo


_USERS_DB = {
    "rm_advisor": {
        "password":   "Password@123",
        "full_name":  "G Yashmieen",
        "role":       "relationship_manager",
        "department": "Private Wealth Advisory",
        "title":      "Senior Relationship Manager",
    },
    "rm": {
        "password":   "rm123",
        "full_name":  "G Yashmieen",
        "role":       "relationship_manager",
        "department": "Private Wealth Advisory",
        "title":      "Senior Relationship Manager",
    },
    "wealth_admin": {
        "password":   "Admin@123",
        "full_name":  "Dodla Bhanu Teja Reddy",
        "role":       "wealth_admin",
        "department": "Wealth Division Governance",
        "title":      "Chief Compliance & Governance Officer",
    },
    "admin": {
        "password":   "admin123",
        "full_name":  "Dodla Bhanu Teja Reddy",
        "role":       "wealth_admin",
        "department": "Wealth Division Governance",
        "title":      "Chief Compliance & Governance Officer",
    },
}


class QueryResponse(BaseModel):
    answer:      str
    sources:     list[Source]
    status:      str
    cached:      bool = False
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    status:     str
    model:      str
    collection: str
    chunks:     int
    streaming:  bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the WealthConnect executive banking frontend."""
    index_path = Path(__file__).resolve().parent / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "WealthConnect RAG API is active. Access docs at /docs"}


@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
def login(payload: LoginRequest):
    """
    Authenticate Relationship Manager or Wealth Admin.
    Demo Credentials:
    - RM: 'rm_advisor' (or 'rm') / 'Password@123' (or 'rm123')
    - Admin: 'wealth_admin' (or 'admin') / 'Admin@123' (or 'admin123')
    """
    uname = payload.username.strip().lower()
    pwd = payload.password.strip()

    user_rec = _USERS_DB.get(uname)
    if not user_rec or user_rec["password"] != pwd:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. For RM use 'rm_advisor' / 'Password@123'. For Admin use 'wealth_admin' / 'Admin@123'."
        )

    token = f"wc_{user_rec['role']}_{int(time.time())}"
    return {
        "status": "success",
        "token": token,
        "user": {
            "username": uname,
            "full_name": user_rec["full_name"],
            "role": user_rec["role"],
            "department": user_rec["department"],
            "title": user_rec["title"],
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    """Liveness & health check."""
    return {
        "status":     "ok",
        "model":      EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "chunks":     len(_collection.records),
        "streaming":  True,
    }


@app.get("/metrics", tags=["Ops"])
def get_metrics():
    """Operational telemetry & metrics."""
    data = metrics.get_metrics()
    data["cache"] = query_cache.stats
    data["collection_size"] = len(_collection.records)
    return data


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest, req: Request):
    """
    Standard batch query endpoint with LRU caching and audit logging.
    """
    start = time.time()
    q = request.question.strip()

    # Check cache
    cached = query_cache.get(q)
    if cached is not None:
        duration = (time.time() - start) * 1000
        metrics.record_query(duration, cached.get("status", "answered"), 0, 0)
        return {
            "answer": cached["answer"],
            "sources": cached.get("sources", []),
            "status": cached.get("status", "answered"),
            "cached": True,
            "duration_ms": round(duration, 2),
        }

    try:
        result = guarded_answer(q)
        duration = (time.time() - start) * 1000

        # Record metrics and audit event
        status = result.get("status", "answered")
        sources = result.get("sources", [])
        client_ip = req.client.host if req.client else "127.0.0.1"

        metrics.record_query(duration, status, tokens_in=len(q)//4, tokens_out=len(result["answer"])//4)
        log_audit_event(q, status, sources, duration, result.get("diagnostics"), client_ip)

        response_payload = {
            "answer": result["answer"],
            "sources": sources,
            "status": status,
            "cached": False,
            "duration_ms": round(duration, 2),
        }

        # Cache successful answered results
        if status == "answered":
            query_cache.set(q, response_payload)

        return response_payload

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG service error: {exc}")


@app.post("/query/stream", tags=["RAG"])
async def stream_query(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits citations before token generation begins.
    """
    async def events():
        try:
            async for event in rag_pipeline_stream(request.question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            err = {"type": "error", "message": "Stream interrupted."}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback", tags=["Feedback"])
def submit_feedback(req: FeedbackRequest):
    """
    Record RM feedback (Helpful / Not Helpful + comments) for compliance and governance (PRD Section 7 FR-08).
    """
    if req.rating not in {"helpful", "not_helpful"}:
        raise HTTPException(status_code=400, detail="Rating must be 'helpful' or 'not_helpful'")

    record = {
        "id": f"fb_{int(time.time() * 1000)}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "question": req.question,
        "answer": req.answer or "",
        "rating": req.rating,
        "comment": req.comment or "",
    }
    _feedback_store.insert(0, record)
    try:
        FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
    return {"status": "recorded", "id": record["id"]}


@app.get("/admin/feedback", tags=["Admin"])
def get_admin_feedback():
    """
    Return RM feedback submissions, ratings, and satisfaction rate metrics (PRD Section 19).
    """
    total = len(_feedback_store)
    helpful_count = sum(1 for f in _feedback_store if f["rating"] == "helpful")
    not_helpful_count = sum(1 for f in _feedback_store if f["rating"] == "not_helpful")
    satisfaction_rate = round((helpful_count / total * 100), 1) if total > 0 else 100.0
    return {
        "total": total,
        "helpful": helpful_count,
        "not_helpful": not_helpful_count,
        "satisfaction_rate": satisfaction_rate,
        "feedback": _feedback_store[:100],
    }


@app.get("/admin/documents", tags=["Admin"])
def list_admin_documents():
    """
    List all catalog documents with version, category, product, chunks, and approval status (PRD Section 19).
    """
    docs = list(_DOCUMENTS_REGISTRY.values())
    total = len(docs)
    approved = sum(1 for d in docs if d.get("approval_status") == "approved")
    draft = sum(1 for d in docs if d.get("approval_status") == "draft")
    superseded = sum(1 for d in docs if d.get("approval_status") == "superseded")
    return {
        "summary": {
            "total": total,
            "approved": approved,
            "draft": draft,
            "superseded": superseded,
        },
        "documents": docs,
    }


@app.patch("/admin/documents/{doc_name}", tags=["Admin"])
def update_document_status(doc_name: str, payload: DocumentStatusUpdate):
    """
    Update document approval status (approved, draft, or superseded) (PRD Section 19).
    FR-04: Superseded and draft documents will be immediately excluded from RM query retrieval.
    """
    new_status = payload.approval_status.lower()
    if new_status not in {"approved", "draft", "superseded"}:
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'approved', 'draft', or 'superseded'.")

    if doc_name not in _DOCUMENTS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found in registry.")

    _DOCUMENTS_REGISTRY[doc_name]["approval_status"] = new_status

    # Update chunks in collection
    updated_chunks = 0
    for r in _collection.records:
        if r.get("metadata", {}).get("source") == doc_name:
            r["metadata"]["approval_status"] = new_status
            updated_chunks += 1

    query_cache.clear()

    return {
        "status": "success",
        "document": _DOCUMENTS_REGISTRY[doc_name],
        "chunks_updated": updated_chunks,
    }


@app.delete("/admin/documents/{doc_name}", tags=["Admin"])
def delete_document(doc_name: str):
    """
    Remove a document from the governance registry and purge its chunks from the active index (PRD Section 19).
    """
    if doc_name not in _DOCUMENTS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found in registry.")

    del _DOCUMENTS_REGISTRY[doc_name]

    initial_len = len(_collection.records)
    _collection.records = [
        r for r in _collection.records
        if r.get("metadata", {}).get("source") != doc_name
    ]
    purged_chunks = initial_len - len(_collection.records)

    query_cache.clear()

    return {
        "status": "deleted",
        "document": doc_name,
        "chunks_purged": purged_chunks,
        "remaining_chunks": len(_collection.records),
    }


@app.get("/admin/questions", tags=["Admin"])
def get_admin_questions():
    """
    Operational question analytics and policy gap identification (PRD Section 8 US-08).
    Pinpoints frequently asked questions and policy gaps (unanswered / refused queries).
    """
    audit_records = []
    audit_file = Path("logs") / "rag_audit.log"
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        audit_records.append(json.loads(line))
        except Exception:
            pass

    total = len(audit_records)
    answered = [r for r in audit_records if r.get("status") == "answered"]
    unanswered = [r for r in audit_records if r.get("status", "").startswith("refused") or r.get("status") == "error"]

    coverage_rate = round((len(answered) / total * 100), 1) if total > 0 else 100.0

    # Unanswered log (documentation gaps)
    unanswered_log = []
    for r in reversed(unanswered):
        diag = r.get("diagnostics") or {}
        reason = diag.get("reason") or r.get("status")
        unanswered_log.append({
            "timestamp": r.get("timestamp", ""),
            "question": r.get("question", ""),
            "status": r.get("status", "refused"),
            "reason": reason,
        })

    # Frequency analysis
    counts = {}
    last_status = {}
    for r in audit_records:
        q = r.get("question", "").strip()
        if q:
            counts[q] = counts.get(q, 0) + 1
            last_status[q] = r.get("status", "answered")

    faq_list = [
        {"question": q, "count": cnt, "status": last_status.get(q, "answered")}
        for q, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ][:10]

    return {
        "total_queries": total,
        "answered_queries": len(answered),
        "unanswered_queries": len(unanswered),
        "coverage_rate": coverage_rate,
        "unanswered_log": unanswered_log[:50],
        "frequent_questions": faq_list,
    }


@app.post("/upload", tags=["Admin"])
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("general"),
    product: str = Form("General Wealth"),
    version: str = Form("v1.0"),
    approval_status: str = Form("approved"),
):
    """
    Runtime document intake, chunking, and live indexing endpoint (PRD FR-03, Milestone 3.45).
    Accepts .txt, .md, .pdf, or .html files, indexes chunks into collection with metadata.
    """
    allowed_exts = {".txt", ".md", ".pdf", ".html", ".htm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {list(allowed_exts)}",
        )

    upload_dir = Path("data") / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename

    content = await file.read()
    with open(destination, "wb") as f:
        f.write(content)

    try:
        raw_text = load_text(destination)
        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Document contained no extractable text after cleaning.")

        status = approval_status.lower()
        if status not in {"approved", "draft", "superseded"}:
            status = "approved"

        chunks = chunk_by_paragraph(cleaned)
        tagged = tag_chunks(
            source=file.filename,
            chunks=chunks,
            extra_meta={
                "document_type": document_type,
                "category": document_type.replace("_", " ").title(),
                "product": product,
                "version": version,
                "approval_status": status,
                "uploaded_at": time.time(),
            },
        )

        for item in tagged:
            vec = _embed(item["text"])
            _collection.add(
                text=item["text"],
                vector=vec,
                metadata=item["metadata"],
            )

        # Update document registry
        _DOCUMENTS_REGISTRY[file.filename] = {
            "name": file.filename,
            "title": file.filename.replace("_", " ").replace("-", " ").title(),
            "category": document_type.replace("_", " ").title(),
            "product": product,
            "version": version,
            "approval_status": status,
            "uploaded_at": time.time(),
            "chunks_count": len(tagged),
        }

        # Invalidate query cache when new documents are added
        query_cache.clear()

        return {
            "status": "success",
            "filename": file.filename,
            "document_type": document_type,
            "product": product,
            "version": version,
            "approval_status": status,
            "chunks_indexed": len(tagged),
            "total_collection_chunks": len(_collection.records),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {exc}")


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=False)
