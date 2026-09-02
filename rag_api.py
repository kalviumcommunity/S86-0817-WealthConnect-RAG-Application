"""
GY3.44 — Backend API for the RAG Service
WealthConnect RAG Application

Exposes the RAG pipeline as a FastAPI HTTP service so any client
(frontend, chatbot UI, another service) can POST a question and receive
a structured JSON response with answer, sources, and status.

Run locally:
    uvicorn rag_api:app --reload --port 8000

Sample request:
    curl -X POST http://localhost:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What was the Q4 portfolio return?"}'
"""

import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Config — all values loaded from environment (never hardcoded)
# ---------------------------------------------------------------------------

EMBED_MODEL     = os.getenv("EMBED_MODEL",     "text-embedding-3-small")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "gpt-4o-mini")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wealthconnect_chunks")

# Guardrail thresholds — override via env in staging / production
MIN_TOP_SCORE         = float(os.getenv("MIN_TOP_SCORE",         "0.06"))
MIN_SUPPORTING_CHUNKS = int(os.getenv("MIN_SUPPORTING_CHUNKS",   "1"))
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K",             "5"))

# ---------------------------------------------------------------------------
# ── RAG pipeline (self-contained, matches prior modules) ──────────────────
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
    "unrelated": [
        "The cafeteria menu has pasta and salad today.",
        "Tomorrow's weather forecast shows heavy rain.",
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
    {"source": "about.html",             "section": "Overview",         "chunk_index": 0,
     "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing."},
    {"source": "about.html",             "section": "Features",         "chunk_index": 1,
     "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents."},
    {"source": "about.html",             "section": "Integrations",     "chunk_index": 2,
     "text": "We integrate with major brokerages to give users a single, holistic view of their portfolio."},
]

REFUSAL_MESSAGE = (
    "I don't have enough reliable context in my knowledge base to answer that question. "
    "Please rephrase your question or contact a WealthConnect advisor directly."
)


def _get_topic(text: str) -> str:
    for topic, members in _TOPIC_MAP.items():
        if text in members:
            return topic
    return "unrelated"


def _get_base(topic: str) -> np.ndarray:
    if topic not in _TOPIC_BASES:
        rng = np.random.default_rng(abs(hash(topic)) % (2 ** 31))
        v   = rng.standard_normal(DIM)
        _TOPIC_BASES[topic] = v / np.linalg.norm(v)
    return _TOPIC_BASES[topic]


def _mock_embed(texts: list[str]) -> list[list[float]]:
    result = []
    for text in texts:
        topic = _get_topic(text)
        base  = _get_base(topic)
        rng   = np.random.default_rng(abs(hash(text)) % (2 ** 31))
        noise = rng.standard_normal(DIM) * 0.08
        vec   = base + noise
        result.append((vec / np.linalg.norm(vec)).tolist())
    return result


def _api_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key,
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _embed(texts: list[str]) -> list[list[float]]:
    """Use real API if key present, otherwise fall back to mock vectors."""
    if os.getenv("OPENAI_API_KEY"):
        return _api_embed(texts)
    return _mock_embed(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class _VectorCollection:
    def __init__(self):
        self._records: list[dict] = []

    def add(self, text: str, embedding: list[float], metadata: dict) -> None:
        self._records.append({"text": text, "embedding": embedding, "metadata": metadata})

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        scored = [
            {"score": _cosine(vector, r["embedding"]),
             "text": r["text"], "metadata": r["metadata"]}
            for r in self._records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def _build_collection() -> _VectorCollection:
    col     = _VectorCollection()
    texts   = [c["text"] for c in CORPUS]
    vectors = _embed(texts)
    for chunk, vector in zip(CORPUS, vectors):
        col.add(
            text=chunk["text"],
            embedding=vector,
            metadata={"source": chunk["source"], "section": chunk["section"],
                      "chunk_index": chunk["chunk_index"]},
        )
    return col


def _retrieve(query: str, col: _VectorCollection) -> list[dict]:
    qv = _embed([query])[0]
    return [
        {"score": round(r["score"], 6), "text": r["text"], "metadata": r["metadata"]}
        for r in col.search(vector=qv, top_k=RETRIEVAL_K)
    ]


def _retrieval_is_strong(chunks: list[dict]) -> tuple[bool, str]:
    if not chunks:
        return False, "no_chunks_retrieved"
    strong = [c for c in chunks if c["score"] >= MIN_TOP_SCORE]
    if len(strong) < MIN_SUPPORTING_CHUNKS:
        return False, f"only {len(strong)} chunk(s) above threshold {MIN_TOP_SCORE}"
    return True, "sufficient_evidence"


def _on_topic(chunks: list[dict], query: str) -> bool:
    import re
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
    """
    Build a grounded answer.
    If OPENAI_API_KEY is present, calls the chat API with an injected context
    prompt. Otherwise returns a template answer from the top chunk.
    """
    # Format context with numbered source markers
    context_parts = []
    for i, c in enumerate(chunks[:5], start=1):
        m = c["metadata"]
        context_parts.append(
            f"[{i}] {m['source']}#{m['chunk_index']} ({m['section']})\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    sources = [
        {
            "source":      c["metadata"]["source"],
            "chunk_id":    f"{c['metadata']['source']}:{c['metadata']['chunk_index']}",
            "score":       c["score"],
            "section":     c["metadata"]["section"],
        }
        for c in chunks[:5]
    ]

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
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
            temperature=0,
            max_tokens=512,
        )
        answer = completion.choices[0].message.content.strip()
    else:
        # Mock answer for offline / no-key environments
        top = chunks[0]
        answer = (
            f"Based on the provided context: {top['text']} "
            f"[1] See {top['metadata']['source']} ({top['metadata']['section']}) for details."
        )

    return {"answer": answer, "sources": sources}


def guarded_answer(question: str) -> dict:
    """
    Full guarded RAG pipeline callable by the API endpoint.
    Returns a result dict consumed by the /query handler.
    """
    chunks = _retrieve(question, _collection)

    strong, reason = _retrieval_is_strong(chunks)
    if not strong:
        return {"answer": REFUSAL_MESSAGE, "sources": [],
                "status": "refused_weak_context", "diagnostics": {"reason": reason}}

    if not _on_topic(chunks, question):
        return {"answer": REFUSAL_MESSAGE, "sources": [],
                "status": "refused_off_topic",
                "diagnostics": {"reason": "no vocabulary overlap with top-3 chunks"}}

    result = _grounded_answer(question, chunks)
    return {**result, "status": "answered",
            "diagnostics": {"reason": "sufficient_evidence"}}


# ---------------------------------------------------------------------------
# Application startup — build collection once at boot
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WealthConnect RAG API",
    description="Ask questions about WealthConnect policies and reports.",
    version="1.0.0",
)

# Build the in-memory vector collection at module load time
_collection: _VectorCollection = _build_collection()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The question to answer from the WealthConnect knowledge base.",
        examples=["What was the Q4 portfolio return?"],
    )


class Source(BaseModel):
    source:   str
    chunk_id: str | None  = None
    score:    float | None = None
    section:  str | None  = None


class QueryResponse(BaseModel):
    answer:  str
    sources: list[Source]
    status:  str


class HealthResponse(BaseModel):
    status:     str
    model:      str
    collection: str
    chunks:     int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    """
    Liveness check — confirms the service is running and reports config.
    Returns the embedding model, collection name, and indexed chunk count.
    """
    return {
        "status":     "ok",
        "model":      EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "chunks":     len(CORPUS),
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """
    Accept a question and return a grounded answer with source citations.

    - Runs full guardrail checks (retrieval strength + topic alignment).
    - Returns `status: refused_*` when evidence is insufficient.
    - Returns `status: answered` with citations when evidence is strong.
    """
    try:
        result = guarded_answer(request.question)
        return {
            "answer":  result["answer"],
            "sources": [
                {
                    "source":   s.get("source"),
                    "chunk_id": s.get("chunk_id"),
                    "score":    s.get("score"),
                    "section":  s.get("section"),
                }
                for s in result.get("sources", [])
            ],
            "status": result.get("status", "answered"),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(status_code=500, detail="RAG service encountered an error.")


# ---------------------------------------------------------------------------
# CLI runner  (python rag_api.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=False)
