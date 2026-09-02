"""
GY3.47 — Streaming Responses & Citation Display
WealthConnect RAG Application

Backend: FastAPI streaming endpoint that emits Server-Sent Events (SSE).
Event types:
  { "type": "citations", "sources": [...] }   — sent before tokens
  { "type": "token",     "text": "..." }       — one per generated token
  { "type": "done" }                           — stream complete
  { "type": "error",     "message": "..." }    — interruption / failure

Run:
    uvicorn streaming_api:app --reload --port 8000

Frontend: open frontend/index.html in a browser while the server is running.
"""

import os
import re
import json
import asyncio
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Config (all values from environment)
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

# ---------------------------------------------------------------------------
# Mock embedder & vector store (same as prior modules)
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
        "What is the capital of France?",
        "How do black holes form?",
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


def _get_topic(text: str) -> str:
    for topic, members in _TOPIC_MAP.items():
        if text in members:
            return topic
    return "unrelated"


def _get_base(topic: str) -> np.ndarray:
    if topic not in _TOPIC_BASES:
        rng = np.random.default_rng(abs(hash(topic)) % (2 ** 31))
        v = rng.standard_normal(DIM)
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


def _embed(texts: list[str]) -> list[list[float]]:
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
    return _mock_embed(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


class _VectorCollection:
    def __init__(self):
        self._records: list[dict] = []

    def add(self, text, embedding, metadata):
        self._records.append({"text": text, "embedding": embedding, "metadata": metadata})

    def search(self, vector, top_k):
        scored = [{"score": _cosine(vector, r["embedding"]),
                   "text": r["text"], "metadata": r["metadata"]}
                  for r in self._records]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def _build_collection() -> _VectorCollection:
    col = _VectorCollection()
    vectors = _embed([c["text"] for c in CORPUS])
    for chunk, vec in zip(CORPUS, vectors):
        col.add(chunk["text"], vec,
                {"source": chunk["source"], "section": chunk["section"],
                 "chunk_index": chunk["chunk_index"]})
    return col


def _retrieve(query: str) -> list[dict]:
    qv = _embed([query])[0]
    return [{"score": round(r["score"], 6), "text": r["text"], "metadata": r["metadata"]}
            for r in _collection.search(qv, RETRIEVAL_K)]


def _strong(chunks: list[dict]) -> tuple[bool, str]:
    if not chunks:
        return False, "no_chunks_retrieved"
    strong = [c for c in chunks if c["score"] >= MIN_TOP_SCORE]
    if len(strong) < MIN_SUPPORTING_CHUNKS:
        return False, f"only {len(strong)} chunk(s) above threshold"
    return True, "ok"


def _on_topic(chunks: list[dict], query: str) -> bool:
    STOP = {"the","a","an","is","are","was","of","in","to","and","or","for",
            "my","i","how","what","when","where","why","do","did"}
    def tok(t): return set(re.sub(r"[^\w]"," ",t.lower()).split()) - STOP
    qt = tok(query)
    return any(qt & tok(c["text"]) for c in chunks[:3])


# ---------------------------------------------------------------------------
# Streaming RAG pipeline  (async generator)
# ---------------------------------------------------------------------------

async def rag_pipeline_stream(question: str):
    """
    Async generator that yields SSE event dicts in order:
      1. citations  — retrieved source metadata
      2. token ...  — answer text tokens
      3. done       — stream complete
    Yields an error event if generation fails mid-stream.
    """
    # --- Retrieve & guardrail checks ---
    chunks = _retrieve(question)
    ok, reason = _strong(chunks)

    if not ok or not _on_topic(chunks, question):
        # Emit refusal as a single token + done (keeps UI consistent)
        yield {"type": "citations", "sources": []}
        yield {"type": "token", "text": REFUSAL_MESSAGE}
        yield {"type": "done"}
        return

    # --- Build source list for citation event ---
    sources = [
        {
            "id":       f"source-{i+1}",
            "label":    f"[{i+1}]",
            "document": c["metadata"]["source"],
            "chunk_id": f"{c['metadata']['source']}:{c['metadata']['chunk_index']}",
            "section":  c["metadata"]["section"],
            "text":     c["text"],
            "score":    c["score"],
        }
        for i, c in enumerate(chunks[:5])
    ]

    # Emit citations BEFORE tokens so the UI can show them immediately
    yield {"type": "citations", "sources": sources}

    # --- Generate answer ---
    context_parts = [
        f"[{i+1}] {s['document']}#{s['chunk_id'].split(':')[-1]} ({s['section']})\n{s['text']}"
        for i, s in enumerate(sources)
    ]
    context = "\n\n---\n\n".join(context_parts)

    system_msg = (
        "You are a grounded financial assistant for WealthConnect. "
        "Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say: "
        "\"I don't have enough information in the provided context.\" "
        "Cite sources using markers like [1] or [2]."
    )

    if os.getenv("OPENAI_API_KEY"):
        # Real streaming via OpenAI SDK
        from openai import OpenAI, APIError
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"},
                ],
                temperature=0,
                max_tokens=512,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "text": delta.content}
                    await asyncio.sleep(0)   # yield control to event loop

        except Exception as exc:
            yield {"type": "error",
                   "message": "The answer stopped streaming. Please retry."}
            return
    else:
        # Mock streaming: split a pre-built answer into word-by-word tokens
        top = chunks[0]
        mock_answer = (
            f"Based on the provided context, {top['text']} "
            f"[1] For more details, refer to {top['metadata']['source']} "
            f"({top['metadata']['section']})."
        )
        for word in mock_answer.split(" "):
            yield {"type": "token", "text": word + " "}
            await asyncio.sleep(0.04)   # simulate token latency

    yield {"type": "done"}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WealthConnect Streaming RAG API",
    description="Streaming SSE endpoint for the WealthConnect RAG assistant.",
    version="1.0.0",
)

# Allow the frontend (any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Build collection at startup
_collection: _VectorCollection = _build_collection()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000,
                          description="Question to answer from the knowledge base.",
                          examples=["What was the Q4 portfolio return?"])


class HealthResponse(BaseModel):
    status:     str
    model:      str
    collection: str
    chunks:     int
    streaming:  bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    """Liveness check — reports model, collection, and streaming capability."""
    return {
        "status":     "ok",
        "model":      EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "chunks":     len(CORPUS),
        "streaming":  True,
    }


@app.post("/query/stream", tags=["RAG"])
async def stream_query(request: QueryRequest):
    """
    Stream a grounded RAG answer as Server-Sent Events.

    Event sequence:
      data: {"type": "citations", "sources": [...]}
      data: {"type": "token",     "text": "..."}   (repeated)
      data: {"type": "done"}

    On error mid-stream:
      data: {"type": "error", "message": "..."}
    """
    async def events():
        try:
            async for event in rag_pipeline_stream(request.question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            error_event = {"type": "error",
                           "message": "The answer stopped streaming. Please retry."}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("streaming_api:app", host="0.0.0.0", port=8000, reload=False)
