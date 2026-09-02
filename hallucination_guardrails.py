"""
GY3.41 — Hallucination Guardrails & Refusal Handling
WealthConnect RAG Application

Demonstrates:
  1. Detecting weak or missing retrieval context
  2. Returning a safe refusal when evidence is insufficient
  3. Using a relevance threshold to control refusal
  4. Still producing confident answers when good context exists
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

# Guardrail thresholds — calibrated from retrieval evaluation (GY3.29).
# Mock vectors produce cosine scores in the ~0.02–0.13 range:
#   - on-topic queries score ~0.06–0.13 (above threshold → answered)
#   - off-topic/unrelated queries score ~0.02–0.05 (below threshold → refused)
# In production, calibrate this from your actual embedding model's score distribution.
MIN_TOP_SCORE          = 0.06   # minimum cosine similarity for the top result
MIN_SUPPORTING_CHUNKS  = 1      # at least this many chunks must pass the threshold
RETRIEVAL_K            = 5      # chunks to retrieve before checking

# Refusal message shown to the user when evidence is too weak
REFUSAL_MESSAGE = (
    "I don't have enough reliable context in my knowledge base to answer that question. "
    "Please rephrase your question or contact a WealthConnect advisor directly."
)

# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ---------------------------------------------------------------------------
# Deterministic mock embedder (topic-seeded, same as prior modules)
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
        "What is the capital of France?",
        "Tell me about ancient Roman history.",
        "What is the boiling point of water?",
        "Recommend a good recipe for chocolate cake.",
        "How do black holes form?",
    ],
}


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


def embed(texts: list[str], dry_run: bool = False) -> list[list[float]]:
    if dry_run:
        return _mock_embed(texts)
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Copy .env.example to .env.")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

# ---------------------------------------------------------------------------
# In-memory vector collection
# ---------------------------------------------------------------------------

class VectorCollection:
    def __init__(self):
        self._records: list[dict] = []

    def add(self, text: str, embedding: list[float], metadata: dict) -> None:
        self._records.append({"text": text, "embedding": embedding, "metadata": metadata})

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        scored = [
            {"score": cosine_similarity(vector, r["embedding"]),
             "text": r["text"], "metadata": r["metadata"]}
            for r in self._records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

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


def build_collection(dry_run: bool = False) -> VectorCollection:
    collection = VectorCollection()
    texts   = [c["text"] for c in CORPUS]
    vectors = embed(texts, dry_run=dry_run)
    for chunk, vector in zip(CORPUS, vectors):
        collection.add(
            text=chunk["text"],
            embedding=vector,
            metadata={"source": chunk["source"], "section": chunk["section"],
                      "chunk_index": chunk["chunk_index"]},
        )
    return collection


def retrieve(
    query: str,
    collection: VectorCollection,
    k: int = RETRIEVAL_K,
    dry_run: bool = False,
) -> list[dict]:
    query_vector = embed([query], dry_run=dry_run)[0]
    results = collection.search(vector=query_vector, top_k=k)
    return [{"score": round(r["score"], 6), "text": r["text"],
             "metadata": r["metadata"]} for r in results]


# ---------------------------------------------------------------------------
# Guardrail 1: retrieval strength check
# ---------------------------------------------------------------------------

def retrieval_is_strong(
    chunks: list[dict],
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting: int = MIN_SUPPORTING_CHUNKS,
) -> tuple[bool, dict]:
    """
    Evaluate whether retrieved chunks provide sufficient evidence.

    Checks:
      - At least one chunk was returned
      - At least `min_supporting` chunks score >= `min_top_score`

    Returns:
        (is_strong, diagnostics_dict)
    """
    diagnostics = {
        "total_chunks":      len(chunks),
        "top_score":         chunks[0]["score"] if chunks else 0.0,
        "min_top_score":     min_top_score,
        "strong_chunks":     0,
        "min_supporting":    min_supporting,
    }

    if not chunks:
        diagnostics["reason"] = "no_chunks_retrieved"
        return False, diagnostics

    strong = [c for c in chunks if c["score"] >= min_top_score]
    diagnostics["strong_chunks"] = len(strong)

    if len(strong) < min_supporting:
        diagnostics["reason"] = (
            f"only {len(strong)} chunk(s) above threshold {min_top_score} "
            f"(need {min_supporting})"
        )
        return False, diagnostics

    diagnostics["reason"] = "sufficient_evidence"
    return True, diagnostics


# ---------------------------------------------------------------------------
# Guardrail 2: additional topic-drift check
# ---------------------------------------------------------------------------

def context_is_on_topic(chunks: list[dict], query: str) -> tuple[bool, str]:
    """
    Secondary guardrail: check that at least one top chunk shares vocabulary
    with the query (guards against embedding space collisions where vectors
    from unrelated topics happen to be close).

    Returns (on_topic, reason_string).
    """
    import re

    def tokens(text: str) -> set[str]:
        return set(re.sub(r"[^\w]", " ", text.lower()).split())

    query_tokens = tokens(query) - {"the", "a", "an", "is", "are", "was", "were",
                                    "of", "in", "to", "and", "or", "for", "my", "i",
                                    "how", "what", "when", "where", "why", "do", "did"}

    for chunk in chunks[:3]:                      # check top-3 only
        overlap = query_tokens & tokens(chunk["text"])
        if len(overlap) >= 1:
            return True, f"topic match via: {overlap}"

    return False, "no vocabulary overlap between query and top-3 chunks"


# ---------------------------------------------------------------------------
# Grounded answer builder (mock — returns template answer from top chunk)
# ---------------------------------------------------------------------------

def generate_grounded_answer(question: str, chunks: list[dict]) -> dict:
    """
    Mock answer generation: builds a template response from the top chunk.
    In production replace with: client.chat.completions.create(...)
    using the assembled prompt from context_injection.py.
    """
    top = chunks[0]
    source_markers = ", ".join(
        f"[{i+1}] {c['metadata']['source']}#{c['metadata']['chunk_index']}"
        for i, c in enumerate(chunks[:3])
    )

    answer = (
        f"Based on the available context: {top['text']} "
        f"(Source: {top['metadata']['source']}, "
        f"section: {top['metadata']['section']})"
    )

    return {
        "answer":  answer,
        "sources": [c["metadata"] for c in chunks],
        "source_markers": source_markers,
    }


# ---------------------------------------------------------------------------
# Main guardrail pipeline
# ---------------------------------------------------------------------------

def guarded_answer(
    question: str,
    collection: VectorCollection,
    dry_run: bool = False,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting: int = MIN_SUPPORTING_CHUNKS,
) -> dict:
    """
    Full guarded RAG pipeline:
      1. Retrieve candidates
      2. Check retrieval strength — refuse if too weak
      3. Check topic alignment  — refuse if context is off-topic
      4. Generate grounded answer if both checks pass

    Returns a result dict with keys:
      status        : "answered" | "refused_weak_context" | "refused_off_topic"
      answer        : answer string or refusal message
      sources       : list of metadata dicts (empty on refusal)
      diagnostics   : retrieval quality signals
    """
    # Step 1: retrieve
    chunks = retrieve(question, collection, k=RETRIEVAL_K, dry_run=dry_run)

    # Step 2: guardrail — retrieval strength
    is_strong, diagnostics = retrieval_is_strong(
        chunks, min_top_score=min_top_score, min_supporting=min_supporting
    )

    if not is_strong:
        return {
            "status":      "refused_weak_context",
            "answer":      REFUSAL_MESSAGE,
            "sources":     [],
            "diagnostics": diagnostics,
        }

    # Step 3: guardrail — topic alignment
    on_topic, topic_reason = context_is_on_topic(chunks, question)
    diagnostics["topic_check"] = topic_reason

    if not on_topic:
        return {
            "status":      "refused_off_topic",
            "answer":      REFUSAL_MESSAGE,
            "sources":     [],
            "diagnostics": diagnostics,
        }

    # Step 4: generate
    result = generate_grounded_answer(question, chunks)
    diagnostics["topic_check"] = topic_reason

    return {
        "status":      "answered",
        "answer":      result["answer"],
        "sources":     result["sources"],
        "source_markers": result.get("source_markers", ""),
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_CASES = [
    # --- Should be ANSWERED ---
    {
        "question": "How can I reset my WealthConnect password?",
        "expected": "answered",
        "note":     "Strong retrieval — policies.md has explicit password reset content.",
    },
    {
        "question": "What was the Q4 portfolio return?",
        "expected": "answered",
        "note":     "Strong retrieval — earnings report has Q4 performance data.",
    },
    {
        "question": "What is the refund policy?",
        "expected": "answered",
        "note":     "Strong retrieval — refund policy clearly in corpus.",
    },
    # --- Should be REFUSED ---
    {
        "question": "What is the capital of France?",
        "expected": "refused",
        "note":     "Off-corpus geography question — no supporting evidence.",
    },
    {
        "question": "How do black holes form?",
        "expected": "refused",
        "note":     "Completely unrelated — should be refused to prevent hallucination.",
    },
    {
        "question": "Recommend a good recipe for chocolate cake.",
        "expected": "refused",
        "note":     "Out-of-domain request — should be refused.",
    },
]


def main():
    lines = []

    lines.append("=" * 65)
    lines.append("GY3.41 — Hallucination Guardrails & Refusal Handling")
    lines.append("=" * 65)
    lines.append(f"\nRelevance threshold   : {MIN_TOP_SCORE}")
    lines.append(f"Min supporting chunks : {MIN_SUPPORTING_CHUNKS}")
    lines.append(f"Retrieval k           : {RETRIEVAL_K}")
    lines.append(f"Corpus size           : {len(CORPUS)} chunks\n")

    collection = build_collection(dry_run=True)

    passed = 0
    total  = len(DEMO_CASES)

    lines.append("=" * 65)
    lines.append("TEST CASES")
    lines.append("=" * 65)

    for i, case in enumerate(DEMO_CASES, start=1):
        result   = guarded_answer(case["question"], collection, dry_run=True)
        status   = result["status"]
        expected = case["expected"]
        diag     = result["diagnostics"]

        # Pass = answered when expected answered, refused when expected refused
        ok = (expected == "answered" and status == "answered") or \
             (expected == "refused"  and status.startswith("refused"))
        if ok:
            passed += 1

        flag = "PASS" if ok else "FAIL"

        lines.append(f"\nTest {i}: [{flag}]  {case['note']}")
        lines.append(f"  Question        : {case['question']}")
        lines.append(f"  Expected        : {expected}")
        lines.append(f"  Status          : {status}")
        lines.append(f"  Top score       : {diag['top_score']}  (threshold: {diag['min_top_score']})")
        lines.append(f"  Strong chunks   : {diag['strong_chunks']} / {diag['total_chunks']}")
        lines.append(f"  Topic check     : {diag.get('topic_check', 'n/a')}")
        lines.append(f"  Reason          : {diag.get('reason', 'n/a')}")

        if status == "answered":
            lines.append(f"  Answer (preview): {result['answer'][:110]}...")
            lines.append(f"  Source markers  : {result.get('source_markers', '')}")
        else:
            lines.append(f"  Refusal message : {result['answer']}")

    lines.append("\n" + "=" * 65)
    lines.append("SUMMARY")
    lines.append("=" * 65)
    pass_rate = round(passed / total * 100, 1)
    lines.append(f"  Tests run : {total}")
    lines.append(f"  Passed    : {passed}")
    lines.append(f"  Failed    : {total - passed}")
    lines.append(f"  Pass rate : {pass_rate}%")

    # -----------------------------------------------------------------------
    # Threshold sensitivity demo
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("THRESHOLD SENSITIVITY DEMO")
    lines.append("=" * 65)
    lines.append(
        "\nSame question run at three threshold levels to show the "
        "refusing-too-often vs answering-too-freely trade-off:\n"
    )

    demo_q = "What was the Q4 portfolio return?"
    for thresh in [0.02, 0.06, 0.99]:
        r = guarded_answer(demo_q, collection, dry_run=True, min_top_score=thresh)
        lines.append(
            f"  threshold={thresh:.2f}  →  status={r['status']:<25}  "
            f"top_score={r['diagnostics']['top_score']}"
        )

    lines.append("""
  LOW threshold (0.02)  : Almost never refuses — risks hallucination on weak context.
  MID threshold (0.06)  : Balanced — refuses when evidence is genuinely absent.
  HIGH threshold (0.99) : Refuses almost everything — too conservative for production.

  Tune the threshold using your retrieval evaluation results (GY3.29),
  not guesswork. In high-stakes domains (finance, health, legal), err toward
  refusing: a confident wrong answer causes more harm than an honest "I don't know."
""".strip())

    lines.append("\n" + "=" * 65)
    lines.append("Hallucination guardrails demo complete.")
    lines.append("=" * 65)

    output = "\n".join(lines)

    output_file = "guardrails_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nResults written to '{output_file}'.")


if __name__ == "__main__":
    main()
