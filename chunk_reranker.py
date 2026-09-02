"""
GY3.35 — Chunk Re-Ranking for Precision
WealthConnect RAG Application

Demonstrates:
  1. Retrieving a candidate set larger than the final k (retrieve-then-rerank)
  2. Re-ranking candidates by relevance to the query
  3. Comparing before-and-after ordering for sample queries
  4. Explaining the cost and latency trade-off of re-ranking
"""

import os
import math
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL     = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
CANDIDATE_K    = 10   # how many chunks to retrieve before re-ranking
FINAL_K        = 3    # how many chunks to keep after re-ranking

# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ---------------------------------------------------------------------------
# Mock embedder — deterministic topic-seeded 1536-dim unit vectors
# ---------------------------------------------------------------------------

_TOPIC_BASES: dict[str, np.ndarray] = {}
DIM = 1536

_TOPIC_MAP = {
    "password": [
        "How can I reset my WealthConnect password?",
        "To reset your WealthConnect password, visit the login page and select Forgot Password.",
        "Account recovery requires identity verification via registered email or phone number.",
        "Two-factor authentication is mandatory for all premium account holders.",
        "Steps to recover access to my login credentials",
    ],
    "portfolio": [
        "What was the Q4 portfolio return?",
        "How did the growth fund perform last quarter?",
        "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
        "WealthConnect Q4 Portfolio Analysis and Earnings Overview.",
        "The tech sector allocation increased 8 percentage points in Q4.",
        "Our bond portfolio underperformed relative to benchmark by 1.2% in Q3.",
    ],
    "refund": [
        "How do I request a refund?",
        "What is the refund policy?",
        "All refund requests must be submitted within 30 days of the original transaction date.",
        "Refunds are processed within 5–7 business days after approval.",
        "Fee waivers are available for accounts inactive for over 90 days.",
    ],
    "platform": [
        "What is WealthConnect?",
        "Tell me about the WealthConnect platform",
        "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing.",
        "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents.",
        "We integrate with major brokerages to give users a single, holistic view of their portfolio.",
        "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant.",
        "The assistant resolves 65% of Tier 1 queries without human intervention.",
    ],
    "unrelated": [
        "The cafeteria menu has pasta and salad today.",
        "Tomorrow's weather forecast shows heavy rain.",
        "The quarterly board meeting is scheduled for next Thursday.",
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
            {
                "score":    cosine_similarity(vector, rec["embedding"]),
                "text":     rec["text"],
                "metadata": rec["metadata"],
            }
            for rec in self._records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def __len__(self):
        return len(self._records)

# ---------------------------------------------------------------------------
# Corpus: WealthConnect document chunks (14 total — enough for k=10 retrieval)
# ---------------------------------------------------------------------------

CORPUS = [
    # policies.md
    {"source": "policies.md", "section": "Password Reset",   "chunk_index": 0,
     "text": "To reset your WealthConnect password, visit the login page and select Forgot Password."},
    {"source": "policies.md", "section": "Account Recovery", "chunk_index": 1,
     "text": "Account recovery requires identity verification via registered email or phone number."},
    {"source": "policies.md", "section": "Security",         "chunk_index": 2,
     "text": "Two-factor authentication is mandatory for all premium account holders."},
    {"source": "policies.md", "section": "Refunds",          "chunk_index": 3,
     "text": "All refund requests must be submitted within 30 days of the original transaction date."},
    {"source": "policies.md", "section": "Refunds",          "chunk_index": 4,
     "text": "Refunds are processed within 5–7 business days after approval."},
    {"source": "policies.md", "section": "Fees",             "chunk_index": 5,
     "text": "Fee waivers are available for accounts inactive for over 90 days."},

    # q4_earnings_report.pdf
    {"source": "q4_earnings_report.pdf", "section": "Overview",    "chunk_index": 0,
     "text": "WealthConnect Q4 Portfolio Analysis and Earnings Overview."},
    {"source": "q4_earnings_report.pdf", "section": "Performance", "chunk_index": 1,
     "text": "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy."},
    {"source": "q4_earnings_report.pdf", "section": "Outlook",     "chunk_index": 2,
     "text": "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds."},
    {"source": "q4_earnings_report.pdf", "section": "Operations",  "chunk_index": 3,
     "text": "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant."},
    {"source": "q4_earnings_report.pdf", "section": "Operations",  "chunk_index": 4,
     "text": "The assistant resolves 65% of Tier 1 queries without human intervention."},
    {"source": "q4_earnings_report.pdf", "section": "Allocation",  "chunk_index": 5,
     "text": "The tech sector allocation increased 8 percentage points in Q4."},
    {"source": "q4_earnings_report.pdf", "section": "Bonds",       "chunk_index": 6,
     "text": "Our bond portfolio underperformed relative to benchmark by 1.2% in Q3."},

    # about.html
    {"source": "about.html", "section": "Overview",     "chunk_index": 0,
     "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing."},
    {"source": "about.html", "section": "Features",     "chunk_index": 1,
     "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents."},
    {"source": "about.html", "section": "Integrations", "chunk_index": 2,
     "text": "We integrate with major brokerages to give users a single, holistic view of their portfolio."},

    # noise
    {"source": "noise.txt", "section": "Unrelated", "chunk_index": 0,
     "text": "The cafeteria menu has pasta and salad today."},
    {"source": "noise.txt", "section": "Unrelated", "chunk_index": 1,
     "text": "Tomorrow's weather forecast shows heavy rain."},
    {"source": "noise.txt", "section": "Unrelated", "chunk_index": 2,
     "text": "The quarterly board meeting is scheduled for next Thursday."},
]


def build_collection(dry_run: bool = False) -> VectorCollection:
    collection = VectorCollection()
    texts   = [c["text"] for c in CORPUS]
    vectors = embed(texts, dry_run=dry_run)
    for chunk, vector in zip(CORPUS, vectors):
        collection.add(
            text=chunk["text"],
            embedding=vector,
            metadata={
                "source":      chunk["source"],
                "section":     chunk["section"],
                "chunk_index": chunk["chunk_index"],
            },
        )
    return collection

# ---------------------------------------------------------------------------
# Step 1: Retrieval (vector similarity — fast, broad)
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: VectorCollection,
    k: int = CANDIDATE_K,
    dry_run: bool = False,
) -> list[dict]:
    """Embed query, run top-k cosine similarity search, return candidates."""
    query_vector = embed([query], dry_run=dry_run)[0]
    results = collection.search(vector=query_vector, top_k=k)
    return [
        {
            "vector_score": round(item["score"], 6),
            "text":         item["text"],
            "metadata":     item["metadata"],
        }
        for item in results
    ]

# ---------------------------------------------------------------------------
# Step 2: Re-ranking (token-overlap scoring — precise, inspects each candidate)
# ---------------------------------------------------------------------------

def token_overlap_rerank_score(query: str, chunk_text: str) -> float:
    """
    Lightweight re-ranking score based on weighted token overlap.

    Computes: |query_tokens ∩ chunk_tokens| / |query_tokens|

    This simulates what a cross-encoder or LLM re-ranker does conceptually —
    it looks at the query and chunk *together* to judge direct relevance,
    rather than comparing two independent vectors.

    For a production pipeline, replace this with:
      - Cohere Rerank API  (cohere.rerank)
      - OpenAI cross-encoder prompt
      - sentence-transformers cross-encoder model
    """
    def tokenise(text: str) -> set[str]:
        # Lowercase, strip punctuation, split on whitespace
        import re
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    query_tokens = tokenise(query)
    chunk_tokens = tokenise(chunk_text)

    if not query_tokens:
        return 0.0

    # Jaccard-style overlap weighted toward query coverage
    intersection = query_tokens & chunk_tokens
    coverage     = len(intersection) / len(query_tokens)       # recall of query terms
    precision    = len(intersection) / len(chunk_tokens) if chunk_tokens else 0.0
    # F1-style harmonic mean
    if coverage + precision == 0:
        return 0.0
    return round(2 * coverage * precision / (coverage + precision), 6)


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Score each candidate against the query and sort by rerank_score (desc).
    Attaches rerank_score to each candidate dict without mutating the originals.
    """
    scored = []
    for chunk in candidates:
        r_score = token_overlap_rerank_score(query, chunk["text"])
        scored.append({**chunk, "rerank_score": r_score})
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

# ---------------------------------------------------------------------------
# Comparison display helper
# ---------------------------------------------------------------------------

def format_ranked_list(label: str, rows: list[dict], final_k: int) -> list[str]:
    lines = []
    lines.append(f"\n  {label}")
    lines.append("  " + "-" * 58)
    for rank, item in enumerate(rows[:final_k], start=1):
        v_score = item.get("vector_score", "n/a")
        r_score = item.get("rerank_score", "n/a")
        lines.append(f"  Rank {rank}")
        lines.append(f"    vector_score : {v_score}")
        lines.append(f"    rerank_score : {r_score}")
        lines.append(f"    source       : {item['metadata']['source']}")
        lines.append(f"    section      : {item['metadata']['section']}")
        lines.append(f"    text         : {item['text'][:85]}{'...' if len(item['text']) > 85 else ''}")
    return lines

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {
        "query": "How can I reset my WealthConnect password?",
        "note":  "Direct intent match — re-ranking should surface the explicit reset chunk above general security ones.",
    },
    {
        "query": "What was the Q4 portfolio return?",
        "note":  "Financial data query — re-ranking should move the performance chunk (12% return) above the broader overview.",
    },
    {
        "query": "How do I get a refund?",
        "note":  "Refund query — re-ranking should prioritise the refund policy chunk over fee-waiver tangents.",
    },
]


def main():
    lines = []

    lines.append("=" * 65)
    lines.append("GY3.35 — Chunk Re-Ranking for Precision")
    lines.append("=" * 65)
    lines.append(f"\nEmbedding model  : {EMBED_MODEL}")
    lines.append(f"Candidate k      : {CANDIDATE_K}  (vector retrieval)")
    lines.append(f"Final k          : {FINAL_K}   (after re-ranking)")
    lines.append(f"Corpus size      : {len(CORPUS)} chunks\n")

    collection = build_collection(dry_run=True)
    lines.append(f"Collection built : {len(collection)} chunks indexed.\n")

    for demo in DEMO_QUERIES:
        query = demo["query"]
        note  = demo["note"]

        lines.append("=" * 65)
        lines.append(f'Query  : "{query}"')
        lines.append(f"Note   : {note}")
        lines.append("=" * 65)

        # Step 1: Vector retrieval — retrieve CANDIDATE_K candidates
        candidates = retrieve(query, collection, k=CANDIDATE_K, dry_run=True)

        # Step 2: Re-rank the candidates
        reranked = rerank(query, candidates)

        # Show before vs after for top FINAL_K
        lines += format_ranked_list(
            f"BEFORE re-ranking  (top {FINAL_K} of {CANDIDATE_K} by vector score)",
            candidates,
            FINAL_K,
        )
        lines += format_ranked_list(
            f"AFTER  re-ranking  (top {FINAL_K} of {CANDIDATE_K} by rerank score)",
            reranked,
            FINAL_K,
        )

        # Highlight any rank changes
        before_texts = [c["text"] for c in candidates[:FINAL_K]]
        after_texts  = [c["text"] for c in reranked[:FINAL_K]]
        moved_in     = [t for t in after_texts if t not in before_texts]
        moved_out    = [t for t in before_texts if t not in after_texts]

        lines.append(f"\n  Rank changes after re-ranking:")
        if moved_in:
            for t in moved_in:
                lines.append(f"    MOVED IN  : {t[:80]}{'...' if len(t) > 80 else ''}")
        if moved_out:
            for t in moved_out:
                lines.append(f"    MOVED OUT : {t[:80]}{'...' if len(t) > 80 else ''}")
        if not moved_in and not moved_out:
            lines.append(f"    No changes — vector order already optimal for this query.")

    # -----------------------------------------------------------------------
    # Trade-off section
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("COST & LATENCY TRADE-OFF")
    lines.append("=" * 65)
    tradeoff = """
  RETRIEVAL (vector search)
    Latency : Sub-millisecond — one ANN (approximate nearest-neighbour)
              lookup against the full index.
    Cost    : Essentially free after indexing; no extra API calls.
    Quality : Good broad recall; may rank loosely related chunks above
              directly relevant ones when topics overlap.

  RE-RANKING (cross-encoder / LLM scoring)
    Latency : Adds N extra scoring calls where N = CANDIDATE_K.
              With LLM scoring each call may take 200–500 ms.
              With a cross-encoder model it is much faster (~10 ms/chunk).
    Cost    : Each scoring call consumes tokens / compute.
              10 candidates × average 50 tokens ≈ 500 extra input tokens.
    Quality : Significantly improves precision for ambiguous queries
              because the scorer sees query + chunk together, not as
              independent vectors.

  WHEN TO USE RE-RANKING
    ✓  Precision matters more than raw speed (e.g. financial advice)
    ✓  Initial retrieval returns mixed-quality results
    ✓  Queries are long or ambiguous
    ✗  Real-time applications where every millisecond counts
    ✗  Corpus is small and vector retrieval already gives clean results

  RECOMMENDED APPROACH
    1. Retrieve a candidate set 3–5× larger than final_k (e.g. k=10 → final 3)
    2. Re-rank with a cross-encoder (faster) or LLM prompt (more flexible)
    3. Send only final_k chunks to the LLM — keeps prompts focused and cheap
    4. Monitor: log both vector_score and rerank_score to detect drift
""".strip()
    lines.append(tradeoff)

    lines.append("\n" + "=" * 65)
    lines.append("Re-ranking demo complete.")
    lines.append("=" * 65)

    output = "\n".join(lines)

    output_file = "reranking_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nResults written to '{output_file}'.")


if __name__ == "__main__":
    main()
