"""
GY3.32 — Similarity Search & Top-K Retrieval
WealthConnect RAG Application

Demonstrates:
  1. Embedding a user query with the same model used for documents
  2. Running top-k similarity search against an in-memory vector collection
  3. Returning chunks with similarity scores and metadata
  4. Showing how changing k changes the retrieved context
"""

import os
import math
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Embedding (real API or deterministic mock)
# ---------------------------------------------------------------------------

# Topic groups used by the mock embedder so semantically related texts
# share a base vector and score HIGH against each other.
_TOPIC_BASES: dict[str, np.ndarray] = {}

_TOPIC_MAP = {
    "password": [
        "How can I reset my WealthConnect password?",
        "Steps to recover access to my login credentials",
        "How do I recover my account?",
        "To reset your WealthConnect password, visit the login page and select Forgot Password.",
        "Account recovery requires identity verification via registered email or phone number.",
        "Two-factor authentication is mandatory for all premium account holders.",
    ],
    "portfolio": [
        "What was the Q4 portfolio return?",
        "Which fund performed best last quarter?",
        "Show me investment performance data",
        "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
        "WealthConnect Q4 Portfolio Analysis and Earnings Overview.",
    ],
    "refund": [
        "How do I request a refund?",
        "What is the refund policy?",
        "All refund requests must be submitted within 30 days of the original transaction date.",
    ],
    "platform": [
        "What is WealthConnect?",
        "Tell me about the platform features",
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

DIM = 1536


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
    """Deterministic 1536-dim unit vectors seeded from topic + text content."""
    result = []
    for text in texts:
        topic = _get_topic(text)
        base = _get_base(topic)
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 31))
        noise = rng.standard_normal(DIM) * 0.08
        vec = base + noise
        result.append((vec / np.linalg.norm(vec)).tolist())
    return result


def embed(texts: list[str], dry_run: bool = False) -> list[list[float]]:
    """Embed texts via OpenAI API, or return mock vectors when dry_run=True."""
    if dry_run:
        return _mock_embed(texts)

    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ---------------------------------------------------------------------------
# In-memory vector collection (simulates a vector DB)
# ---------------------------------------------------------------------------

class VectorCollection:
    """
    Lightweight in-memory vector store.
    Mimics the query interface of Chroma / Pinecone / Qdrant so the
    retrieve() function below works identically against a real DB.
    """

    def __init__(self):
        self._records: list[dict] = []

    def add(self, text: str, embedding: list[float], metadata: dict) -> None:
        self._records.append({
            "text":      text,
            "embedding": embedding,
            "metadata":  metadata,
        })

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        """Return top_k records sorted by cosine similarity (highest first)."""
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
# Corpus: WealthConnect document chunks
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

    # about.html
    {"source": "about.html", "section": "Overview",      "chunk_index": 0,
     "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing."},
    {"source": "about.html", "section": "Features",      "chunk_index": 1,
     "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents."},
    {"source": "about.html", "section": "Integrations",  "chunk_index": 2,
     "text": "We integrate with major brokerages to give users a single, holistic view of their portfolio."},

    # noise
    {"source": "noise.txt", "section": "Unrelated", "chunk_index": 0,
     "text": "The cafeteria menu has pasta and salad today."},
    {"source": "noise.txt", "section": "Unrelated", "chunk_index": 1,
     "text": "Tomorrow's weather forecast shows heavy rain."},
]


def build_collection(dry_run: bool = False) -> VectorCollection:
    """Embed all corpus chunks and load them into the vector collection."""
    collection = VectorCollection()
    texts = [c["text"] for c in CORPUS]
    vectors = embed(texts, dry_run=dry_run)
    for chunk, vector in zip(CORPUS, vectors):
        metadata = {
            "source":      chunk["source"],
            "section":     chunk["section"],
            "chunk_index": chunk["chunk_index"],
        }
        collection.add(text=chunk["text"], embedding=vector, metadata=metadata)
    return collection


# ---------------------------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: VectorCollection,
    k: int = 3,
    dry_run: bool = False,
) -> list[dict]:
    """
    Embed the query with the same model used for documents, run top-k
    similarity search, and return ranked chunks with scores and metadata.

    Args:
        query      : User's natural-language question.
        collection : Pre-built vector collection.
        k          : Number of top results to return.
        dry_run    : Use mock embeddings (no API key required).

    Returns:
        List of dicts: score, text, metadata — sorted highest score first.
    """
    # IMPORTANT: must use the same model as the corpus embeddings
    query_vector = embed([query], dry_run=dry_run)[0]

    results = collection.search(vector=query_vector, top_k=k)

    return [
        {
            "score":    round(item["score"], 6),
            "text":     item["text"],
            "metadata": item["metadata"],
        }
        for item in results
    ]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "How can I reset my WealthConnect password?",
    "What was the Q4 portfolio return?",
    "How do I request a refund?",
    "Tell me about the WealthConnect platform",
]

K_VALUES = [1, 3, 5]


def main():
    lines = []

    lines.append("=" * 65)
    lines.append("GY3.32 — Similarity Search & Top-K Retrieval")
    lines.append("=" * 65)
    lines.append(f"\nEmbedding model : {EMBED_MODEL}")
    lines.append(f"Collection size : {len(CORPUS)} chunks\n")

    # Build the collection once
    collection = build_collection(dry_run=True)
    lines.append(f"Collection built: {len(collection)} chunks indexed.\n")

    # -----------------------------------------------------------------------
    # Section 1: Full ranked retrieval for each demo query at k=3
    # -----------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("SECTION 1 — Top-K Retrieval (k=3) with Scores & Metadata")
    lines.append("=" * 65)

    for query in DEMO_QUERIES:
        lines.append(f"\nQuery: \"{query}\"")
        lines.append("-" * 55)
        results = retrieve(query, collection, k=3, dry_run=True)
        for rank, result in enumerate(results, start=1):
            lines.append(f"  Rank {rank}")
            lines.append(f"    score        : {result['score']}")
            lines.append(f"    source       : {result['metadata']['source']}")
            lines.append(f"    section      : {result['metadata']['section']}")
            lines.append(f"    chunk_index  : {result['metadata']['chunk_index']}")
            lines.append(f"    text         : {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")

    # -----------------------------------------------------------------------
    # Section 2: How changing k changes retrieved context
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("SECTION 2 — Effect of Changing k on Retrieved Context")
    lines.append("=" * 65)

    focus_query = "How can I reset my WealthConnect password?"
    lines.append(f"\nQuery: \"{focus_query}\"\n")

    for k in K_VALUES:
        results = retrieve(focus_query, collection, k=k, dry_run=True)
        sources = [r["metadata"]["source"] for r in results]
        scores  = [r["score"] for r in results]
        lines.append(f"  k={k}:")
        lines.append(f"    chunks returned : {len(results)}")
        lines.append(f"    score range     : {min(scores):.6f} – {max(scores):.6f}")
        lines.append(f"    sources         : {sources}")
        # Assess quality
        unique_sources = set(sources)
        if len(unique_sources) == 1 and "policies.md" in unique_sources:
            note = "All results from the expected source — focused."
        elif "policies.md" in unique_sources:
            note = "Expected source present but extra sources included — broader context."
        else:
            note = "Expected source missing — potential retrieval noise."
        lines.append(f"    assessment      : {note}")

    # -----------------------------------------------------------------------
    # Section 3: Trade-off analysis
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("SECTION 3 — Top-K Trade-Off Analysis")
    lines.append("=" * 65)
    tradeoff = """
  k=1  → Fastest, lowest cost, but risks missing complementary context.
          Best when one chunk fully answers the query (e.g., exact fact lookup).

  k=3  → Sweet spot for most RAG queries. Enough context for multi-part
          answers; unlikely to overflow a typical LLM context window.

  k=5  → Higher recall: useful when the answer spans multiple sections or
          documents. Risk: loosely related chunks can distract the LLM and
          increase prompt token cost.

  Rule of thumb: start at k=3, tune upward if answers are incomplete,
  tune downward if the LLM returns contradictory or off-topic content.

  CRITICAL: query and corpus must use the same embedding model.
  Mixing models (e.g. text-embedding-3-small for corpus, ada-002 for query)
  puts vectors in different spaces — scores are numbers but rankings are wrong.
""".strip()
    lines.append(tradeoff)

    lines.append("\n" + "=" * 65)
    lines.append("Similarity search & top-k retrieval demo complete.")
    lines.append("=" * 65)

    output = "\n".join(lines)

    output_file = "similarity_search_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nResults written to '{output_file}'.")


if __name__ == "__main__":
    main()
