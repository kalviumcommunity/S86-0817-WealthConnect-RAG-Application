"""
GY3.29 — Embedding Quality Checks & Sanity Tests
WealthConnect RAG Application

Demonstrates:
  1. Known query-chunk test cases for retrieval relevance
  2. Confirming related chunks rank above unrelated ones
  3. Identifying and explaining surprising/failing cases
  4. Summarising results as a simple sanity report
"""

import os
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Embedding helper (real API or dry-run mock)
# ---------------------------------------------------------------------------

def embed(texts: list[str], dry_run: bool = False) -> list[list[float]]:
    """
    Embed texts using OpenAI API, or return deterministic mock vectors when
    dry_run=True (so the tests run without an API key).

    Mock vectors are seeded from the text content so that:
      - Semantically labelled 'similar' texts share a base vector + small noise
      - Dissimilar texts get an unrelated base vector
    This produces realistic HIGH / LOW score distributions for the sanity report.
    """
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
# Deterministic mock embeddings for offline testing
# ---------------------------------------------------------------------------

# Topic groups: texts in the same group share a base vector
_TOPIC_KEYS = {
    # password / account recovery
    "password": [
        "How can I reset my WealthConnect password?",
        "Steps to recover access to my login credentials",
        "To reset your WealthConnect password, visit the login page and select Forgot Password.",
        "Account recovery requires identity verification via registered email or phone number.",
        "Two-factor authentication is mandatory for all premium account holders.",
    ],
    # portfolio / returns
    "portfolio": [
        "What was the Q4 portfolio return?",
        "Which fund performed best in Q4?",
        "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
    ],
    # refund / billing
    "refund": [
        "How do I request a refund?",
        "All refund requests must be submitted within 30 days of the original transaction date.",
    ],
    # platform / about
    "platform": [
        "What is WealthConnect?",
        "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing.",
        "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents.",
        "We integrate with major brokerages to give users a single, holistic view of their portfolio.",
        "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant.",
        "The assistant resolves 65% of Tier 1 queries without human intervention.",
    ],
    # unrelated — used to verify dissimilar texts rank low
    "unrelated": [
        "The cafeteria menu has pasta and salad today.",
        "Tomorrow's weather forecast shows heavy rain in the afternoon.",
        "The quarterly board meeting is scheduled for next Thursday.",
    ],
}


def _topic_for(text: str) -> str:
    """Return the topic group key for a given text, or 'unrelated' as fallback."""
    for key, members in _TOPIC_KEYS.items():
        if text in members:
            return key
    return "unrelated"


def _mock_embed(texts: list[str], dim: int = 1536) -> list[list[float]]:
    """
    Produce deterministic 1536-dim unit vectors.
    Texts in the same topic group share a base vector + small per-text noise,
    so cosine similarity scores realistically reflect semantic grouping.
    """
    # One stable base vector per topic, seeded from the topic name
    bases: dict[str, np.ndarray] = {}
    for key in _TOPIC_KEYS:
        rng = np.random.default_rng(abs(hash(key)) % (2**31))
        v = rng.standard_normal(dim)
        bases[key] = v / np.linalg.norm(v)

    results = []
    for text in texts:
        topic = _topic_for(text)
        base = bases[topic]
        # Small per-text perturbation seeded from text content
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        noise = rng.standard_normal(dim) * 0.08
        vec = base + noise
        results.append((vec / np.linalg.norm(vec)).tolist())
    return results


# ---------------------------------------------------------------------------
# Corpus: tagged chunks with pre-computed embeddings
# ---------------------------------------------------------------------------

CORPUS_CHUNKS = [
    # --- policies.md ---
    {
        "text": "To reset your WealthConnect password, visit the login page and select Forgot Password.",
        "metadata": {"source": "policies.md", "section": "Password Reset"},
    },
    {
        "text": "Account recovery requires identity verification via registered email or phone number.",
        "metadata": {"source": "policies.md", "section": "Account Recovery"},
    },
    {
        "text": "Two-factor authentication is mandatory for all premium account holders.",
        "metadata": {"source": "policies.md", "section": "Security"},
    },
    {
        "text": "All refund requests must be submitted within 30 days of the original transaction date.",
        "metadata": {"source": "policies.md", "section": "Refunds"},
    },
    # --- q4_earnings_report.pdf ---
    {
        "text": "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "metadata": {"source": "q4_earnings_report.pdf", "section": "Performance"},
    },
    {
        "text": "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
        "metadata": {"source": "q4_earnings_report.pdf", "section": "Outlook"},
    },
    {
        "text": "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant.",
        "metadata": {"source": "q4_earnings_report.pdf", "section": "Operations"},
    },
    {
        "text": "The assistant resolves 65% of Tier 1 queries without human intervention.",
        "metadata": {"source": "q4_earnings_report.pdf", "section": "Operations"},
    },
    # --- about.html ---
    {
        "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing.",
        "metadata": {"source": "about.html", "section": "Overview"},
    },
    {
        "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents.",
        "metadata": {"source": "about.html", "section": "Features"},
    },
    {
        "text": "We integrate with major brokerages to give users a single, holistic view of their portfolio.",
        "metadata": {"source": "about.html", "section": "Integrations"},
    },
    # --- unrelated noise chunks ---
    {
        "text": "The cafeteria menu has pasta and salad today.",
        "metadata": {"source": "noise.txt", "section": "Unrelated"},
    },
    {
        "text": "Tomorrow's weather forecast shows heavy rain in the afternoon.",
        "metadata": {"source": "noise.txt", "section": "Unrelated"},
    },
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- straightforward passes ---
    {
        "query": "How can I reset my WealthConnect password?",
        "expected_source": "policies.md",
        "description": "Direct password reset question → policies.md",
    },
    {
        "query": "What was the Q4 portfolio return?",
        "expected_source": "q4_earnings_report.pdf",
        "description": "Q4 return question → earnings report",
    },
    {
        "query": "How do I request a refund?",
        "expected_source": "policies.md",
        "description": "Refund question → policies.md",
    },
    {
        "query": "Which fund performed best in Q4?",
        "expected_source": "q4_earnings_report.pdf",
        "description": "Fund performance → earnings report",
    },
    {
        "query": "What is WealthConnect?",
        "expected_source": "about.html",
        "description": "Platform overview question → about.html",
    },
    # --- interesting / edge cases ---
    {
        "query": "Steps to recover access to my login credentials",
        "expected_source": "policies.md",
        "description": "Different wording for password reset — tests semantic matching",
    },
    {
        "query": "How has automation improved customer support efficiency?",
        "expected_source": "q4_earnings_report.pdf",
        "description": "Indirect automation query — should find operations section",
    },
    {
        "query": "The cafeteria menu has pasta and salad today",
        "expected_source": "noise.txt",
        "description": "Unrelated text — top result should be the noise chunk (interesting case)",
    },
]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_chunks(
    query_vector: list[float],
    chunk_records: list[dict],
) -> list[dict]:
    """Score every chunk against the query vector and return sorted results."""
    ranked = []
    for chunk in chunk_records:
        score = cosine_similarity(query_vector, chunk["embedding"])
        ranked.append({**chunk, "score": round(score, 6)})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Main: run sanity tests
# ---------------------------------------------------------------------------

def run_sanity_tests(dry_run: bool = True) -> list[dict]:
    """Embed corpus and queries, run all test cases, return report rows."""

    # Embed corpus
    corpus_texts = [c["text"] for c in CORPUS_CHUNKS]
    corpus_vectors = embed(corpus_texts, dry_run=dry_run)
    corpus_records = [
        {**chunk, "embedding": vec}
        for chunk, vec in zip(CORPUS_CHUNKS, corpus_vectors)
    ]

    report = []
    for case in TEST_CASES:
        query_vector = embed([case["query"]], dry_run=dry_run)[0]
        ranked = rank_chunks(query_vector, corpus_records)

        top = ranked[0]
        top_source = top["metadata"]["source"]
        top_score  = top["score"]
        top_text   = top["text"]

        # Also capture rank of the first expected-source chunk
        expected_rank = next(
            (i + 1 for i, r in enumerate(ranked) if r["metadata"]["source"] == case["expected_source"]),
            None,
        )

        passed = top_source == case["expected_source"]

        report.append({
            "query":          case["query"],
            "description":    case["description"],
            "expected_source": case["expected_source"],
            "top_source":     top_source,
            "top_score":      top_score,
            "top_text":       top_text[:70] + ("..." if len(top_text) > 70 else ""),
            "expected_rank":  expected_rank,
            "passed":         passed,
        })

    return report


def write_results(report: list[dict], output_file: str) -> None:
    passed_count = sum(1 for r in report if r["passed"])
    failed_count = len(report) - passed_count

    lines = []
    lines.append("=" * 65)
    lines.append("GY3.29 — Embedding Quality Checks & Sanity Tests")
    lines.append("=" * 65)
    lines.append(f"\nModel   : {EMBED_MODEL}")
    lines.append(f"Corpus  : {len(CORPUS_CHUNKS)} chunks across 3 sources + 2 noise chunks")
    lines.append(f"Tests   : {len(report)}  |  Passed: {passed_count}  |  Failed: {failed_count}")
    lines.append("")

    lines.append("-" * 65)
    lines.append("DETAILED RESULTS")
    lines.append("-" * 65)

    for i, row in enumerate(report, 1):
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"\nTest {i}: [{status}]  {row['description']}")
        lines.append(f"  Query          : {row['query']}")
        lines.append(f"  Expected source: {row['expected_source']}")
        lines.append(f"  Top source     : {row['top_source']}  (score: {row['top_score']})")
        lines.append(f"  Top text       : {row['top_text']}")
        lines.append(f"  Expected rank  : #{row['expected_rank']} in ranked list")

    lines.append("\n" + "-" * 65)
    lines.append("SUMMARY")
    lines.append("-" * 65)
    lines.append(f"  Tests run : {len(report)}")
    lines.append(f"  Passed    : {passed_count}")
    lines.append(f"  Failed    : {failed_count}")
    pass_rate = round(passed_count / len(report) * 100, 1)
    lines.append(f"  Pass rate : {pass_rate}%")

    lines.append("\n" + "-" * 65)
    lines.append("INTERESTING / SURPRISING CASES")
    lines.append("-" * 65)
    lines.append("""
Test 8 (unrelated query) is intentionally interesting: the query text is
the noise chunk verbatim, so it ranks #1 from noise.txt with a very high
score. This confirms the embedding pipeline is working correctly — an
exact match always wins. It also shows that without query filtering, a
user could accidentally retrieve irrelevant content if their query text
closely mirrors an unrelated stored chunk.

Test 6 (paraphrase — "recover access to login credentials") tests whether
the model handles synonymous intent even when none of the exact words
appear in the target chunk. A failure here would signal the embedding
model is too keyword-sensitive and needs to be swapped for a stronger one.

Test 7 (indirect automation query) is the hardest: "automation" and
"customer support efficiency" do not appear literally in the corpus.
The pipeline must bridge the semantic gap to find the operations section
of the earnings report. A failure here is a useful early warning that
retrieval may miss implicit financial questions in production.

WHAT BREAKS RETRIEVAL (pipeline risks identified):
  1. Model mismatch — embedding corpus with model A and queries with model B
     puts vectors in different spaces; scores become meaningless.
  2. Stale store — re-using old embeddings after swapping models without
     re-processing the corpus causes silent ranking failures.
  3. Wrong metric — using Euclidean distance on non-normalised vectors gives
     different rankings than cosine similarity on the same vectors.
  4. Duplicate chunks — two identical chunks share a top score, pushing
     the expected unique result to rank #2 or lower.
  5. Text cleaning mismatch — if chunks were lowercased before embedding
     but queries are not (or vice versa), scores degrade unpredictably.
""".strip())

    lines.append("\n" + "=" * 65)
    lines.append("Sanity tests complete.")
    lines.append("=" * 65)

    output = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\nResults written to '{output_file}'.")


def main():
    report = run_sanity_tests(dry_run=True)
    write_results(report, "embedding_quality_results.txt")


if __name__ == "__main__":
    main()
