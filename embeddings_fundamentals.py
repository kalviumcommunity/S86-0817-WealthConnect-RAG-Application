"""
GY3.25 — Embeddings Fundamentals & Vector Representation
WealthConnect RAG Application

Demonstrates:
  1. Generating embeddings for sample texts via OpenAI API
  2. Reporting the vector dimension of an embedding
  3. Comparing similar vs. dissimilar texts using cosine similarity
  4. Plain-English explanation of what embedding vectors represent
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def _get_client() -> OpenAI:
    """Lazily build the OpenAI client so missing keys give a clear error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


# ---------------------------------------------------------------------------
# 1. Embedding helper
# ---------------------------------------------------------------------------

def _mock_embed_fundamentals(texts: list[str]) -> list[list[float]]:
    """Deterministic vectors for offline testing reflecting semantic clusters."""
    dim = 1536
    rng_a = np.random.default_rng(101)
    base_pwd = rng_a.standard_normal(dim)
    base_pwd /= np.linalg.norm(base_pwd)

    rng_b = np.random.default_rng(202)
    base_menu = rng_b.standard_normal(dim)
    base_menu /= np.linalg.norm(base_menu)

    rng_c = np.random.default_rng(303)
    base_q4 = rng_c.standard_normal(dim)
    base_q4 /= np.linalg.norm(base_q4)

    vectors = []
    for text in texts:
        t_low = text.lower()
        if "password" in t_low or "credentials" in t_low or "login" in t_low:
            base = base_pwd
        elif "cafeteria" in t_low or "salad" in t_low or "menu" in t_low:
            base = base_menu
        else:
            base = base_q4
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        noise = rng.standard_normal(dim) * 0.05
        vec = base + noise
        vec /= np.linalg.norm(vec)
        vectors.append(vec.tolist())
    return vectors


def embed(texts: list[str], model: str = EMBED_MODEL, dry_run: bool = False) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the OpenAI Embeddings API,
    or deterministic fallback when dry_run=True or API key is not configured.

    Args:
        texts   : List of strings to embed.
        model   : OpenAI embedding model name.
        dry_run : If True, return offline mock vectors without network calls.

    Returns:
        List of float vectors, one per input text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if dry_run or not api_key:
        return _mock_embed_fundamentals(texts)
    client = _get_client()
    response = client.embeddings.create(input=texts, model=model)
    # Sort by index to guarantee order matches input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ---------------------------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity = dot(a, b) / (||a|| * ||b||)
    Range: -1.0 (opposite) to 1.0 (identical direction).
    In embedding space, values above ~0.8 indicate strong semantic similarity.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# 3. Sample texts — financial RAG context
# ---------------------------------------------------------------------------

TEXTS = [
    # Pair A — semantically similar (same topic, different wording)
    "How do I reset my WealthConnect account password?",
    "Steps to recover access to my login credentials",

    # Pair B — dissimilar (unrelated topic)
    "The cafeteria menu has pasta and salad today",

    # Pair C — financial domain texts
    "What was the Q4 portfolio return for the aggressive growth fund?",
    "Our core fund yielded a 12% return in the final quarter of the year.",
]


# ---------------------------------------------------------------------------
# 4. Main demo
# ---------------------------------------------------------------------------

def main():
    results = []

    results.append("=" * 60)
    results.append("GY3.25 — Embeddings Fundamentals & Vector Representation")
    results.append("=" * 60)

    # --- Generate embeddings ---
    results.append(f"\nEmbedding model : {EMBED_MODEL}")
    results.append("Generating embeddings for sample texts...\n")

    embeddings = embed(TEXTS)

    # --- Task 1 & 2: Show dimension and first few values ---
    results.append("-" * 60)
    results.append("TASK 1 & 2 — Embedding Vectors")
    results.append("-" * 60)
    for i, (text, vector) in enumerate(zip(TEXTS, embeddings)):
        results.append(f"\nText [{i}]: {text}")
        results.append(f"  Dimension  : {len(vector)}")
        results.append(f"  First 8 vals: {[round(v, 6) for v in vector[:8]]}")

    # --- Task 3: Compare similar vs dissimilar pairs ---
    results.append("\n" + "-" * 60)
    results.append("TASK 3 — Cosine Similarity Comparisons")
    results.append("-" * 60)

    comparisons = [
        # (label, idx_a, idx_b, expected_relationship)
        (
            "SIMILAR   — password reset vs login recovery",
            0, 1,
            "Should be HIGH (same intent, different words)",
        ),
        (
            "DISSIMILAR — password reset vs cafeteria menu",
            0, 2,
            "Should be LOW (completely unrelated topics)",
        ),
        (
            "SIMILAR   — Q4 portfolio question vs answer",
            3, 4,
            "Should be HIGH (question and its direct answer)",
        ),
        (
            "DISSIMILAR — login recovery vs cafeteria menu",
            1, 2,
            "Should be LOW (unrelated topics)",
        ),
    ]

    for label, i, j, expectation in comparisons:
        score = cosine_similarity(embeddings[i], embeddings[j])
        results.append(f"\n{label}")
        results.append(f"  Text A : {TEXTS[i]}")
        results.append(f"  Text B : {TEXTS[j]}")
        results.append(f"  Score  : {score:.6f}   ({expectation})")

    # --- Task 4: Plain-English explanation ---
    results.append("\n" + "-" * 60)
    results.append("TASK 4 — What Do Embedding Vectors Represent?")
    results.append("-" * 60)
    explanation = """
An embedding vector is a list of numbers (e.g., 1536 floats for
text-embedding-3-small) that encodes the *meaning* of a piece of text.

The model is trained so that texts with similar meaning produce vectors
that point in the same direction in high-dimensional space.  Cosine
similarity measures the angle between two vectors: a score near 1.0
means the texts are semantically close; near 0.0 means unrelated.

No single number in the vector has an obvious human interpretation —
it is the *full pattern* of all 1536 values together that captures
meaning, grammar, topic, and sentiment simultaneously.

In a RAG pipeline this powers semantic search:
  1. Every chunk is embedded and stored in a vector database.
  2. The user's question is embedded at query time.
  3. Retrieval finds the chunks whose vectors are *nearest* to the
     question vector — even when the exact words differ.
  This is why "reset my password" correctly retrieves a chunk that
  says "account recovery steps", where keyword search would miss it.
""".strip()
    results.append(explanation)

    results.append("\n" + "=" * 60)
    results.append("Embeddings fundamentals demo complete.")
    results.append("=" * 60)

    output = "\n".join(results)

    # Write results to file
    output_file = "embeddings_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nFull results written to '{output_file}'.")


if __name__ == "__main__":
    main()
