"""
GY3.28 — Batch Embedding & Rate/Cost Management
WealthConnect RAG Application

Demonstrates:
  1. Embedding chunks in batches (multiple texts per API request)
  2. Retrying rate-limit and transient failures with exponential backoff
  3. Reporting total embeddings generated and approximate cost
  4. Skipping already-embedded chunks on re-runs (resume-safe pipeline)
"""

import os
import json
import time
import hashlib
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# text-embedding-3-small: $0.020 per 1M tokens = $0.00002 per 1K tokens
PRICE_PER_1K_TOKENS = 0.00002

# Batch size: how many chunks to send in a single API call (max 2048 for OpenAI)
BATCH_SIZE = 64

# Retry settings
MAX_ATTEMPTS = 5

# Persistent store for already-embedded chunk IDs (simulates a vector DB index)
EMBEDDING_STORE_FILE = "embedding_store.json"


# ---------------------------------------------------------------------------
# Client (lazy init)
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


# ---------------------------------------------------------------------------
# 1. Batch generator
# ---------------------------------------------------------------------------

def batches(items: list, size: int):
    """
    Yield successive slices of `items`, each of length `size`.
    The final batch may be smaller.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# 2. Retry with exponential backoff
# ---------------------------------------------------------------------------

def embed_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str = EMBED_MODEL,
    max_attempts: int = MAX_ATTEMPTS,
) -> list[list[float]]:
    """
    Call the OpenAI embeddings endpoint with exponential backoff on transient errors.

    Retries on:
      - RateLimitError  (HTTP 429) — slow down, then retry
      - APIConnectionError         — network hiccup
      - APIStatusError 5xx         — server-side transient error

    Raises immediately on permanent errors (bad request, auth failure, etc.).

    Returns:
        List of float vectors ordered to match `texts`.
    """
    for attempt in range(max_attempts):
        try:
            response = client.embeddings.create(model=model, input=texts)
            # Sort by index to guarantee order matches input list
            return [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]

        except RateLimitError as err:
            if attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt          # 1s, 2s, 4s, 8s, 16s
            print(f"  [rate limit] attempt {attempt + 1}/{max_attempts} — waiting {wait}s | {err}")
            time.sleep(wait)

        except APIConnectionError as err:
            if attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"  [connection error] attempt {attempt + 1}/{max_attempts} — waiting {wait}s | {err}")
            time.sleep(wait)

        except APIStatusError as err:
            # Only retry on server errors (5xx); re-raise client errors (4xx)
            if err.status_code < 500 or attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"  [server error {err.status_code}] attempt {attempt + 1}/{max_attempts} — waiting {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# 3. Token estimator (no API call needed — ~4 chars per token heuristic)
# ---------------------------------------------------------------------------

def estimate_tokens(texts: list[str]) -> int:
    """
    Fast token approximation: ~4 characters per token.
    Good enough for cost estimation; use tiktoken for exact counts.
    """
    return sum(len(t) for t in texts) // 4


# ---------------------------------------------------------------------------
# 4. Persistent embedding store  (skip already-embedded chunks on re-runs)
# ---------------------------------------------------------------------------

def chunk_id(text: str) -> str:
    """Deterministic ID for a chunk — SHA-256 of its text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_store(path: str) -> dict:
    """Load existing embeddings from a JSON file. Returns {} if file absent."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store: dict, path: str) -> None:
    """Persist the embedding store to disk after every batch (resume-safe)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f)


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

# --- Sample corpus: WealthConnect financial document chunks -----------------

ALL_CHUNKS = [
    # Document 1: Q4 Earnings Report
    {"source": "q4_earnings_report.pdf", "text": "WealthConnect Q4 Portfolio Analysis and Earnings Overview."},
    {"source": "q4_earnings_report.pdf", "text": "Our core aggressive growth fund yielded a 12% return, driven by AI and renewable energy investments."},
    {"source": "q4_earnings_report.pdf", "text": "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered support assistant."},
    {"source": "q4_earnings_report.pdf", "text": "The assistant resolves 65% of Tier 1 queries without human intervention, cutting support load significantly."},
    {"source": "q4_earnings_report.pdf", "text": "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds."},
    # Document 2: Refund & Account Policy
    {"source": "policies.md", "text": "All refund requests must be submitted within 30 days of the original transaction date."},
    {"source": "policies.md", "text": "To reset your WealthConnect password, visit the login page and select 'Forgot Password'."},
    {"source": "policies.md", "text": "Account recovery requires identity verification via registered email or phone number."},
    {"source": "policies.md", "text": "Two-factor authentication is mandatory for all premium account holders."},
    # Document 3: About WealthConnect (web)
    {"source": "https://wealthconnect.io/about", "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing."},
    {"source": "https://wealthconnect.io/about", "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your own documents."},
    {"source": "https://wealthconnect.io/about", "text": "We integrate with major brokerages to give users a single, holistic view of their entire portfolio."},
]

# Attach deterministic IDs to every chunk
for chunk in ALL_CHUNKS:
    chunk["id"] = chunk_id(chunk["text"])


def run_pipeline(dry_run: bool = False) -> dict:
    """
    Run the full batch embedding pipeline.

    Args:
        dry_run: If True, skip the actual API call (for offline testing).

    Returns:
        Run summary dict.
    """
    client = None if dry_run else _get_client()

    # --- Load existing embeddings to enable skip-on-rerun ---
    store = load_store(EMBEDDING_STORE_FILE)
    existing_ids = set(store.keys())

    # --- Filter to only chunks that need embedding ---
    pending = [c for c in ALL_CHUNKS if c["id"] not in existing_ids]

    summary = {
        "model": EMBED_MODEL,
        "total_chunks": len(ALL_CHUNKS),
        "skipped_existing": len(ALL_CHUNKS) - len(pending),
        "pending": len(pending),
        "embedded": 0,
        "failed": 0,
        "input_tokens_estimated": 0,
        "batches_processed": 0,
    }

    print(f"\nTotal chunks      : {summary['total_chunks']}")
    print(f"Already embedded  : {summary['skipped_existing']}  (skipping)")
    print(f"Pending           : {summary['pending']}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Model             : {EMBED_MODEL}\n")

    # --- Process batches ---
    for batch_num, batch in enumerate(batches(pending, BATCH_SIZE), start=1):
        texts = [c["text"] for c in batch]
        token_estimate = estimate_tokens(texts)
        summary["input_tokens_estimated"] += token_estimate

        print(f"  Batch {batch_num}: {len(batch)} chunks | ~{token_estimate} tokens")

        try:
            if dry_run:
                # Simulate vectors (1536-dim zeros) for offline demo
                import numpy as np
                vectors = [list(np.random.randn(1536).tolist()) for _ in batch]
            else:
                vectors = embed_with_retry(client, texts)

            # Save each chunk's embedding to the store immediately
            for chunk, vector in zip(batch, vectors):
                store[chunk["id"]] = {
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "embedding": vector[:8],   # store first 8 dims to keep file small
                    "dim": len(vector),
                }
            save_store(store, EMBEDDING_STORE_FILE)   # persist after every batch

            summary["embedded"] += len(batch)
            summary["batches_processed"] += 1

        except Exception as err:
            print(f"  [FAILED] batch {batch_num}: {err}")
            summary["failed"] += len(batch)

    # --- Cost estimate ---
    estimated_cost = summary["input_tokens_estimated"] / 1000 * PRICE_PER_1K_TOKENS

    summary["estimated_cost_usd"] = round(estimated_cost, 8)

    return summary


def write_results(summary: dict, store: dict, output_file: str) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("GY3.28 - Batch Embedding & Rate/Cost Management")
    lines.append("=" * 60)

    lines.append("\n--- RUN SUMMARY ---")
    for key, val in summary.items():
        lines.append(f"  {key:<28}: {val}")

    lines.append(f"\n  Price per 1K tokens (USD) : ${PRICE_PER_1K_TOKENS}")
    lines.append(f"  Estimated total cost (USD): ${summary['estimated_cost_usd']:.8f}")

    lines.append("\n--- EMBEDDING STORE (first 8 dims shown) ---")
    for cid, entry in store.items():
        lines.append(f"\n  ID      : {cid}")
        lines.append(f"  Source  : {entry['source']}")
        lines.append(f"  Text    : {entry['text'][:70]}{'...' if len(entry['text']) > 70 else ''}")
        lines.append(f"  Dim     : {entry['dim']}")
        lines.append(f"  Vec[:8] : {[round(v, 6) for v in entry['embedding']]}")

    lines.append("\n--- RESUME SAFETY DEMO ---")
    lines.append("  Re-running the pipeline now skips all already-embedded chunks.")
    lines.append(f"  {len(store)} chunk(s) are in the store — they will be skipped next run.")
    lines.append("  Only new or previously-failed chunks will be sent to the API.")

    lines.append("\n--- HOW BACKOFF WORKS ---")
    lines.append("  On a RateLimitError or transient server error the pipeline waits:")
    lines.append("    attempt 1 → wait 1s")
    lines.append("    attempt 2 → wait 2s")
    lines.append("    attempt 3 → wait 4s")
    lines.append("    attempt 4 → wait 8s")
    lines.append("    attempt 5 → raise (permanent failure logged in summary)")
    lines.append("  Permanent errors (auth, bad request) raise immediately.")

    lines.append("\n" + "=" * 60)
    lines.append("Batch embedding pipeline complete.")
    lines.append("=" * 60)

    output = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)
    print("\n" + output)
    print(f"\nResults written to '{output_file}'.")


def main():
    # Run with dry_run=True so the demo works without an API key.
    # Set dry_run=False (and provide OPENAI_API_KEY in .env) for a live run.
    summary = run_pipeline(dry_run=True)
    store = load_store(EMBEDDING_STORE_FILE)
    write_results(summary, store, "batch_embedding_results.txt")

    # --- Demonstrate resume: second run should skip all chunks ---
    print("\n--- SIMULATING RE-RUN (all chunks already embedded) ---")
    summary2 = run_pipeline(dry_run=True)
    print(f"  Re-run embedded : {summary2['embedded']}  (expected 0 — all skipped)")
    print(f"  Re-run skipped  : {summary2['skipped_existing']}  (expected {len(ALL_CHUNKS)})")


if __name__ == "__main__":
    main()
