"""
GY3.38 — Context Injection & Prompt Augmentation
WealthConnect RAG Application

Demonstrates:
  1. Formatting retrieved chunks with source markers for citations
  2. Keeping assembled context within a token budget
  3. Building an augmented prompt that instructs the model to answer
     only from the provided context
  4. Showing what happens when chunks exceed the token budget
"""

import os
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL        = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Token budget split (for an 8 192-token context window)
MODEL_CONTEXT_WINDOW = 8_192
RESERVED_FOR_ANSWER  = 1_500   # leave room for the model's reply
RESERVED_FOR_SYSTEM  = 400     # system instruction overhead
RESERVED_FOR_QUESTION = 200    # question overhead
MAX_CONTEXT_TOKENS   = (
    MODEL_CONTEXT_WINDOW
    - RESERVED_FOR_ANSWER
    - RESERVED_FOR_SYSTEM
    - RESERVED_FOR_QUESTION
)   # ≈ 6 092 tokens available for chunk context


# ---------------------------------------------------------------------------
# Token counter (no tiktoken required — portable 4-chars/token heuristic)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Fast token approximation: ~4 characters per token.
    Accurate enough for budget enforcement; use tiktoken for exact counts.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Mock embedder (same deterministic approach as prior modules)
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
        "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy.",
        "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds.",
        "WealthConnect Q4 Portfolio Analysis and Earnings Overview.",
        "The tech sector allocation increased 8 percentage points in Q4.",
    ],
    "refund": [
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
# In-memory vector collection (same as prior modules)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


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
    {"source": "policies.md",          "section": "Password Reset",   "chunk_index": 0,
     "text": "To reset your WealthConnect password, visit the login page and select Forgot Password."},
    {"source": "policies.md",          "section": "Account Recovery", "chunk_index": 1,
     "text": "Account recovery requires identity verification via registered email or phone number."},
    {"source": "policies.md",          "section": "Security",         "chunk_index": 2,
     "text": "Two-factor authentication is mandatory for all premium account holders."},
    {"source": "policies.md",          "section": "Refunds",          "chunk_index": 3,
     "text": "All refund requests must be submitted within 30 days of the original transaction date."},
    {"source": "policies.md",          "section": "Refunds",          "chunk_index": 4,
     "text": "Refunds are processed within 5–7 business days after approval."},
    {"source": "q4_earnings_report.pdf", "section": "Overview",       "chunk_index": 0,
     "text": "WealthConnect Q4 Portfolio Analysis and Earnings Overview."},
    {"source": "q4_earnings_report.pdf", "section": "Performance",    "chunk_index": 1,
     "text": "Our core aggressive growth fund yielded a 12% return in Q4, driven by AI and renewable energy."},
    {"source": "q4_earnings_report.pdf", "section": "Outlook",        "chunk_index": 2,
     "text": "We project interest rate stabilisation next year and recommend rebalancing to 30% high-yield bonds."},
    {"source": "q4_earnings_report.pdf", "section": "Operations",     "chunk_index": 3,
     "text": "Customer acquisition costs dropped by $15 per head after deploying the RAG-powered assistant."},
    {"source": "q4_earnings_report.pdf", "section": "Allocation",     "chunk_index": 4,
     "text": "The tech sector allocation increased 8 percentage points in Q4."},
    {"source": "about.html",            "section": "Overview",        "chunk_index": 0,
     "text": "WealthConnect is an AI-powered financial advisory platform for smart, personalised investing."},
    {"source": "about.html",            "section": "Features",        "chunk_index": 1,
     "text": "Our RAG assistant answers financial questions with cited, up-to-date sources from your documents."},
    {"source": "about.html",            "section": "Integrations",    "chunk_index": 2,
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


def retrieve(query: str, collection: VectorCollection,
             k: int = 5, dry_run: bool = False) -> list[dict]:
    query_vector = embed([query], dry_run=dry_run)[0]
    results = collection.search(vector=query_vector, top_k=k)
    return [{"score": round(r["score"], 6), "text": r["text"],
             "metadata": r["metadata"]} for r in results]


# ---------------------------------------------------------------------------
# 1. Format chunk with source marker
# ---------------------------------------------------------------------------

def format_chunk(index: int, chunk: dict) -> str:
    """
    Label each chunk with a numbered source marker.

    Format:
        [1] policies.md#0  (Password Reset)
        To reset your WealthConnect password ...

    The marker gives the LLM a citable reference it can include in its answer.
    """
    meta        = chunk["metadata"]
    source      = meta["source"]
    chunk_index = meta.get("chunk_index", "?")
    section     = meta.get("section", "")
    section_tag = f"  ({section})" if section else ""
    marker      = f"[{index}] {source}#{chunk_index}{section_tag}"
    return f"{marker}\n{chunk['text']}"


# ---------------------------------------------------------------------------
# 2. Assemble context within token budget
# ---------------------------------------------------------------------------

def assemble_context(
    chunks: list[dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[str, int, list[dict]]:
    """
    Greedily include chunks (best-ranked first) until the token budget is reached.

    Returns:
        context_str    : Formatted string of all included chunks.
        tokens_used    : Approximate tokens consumed by the context.
        included_chunks: Subset of chunks that fit within the budget.
    """
    selected_parts: list[str] = []
    included_chunks: list[dict] = []
    used_tokens = 0

    for index, chunk in enumerate(chunks, start=1):
        formatted   = format_chunk(index, chunk)
        token_count = count_tokens(formatted)

        if used_tokens + token_count > max_tokens:
            # Over budget — skip this chunk (could also trim/summarise here)
            continue

        selected_parts.append(formatted)
        included_chunks.append(chunk)
        used_tokens += token_count

    context_str = "\n\n---\n\n".join(selected_parts)
    return context_str, used_tokens, included_chunks


# ---------------------------------------------------------------------------
# 3. Build augmented prompt
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a grounded financial assistant for WealthConnect. "
    "Answer the question using ONLY the provided context. "
    'If the answer is not in the context, say: '
    '"I don\'t have enough information in the provided context." '
    "When possible, cite sources using the markers like [1] or [2]."
)


def build_prompt(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Assemble a complete, token-budget-respecting prompt for the LLM.

    Args:
        question         : The user's natural-language question.
        retrieved_chunks : Ranked list of chunks from retrieval.

    Returns:
        dict with keys:
          system_instruction  : The grounding instruction for the LLM.
          prompt              : The full user-turn message (context + question).
          context_tokens      : Tokens used by the injected context.
          total_tokens_est    : Total estimated tokens for the full prompt.
          sources_used        : Metadata of every chunk that made it in.
          chunks_retrieved    : Total chunks offered before budget check.
          chunks_included     : Chunks that fit within the budget.
          budget_remaining    : Unused context token budget.
    """
    context_str, context_tokens, included = assemble_context(retrieved_chunks)

    prompt = (
        f"Context:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    system_tokens  = count_tokens(SYSTEM_INSTRUCTION)
    question_tokens = count_tokens(question)
    total_tokens   = system_tokens + context_tokens + question_tokens

    return {
        "system_instruction":  SYSTEM_INSTRUCTION,
        "prompt":              prompt,
        "context_tokens":      context_tokens,
        "total_tokens_est":    total_tokens,
        "sources_used":        [c["metadata"] for c in included],
        "chunks_retrieved":    len(retrieved_chunks),
        "chunks_included":     len(included),
        "budget_remaining":    MAX_CONTEXT_TOKENS - context_tokens,
    }


# ---------------------------------------------------------------------------
# 4. Main demo
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {
        "question": "How can I reset my WealthConnect password?",
        "note": "Straightforward policy lookup — should cite policies.md sections.",
    },
    {
        "question": "What was the Q4 portfolio return and what is the investment outlook?",
        "note": "Multi-part financial question — should pull multiple earnings chunks.",
    },
    {
        "question": "What is the meaning of life?",
        "note": "Out-of-corpus question — should trigger the 'not enough information' response path.",
    },
]

# Budget-overflow demo: create an artificially tiny budget to show truncation
TIGHT_BUDGET = 50   # tokens — forces early cutoff


def main():
    lines = []

    lines.append("=" * 65)
    lines.append("GY3.38 — Context Injection & Prompt Augmentation")
    lines.append("=" * 65)
    lines.append(f"\nModel context window : {MODEL_CONTEXT_WINDOW} tokens")
    lines.append(f"Reserved for answer  : {RESERVED_FOR_ANSWER} tokens")
    lines.append(f"Reserved for system  : {RESERVED_FOR_SYSTEM} tokens")
    lines.append(f"Reserved for question: {RESERVED_FOR_QUESTION} tokens")
    lines.append(f"Max context budget   : {MAX_CONTEXT_TOKENS} tokens\n")

    collection = build_collection(dry_run=True)

    # -----------------------------------------------------------------------
    # Section 1: Full prompt assembly for each demo query
    # -----------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("SECTION 1 — Augmented Prompt Assembly (normal budget)")
    lines.append("=" * 65)

    for demo in DEMO_QUERIES:
        question = demo["question"]
        note     = demo["note"]

        chunks = retrieve(question, collection, k=5, dry_run=True)
        result = build_prompt(question, chunks)

        lines.append(f'\nQuestion : "{question}"')
        lines.append(f"Note     : {note}")
        lines.append("-" * 55)
        lines.append(f"  Chunks retrieved  : {result['chunks_retrieved']}")
        lines.append(f"  Chunks included   : {result['chunks_included']}")
        lines.append(f"  Context tokens    : {result['context_tokens']}")
        lines.append(f"  Total tokens est. : {result['total_tokens_est']}")
        lines.append(f"  Budget remaining  : {result['budget_remaining']} tokens")
        lines.append(f"  Sources used      :")
        for src in result["sources_used"]:
            lines.append(
                f"    [{src['chunk_index']}] {src['source']}  — {src['section']}"
            )
        lines.append(f"\n  --- FULL PROMPT ---")
        lines.append(f"  SYSTEM: {result['system_instruction']}")
        lines.append(f"")
        # Indent each prompt line for readability
        for pline in result["prompt"].splitlines():
            lines.append(f"  {pline}")

    # -----------------------------------------------------------------------
    # Section 2: Token budget overflow demo
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("SECTION 2 — Token Budget Overflow (tight budget demo)")
    lines.append("=" * 65)
    lines.append(f"\nArtificial budget cap: {TIGHT_BUDGET} tokens")
    lines.append("(Normally {MAX_CONTEXT_TOKENS} tokens — reduced here to show truncation)\n")

    tight_question = "How can I reset my WealthConnect password?"
    tight_chunks   = retrieve(tight_question, collection, k=5, dry_run=True)

    _, _, included_tight = assemble_context(tight_chunks, max_tokens=TIGHT_BUDGET)
    _, _, included_normal = assemble_context(tight_chunks, max_tokens=MAX_CONTEXT_TOKENS)

    lines.append(f"  Chunks retrieved          : {len(tight_chunks)}")
    lines.append(f"  Chunks included (tight)   : {len(included_tight)}")
    lines.append(f"  Chunks included (normal)  : {len(included_normal)}")
    lines.append(f"\n  Tight budget included chunks:")
    for c in included_tight:
        lines.append(f"    [{c['metadata']['chunk_index']}] {c['metadata']['source']} — {c['text'][:60]}...")
    if not included_tight:
        lines.append("    (none — all chunks exceed the tight budget)")
    lines.append(f"\n  Strategies when over budget:")
    lines.append("    1. Use best-ranked chunks first (already done — greedy by score)")
    lines.append("    2. Reduce k to retrieve fewer candidates")
    lines.append("    3. Trim long chunks to N sentences before assembly")
    lines.append("    4. Summarise lower-ranked chunks to compress them")
    lines.append("    5. Re-rank before assembly to maximise relevance per token")

    # -----------------------------------------------------------------------
    # Section 3: Why source markers matter
    # -----------------------------------------------------------------------
    lines.append("\n" + "=" * 65)
    lines.append("SECTION 3 — Source Markers & Citation")
    lines.append("=" * 65)
    lines.append("""
  Every chunk is labelled with a numbered marker before being injected:

      [1] policies.md#0  (Password Reset)
      To reset your WealthConnect password ...

  The system instruction tells the model to cite these markers in its answer.
  A grounded answer might look like:

      "To reset your password, visit the login page and select
       'Forgot Password' [1]. If you cannot access your account,
       identity verification via your registered email is required [2]."

  Without markers:
    - The model may use the context but can't point back to the source.
    - Auditing which document drove the answer becomes impossible.
    - Users cannot verify claims against the original documents.

  The "answer only from context" instruction is equally critical:
    - Without it, the model may blend retrieved facts with its own training
      knowledge, producing confident but uncited and unverifiable answers.
    - With it, any gap in retrieved context surfaces as an explicit
      "I don't have enough information" — a signal to improve retrieval.
""".strip())

    lines.append("\n" + "=" * 65)
    lines.append("Context injection & prompt augmentation demo complete.")
    lines.append("=" * 65)

    output = "\n".join(lines)

    output_file = "context_injection_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nResults written to '{output_file}'.")


if __name__ == "__main__":
    main()
