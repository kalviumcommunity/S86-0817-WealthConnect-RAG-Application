"""
prompt_builder.py — Prompt Construction & System/User Roles for WealthConnect RAG
Assignment: GY3.13

Demonstrates:
  - Separation of system and user roles
  - Loading prompt templates from prompts/ (kept out of code)
  - Building a structured user message with injected RAG context
  - Multiple prompt variants (strict prose, JSON output, concise)
  - Side-by-side prompt comparison to show how wording changes output

Key principle:
  system  = who the assistant is + rules it must always follow
  user    = the actual question for this turn + the retrieved context

The system message is the control panel for grounding —
"answer only from context" and "say you don't know when unsure"
both live here.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Client — credentials from .env, never hard-coded
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PROMPTS_DIR = Path("prompts")

# ---------------------------------------------------------------------------
# Available prompt variants
# Maps a short name to the template file in prompts/
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "strict":   "system_prompt_strict.txt",   # Full rules, prose output, source line
    "json":     "system_prompt_json.txt",      # Rules + structured JSON output
    "concise":  "system_prompt_concise.txt",   # Minimal, 2–3 sentence cap
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_system_prompt(variant: str = "strict") -> str:
    """
    Load a system prompt template from the prompts/ directory.

    Keeping prompts in files (not hard-coded strings) means the Wealth team
    can adjust tone, rules, and fallback wording without touching Python code.

    Args:
        variant: One of 'strict', 'json', 'concise'

    Returns:
        The system prompt string.
    """
    filename = PROMPT_VARIANTS.get(variant)
    if not filename:
        raise ValueError(
            f"Unknown prompt variant '{variant}'. "
            f"Choose from: {list(PROMPT_VARIANTS.keys())}"
        )

    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    return filepath.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------

def build_user_message(question: str, context: str) -> str:
    """
    Build the user-role message by injecting retrieved document context
    alongside the Relationship Manager's question.

    Why context goes in the user message (not the system message):
      - The system message is static per session — it sets the rules.
      - The user message changes every turn — it carries the question
        and the retrieved chunks for that specific query.
      - This separation keeps the architecture clean and the system
        prompt reusable across different questions.

    Args:
        question : The RM's natural-language question.
        context  : Formatted string of retrieved approved document chunks.

    Returns:
        A structured user message string ready for the API call.
    """
    if context.strip():
        return (
            "The following context has been retrieved from the bank's current "
            "approved wealth-management documents. Use ONLY this context to answer.\n\n"
            f"--- DOCUMENT CONTEXT START ---\n{context}\n--- DOCUMENT CONTEXT END ---\n\n"
            f"Relationship Manager Question: {question}"
        )
    else:
        # No context retrieved — signal the model to use the fallback
        return (
            "No relevant information was found in the current approved wealth documents.\n\n"
            f"Relationship Manager Question: {question}"
        )


# ---------------------------------------------------------------------------
# Single completion with a chosen variant
# ---------------------------------------------------------------------------

def complete(
    question: str,
    context: str = "",
    variant: str = "strict",
) -> str:
    """
    Call the LLM with the chosen prompt variant.

    Args:
        question : The RM's question.
        context  : Retrieved document context (empty string if none).
        variant  : Prompt variant — 'strict', 'json', or 'concise'.

    Returns:
        The model's raw response string.
    """
    system_prompt = load_system_prompt(variant)
    user_message = build_user_message(question, context)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,  # Deterministic — grounding requires consistency
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Prompt comparison — core concept from GY3.13
# Same question, different prompts → different outputs
# ---------------------------------------------------------------------------

def compare_prompt_variants(
    question: str,
    context: str = "",
    variants: list[str] | None = None,
) -> dict[str, str]:
    """
    Run the same question through multiple prompt variants and return
    all responses side by side.

    This demonstrates how system message wording directly controls output
    shape, length, and format — the central lesson of GY3.13.

    Args:
        question : The RM's question.
        context  : Retrieved document context.
        variants : Which variants to compare. Defaults to all three.

    Returns:
        Dict mapping variant name → model response.
    """
    if variants is None:
        variants = list(PROMPT_VARIANTS.keys())

    results = {}
    for variant in variants:
        print(f"[prompt_builder] Running variant: '{variant}'...")
        results[variant] = complete(question, context, variant)

    return results


def print_comparison(question: str, results: dict[str, str]) -> None:
    """
    Pretty-print a side-by-side comparison of prompt variant outputs.
    Makes it easy to see how prompt wording drives output differences.
    """
    print("\n" + "=" * 70)
    print(f"PROMPT VARIANT COMPARISON")
    print(f"Question: {question}")
    print("=" * 70)

    for variant, response in results.items():
        print(f"\n── Variant: '{variant}' ──")
        print(f"  System prompt file: {PROMPT_VARIANTS[variant]}")
        print(f"  Response:\n")

        # Indent the response for readability
        for line in response.splitlines():
            print(f"    {line}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Ambiguous vs clear prompt demonstration
# GY3.13: "A clear prompt states: the task, the scope, the format, the fallback"
# ---------------------------------------------------------------------------

AMBIGUOUS_PROMPTS = [
    "Tell me about this investment product.",
    "What are the tax rules?",
    "Explain eligibility.",
]

CLEAR_PROMPTS = [
    (
        "In 2–3 sentences, summarise the key features of the investment product "
        "described in the context. Cite the document name and version."
    ),
    (
        "List the applicable tax rules for this product as bullet points, "
        "using only the information in the provided context. "
        "If tax rules are not covered, use the fallback message."
    ),
    (
        "State the eligibility requirements for this investment product "
        "in a numbered list. Include any age, income, or residency conditions "
        "mentioned in the context. Cite the source document."
    ),
]


def demo_ambiguous_vs_clear(context: str = "") -> None:
    """
    Run each ambiguous/clear prompt pair through the model and print both
    responses to illustrate how prompt clarity changes output quality.

    This is the core demonstration from GY3.13:
    "Tiny wording changes move the output a lot."
    """
    system_prompt = load_system_prompt("strict")

    print("\n" + "=" * 70)
    print("AMBIGUOUS vs CLEAR PROMPT DEMONSTRATION")
    print("=" * 70)

    for ambiguous, clear in zip(AMBIGUOUS_PROMPTS, CLEAR_PROMPTS):
        print(f"\n── Pair ──")

        for label, prompt_text in [("AMBIGUOUS", ambiguous), ("CLEAR", clear)]:
            user_msg = build_user_message(prompt_text, context)
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()

            print(f"\n  [{label}] Prompt:\n    {prompt_text}")
            print(f"  Response:")
            for line in answer.splitlines():
                print(f"    {line}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# JSON output parsing helper
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict:
    """
    Parse the model's response when using the 'json' variant.
    Returns the parsed dict, or an error dict if parsing fails.

    The 'json' prompt instructs the model to reply with ONLY a JSON object:
        { "answer": ..., "source": ..., "confidence": ... }
    """
    try:
        # Strip markdown code fences if the model wraps the JSON
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "answer": raw,
            "source": None,
            "confidence": "unknown",
            "parse_error": str(e),
        }


# ---------------------------------------------------------------------------
# Entry point — runs all demonstrations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample question and a minimal mock context for demonstration.
    # In the full RAG pipeline, context comes from src/retrieval.py.
    sample_question = "What are the tax implications of the ABC Investment Product?"

    sample_context = (
        "[Source 1] ABC Investment Product Brochure | Version: 4.1 | Type: product_brochure\n"
        "The ABC Investment Product is subject to capital gains tax on returns above the "
        "annual exemption threshold. Interest income is taxed at the investor's marginal rate. "
        "Tax-free allowances may apply subject to current HMRC guidelines.\n\n"
        "[Source 2] Tax Rules — Wealth Division | Version: 3.2 | Type: tax_rules\n"
        "Investments held for more than 12 months qualify for the reduced long-term capital "
        "gains rate. Early redemption within 12 months is subject to standard income tax rates."
    )

    # 1. Compare all three prompt variants on the same question
    print("\n[1] Comparing all prompt variants...")
    results = compare_prompt_variants(sample_question, sample_context)
    print_comparison(sample_question, results)

    # 2. Parse the JSON variant response
    print("\n[2] Parsed JSON variant output:")
    parsed = parse_json_response(results["json"])
    print(f"  Answer     : {parsed.get('answer')}")
    print(f"  Source     : {parsed.get('source')}")
    print(f"  Confidence : {parsed.get('confidence')}")

    # 3. Ambiguous vs clear prompt demonstration
    print("\n[3] Ambiguous vs Clear prompt demonstration...")
    demo_ambiguous_vs_clear(sample_context)
