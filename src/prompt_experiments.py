"""
prompt_experiments.py — GY3.13: Prompt Construction & System/User Roles
Assignment demonstration script for WealthConnect RAG

Runs four experiments that illustrate the core lessons of GY3.13:

  Experiment 1 — System vs User role separation
    Shows how the system message sets fixed rules while the user message
    carries the per-turn question and context.

  Experiment 2 — Same question, three prompt variants
    strict  → prose answer with rules, source line
    concise → 2–3 sentence cap
    json    → structured { answer, source, confidence } output

  Experiment 3 — Ambiguous vs clear user prompt
    Demonstrates how task + scope + format + fallback in the user prompt
    produces far more usable output than vague phrasing.

  Experiment 4 — Fallback behaviour
    Shows that when no context is provided, all variants produce the safe
    fallback message instead of hallucinating an answer.

Run with:
    python -m src.prompt_experiments
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.prompt_builder import (
    load_system_prompt,
    build_user_message,
    compare_prompt_variants,
    print_comparison,
    parse_json_response,
    PROMPT_VARIANTS,
)

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "What are the tax implications of the ABC Investment Product?"

SAMPLE_CONTEXT = (
    "[Source 1] ABC Investment Product Brochure | Version: 4.1 | Type: product_brochure\n"
    "The ABC Investment Product is subject to capital gains tax on returns above the "
    "annual exemption threshold. Interest income is taxed at the investor's marginal rate. "
    "Tax-free allowances may apply subject to current HMRC guidelines.\n\n"
    "[Source 2] Tax Rules — Wealth Division | Version: 3.2 | Type: tax_rules\n"
    "Investments held for more than 12 months qualify for the reduced long-term capital "
    "gains rate. Early redemption within 12 months is subject to standard income tax rates."
)

DIVIDER = "=" * 70


# ---------------------------------------------------------------------------
# Experiment 1 — System vs User role separation
# ---------------------------------------------------------------------------

def experiment_1_role_separation() -> None:
    """
    GY3.13 core concept: system sets the rules, user carries the question.

    Prints the raw messages dict that will be sent to the API so the
    separation is explicit and visible.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 1 — System vs User Role Separation")
    print(DIVIDER)

    system_prompt = load_system_prompt("strict")
    user_message  = build_user_message(SAMPLE_QUESTION, SAMPLE_CONTEXT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    print("\n  role: 'system'")
    print("  Purpose: Sets who the assistant is and the rules it must follow.")
    print("  Content preview (first 300 chars):")
    print(f"    {system_prompt[:300]}...")

    print("\n  role: 'user'")
    print("  Purpose: Carries this turn's retrieved context + RM's question.")
    print("  Content preview (first 300 chars):")
    print(f"    {user_message[:300]}...")

    print(f"\n  Sending to model: {CHAT_MODEL}")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
    )
    answer = response.choices[0].message.content.strip()

    print(f"\n  Model Response:")
    for line in answer.splitlines():
        print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
# Experiment 2 — Same question, three prompt variants
# ---------------------------------------------------------------------------

def experiment_2_prompt_variants() -> None:
    """
    GY3.13: "Test variations side by side instead of guessing."

    Runs SAMPLE_QUESTION through strict, concise, and json variants.
    Shows how the system message wording shapes length, format, and structure.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 2 — Same Question, Three Prompt Variants")
    print(DIVIDER)

    results = compare_prompt_variants(
        question=SAMPLE_QUESTION,
        context=SAMPLE_CONTEXT,
        variants=["strict", "concise", "json"],
    )
    print_comparison(SAMPLE_QUESTION, results)

    # Parse and display the JSON variant as a structured object
    print("\n  Parsed JSON variant:")
    parsed = parse_json_response(results["json"])
    print(f"    answer     : {parsed.get('answer', '')[:120]}...")
    print(f"    source     : {parsed.get('source')}")
    print(f"    confidence : {parsed.get('confidence')}")


# ---------------------------------------------------------------------------
# Experiment 3 — Ambiguous vs clear user prompt
# ---------------------------------------------------------------------------

def experiment_3_ambiguous_vs_clear() -> None:
    """
    GY3.13: "A clear prompt states: the task, the scope, the format,
    and the fallback."

    Runs three ambiguous/clear pairs through the model to show how
    specific phrasing produces more useful, consistent output.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 3 — Ambiguous vs Clear User Prompt")
    print(DIVIDER)

    system_prompt = load_system_prompt("strict")

    pairs = [
        (
            # Ambiguous — no task, no format, no scope
            "Tell me about this investment product.",
            # Clear — task + scope + format + source instruction
            (
                "In 2–3 sentences, summarise the key features of the investment product "
                "described in the context. Cite the document name and version."
            ),
        ),
        (
            "What are the tax rules?",
            (
                "List the applicable tax rules for this product as bullet points, "
                "using only the information in the provided context. "
                "If tax rules are not covered, use the fallback message."
            ),
        ),
    ]

    for i, (ambiguous, clear) in enumerate(pairs, start=1):
        print(f"\n  ── Pair {i} ──")
        for label, prompt_text in [("AMBIGUOUS", ambiguous), ("CLEAR", clear)]:
            user_msg = build_user_message(prompt_text, SAMPLE_CONTEXT)
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()
            print(f"\n  [{label}]")
            print(f"  Prompt : {prompt_text}")
            print(f"  Output :")
            for line in answer.splitlines():
                print(f"    {line}")


# ---------------------------------------------------------------------------
# Experiment 4 — Fallback when no context is found
# ---------------------------------------------------------------------------

def experiment_4_fallback_behaviour() -> None:
    """
    GY3.13: "Constrain the fallback — what to do when it can't answer."

    Runs the question with an empty context to verify all variants
    return the safe fallback message instead of hallucinating.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 4 — Fallback Behaviour (No Context)")
    print(DIVIDER)

    for variant in PROMPT_VARIANTS:
        system_prompt = load_system_prompt(variant)
        user_message  = build_user_message(SAMPLE_QUESTION, context="")  # no context

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()

        print(f"\n  Variant: '{variant}'")
        print(f"  Response:")
        for line in answer.splitlines():
            print(f"    {line}")

    print()


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'#' * 70}")
    print("  GY3.13 — Prompt Construction & System/User Roles")
    print("  WealthConnect RAG — Experiment Runner")
    print(f"{'#' * 70}")

    experiment_1_role_separation()
    experiment_2_prompt_variants()
    experiment_3_ambiguous_vs_clear()
    experiment_4_fallback_behaviour()

    print(f"\n{'#' * 70}")
    print("  All experiments complete.")
    print(f"{'#' * 70}\n")
