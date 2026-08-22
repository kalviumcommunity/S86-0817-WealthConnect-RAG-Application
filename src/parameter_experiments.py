"""
parameter_experiments.py — GY3.16: Model Parameters & Output Control
Assignment demonstration script for WealthConnect RAG

Runs five experiments that cover every concept in GY3.16:

  Experiment 1 — temperature comparison (0.0 vs 1.0)
    Same prompt, same context — low temp is stable and factual,
    high temp drifts and embellishes.

  Experiment 2 — max_tokens cap
    Shows how output is truncated at different token limits.
    Illustrates the cost/length trade-off.

  Experiment 3 — stop sequences
    Demonstrates how a stop string cuts generation at a defined boundary,
    preventing the model from rambling past the answer.

  Experiment 4 — top_p vs temperature
    Compares top_p=0.1 (tight nucleus) to temperature=0.0 (greedy),
    showing two routes to focused output and why you tune one, not both.

  Experiment 5 — production preset comparison
    Runs all named WealthConnect presets on the same wealth question
    and prints the results side by side.

Run with:
    python -m src.parameter_experiments
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from src.prompt_builder import load_system_prompt, build_user_message
from src.model_params import (
    ModelParams,
    RAG_GROUNDED,
    RAG_STRICT,
    RAG_JSON,
    HIGH_TEMPERATURE,
    LOW_TOP_P,
    PRESETS,
    describe_presets,
)

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

DIVIDER = "=" * 70

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


# ---------------------------------------------------------------------------
# Core helper — single LLM call with explicit ModelParams
# ---------------------------------------------------------------------------

def call_model(question: str, context: str, params: ModelParams, variant: str = "strict") -> dict:
    """
    Make one LLM call using the given ModelParams preset.

    Returns a dict with:
        response      : model's raw text response
        finish_reason : why generation stopped ('stop', 'length', etc.)
        usage         : token usage breakdown
        params        : the ModelParams used
    """
    system_prompt = load_system_prompt(variant)
    user_message  = build_user_message(question, context)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        **params.to_api_kwargs(),
    )

    choice = response.choices[0]
    return {
        "response":      choice.message.content.strip(),
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
        "params": params,
    }


def print_result(label: str, result: dict) -> None:
    """Pretty-print a single experiment result."""
    p = result["params"]
    u = result["usage"]
    print(f"\n  ── {label} ──")
    print(f"  Params : temperature={p.temperature}  max_tokens={p.max_tokens}  "
          f"top_p={p.top_p}  stop={p.stop or 'none'}")
    print(f"  Tokens : prompt={u['prompt_tokens']}  "
          f"completion={u['completion_tokens']}  total={u['total_tokens']}")
    print(f"  Finish : {result['finish_reason']}")
    print(f"  Response:")
    for line in result["response"].splitlines():
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Experiment 1 — temperature: 0.0 vs 1.0
# ---------------------------------------------------------------------------

def experiment_1_temperature() -> None:
    """
    GY3.16: "At 0.0 the answer is stable across runs; at 1.0 it varies
    and embellishes — fine for brainstorming, dangerous for grounded facts."

    Runs the same prompt twice at each temperature so the stability
    difference is visible.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 1 — Temperature: 0.0 (factual) vs 1.0 (creative)")
    print(DIVIDER)
    print("  Same prompt, same context — only temperature changes.")
    print("  Low = stable, repeatable, faithful to context.")
    print("  High = varied, may embellish or drift from source.")

    for temp_label, temp_value in [("LOW (0.0)", 0.0), ("HIGH (1.0)", 1.0)]:
        params = ModelParams(
            name        = f"temp_{temp_value}",
            temperature = temp_value,
            max_tokens  = 300,
        )
        # Run twice to show consistency at 0.0 and variance at 1.0
        for run in range(1, 3):
            result = call_model(SAMPLE_QUESTION, SAMPLE_CONTEXT, params)
            print_result(f"temperature={temp_value}  Run {run}", result)


# ---------------------------------------------------------------------------
# Experiment 2 — max_tokens cap
# ---------------------------------------------------------------------------

def experiment_2_max_tokens() -> None:
    """
    GY3.16: "max_tokens caps output length. Protects against runaway answers
    and directly limits cost (output tokens are billed)."

    Runs the same prompt at 50, 150, and 400 token caps.
    Shows how finish_reason flips from 'stop' to 'length' when truncated.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 2 — max_tokens cap (50 / 150 / 400)")
    print(DIVIDER)
    print("  finish_reason='length' means the model was cut off mid-answer.")
    print("  finish_reason='stop'   means the model finished naturally.")

    for cap in [50, 150, 400]:
        params = ModelParams(
            name       = f"max_tokens_{cap}",
            temperature = 0.0,
            max_tokens  = cap,
        )
        result = call_model(SAMPLE_QUESTION, SAMPLE_CONTEXT, params)
        print_result(f"max_tokens={cap}", result)


# ---------------------------------------------------------------------------
# Experiment 3 — stop sequences
# ---------------------------------------------------------------------------

def experiment_3_stop_sequences() -> None:
    """
    GY3.16: "stop — sequences that end generation early."

    Compares no stop sequence vs stop=['\n\n'] to show how a boundary
    string cuts generation and keeps the answer to one paragraph.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 3 — Stop Sequences")
    print(DIVIDER)
    print("  stop=['\\n\\n'] cuts generation after the first paragraph break.")
    print("  Prevents the model from rambling past the answer.")

    for label, stop_val in [("No stop sequence", []), ("stop=['\\n\\n']", ["\n\n"])]:
        params = ModelParams(
            name        = label,
            temperature = 0.0,
            max_tokens  = 400,
            stop        = stop_val,
        )
        result = call_model(SAMPLE_QUESTION, SAMPLE_CONTEXT, params)
        print_result(label, result)


# ---------------------------------------------------------------------------
# Experiment 4 — top_p vs temperature
# ---------------------------------------------------------------------------

def experiment_4_top_p_vs_temperature() -> None:
    """
    GY3.16: "top_p — an alternative to temperature (nucleus sampling);
    usually tune one, not both."

    Compares three approaches to focused output:
      A) temperature=0.0, top_p=1.0  (greedy — standard RAG setting)
      B) temperature=1.0, top_p=0.1  (tight nucleus sampling)
      C) temperature=0.5, top_p=0.5  (WRONG — tuning both; shown for contrast)
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 4 — top_p vs temperature")
    print(DIVIDER)
    print("  Rule: tune temperature OR top_p — not both simultaneously.")

    configs = [
        ("A) temperature=0.0, top_p=1.0  [standard RAG — greedy]",
         ModelParams(name="greedy",   temperature=0.0, top_p=1.0, max_tokens=300)),
        ("B) temperature=1.0, top_p=0.1  [tight nucleus — alternative approach]",
         ModelParams(name="nucleus",  temperature=1.0, top_p=0.1, max_tokens=300)),
        ("C) temperature=0.5, top_p=0.5  [both tuned — NOT recommended]",
         ModelParams(name="both",     temperature=0.5, top_p=0.5, max_tokens=300)),
    ]

    for label, params in configs:
        result = call_model(SAMPLE_QUESTION, SAMPLE_CONTEXT, params)
        print_result(label, result)

    print("\n  Takeaway: A and B both produce focused output via different routes.")
    print("  C introduces unpredictable interaction — avoid tuning both at once.")


# ---------------------------------------------------------------------------
# Experiment 5 — production preset comparison
# ---------------------------------------------------------------------------

def experiment_5_production_presets() -> None:
    """
    Runs all named WealthConnect ModelParams presets on the same question
    to show the recommended settings for each use-case in one view.
    """
    print(f"\n{DIVIDER}")
    print("EXPERIMENT 5 — WealthConnect Production Presets")
    print(DIVIDER)
    print("  Comparing all named presets defined in model_params.py.\n")

    describe_presets()

    preset_to_variant = {
        "rag_grounded":    "strict",
        "rag_strict":      "concise",
        "rag_json":        "json",
        "high_temperature":"strict",
        "low_top_p":       "strict",
    }

    for preset_name, preset in PRESETS.items():
        variant = preset_to_variant.get(preset_name, "strict")
        result  = call_model(SAMPLE_QUESTION, SAMPLE_CONTEXT, preset, variant=variant)
        print_result(f"Preset: '{preset_name}' | prompt_variant='{variant}'", result)


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'#' * 70}")
    print("  GY3.16 — Model Parameters & Output Control")
    print("  WealthConnect RAG — Parameter Experiment Runner")
    print(f"{'#' * 70}")

    experiment_1_temperature()
    experiment_2_max_tokens()
    experiment_3_stop_sequences()
    experiment_4_top_p_vs_temperature()
    experiment_5_production_presets()

    print(f"\n{'#' * 70}")
    print("  All experiments complete.")
    print(f"  Recommended production setting: RAG_GROUNDED preset")
    print(f"  temperature=0.0  max_tokens=400  top_p=1.0  stop=none")
    print(f"{'#' * 70}\n")
