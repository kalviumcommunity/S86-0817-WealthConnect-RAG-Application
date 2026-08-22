"""
model_params.py — Model Parameters & Output Control for WealthConnect RAG
Assignment: GY3.16

Centralises every parameter that controls LLM output behaviour:
  temperature  — randomness / creativity (0 = deterministic, 2 = very random)
  max_tokens   — hard cap on output length; directly controls cost
  top_p        — nucleus sampling; alternative to temperature (tune one, not both)
  stop         — sequences that end generation early

Why this matters for a RAG assistant:
  A grounded, factual assistant must be CONSISTENT and SHORT.
  - temperature=0.0  → same question always gets the same answer
  - max_tokens=300   → caps output; output tokens are billed, runaway = expensive
  - stop sequences   → prevents the model from rambling past the answer

This module defines named parameter presets so the right settings are
always applied by name, not by guess.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    All generation parameters for a single LLM call.

    Attributes:
        temperature : 0.0–2.0. Low = focused/repeatable. High = creative/varied.
        max_tokens  : Maximum tokens in the response. Caps cost and length.
        top_p       : Nucleus sampling threshold (0.0–1.0). Tune temperature OR
                      top_p, not both.
        stop        : List of sequences that will end generation early.
        name        : Human-readable label for this preset.
        description : Why you'd use these settings.
    """
    temperature : float       = 0.0
    max_tokens  : int         = 400
    top_p       : float       = 1.0
    stop        : list[str]   = field(default_factory=list)
    name        : str         = "default"
    description : str         = ""

    def to_api_kwargs(self) -> dict:
        """
        Return a dict of only the parameters the OpenAI API accepts.
        Excludes internal fields (name, description).
        Drops stop if the list is empty — the API treats [] differently from omitted.
        """
        kwargs = {
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "top_p":       self.top_p,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        return kwargs


# ---------------------------------------------------------------------------
# Named presets — the recommended settings for each use-case
# ---------------------------------------------------------------------------

# Production preset for WealthConnect RAG answers.
# temperature=0.0  → fully deterministic; same question always gives same answer.
# max_tokens=400   → enough for a concise factual answer with a source line.
# top_p=1.0        → no nucleus filtering needed when temperature is already 0.
RAG_GROUNDED = ModelParams(
    name        = "rag_grounded",
    description = (
        "Production preset for grounded RAG answers. "
        "Deterministic, consistent, source-faithful. "
        "Low cost — hard cap at 400 output tokens."
    ),
    temperature = 0.0,
    max_tokens  = 400,
    top_p       = 1.0,
    stop        = [],
)

# Strict preset — shorter answers, stop sequence prevents rambling.
# stop=["\n\n"] cuts generation after the first double newline,
# keeping the answer to one tight paragraph.
RAG_STRICT = ModelParams(
    name        = "rag_strict",
    description = (
        "Strict preset: one paragraph max. "
        "Stop sequence cuts generation after the first paragraph break. "
        "Useful for the concise prompt variant."
    ),
    temperature = 0.0,
    max_tokens  = 200,
    top_p       = 1.0,
    stop        = ["\n\n"],
)

# JSON preset — used with system_prompt_json.txt.
# Slightly more tokens to accommodate the JSON wrapper.
# temperature=0.0 keeps the JSON structure stable and parseable.
RAG_JSON = ModelParams(
    name        = "rag_json",
    description = (
        "For the JSON output variant. "
        "Extra tokens to fit the JSON envelope. "
        "temperature=0.0 keeps the structure consistent and parseable."
    ),
    temperature = 0.0,
    max_tokens  = 500,
    top_p       = 1.0,
    stop        = [],
)

# High-temperature preset — for comparison/demo only, NOT for production.
# Shows how output drifts and embellishes at higher randomness.
# Never use this for grounded wealth advice.
HIGH_TEMPERATURE = ModelParams(
    name        = "high_temperature",
    description = (
        "DEMO ONLY — shows how output varies and drifts at high temperature. "
        "Never use for grounded wealth advice."
    ),
    temperature = 1.0,
    max_tokens  = 400,
    top_p       = 1.0,
    stop        = [],
)

# Low top_p — alternative to temperature for tighter nucleus sampling.
# top_p=0.1 means only tokens comprising the top 10% of probability mass
# are considered — very focused output without setting temperature=0.
LOW_TOP_P = ModelParams(
    name        = "low_top_p",
    description = (
        "Tight nucleus sampling via top_p=0.1. "
        "Alternative to temperature=0 — tune one, not both. "
        "Produces focused, high-probability-token output."
    ),
    temperature = 1.0,   # Leave temperature at default when tuning top_p
    max_tokens  = 400,
    top_p       = 0.1,
    stop        = [],
)


# Registry — all presets by name for easy lookup
PRESETS: dict[str, ModelParams] = {
    p.name: p
    for p in [RAG_GROUNDED, RAG_STRICT, RAG_JSON, HIGH_TEMPERATURE, LOW_TOP_P]
}


def get_preset(name: str) -> ModelParams:
    """
    Retrieve a named parameter preset.

    Args:
        name: One of 'rag_grounded', 'rag_strict', 'rag_json',
              'high_temperature', 'low_top_p'

    Returns:
        ModelParams instance for that preset.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]


def describe_presets() -> None:
    """Print a summary of all available parameter presets."""
    print("\nAvailable Model Parameter Presets:")
    print("-" * 60)
    for preset in PRESETS.values():
        print(f"  {preset.name}")
        print(f"    temperature={preset.temperature}  "
              f"max_tokens={preset.max_tokens}  "
              f"top_p={preset.top_p}  "
              f"stop={preset.stop or 'none'}")
        print(f"    {preset.description}")
        print()
