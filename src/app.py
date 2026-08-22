"""
app.py — Main conversational entry point for WealthConnect RAG

Orchestrates the full RAG pipeline:
  1. Accept a natural-language question from a Relationship Manager
  2. Retrieve relevant approved document chunks
  3. Build a structured prompt via prompt_builder (system + user roles)
  4. Apply model parameters via model_params (temperature, max_tokens, etc.)
  5. Call the LLM
  6. Return the answer with source citations and token usage

Secrets are loaded from .env via python-dotenv — never hard-coded here.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval import retrieve, format_context, build_sources_list
from src.prompt_builder import load_system_prompt, build_user_message, parse_json_response
from src.model_params import ModelParams, RAG_GROUNDED, RAG_JSON, get_preset

load_dotenv()

# ---------------------------------------------------------------------------
# Client — credentials from environment, not source code
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def ask(
    question: str,
    n_results: int = 5,
    prompt_variant: str = "strict",
    params: ModelParams | None = None,
) -> dict:
    """
    Main RAG function — takes a Relationship Manager's question and returns
    a grounded answer with source references and token usage.

    Args:
        question       : Natural-language question from the RM.
        n_results      : Number of document chunks to retrieve.
        prompt_variant : Prompt style — 'strict' (default), 'json', or 'concise'.
        params         : ModelParams instance controlling temperature, max_tokens,
                         top_p, and stop sequences. Defaults to RAG_GROUNDED preset
                         (temperature=0.0, max_tokens=400) — the correct setting
                         for a grounded, factual wealth assistant.

    Returns:
        {
            "question"        : str,
            "answer"          : str,
            "sources"         : list[str],
            "prompt_variant"  : str,
            "model_params"    : dict,   # parameters used for this call
            "usage"           : dict,   # token counts from the API response
        }
    """
    # Default to the production RAG preset if no params supplied.
    # JSON variant gets its own preset for the slightly larger token budget.
    if params is None:
        params = RAG_JSON if prompt_variant == "json" else RAG_GROUNDED

    # Step 1: Retrieve relevant approved chunks from the vector store
    results = retrieve(question, n_results=n_results)

    # Step 2: Build formatted context block and human-readable sources list
    context = format_context(results)
    sources = build_sources_list(results)

    # Step 3: Build system and user messages
    #   system role → who the assistant is + grounding rules (from prompts/)
    #   user role   → the RM's question + retrieved context for this turn
    system_prompt = load_system_prompt(prompt_variant)
    user_message  = build_user_message(question, context)

    # Step 4: Call the LLM — model parameters applied via to_api_kwargs()
    #   temperature, max_tokens, top_p, stop are all controlled here.
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        **params.to_api_kwargs(),
    )

    raw_answer    = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    # Step 5: If JSON variant was used, parse the structured output
    if prompt_variant == "json":
        parsed = parse_json_response(raw_answer)
        answer = parsed.get("answer", raw_answer)
    else:
        answer = raw_answer

    return {
        "question":       question,
        "answer":         answer,
        "sources":        sources,
        "prompt_variant": prompt_variant,
        "finish_reason":  finish_reason,
        "model_params": {
            "temperature": params.temperature,
            "max_tokens":  params.max_tokens,
            "top_p":       params.top_p,
            "stop":        params.stop,
            "preset_name": params.name,
        },
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
    }


def print_response(result: dict) -> None:
    """Pretty-print the RAG response to the terminal."""
    mp = result["model_params"]
    u  = result["usage"]
    print("\n" + "=" * 60)
    print(f"Prompt Variant : {result['prompt_variant']}")
    print(f"Preset         : {mp['preset_name']}")
    print(f"Params         : temperature={mp['temperature']}  "
          f"max_tokens={mp['max_tokens']}  top_p={mp['top_p']}")
    print(f"Tokens         : prompt={u['prompt_tokens']}  "
          f"completion={u['completion_tokens']}  total={u['total_tokens']}")
    print(f"Finish reason  : {result['finish_reason']}")
    print(f"Question       : {result['question']}")
    print("-" * 60)
    print(f"Answer:\n  {result['answer']}")
    print("-" * 60)
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s}")
    else:
        print("Sources: None found in approved documents.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    sample_question = "What are the tax implications of the ABC Investment Product?"

    # Smoke-test: strict variant with production grounded preset
    result = ask(sample_question, prompt_variant="strict")
    print_response(result)

    # Smoke-test: json variant with json preset
    result = ask(sample_question, prompt_variant="json")
    print_response(result)

    # Smoke-test: concise variant with a custom tight preset
    from src.model_params import RAG_STRICT
    result = ask(sample_question, prompt_variant="concise", params=RAG_STRICT)
    print_response(result)
