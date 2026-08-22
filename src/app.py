"""
app.py — Main conversational entry point for WealthConnect RAG

Orchestrates the full RAG pipeline:
  1. Accept a natural-language question from a Relationship Manager
  2. Retrieve relevant approved document chunks
  3. Build a structured prompt via prompt_builder (system + user roles)
  4. Call the LLM
  5. Return the answer with source citations

Secrets are loaded from .env via python-dotenv — never hard-coded here.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval import retrieve, format_context, build_sources_list
from src.prompt_builder import load_system_prompt, build_user_message, parse_json_response

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
) -> dict:
    """
    Main RAG function — takes a Relationship Manager's question and returns
    a grounded answer with source references.

    Args:
        question       : Natural-language question from the RM.
        n_results      : Number of document chunks to retrieve.
        prompt_variant : Prompt style — 'strict' (default), 'json', or 'concise'.

    Returns:
        {
            "question"       : str,
            "answer"         : str,
            "sources"        : list[str],
            "prompt_variant" : str,
        }
    """
    # Step 1: Retrieve relevant approved chunks from the vector store
    results = retrieve(question, n_results=n_results)

    # Step 2: Build formatted context block and human-readable sources list
    context = format_context(results)
    sources = build_sources_list(results)

    # Step 3: Build system and user messages using prompt_builder
    #   system role  → who the assistant is + grounding rules (loaded from prompts/)
    #   user role    → the RM's question + retrieved context for this turn
    system_prompt = load_system_prompt(prompt_variant)
    user_message  = build_user_message(question, context)

    # Step 4: Call the LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,  # Deterministic — grounding requires consistency
    )

    raw_answer = response.choices[0].message.content.strip()

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
    }


def print_response(result: dict) -> None:
    """Pretty-print the RAG response to the terminal."""
    print("\n" + "=" * 60)
    print(f"Prompt Variant : {result['prompt_variant']}")
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
    # Smoke-test with all three prompt variants
    sample_question = "What are the tax implications of the ABC Investment Product?"

    for variant in ["strict", "concise", "json"]:
        result = ask(sample_question, prompt_variant=variant)
        print_response(result)
