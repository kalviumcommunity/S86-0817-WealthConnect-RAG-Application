"""
app.py — Main conversational entry point for WealthConnect RAG

Orchestrates the full RAG pipeline:
  1. Accept a natural-language question from a Relationship Manager
  2. Retrieve relevant approved document chunks
  3. Build a grounded prompt
  4. Call the LLM
  5. Return the answer with source citations

Secrets are loaded from .env via python-dotenv — never hard-coded here.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval import retrieve, format_context, build_sources_list

load_dotenv()

# ---------------------------------------------------------------------------
# Client — credentials from environment, not source code
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Prompt template (kept here; move to prompts/ for team-editable templates)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are WealthConnect, an AI-powered knowledge assistant for a bank's wealth division.

Your role is to help Relationship Managers find accurate, approved information from the bank's 
wealth-management documents including investment policies, product brochures, tax rules, 
eligibility guidelines, and risk documents.

Rules you must always follow:
1. Answer ONLY using the provided document context. Do not use outside knowledge.
2. If the context does not contain enough information, respond with the fallback message exactly.
3. Never invent investment, tax, legal, or policy information.
4. Always reference the source document(s) in your answer.
5. Be concise and professional.

Fallback message (use when context is insufficient):
"I couldn't find enough information in the current approved wealth documents to answer this 
question. Please verify with the appropriate wealth, tax, legal, or compliance team."
"""


def ask(question: str, n_results: int = 5) -> dict:
    """
    Main RAG function — takes a Relationship Manager's question and returns
    a grounded answer with source references.

    Returns:
        {
            "question": str,
            "answer":   str,
            "sources":  list[str]
        }
    """
    # Step 1: Retrieve relevant approved chunks
    results = retrieve(question, n_results=n_results)

    # Step 2: Build context block and sources list
    context = format_context(results)
    sources = build_sources_list(results)

    # Step 3: Build the user message with context injected
    if context:
        user_message = (
            f"Context from approved wealth documents:\n\n{context}\n\n"
            f"Relationship Manager Question: {question}"
        )
    else:
        # No documents retrieved — force the safe fallback
        user_message = (
            f"No relevant approved documents were found.\n\n"
            f"Relationship Manager Question: {question}"
        )

    # Step 4: Call the LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,   # Deterministic — we want consistent, grounded answers
    )

    answer = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


def print_response(result: dict) -> None:
    """Pretty-print the RAG response to the terminal."""
    print("\n" + "=" * 60)
    print(f"Question:\n  {result['question']}")
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
    # Quick smoke-test — replace with your actual question
    sample_question = "What are the tax implications of the ABC Investment Product?"
    result = ask(sample_question)
    print_response(result)
