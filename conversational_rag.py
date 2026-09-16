"""
GY3.42 — Conversational RAG & Follow-Up Context
WealthConnect RAG Application

Handles multi-turn relationship manager dialogues:
1. Maintains conversation history within context window limits.
2. Rewrites ambiguous follow-up questions into standalone search queries
   using conversational context.
3. Dispatches the condensed query to the RAG retrieval pipeline.
"""

import os
import re
from dotenv import load_dotenv
from chat_history_manager import count_tokens

load_dotenv()


def condense_followup_question(
    question: str,
    history: list[dict],
    chat_model: str = "gpt-4o-mini",
    client=None,
) -> str:
    """
    Rewrite a follow-up question into an independent, standalone search query
    that contains all necessary entities and context from previous turns.

    Args:
        question   : The user's latest follow-up question (e.g. "What is the tax rate?").
        history    : List of prior turns [{'role': 'user'|'assistant', 'content': '...'}]
        chat_model : Model identifier for rewriting.
        client     : Optional OpenAI client instance.

    Returns:
        Standalone search query string.
    """
    if not history:
        return question.strip()

    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    if not user_turns:
        return question.strip()

    last_user_query = user_turns[-1]

    # If OpenAI client is available, use LLM for query rewriting
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and client is None:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                max_retries=0,
            )
        except Exception:
            client = None

    if client is not None:
        prompt = (
            "Given the following chat history and a follow-up question from a bank relationship manager, "
            "rephrase the follow-up question to be a standalone, self-contained search query. "
            "Do not answer the question; only return the rephrased search query.\n\n"
            f"Chat History:\n"
            + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]])
            + f"\n\nFollow-up Question: {question}\n"
            "Standalone Search Query:"
        )
        try:
            response = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            condensed = response.choices[0].message.content.strip()
            # If the model added quotes, clean them
            return condensed.strip("\"'")
        except Exception:
            pass

    # Deterministic offline heuristic for standalone query condensation
    q_lower = question.lower()
    q_words = question.strip().split()
    pronouns = ["it", "this", "that", "these", "those", "they", "them", "the fund", "the policy"]
    followup_stems = [
        "what about", "and for", "how about", "what is the", "what are the",
        "is there a", "are there any", "how much", "who is eligible", "tell me more",
    ]
    is_followup = (
        any(re.search(r"\b" + re.escape(p) + r"\b", q_lower) for p in pronouns)
        or any(q_lower.startswith(stem) for stem in followup_stems)
        or len(q_words) <= 6
    )

    if is_followup:
        # Extract core subject from the previous turn
        clean_last = re.sub(
            r"^(what is|tell me about|how does|what are|can you explain|explain|describe)\s+",
            "",
            last_user_query,
            flags=re.IGNORECASE,
        ).rstrip("?. ")
        if clean_last and not any(w in q_lower for w in clean_last.lower().split() if len(w) > 3):
            return f"{question.rstrip('?. ')} regarding {clean_last}"

    return question.strip()


class ConversationalRAGSession:
    """
    Session-level manager for conversational RAG queries.
    Stores multi-turn messages, condenses follow-up queries, and invokes
    the underlying answer pipeline.
    """
    def __init__(self, system_prompt: str, max_tokens: int = 4000):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def ask(self, question: str, rag_func) -> dict:
        """
        Process a user question in conversation context.

        Args:
            question : Relationship manager's query
            rag_func : Callable(condensed_query: str) -> dict with 'answer', 'sources'

        Returns:
            RAG result dict including original question and condensed query.
        """
        # Step 1: Condense follow-up using previous turns
        prior_turns = [m for m in self.history if m["role"] in ("user", "assistant")]
        search_query = condense_followup_question(question, prior_turns)

        # Step 2: Run RAG retrieval & answer generation with the condensed query
        result = rag_func(search_query)

        # Step 3: Record conversation history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result.get("answer", "")})

        # Step 4: Budget conversation history tokens
        while count_tokens(self.history) > self.max_tokens and len(self.history) > 2:
            # Drop the oldest user/assistant pair (preserve system prompt)
            self.history.pop(1)
            if len(self.history) > 1 and self.history[1]["role"] == "assistant":
                self.history.pop(1)

        result["original_question"] = question
        result["condensed_query"] = search_query
        return result
