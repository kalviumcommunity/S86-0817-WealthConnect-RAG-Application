"""
Centralized storage for all system and user prompt templates.
Keep logic out of this file.
"""

class PromptTemplates:

    RAG_QA_PROMPT = (
        "You are a helpful financial assistant.\n"
        "Use the provided context to answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Please provide a structured and concise answer."
    )

    RAG_USER_MESSAGE = (
        "The following context has been retrieved from the bank's current "
        "approved wealth-management documents. Use ONLY this context to answer.\n\n"
        "--- DOCUMENT CONTEXT START ---\n{context}\n"
        "--- DOCUMENT CONTEXT END ---\n\n"
        "Relationship Manager Question: {question}"
    )

    SUMMARIZATION_PROMPT = (
        "Please summarize the following financial document in {max_sentences} sentences:\n\n"
        "{document_text}"
    )

    @staticmethod
    def render(template_str: str, **kwargs) -> str:
        """
        Render a named-placeholder template with runtime values.
        """
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
