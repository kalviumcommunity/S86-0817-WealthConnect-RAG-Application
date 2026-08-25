"""
Centralized storage for all system and user prompt templates.
Keep logic out of this file.
"""

class PromptTemplates:
    
    # Task 1: Define a template with named placeholders
    RAG_QA_PROMPT = (
        "You are a helpful financial assistant.\n"
        "Use the provided context to answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Please provide a structured and concise answer."
    )

    SUMMARIZATION_PROMPT = (
        "Please summarize the following financial document in {max_sentences} sentences:\n\n"
        "{document_text}"
    )

    @staticmethod
    def render(template_str: str, **kwargs) -> str:
        """
        Task 1: Render function that fills the placeholders.
        """
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
