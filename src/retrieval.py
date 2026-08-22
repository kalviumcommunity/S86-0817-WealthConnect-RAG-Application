"""
retrieval.py — Retrieval layer for WealthConnect RAG

Searches the vector store for chunks relevant to a Relationship Manager's
question, applies metadata filters (approved + current versions only),
and returns ranked results with full source attribution.
"""

from src.embeddings import query_collection


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve the top-n approved document chunks most relevant to the question.

    Returns a list of result dicts, each containing:
        - text      : the chunk content
        - metadata  : document name, version, approval status, etc.
        - distance  : cosine distance (lower = more similar)
    """
    results = query_collection(
        query_text=question,
        n_results=n_results,
        approval_status="approved",
    )

    if not results:
        print("[retrieval] No relevant approved documents found.")

    return results


def format_context(results: list[dict]) -> str:
    """
    Concatenate retrieved chunks into a single context block that will be
    passed to the LLM prompt. Each chunk is prefixed with its source so the
    model can cite it in the answer.
    """
    if not results:
        return ""

    context_parts = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        source_label = (
            f"[Source {i}] {meta.get('document_name', 'Unknown')} "
            f"| Version: {meta.get('version', 'N/A')} "
            f"| Type: {meta.get('document_type', 'N/A')}"
        )
        context_parts.append(f"{source_label}\n{r['text']}")

    return "\n\n".join(context_parts)


def build_sources_list(results: list[dict]) -> list[str]:
    """
    Build a human-readable list of source references to display
    alongside the answer in the UI (satisfies FR-06).
    """
    sources = []
    for r in results:
        meta = r["metadata"]
        source = (
            f"{meta.get('document_name', 'Unknown Document')} "
            f"— Version: {meta.get('version', 'N/A')} "
            f"| Approved | "
            f"Effective: {meta.get('effective_date', 'N/A')}"
        )
        sources.append(source)
    return sources
