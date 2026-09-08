"""Stage-separated, testable RAG pipeline orchestration."""

from collections.abc import Callable
import re

from similarity_search import VectorCollection, build_collection, embed


FALLBACK_ANSWER = "I could not find relevant context for that question."


def build_citation_map(chunks: list[dict]) -> dict[str, dict]:
    """Map each displayed marker to the exact retrieved evidence behind it."""
    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        citation_map[f"[{index}]"] = {
            "source": metadata.get("source", metadata.get("document_name", "Unknown")),
            "chunk_id": metadata.get("chunk_id", chunk.get("id")),
            "chunk_index": metadata.get("chunk_index"),
            "section": metadata.get("section"),
            "text": chunk["text"],
        }
    return citation_map


def validate_citations(answer: str, citation_map: dict[str, dict]) -> bool:
    """Return whether every citation marker in an answer maps to evidence."""
    markers = re.findall(r"\[\d+\]", answer)
    return bool(markers) and all(marker in citation_map for marker in markers)


def build_cited_prompt(question: str, chunks: list[dict]) -> str:
    """Build generation instructions that constrain citations to retrieved chunks."""
    context = assemble_context(chunks)
    return (
        "Answer using only the context below. Cite every factual claim using "
        "source markers like [1] or [2]. Only use markers that appear in the "
        "context. If the context does not support an answer, say you do not "
        "have enough information and do not invent citations.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


def embed_query(
    query: str,
    embedder: Callable[[list[str]], list[list[float]]],
) -> list[float]:
    """Embed exactly one user query with the same model as the corpus."""
    if not query.strip():
        raise ValueError("query must not be empty")
    vectors = embedder([query])
    if len(vectors) != 1:
        raise ValueError("query embedder must return exactly one vector")
    return vectors[0]


def retrieve_context(
    query_vector: list[float],
    collection: VectorCollection,
    k: int = 4,
    metadata_filter: dict | None = None,
    min_score: float = float("-inf"),
) -> list[dict]:
    """Retrieve ranked evidence without knowing anything about generation."""
    return collection.search(
        query_vector,
        top_k=k,
        metadata_filter=metadata_filter,
        min_score=min_score,
    )


def assemble_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with numbered citation markers."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        source = metadata.get("source", metadata.get("document_name", "Unknown"))
        section = metadata.get("section")
        section_label = f" ({section})" if section else ""
        parts.append(f"[{index}] Source: {source}{section_label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    generator: Callable[[str, str], str] | None = None,
) -> str:
    """Generate from assembled context, or provide an offline grounded answer."""
    if not context:
        return FALLBACK_ANSWER
    if generator is not None:
        return generator(query, context)
    if "Context:\n" in context:
        context = context.split("Context:\n", 1)[1]
        context = context.split("\n\nQuestion:", 1)[0]
    first_evidence = context.split("\n\n---", 1)[0]
    return f"Based on the provided context: {first_evidence}"


def answer_query(
    query: str,
    collection: VectorCollection,
    embedder: Callable[[list[str]], list[list[float]]],
    generator: Callable[[str, str], str] | None = None,
    k: int = 4,
    metadata_filter: dict | None = None,
    min_score: float = float("-inf"),
) -> dict:
    """Run embed -> retrieve -> cite -> generate and verify attribution."""
    query_vector = embed_query(query, embedder)
    chunks = retrieve_context(
        query_vector,
        collection,
        k=k,
        metadata_filter=metadata_filter,
        min_score=min_score,
    )
    context = assemble_context(chunks)
    citation_map = build_citation_map(chunks)
    if not chunks:
        answer = FALLBACK_ANSWER
    else:
        generation_context = build_cited_prompt(query, chunks)
        answer = generate_answer(query, generation_context, generator)
    citations_valid = not chunks or validate_citations(answer, citation_map)
    if not citations_valid:
        answer = FALLBACK_ANSWER
        citation_map = {}
    return {
        "query": query,
        "answer": answer,
        "sources": list(citation_map.values()),
        "citations": citation_map,
        "citations_valid": citations_valid,
        "context": context,
        "retrieved_count": len(chunks),
    }


def offline_embed(texts: list[str]) -> list[list[float]]:
    """Use deterministic repository vectors for the local demo."""
    return embed(texts, dry_run=True)


def main() -> None:
    collection = build_collection(dry_run=True)
    result = answer_query(
        "What was the Q4 portfolio return?",
        collection,
        embedder=offline_embed,
        k=3,
    )
    print(result["answer"])
    print("Sources:", result["sources"])


if __name__ == "__main__":
    main()