"""Stage-separated, testable RAG pipeline orchestration."""

from collections.abc import Callable

from similarity_search import VectorCollection, build_collection, embed


FALLBACK_ANSWER = "I could not find relevant context for that question."


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
    first_evidence = context.split("\n", 1)[-1].split("\n\n---", 1)[0]
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
    """Run embed -> retrieve -> assemble -> generate and return sources."""
    query_vector = embed_query(query, embedder)
    chunks = retrieve_context(
        query_vector,
        collection,
        k=k,
        metadata_filter=metadata_filter,
        min_score=min_score,
    )
    context = assemble_context(chunks)
    answer = generate_answer(query, context, generator)
    sources = [chunk["metadata"] for chunk in chunks]
    return {
        "query": query,
        "answer": answer,
        "sources": sources if context else [],
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