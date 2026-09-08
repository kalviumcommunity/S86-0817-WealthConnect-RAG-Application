"""Offline retrieval evaluation and parameter tuning for WealthConnect."""

from similarity_search import VectorCollection, build_collection, embed


TEST_QUERIES = [
    {
        "query": "How can I reset my WealthConnect password?",
        "expected_source": "policies.md",
    },
    {
        "query": "What was the Q4 portfolio return?",
        "expected_source": "q4_earnings_report.pdf",
    },
    {
        "query": "How do I request a refund?",
        "expected_source": "policies.md",
    },
    {
        "query": "What is WealthConnect?",
        "expected_source": "about.html",
    },
]

SETTINGS = [
    {"name": "baseline_k1", "k": 1, "metadata_filter": None, "min_score": 0.0},
    {"name": "baseline_k3", "k": 3, "metadata_filter": None, "min_score": 0.0},
    {"name": "filtered_k3", "k": 3, "metadata_filter": {"source": "policies.md"}, "min_score": 0.0},
]


def offline_embed(texts: list[str]) -> list[list[float]]:
    """Use the repository's deterministic vectors for local evaluation."""
    return embed(texts, dry_run=True)


def retrieve_with_settings(
    query: str,
    collection: VectorCollection,
    setting: dict,
    embedder=offline_embed,
) -> list[dict]:
    """Retrieve chunks using one evaluated setting."""
    query_vector = embedder([query])[0]
    return collection.search(
        query_vector,
        top_k=setting["k"],
        metadata_filter=setting.get("metadata_filter"),
        min_score=setting.get("min_score", 0.0),
    )


def evaluate_setting(
    setting: dict,
    test_queries: list[dict],
    collection: VectorCollection,
    embedder=offline_embed,
) -> dict:
    """Measure source hit rate and return per-query evidence."""
    rows = []
    for item in test_queries:
        results = retrieve_with_settings(item["query"], collection, setting, embedder)
        sources = [result["metadata"].get("source") for result in results]
        rows.append({
            "query": item["query"],
            "expected_source": item["expected_source"],
            "returned_sources": sources,
            "top_score": results[0]["score"] if results else 0.0,
            "hit": item["expected_source"] in sources,
        })

    hits = sum(row["hit"] for row in rows)
    return {
        "setting": setting["name"],
        "hit_rate": hits / len(rows) if rows else 0.0,
        "hits": hits,
        "queries": len(rows),
        "details": rows,
    }


def compare_settings(
    settings: list[dict],
    test_queries: list[dict] = TEST_QUERIES,
    collection: VectorCollection | None = None,
    embedder=offline_embed,
) -> list[dict]:
    """Evaluate every setting against the same query set."""
    if collection is None:
        collection = build_collection(dry_run=True)
    return [evaluate_setting(setting, test_queries, collection, embedder) for setting in settings]


def choose_best_setting(results: list[dict], settings: list[dict] = SETTINGS) -> dict:
    """Choose the highest-hit-rate setting, preferring smaller k on ties."""
    if not results:
        raise ValueError("results must not be empty")
    sizes = {setting["name"]: setting["k"] for setting in settings}
    return max(results, key=lambda result: (
        result["hit_rate"],
        -sizes[result["setting"]],
    ))


def main() -> None:
    results = compare_settings(SETTINGS)
    best = choose_best_setting(results)
    lines = ["Retrieval Relevance Tuning", "=" * 30]
    for result in results:
        lines.append(
            f"{result['setting']}: hit_rate={result['hit_rate']:.2f} "
            f"({result['hits']}/{result['queries']})"
        )
    lines.append(f"Selected setting: {best['setting']}")
    lines.append("Selection basis: highest measured hit rate; ties prefer smaller k.")
    output = "\n".join(lines)
    print(output)
    with open("retrieval_tuning_results.txt", "w", encoding="utf-8") as file:
        file.write(output + "\n")


if __name__ == "__main__":
    main()