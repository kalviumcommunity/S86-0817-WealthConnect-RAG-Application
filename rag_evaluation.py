"""End-to-end answer quality evaluation for the WealthConnect RAG pipeline."""

import re
from collections.abc import Callable

from rag_pipeline import FALLBACK_ANSWER, answer_query, offline_embed
from similarity_search import build_collection


TEST_SET = [
    {
        "question": "What was the Q4 portfolio return?",
        "expected_points": ["12% return", "aggressive growth fund"],
        "expected_sources": {"q4_earnings_report.pdf"},
    },
    {
        "question": "How do I request a refund?",
        "expected_points": ["refund requests", "30 days"],
        "expected_sources": {"policies.md"},
    },
    {
        "question": "What is WealthConnect?",
        "expected_points": ["financial advisory platform", "investing"],
        "expected_sources": {"about.html"},
    },
]


def score_correctness(answer: str, expected_points: list[str]) -> float:
    """Return the fraction of expected answer points present in the answer."""
    if not expected_points:
        return 1.0
    normalized = answer.casefold()
    matched = sum(point.casefold() in normalized for point in expected_points)
    return matched / len(expected_points)


def score_grounding(answer: str, citations: dict[str, dict]) -> float:
    """Score whether answer markers resolve to non-empty retrieved evidence."""
    if answer == FALLBACK_ANSWER and not citations:
        return 1.0
    markers = re.findall(r"\[\d+\]", answer)
    if not markers or any(marker not in citations for marker in markers):
        return 0.0
    return float(all(citations[marker].get("text", "").strip() for marker in markers))


def score_citation_accuracy(
    citations: dict[str, dict],
    expected_sources: set[str],
) -> float:
    """Return 1 when expected sources are represented, otherwise 0."""
    actual_sources = {citation.get("source") for citation in citations.values()}
    if not expected_sources:
        return float(not actual_sources)
    return float(expected_sources.issubset(actual_sources))


def score_example(
    example: dict,
    answerer: Callable[[str], dict],
) -> dict:
    """Run one labeled question and score its three RAG quality dimensions."""
    result = answerer(example["question"])
    citations = result.get("citations", {})
    row = {
        "question": example["question"],
        "answer": result.get("answer", ""),
        "correctness": score_correctness(
            result.get("answer", ""), example["expected_points"]
        ),
        "grounding": score_grounding(result.get("answer", ""), citations),
        "citation_accuracy": score_citation_accuracy(
            citations, example["expected_sources"]
        ),
        "citations": citations,
    }
    row["passed"] = min(
        row["correctness"], row["grounding"], row["citation_accuracy"]
    ) == 1.0
    return row


def summarize_results(rows: list[dict]) -> dict:
    """Aggregate dimension averages and retain examples needing attention."""
    if not rows:
        return {
            "questions": 0,
            "avg_correctness": 0.0,
            "avg_grounding": 0.0,
            "avg_citation_accuracy": 0.0,
            "failures": [],
        }

    count = len(rows)
    summary = {
        "questions": count,
        "avg_correctness": sum(row["correctness"] for row in rows) / count,
        "avg_grounding": sum(row["grounding"] for row in rows) / count,
        "avg_citation_accuracy": sum(row["citation_accuracy"] for row in rows) / count,
        "failures": [row for row in rows if not row["passed"]],
    }
    return summary


def evaluate_test_set(
    test_set: list[dict],
    answerer: Callable[[str], dict],
) -> dict:
    """Score all examples and return rows plus an aggregate summary."""
    rows = [score_example(example, answerer) for example in test_set]
    return {"rows": rows, "summary": summarize_results(rows)}


def main() -> None:
    collection = build_collection(dry_run=True)

    def answerer(question: str) -> dict:
        return answer_query(question, collection, offline_embed, k=3)

    report = evaluate_test_set(TEST_SET, answerer)
    summary = report["summary"]
    print(f"Questions evaluated: {summary['questions']}")
    print(f"Average correctness: {summary['avg_correctness']:.2f}")
    print(f"Average grounding: {summary['avg_grounding']:.2f}")
    print(f"Average citation accuracy: {summary['avg_citation_accuracy']:.2f}")
    print(f"Failures: {len(summary['failures'])}")
    for failure in summary["failures"]:
        print(f"  - {failure['question']}")


if __name__ == "__main__":
    main()