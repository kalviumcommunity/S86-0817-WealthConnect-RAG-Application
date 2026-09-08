import unittest

from rag_evaluation import (
    score_correctness,
    score_example,
    score_grounding,
    summarize_results,
)


class RagEvaluationTests(unittest.TestCase):
    def test_correctness_scores_expected_point_coverage(self):
        score = score_correctness("The return was 12% for the fund.", ["12%", "fund"])

        self.assertEqual(score, 1.0)
        self.assertEqual(score_correctness("Only 12%.", ["12%", "fund"]), 0.5)

    def test_grounding_rejects_unknown_markers(self):
        citations = {"[1]": {"source": "policy.md", "text": "Refunds."}}

        self.assertEqual(score_grounding("Refunds [1].", citations), 1.0)
        self.assertEqual(score_grounding("Refunds [2].", citations), 0.0)

    def test_score_example_and_summary_keep_failure_details(self):
        example = {
            "question": "How do I request a refund?",
            "expected_points": ["refund", "30 days"],
            "expected_sources": {"policy.md"},
        }

        def answerer(question):
            return {
                "answer": "Refunds are available [1].",
                "citations": {
                    "[1]": {"source": "other.md", "text": "Refunds."}
                },
            }

        row = score_example(example, answerer)
        summary = summarize_results([row])

        self.assertEqual(row["correctness"], 0.5)
        self.assertEqual(row["grounding"], 1.0)
        self.assertEqual(row["citation_accuracy"], 0.0)
        self.assertFalse(row["passed"])
        self.assertEqual(len(summary["failures"]), 1)


if __name__ == "__main__":
    unittest.main()