import unittest

from rag_pipeline import (
    FALLBACK_ANSWER,
    answer_query,
    assemble_context,
    embed_query,
    retrieve_context,
)
from similarity_search import VectorCollection


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        self.collection = VectorCollection()
        self.collection.add(
            "Approved refund policy.",
            [1.0, 0.0],
            {"source": "policy.md", "section": "Refunds"},
        )
        self.collection.add(
            "Cafeteria menu.",
            [0.0, 1.0],
            {"source": "campus.md", "section": "Food"},
        )
        self.vectors = {
            "refund question": [1.0, 0.0],
            "empty question": [0.0, 1.0],
        }

    def embedder(self, texts):
        return [self.vectors[text] for text in texts]

    def test_stages_embed_retrieve_and_assemble_citations(self):
        vector = embed_query("refund question", self.embedder)
        chunks = retrieve_context(vector, self.collection, k=1)
        context = assemble_context(chunks)

        self.assertEqual(vector, [1.0, 0.0])
        self.assertEqual(chunks[0]["text"], "Approved refund policy.")
        self.assertIn("[1] Source: policy.md (Refunds)", context)

    def test_answer_query_injects_context_into_generator(self):
        captured = {}

        def generator(query, context):
            captured["query"] = query
            captured["context"] = context
            return "Refunds are available."

        result = answer_query(
            "refund question",
            self.collection,
            self.embedder,
            generator=generator,
            k=1,
        )

        self.assertEqual(result["answer"], "Refunds are available.")
        self.assertEqual(result["retrieved_count"], 1)
        self.assertEqual(result["sources"][0]["source"], "policy.md")
        self.assertIn("Approved refund policy.", captured["context"])

    def test_empty_retrieval_returns_fallback_without_generation(self):
        empty_collection = VectorCollection()
        called = False

        def generator(query, context):
            nonlocal called
            called = True
            return "should not be used"

        result = answer_query(
            "empty question",
            empty_collection,
            self.embedder,
            generator=generator,
        )

        self.assertEqual(result["answer"], FALLBACK_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()