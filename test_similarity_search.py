import unittest

from similarity_search import VectorCollection, cosine_similarity, rank_by_similarity


class SimilarityRankingTests(unittest.TestCase):
    def test_cosine_similarity_scores_direction(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_rank_by_similarity_returns_descending_top_k(self):
        records = [
            {"text": "off topic", "embedding": [0, 1], "metadata": {"id": 1}},
            {"text": "best match", "embedding": [1, 0], "metadata": {"id": 2}},
            {"text": "partial match", "embedding": [1, 1], "metadata": {"id": 3}},
        ]

        ranked = rank_by_similarity([1, 0], records, top_k=2)

        self.assertEqual([item["text"] for item in ranked], ["best match", "partial match"])
        self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])

    def test_collection_search_and_invalid_dimensions(self):
        collection = VectorCollection()
        collection.add("match", [1, 0], {"source": "policy.md"})

        self.assertEqual(collection.search([1, 0], top_k=1)[0]["text"], "match")
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 0])
        with self.assertRaises(ValueError):
            collection.search([1, 0], top_k=-1)


if __name__ == "__main__":
    unittest.main()