"""
Unit tests for Milestone 3.33 — Metadata Filtering & Hybrid Search
"""

import unittest
from hybrid_search import BM25Index, HybridSearchEngine, tokenize


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {
                "text": "Long-term capital gains tax exemption for municipal bonds and clean energy.",
                "metadata": {"document_name": "tax_rules.md", "document_type": "tax_rules", "status": "approved"},
            },
            {
                "text": "Corporate wealth accounts standard 15 percent surtax with no threshold relief.",
                "metadata": {"document_name": "tax_rules.md", "document_type": "tax_rules", "status": "approved"},
            },
            {
                "text": "Moderate Growth Fund requires minimum investment of $10,000.",
                "metadata": {"document_name": "portfolio.txt", "document_type": "investment_policy", "status": "approved"},
            },
            {
                "text": "Draft draft experimental fund without executive approval.",
                "metadata": {"document_name": "draft.txt", "document_type": "investment_policy", "status": "draft"},
            },
        ]
        self.engine = HybridSearchEngine()
        # Mock deterministic embedder for dense vectors
        def mock_embedder(texts):
            vectors = []
            for t in texts:
                # 4-dimensional simple mock vector
                v = [
                    1.0 if "tax" in t.lower() or "capital" in t.lower() else 0.0,
                    1.0 if "corporate" in t.lower() or "surtax" in t.lower() else 0.0,
                    1.0 if "growth" in t.lower() or "investment" in t.lower() else 0.0,
                    1.0 if "draft" in t.lower() else 0.0,
                ]
                vectors.append(v)
            return vectors

        self.engine.index_corpus(self.corpus, embedder=mock_embedder)

    def test_bm25_tokenization_and_scoring(self):
        bm25 = BM25Index()
        bm25.fit(self.corpus)
        scores = bm25.score("capital gains tax")
        self.assertEqual(len(scores), len(self.corpus))
        # First document should have highest BM25 score
        self.assertEqual(scores.index(max(scores)), 0)
        self.assertGreater(scores[0], 0.0)

    def test_hybrid_search_ranks_relevant_document_first(self):
        query = "tax exemption on capital gains"
        query_vector = [1.0, 0.0, 0.0, 0.0]
        results = self.engine.search(query=query, query_vector=query_vector, top_k=2)

        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("capital gains", top["text"].lower())
        self.assertGreater(top["score"], 0.0)

    def test_metadata_filtering_excludes_drafts(self):
        query = "fund"
        query_vector = [0.0, 0.0, 1.0, 1.0]
        # Filter for approved documents only
        results = self.engine.search(
            query=query,
            query_vector=query_vector,
            top_k=5,
            metadata_filter={"status": "approved"},
        )

        statuses = [r["metadata"]["status"] for r in results]
        self.assertTrue(all(s == "approved" for s in statuses))
        self.assertNotIn("draft", [r["metadata"]["status"] for r in results])

    def test_metadata_filtering_by_document_type(self):
        results = self.engine.search(
            query="policy and tax",
            query_vector=[1.0, 0.0, 1.0, 0.0],
            top_k=5,
            metadata_filter={"document_type": "tax_rules"},
        )
        self.assertTrue(all(r["metadata"]["document_type"] == "tax_rules" for r in results))


if __name__ == "__main__":
    unittest.main()
