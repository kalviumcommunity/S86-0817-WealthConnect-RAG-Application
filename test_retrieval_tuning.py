import unittest

from retrieval_tuning import choose_best_setting, compare_settings
from similarity_search import VectorCollection


class RetrievalTuningTests(unittest.TestCase):
    def setUp(self):
        self.collection = VectorCollection()
        self.collection.add("policy", [1.0, 0.0], {"source": "policy.md", "type": "policy"})
        self.collection.add("guide", [0.0, 1.0], {"source": "guide.md", "type": "guide"})

        self.vectors = {
            "policy question": [1.0, 0.0],
            "guide question": [0.0, 1.0],
        }

    def embedder(self, texts):
        return [self.vectors[text] for text in texts]

    def test_compare_settings_reports_hits_and_filters(self):
        settings = [
            {"name": "k1", "k": 1, "metadata_filter": None, "min_score": 0.0},
            {"name": "guide_only", "k": 2, "metadata_filter": {"type": "guide"}, "min_score": 0.0},
        ]
        queries = [
            {"query": "policy question", "expected_source": "policy.md"},
            {"query": "guide question", "expected_source": "guide.md"},
        ]

        results = compare_settings(settings, queries, self.collection, self.embedder)

        self.assertEqual(results[0]["hit_rate"], 1.0)
        self.assertEqual(results[1]["hit_rate"], 0.5)
        self.assertEqual(results[1]["details"][0]["returned_sources"], ["guide.md"])

    def test_score_threshold_can_remove_results(self):
        settings = [{"name": "strict", "k": 2, "metadata_filter": None, "min_score": 1.1}]
        queries = [{"query": "policy question", "expected_source": "policy.md"}]

        results = compare_settings(settings, queries, self.collection, self.embedder)

        self.assertEqual(results[0]["hit_rate"], 0.0)
        self.assertEqual(results[0]["details"][0]["returned_sources"], [])

    def test_best_setting_prefers_hit_rate_then_smaller_k(self):
        results = [
            {"setting": "k3", "hit_rate": 1.0},
            {"setting": "k1", "hit_rate": 1.0},
        ]
        settings = [
            {"name": "k3", "k": 3}, {"name": "k1", "k": 1}
        ]

        self.assertEqual(choose_best_setting(results, settings)["setting"], "k1")


if __name__ == "__main__":
    unittest.main()