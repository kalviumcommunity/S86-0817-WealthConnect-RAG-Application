import unittest

from src.embeddings import chunk_record_id, index_chunks_verified, to_vector_record


class FakeCollection:
    def __init__(self):
        self.records = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for record_id, vector, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            self.records[record_id] = {
                "embedding": vector,
                "document": document,
                "metadata": metadata,
            }

    def count(self):
        return len(self.records)

    def get(self, ids, include):
        record = self.records[ids[0]]
        return {
            "ids": ids,
            "documents": [record["document"]],
            "metadatas": [record["metadata"]],
            "embeddings": [record["embedding"]],
        }


class EmbeddingIndexTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "text": "Refunds are available within 30 days.",
                "metadata": {"document_name": "policy.md", "chunk_index": 0},
            },
            {
                "text": "Password resets require identity verification.",
                "metadata": {"document_name": "policy.md", "chunk_index": 1},
            },
        ]

    def test_records_have_stable_ids_and_metadata(self):
        vector = [0.1, 0.2, 0.3]
        record = to_vector_record(self.chunks[0], vector)

        self.assertEqual(record["id"], chunk_record_id(self.chunks[0]))
        self.assertEqual(record["text"], self.chunks[0]["text"])
        self.assertEqual(record["metadata"], self.chunks[0]["metadata"])
        self.assertEqual(record["vector"], vector)

    def test_batch_indexing_verifies_count_and_spot_check(self):
        collection = FakeCollection()

        report = index_chunks_verified(
            self.chunks,
            batch_size=1,
            collection=collection,
            embedder=lambda texts: [[float(index), 1.0] for index, _ in enumerate(texts)],
        )

        self.assertEqual(report["expected_count"], 2)
        self.assertEqual(report["inserted_count"], 2)
        self.assertEqual(report["indexed_count"], 2)
        self.assertTrue(report["count_matches"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["spot_check"]["vector_dimensions"], 2)

    def test_failed_batch_is_reported(self):
        report = index_chunks_verified(
            self.chunks,
            collection=FakeCollection(),
            embedder=lambda texts: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
        )

        self.assertEqual(report["inserted_count"], 0)
        self.assertEqual(len(report["failures"]), 1)
        self.assertFalse(report["count_matches"])


if __name__ == "__main__":
    unittest.main()