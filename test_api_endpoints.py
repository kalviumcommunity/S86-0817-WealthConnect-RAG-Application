"""
Integration tests for the unified FastAPI backend in rag_api.py
Tests /health, /metrics, /query (batch + cache), /query/stream, and /upload.
"""

import json
import io
import unittest
from fastapi.testclient import TestClient

from rag_api import app, query_cache


class ApiEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        query_cache.clear()

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["streaming"])
        self.assertGreater(data["chunks"], 0)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_queries", data)
        self.assertIn("cache", data)
        self.assertIn("uptime_seconds", data)

    def test_batch_query_and_caching(self):
        q = "What was the Q4 portfolio return?"
        # First query: cache miss
        r1 = self.client.post("/query", json={"question": q})
        self.assertEqual(r1.status_code, 200)
        data1 = r1.json()
        self.assertEqual(data1["status"], "answered")
        self.assertFalse(data1["cached"])
        self.assertGreater(len(data1["sources"]), 0)

        # Second identical query: cache hit
        r2 = self.client.post("/query", json={"question": q})
        self.assertEqual(r2.status_code, 200)
        data2 = r2.json()
        self.assertEqual(data2["status"], "answered")
        self.assertTrue(data2["cached"])

    def test_guardrail_refusal_for_unrelated_query(self):
        r = self.client.post("/query", json={"question": "What is the capital of France?"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["status"].startswith("refused"))
        self.assertEqual(len(data["sources"]), 0)
        self.assertIn("reliable context", data["answer"])

    def test_stream_query_sse_events(self):
        r = self.client.post("/query/stream", json={"question": "What was the Q4 portfolio return?"})
        self.assertEqual(r.status_code, 200)

        events = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e.get("type") for e in events]
        self.assertIn("citations", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

    def test_document_upload_and_indexing(self):
        fake_content = (
            "Tax Circular 2026-B.\n\n"
            "High net worth individuals are eligible for a 10 percent credit on municipal green bonds.\n\n"
            "All claims must be audited by branch compliance."
        )
        file_bytes = io.BytesIO(fake_content.encode("utf-8"))

        response = self.client.post(
            "/upload",
            files={"file": ("tax_circular_2026.txt", file_bytes, "text/plain")},
            data={"document_type": "tax_rules"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "tax_circular_2026.txt")
        self.assertGreater(data["chunks_indexed"], 0)


if __name__ == "__main__":
    unittest.main()
