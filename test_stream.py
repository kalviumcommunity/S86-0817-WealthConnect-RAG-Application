"""Unit and integration test for the streaming endpoint using TestClient."""
import json
import unittest
from fastapi.testclient import TestClient

from streaming_api import app


class StreamingApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data.get("streaming", False))
        self.assertGreater(data.get("chunks", 0), 0)

    def test_stream_valid_query(self):
        response = self.client.post(
            "/query/stream",
            json={"question": "What was the Q4 portfolio return?"},
        )
        self.assertEqual(response.status_code, 200)

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e.get("type") for e in events]
        self.assertIn("citations", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

        # Check citations
        citations_event = next(e for e in events if e.get("type") == "citations")
        self.assertGreater(len(citations_event.get("sources", [])), 0)

        # Check full reconstructed answer
        tokens = [e["text"] for e in events if e.get("type") == "token"]
        full_text = "".join(tokens)
        self.assertIn("q4", full_text.lower())

    def test_stream_refusal_for_unrelated_query(self):
        response = self.client.post(
            "/query/stream",
            json={"question": "What is the capital of France?"},
        )
        self.assertEqual(response.status_code, 200)

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e.get("type") for e in events]
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

        tokens = [e["text"] for e in events if e.get("type") == "token"]
        full_text = "".join(tokens)
        self.assertIn("don't have enough reliable context", full_text)


if __name__ == "__main__":
    unittest.main()
