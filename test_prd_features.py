"""
Unit tests for PRD-specified features in WealthConnect:
1. Document governance catalog & status lifecycle (approved, draft, superseded).
2. FR-04: Current-version filtering (excluding superseded and draft documents from RM retrieval).
3. FR-08: RM feedback recording (helpful/not_helpful) & satisfaction analytics.
4. US-08: Operational question analytics & policy gap identification.
5. Runtime upload with versioning & product metadata.
"""

import json
import io
import unittest
from fastapi.testclient import TestClient

from rag_api import app, query_cache, _DOCUMENTS_REGISTRY, _collection, _feedback_store


class PrdFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        query_cache.clear()

    def test_admin_documents_list_and_summary(self):
        """Verify GET /admin/documents returns all catalog documents and summary counts."""
        res = self.client.get("/admin/documents")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("summary", data)
        self.assertIn("documents", data)
        summary = data["summary"]
        self.assertGreater(summary["total"], 0)
        self.assertGreaterEqual(summary["approved"], 1)

        # Spot check a known seed document
        doc_names = [d["name"] for d in data["documents"]]
        self.assertIn("sample_investment_policy.txt", doc_names)
        inv_doc = next(d for d in data["documents"] if d["name"] == "sample_investment_policy.txt")
        self.assertEqual(inv_doc["version"], "v3.2")
        self.assertEqual(inv_doc["category"], "Investment Policy")
        self.assertEqual(inv_doc["approval_status"], "approved")

    def test_admin_update_document_status_lifecycle(self):
        """Verify PATCH /admin/documents/{name} updates status in registry and collection."""
        doc_name = "sample_tax_rules.md"
        # 1. Update to superseded
        patch_res = self.client.patch(
            f"/admin/documents/{doc_name}",
            json={"approval_status": "superseded"},
        )
        self.assertEqual(patch_res.status_code, 200)
        pdata = patch_res.json()
        self.assertEqual(pdata["status"], "success")
        self.assertEqual(pdata["document"]["approval_status"], "superseded")
        self.assertGreater(pdata["chunks_updated"], 0)

        # Verify in registry
        self.assertEqual(_DOCUMENTS_REGISTRY[doc_name]["approval_status"], "superseded")

        # 2. Restore to approved
        restore_res = self.client.patch(
            f"/admin/documents/{doc_name}",
            json={"approval_status": "approved"},
        )
        self.assertEqual(restore_res.status_code, 200)
        self.assertEqual(_DOCUMENTS_REGISTRY[doc_name]["approval_status"], "approved")

    def test_current_version_filtering_excludes_superseded(self):
        """
        PRD FR-04: Test that marking a document 'superseded' excludes its chunks
        from Relationship Manager query retrieval.
        """
        doc_name = "sample_investment_policy.txt"
        query_text = "What is the Moderate Growth Fund asset allocation split?"

        # Ensure approved first
        self.client.patch(f"/admin/documents/{doc_name}", json={"approval_status": "approved"})
        query_cache.clear()

        r_approved = self.client.post("/query", json={"question": query_text})
        self.assertEqual(r_approved.status_code, 200)
        data_approved = r_approved.json()
        self.assertEqual(data_approved["status"], "answered")
        sources_approved = [s["source"] for s in data_approved["sources"]]
        self.assertIn(doc_name, sources_approved)

        # Now mark as superseded
        self.client.patch(f"/admin/documents/{doc_name}", json={"approval_status": "superseded"})
        query_cache.clear()

        r_superseded = self.client.post("/query", json={"question": query_text})
        self.assertEqual(r_superseded.status_code, 200)
        data_superseded = r_superseded.json()

        # The superseded document's chunks MUST NOT appear in the retrieved sources
        sources_superseded = [s["source"] for s in data_superseded.get("sources", [])]
        self.assertNotIn(doc_name, sources_superseded)

        # Revert back to approved
        self.client.patch(f"/admin/documents/{doc_name}", json={"approval_status": "approved"})

    def test_feedback_submission_and_metrics(self):
        """PRD Section 7 FR-08: Test POST /feedback and GET /admin/feedback."""
        # Invalid rating rejected
        r_bad = self.client.post("/feedback", json={
            "question": "What is the tax rate?",
            "rating": "maybe",
        })
        self.assertEqual(r_bad.status_code, 400)

        # Helpful feedback submission
        r_good = self.client.post("/feedback", json={
            "question": "What was the Q4 portfolio return?",
            "answer": "Our aggressive growth fund yielded 12% in Q4.",
            "rating": "helpful",
            "comment": "Accurate and client-ready answer.",
        })
        self.assertEqual(r_good.status_code, 200)
        self.assertEqual(r_good.json()["status"], "recorded")

        # Not helpful feedback submission
        r_not_helpful = self.client.post("/feedback", json={
            "question": "Explain offshore trust tax rules.",
            "answer": "I don't have enough reliable context...",
            "rating": "not_helpful",
            "comment": "Policy gap: offshore trust rules are missing.",
        })
        self.assertEqual(r_not_helpful.status_code, 200)

        # Check admin feedback metrics
        res_admin = self.client.get("/admin/feedback")
        self.assertEqual(res_admin.status_code, 200)
        f_data = res_admin.json()
        self.assertGreaterEqual(f_data["total"], 2)
        self.assertGreaterEqual(f_data["helpful"], 1)
        self.assertGreaterEqual(f_data["not_helpful"], 1)
        self.assertGreaterEqual(f_data["satisfaction_rate"], 0.0)
        self.assertLessEqual(f_data["satisfaction_rate"], 100.0)
        self.assertGreater(len(f_data["feedback"]), 0)

    def test_admin_questions_analytics_and_policy_gaps(self):
        """PRD Section 8 US-08: Test GET /admin/questions for gap identification & FAQs."""
        # Trigger an answered query and an out-of-scope/gap query to ensure audit logging
        self.client.post("/query", json={"question": "What was the Q4 portfolio return?"})
        self.client.post("/query", json={"question": "What is the capital of Mars?"})

        res = self.client.get("/admin/questions")
        self.assertEqual(res.status_code, 200)
        q_data = res.json()

        self.assertIn("total_queries", q_data)
        self.assertIn("answered_queries", q_data)
        self.assertIn("unanswered_queries", q_data)
        self.assertIn("coverage_rate", q_data)
        self.assertIn("unanswered_log", q_data)
        self.assertIn("frequent_questions", q_data)

        # Verify unanswered log captures refusals
        if q_data["unanswered_queries"] > 0:
            top_gap = q_data["unanswered_log"][0]
            self.assertIn("question", top_gap)
            self.assertIn("status", top_gap)
            self.assertIn("reason", top_gap)

    def test_upload_with_version_and_product_metadata(self):
        """Test POST /upload with custom product, version, and status metadata."""
        content = (
            "Global Wealth Mandate v4.0.\n\n"
            "Clients with accounts exceeding $500,000 qualify for dedicated relationship officers.\n\n"
            "Asset allocations are rebalanced quarterly."
        )
        file_bytes = io.BytesIO(content.encode("utf-8"))

        res = self.client.post(
            "/upload",
            files={"file": ("global_wealth_mandate_v4.txt", file_bytes, "text/plain")},
            data={
                "document_type": "investment_policy",
                "product": "Global Mandate High-Tier",
                "version": "v4.0",
                "approval_status": "approved",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["version"], "v4.0")
        self.assertEqual(data["product"], "Global Mandate High-Tier")
        self.assertEqual(data["approval_status"], "approved")

        # Verify registry was updated
        self.assertIn("global_wealth_mandate_v4.txt", _DOCUMENTS_REGISTRY)
        reg_item = _DOCUMENTS_REGISTRY["global_wealth_mandate_v4.txt"]
        self.assertEqual(reg_item["version"], "v4.0")
        self.assertEqual(reg_item["product"], "Global Mandate High-Tier")

        # Test DELETE /admin/documents/{name}
        del_res = self.client.delete("/admin/documents/global_wealth_mandate_v4.txt")
        self.assertEqual(del_res.status_code, 200)
        self.assertNotIn("global_wealth_mandate_v4.txt", _DOCUMENTS_REGISTRY)

    def test_auth_login_relationship_manager(self):
        """Test authentication as Relationship Manager."""
        res = self.client.post("/auth/login", json={
            "username": "rm_advisor",
            "password": "Password@123",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user"]["role"], "relationship_manager")
        self.assertIn("G Yashmieen", data["user"]["full_name"])
        self.assertTrue(data["token"].startswith("wc_relationship_manager_"))

    def test_auth_login_wealth_admin(self):
        """Test authentication as Wealth Admin."""
        res = self.client.post("/auth/login", json={
            "username": "wealth_admin",
            "password": "Admin@123",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user"]["role"], "wealth_admin")
        self.assertIn("Dodla Bhanu Teja Reddy", data["user"]["full_name"])
        self.assertTrue(data["token"].startswith("wc_wealth_admin_"))

    def test_auth_login_invalid_credentials(self):
        """Test rejection of invalid credentials."""
        res = self.client.post("/auth/login", json={
            "username": "rm_advisor",
            "password": "WrongPassword!",
        })
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid credentials", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
