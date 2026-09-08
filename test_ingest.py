import tempfile
import unittest
from pathlib import Path

from src.ingest import ingest_corpus


class CorpusIngestionTests(unittest.TestCase):
    def test_report_reconciles_successes_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.txt").write_text(
                "Page 1 of 2\n\nApproved refund policy.\n\nCONFIDENTIAL DOCUMENT",
                encoding="utf-8",
            )
            (root / "product.md").write_text(
                "# Product\n\nGrowth portfolio.",
                encoding="utf-8",
            )
            (root / "unsupported.csv").write_text("name,value\nfee,10", encoding="utf-8")

            report = ingest_corpus(directory, chunk_size=40, overlap=5, verbose=False)

        self.assertEqual(report["files_found"], 3)
        self.assertEqual(report["documents_ingested"], 2)
        self.assertEqual(len(report["failures"]), 1)
        self.assertTrue(report["reconciled"])
        self.assertGreater(report["chunks_produced"], 0)
        self.assertEqual(report["sample_chunk"]["metadata"]["approval_status"], "approved")

    def test_empty_document_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "empty.txt").write_text("Page 1 of 1", encoding="utf-8")

            report = ingest_corpus(directory, verbose=False)

        self.assertEqual(report["documents_ingested"], 0)
        self.assertEqual(len(report["failures"]), 1)
        self.assertTrue(report["reconciled"])


if __name__ == "__main__":
    unittest.main()