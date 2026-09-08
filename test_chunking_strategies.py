import unittest

from chunking_strategies import (
    chunk_by_fixed_size,
    chunk_by_paragraph,
    compare_strategies,
    compute_stats,
)


class ChunkingStrategyTests(unittest.TestCase):
    def test_fixed_chunks_use_overlap(self):
        chunks = chunk_by_fixed_size("abcdefghij", chunk_size=6, overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij", "ij"])

    def test_fixed_chunks_validate_configuration_and_empty_text(self):
        self.assertEqual(chunk_by_fixed_size("   ", chunk_size=6, overlap=2), [])
        with self.assertRaises(ValueError):
            chunk_by_fixed_size("text", chunk_size=4, overlap=4)

    def test_paragraph_chunks_preserve_semantic_boundaries(self):
        text = " First paragraph. \n\n\n Second paragraph. "

        self.assertEqual(
            chunk_by_paragraph(text),
            ["First paragraph.", "Second paragraph."],
        )

    def test_comparison_reports_count_and_average_size(self):
        comparison = compare_strategies("one two\n\nthree four")

        self.assertEqual(comparison["paragraph"]["stats"], {"count": 2, "avg_size": 8.5})
        self.assertEqual(compute_stats([]), {"count": 0, "avg_size": 0})


if __name__ == "__main__":
    unittest.main()