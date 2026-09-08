import unittest

from prompts.templates import PromptTemplates


class PromptTemplateTests(unittest.TestCase):
    def test_render_injects_named_values(self):
        rendered = PromptTemplates.render(
            PromptTemplates.RAG_USER_MESSAGE,
            context="Approved balance: $50,000.",
            question="What is the balance?",
        )

        self.assertIn("Approved balance: $50,000.", rendered)
        self.assertIn("What is the balance?", rendered)
        self.assertIn("DOCUMENT CONTEXT START", rendered)

    def test_render_reports_missing_named_value(self):
        with self.assertRaisesRegex(ValueError, "question"):
            PromptTemplates.render(
                PromptTemplates.RAG_USER_MESSAGE,
                context="Approved balance: $50,000.",
            )


if __name__ == "__main__":
    unittest.main()