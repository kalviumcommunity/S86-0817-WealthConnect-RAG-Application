"""
Unit tests for Milestone 3.42 — Conversational RAG & Follow-Up Context
"""

import unittest
from conversational_rag import condense_followup_question, ConversationalRAGSession


class ConversationalRAGTests(unittest.TestCase):
    def test_condense_standalone_question_remains_unchanged(self):
        question = "What is the Q4 portfolio return?"
        history = []
        condensed = condense_followup_question(question, history)
        self.assertEqual(condensed, question)

    def test_condense_followup_attaches_prior_subject(self):
        history = [
            {"role": "user", "content": "What is the investment policy for the Moderate Growth Fund?"},
            {"role": "assistant", "content": "The Moderate Growth Fund requires a $10,000 minimum investment."},
        ]
        followup = "What about the tax rate?"
        condensed = condense_followup_question(followup, history)

        self.assertIn("tax rate", condensed.lower())
        self.assertIn("moderate growth fund", condensed.lower())

    def test_session_maintains_multi_turn_history(self):
        session = ConversationalRAGSession("System prompt", max_tokens=1000)

        # Mock RAG function that returns a dummy answer
        def mock_rag(query: str):
            return {
                "answer": f"Grounded response for: {query}",
                "sources": [{"source": "test_doc.md"}],
            }

        res1 = session.ask("Tell me about municipal bonds", mock_rag)
        self.assertEqual(res1["original_question"], "Tell me about municipal bonds")
        self.assertEqual(len(session.history), 3) # system, user, assistant

        res2 = session.ask("What is the exemption limit?", mock_rag)
        self.assertIn("municipal bonds", res2["condensed_query"].lower())
        self.assertEqual(len(session.history), 5) # system, user1, asst1, user2, asst2


if __name__ == "__main__":
    unittest.main()
