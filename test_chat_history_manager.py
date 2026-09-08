import unittest

from chat_history_manager import ChatHistory, count_tokens, enforce_token_limit


class ChatHistoryTests(unittest.TestCase):
    def test_trimming_preserves_system_and_complete_turns(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "old question " * 20},
            {"role": "assistant", "content": "old answer " * 20},
            {"role": "user", "content": "current question"},
        ]

        result = enforce_token_limit(messages, 60)

        self.assertIs(result, messages)
        self.assertEqual(result[0]["role"], "system")
        self.assertLessEqual(count_tokens(result), 60)
        self.assertNotEqual(
            [(message["role"]) for message in result[1:3]],
            ["assistant", "user"],
        )

    def test_summary_keeps_system_prompt(self):
        history = ChatHistory("You are a financial assistant.", max_tokens=100)
        history.add_message("user", "What is my balance?")
        history.add_message("assistant", "Your balance is $10,500.")

        history.summarize("The user asked about their account balance.")

        self.assertEqual(history.messages[0]["role"], "system")
        self.assertIn("Conversation summary:", history.messages[1]["content"])
        self.assertLessEqual(history.token_count, 100)


if __name__ == "__main__":
    unittest.main()