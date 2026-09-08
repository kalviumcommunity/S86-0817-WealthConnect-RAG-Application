import tiktoken


def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    """Counts tokens in a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2 # every reply is primed with <im_start>assistant
    return num_tokens


def enforce_token_limit(messages: list, max_tokens: int, model: str = "gpt-4o") -> list:
    """
    Trims the conversation history to stay under max_tokens.
    Always preserves the first message (System message).
    Removes oldest messages first.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    while count_tokens(messages, model) > max_tokens and len(messages) > 1:
        # Remove a complete oldest turn when the history has a user/assistant pair.
        removed = [messages.pop(1)]
        if removed[0].get("role") == "user" and len(messages) > 1:
            if messages[1].get("role") == "assistant":
                removed.append(messages.pop(1))
        for message in removed:
            print(f"  [Trimming] Removed oldest message (Role: {message['role']}) to stay under budget.")
    
    return messages


def summarize_history(
    messages: list,
    summary: str,
    max_tokens: int,
    model: str = "gpt-4o",
) -> list:
    """Replace older turns with a caller-generated summary and enforce the budget.

    The summary is kept as a system message after the original system prompt.
    Generating the summary is intentionally left to the caller so this helper
    does not make an unexpected model/API request.
    """
    if not messages or messages[0].get("role") != "system":
        raise ValueError("messages must start with a system message")
    if not summary.strip():
        raise ValueError("summary must not be empty")

    messages[1:] = [{"role": "system", "content": f"Conversation summary: {summary}"}]
    return enforce_token_limit(messages, max_tokens, model)


class ChatHistory:
    """Stateful history that budgets messages before each model request."""

    def __init__(self, system_prompt: str, max_tokens: int = 6_000, model: str = "gpt-4o"):
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        self.max_tokens = max_tokens
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    @property
    def token_count(self) -> int:
        return count_tokens(self.messages, self.model)

    def add_message(self, role: str, content: str) -> list:
        """Add a turn and return the budgeted messages for the next request."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not content.strip():
            raise ValueError("content must not be empty")
        self.messages.append({"role": role, "content": content})
        enforce_token_limit(self.messages, self.max_tokens, self.model)
        return self.messages

    def summarize(self, summary: str) -> list:
        """Replace previous turns with a compact summary before the next request."""
        return summarize_history(self.messages, summary, self.max_tokens, self.model)

def simulate_chat():
    MAX_BUDGET = 100 # Artificially small budget to demonstrate trimming
    
    # Task 1: Maintain multi-turn history
    history = [
        {"role": "system", "content": "You are a helpful financial RAG assistant."}
    ]
    
    # Task 4: Overflowing conversation
    turns = [
        "Hi, what is my account balance?",
        "Your account balance is $10,500.",
        "What about my retirement fund?",
        "Your retirement fund is currently valued at $45,200.",
        "Can you summarize my total net worth across all accounts you can see?",
        "Based on the provided documents, your total net worth is $55,700.",
        "Wait, what was my account balance again?", # This should force trimming of the earliest turns
        "Your account balance is $10,500."
    ]

    print(f"--- Chat History Manager (Max Tokens: {MAX_BUDGET}) ---\n")

    for i, content in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        message = {"role": role, "content": content}
        
        # Add new message to history
        history.append(message)
        
        # Task 2: Measure tokens before each request
        current_tokens = count_tokens(history)
        print(f"Turn {i+1} ({role}):")
        print(f"  Message: {content}")
        print(f"  Current Token Count: {current_tokens} / {MAX_BUDGET}")
        
        # Task 3: Trim or summarize old turns
        if current_tokens > MAX_BUDGET:
            print("  ! Token limit exceeded. Trimming history...")
            history = enforce_token_limit(history, MAX_BUDGET)
            print(f"  New Token Count after trimming: {count_tokens(history)}")
        
        print("-" * 50)

if __name__ == "__main__":
    simulate_chat()
