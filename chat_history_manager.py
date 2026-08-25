import tiktoken

def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    """Counts tokens in a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2 # every reply is primed with <im_start>assistant
    return num_tokens

def enforce_token_limit(messages: list, max_tokens: int) -> list:
    """
    Trims the conversation history to stay under max_tokens.
    Always preserves the first message (System message).
    Removes oldest messages first.
    """
    while count_tokens(messages) > max_tokens and len(messages) > 1:
        # We want to remove the oldest message after the system message
        # But we should probably remove user/assistant pairs if we can,
        # or just simply pop the message at index 1.
        removed = messages.pop(1)
        print(f"  [Trimming] Removed oldest message (Role: {removed['role']}) to stay under budget.")
    
    return messages

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
