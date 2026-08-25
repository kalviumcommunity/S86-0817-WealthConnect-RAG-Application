from prompts.templates import PromptTemplates

# Task 3: Feature 1 - Chat Path (Interactive QA)
def handle_chat_request(user_question: str, retrieved_docs: str):
    print("--- Feature 1: Chat Path ---")
    
    # Task 2: Inject dynamic values at runtime
    rendered_prompt = PromptTemplates.render(
        PromptTemplates.RAG_QA_PROMPT,
        context=retrieved_docs,
        question=user_question
    )
    
    print("Generated Prompt for Chat API:")
    print(rendered_prompt)
    print("-" * 50)


# Task 3: Feature 2 - Batch/CLI Path (Automated Report Generation)
def handle_batch_job(batch_questions: list, global_context: str):
    print("--- Feature 2: Batch Processing Path ---")
    
    for idx, question in enumerate(batch_questions):
        # Reuses the exact same template structure
        rendered_prompt = PromptTemplates.render(
            PromptTemplates.RAG_QA_PROMPT,
            context=global_context,
            question=question
        )
        print(f"Generated Prompt for Batch Job #{idx + 1}:")
        print(rendered_prompt)
        print("-" * 30)


def main():
    # Example usage for Chat Path
    mock_chat_docs = "- Account balance: $50,000\n- Recent transaction: -$500 (Amazon)"
    mock_chat_question = "What was my last transaction?"
    handle_chat_request(mock_chat_question, mock_chat_docs)

    # Example usage for Batch Path
    mock_batch_context = "User is a premium member. Portfolio is high risk."
    mock_batch_questions = [
        "Should this user invest in bonds?",
        "What is the user's membership tier?"
    ]
    handle_batch_job(mock_batch_questions, mock_batch_context)

if __name__ == "__main__":
    main()
