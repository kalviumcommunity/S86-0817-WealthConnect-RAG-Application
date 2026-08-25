import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def calculate_cost(num_tokens: int, cost_per_1m_tokens: float) -> float:
    """Calculates the cost based on token count and price per 1M tokens."""
    return (num_tokens / 1_000_000) * cost_per_1m_tokens

def main():
    # Define pricing (e.g., GPT-4o pricing)
    INPUT_COST_PER_1M = 5.00   # $5.00 per 1M input tokens
    OUTPUT_COST_PER_1M = 15.00 # $15.00 per 1M output tokens

    # Task 2: Define three text samples of varying length
    sample_short = "What is the current balance of my retirement account?"
    
    sample_paragraph = (
        "WealthConnect is an AI-powered financial advisory platform designed to help users "
        "manage their assets effectively. By integrating with various financial institutions, "
        "it provides a holistic view of a user's portfolio and uses machine learning to offer "
        "personalized investment recommendations."
    )
    
    sample_document = (
        "Q3 Financial Performance Report\n\n"
        "Executive Summary:\n"
        "This quarter saw a steady increase in user acquisition and total assets under management (AUM). "
        "Our core metrics indicate a 15% quarter-over-quarter growth in recurring revenue.\n\n"
        "Market Analysis:\n"
        "Despite volatility in the tech sector, our diversified investment strategies have yielded stable returns. "
        "The integration of our new Retrieval-Augmented Generation (RAG) system has improved customer query resolution "
        "times by 40%, significantly reducing the load on human advisors.\n\n"
        "Future Outlook:\n"
        "In Q4, we plan to expand our service offerings to include automated tax-loss harvesting and estate planning "
        "tools. We anticipate these features will drive a 20% increase in premium subscriptions."
        " " * 100 # Add some spaces to show character vs token difference further
    )

    samples = {
        "Short Question": sample_short,
        "Paragraph": sample_paragraph,
        "Full Document (Mock)": sample_document
    }

    print("--- Token Count, Relationship, and Cost Estimate ---\n")

    for name, text in samples.items():
        # Task 1: Count tokens
        token_count = count_tokens(text)
        
        # Task 4: Show the length-token relationship
        char_count = len(text)
        chars_per_token = char_count / token_count if token_count > 0 else 0
        
        # Task 3: Estimate cost
        input_cost = calculate_cost(token_count, INPUT_COST_PER_1M)
        output_cost = calculate_cost(token_count, OUTPUT_COST_PER_1M)

        print(f"Sample: {name}")
        print(f"Character Count: {char_count}")
        print(f"Token Count: {token_count}")
        print(f"Relationship (Chars/Token): {chars_per_token:.2f}")
        print(f"Estimated Input Cost: ${input_cost:.6f}")
        print(f"Estimated Output Cost: ${output_cost:.6f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
