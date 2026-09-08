import os
import logging
import time
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

MAX_RATE_LIMIT_RETRIES = 3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_interaction.log")
    ]
)

def request_completion(client: OpenAI, model: str, messages: list[dict]):
    """Request a completion, retrying temporary rate-limit failures."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except RateLimitError:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = 2**attempt
            logging.warning(
                "Rate limited (429), retrying in %ss (%s/%s)",
                delay,
                attempt + 1,
                MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(delay)


def main() -> int:
    # Task 1: Configure the client from environment
    load_dotenv()
    
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        return 1

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    # Define messages
    messages = [
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": "Hello! What is your purpose?"}
    ]
    
    try:
        # Task 3: Log the request
        logging.info(f"Sending request to {base_url} for model {model}")
        logging.info(f"Request messages: {messages}")
        
        # Task 2: Send a request, retrying temporary rate limits.
        response = request_completion(client, model, messages)
        
        # Task 2 & 3: Log and print response
        reply = response.choices[0].message.content
        usage = response.usage
        
        logging.info("Successfully received response.")
        logging.info("Response Content: %s", reply)
        if usage:
            logging.info(
                "Token Usage - Prompt: %s, Completion: %s, Total: %s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            
        print("\n--- Model Reply ---")
        print(reply)
        print("-------------------\n")
        return 0

    # Task 4: Handle errors clearly
    except AuthenticationError:
        logging.error("Authentication Error (401): The provided API key is invalid or lacks permissions.")
    except RateLimitError:
        logging.error("Rate Limit Error (429): retries exhausted; try again later.")
    except APIError as error:
        logging.error("API Error: An error occurred on the server side. Details: %s", error)
    except Exception as error:
        logging.error("Unexpected Error: %s", error)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
