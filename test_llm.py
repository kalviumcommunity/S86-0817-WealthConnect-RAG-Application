import os
import logging
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_interaction.log")
    ]
)

def main():
    # Task 1: Configure the client from environment
    load_dotenv()
    
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        return

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
        
        # Task 2: Send a request
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # Task 2 & 3: Log and print response
        reply = response.choices[0].message.content
        usage = response.usage
        
        logging.info("Successfully received response.")
        logging.info(f"Response Content: {reply}")
        if usage:
            logging.info(f"Token Usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
        print("\n--- Model Reply ---")
        print(reply)
        print("-------------------\n")

    # Task 4: Handle errors clearly
    except AuthenticationError as e:
        logging.error("Authentication Error (401): The provided API key is invalid or lacks permissions.")
    except RateLimitError as e:
        logging.error("Rate Limit Error (429): You have exceeded your rate limit. Please try again later.")
    except APIError as e:
        logging.error(f"API Error: An error occurred on the server side. Details: {e}")
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
