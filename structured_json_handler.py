import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_llm_response(response_text: str) -> dict:
    """
    Parses LLM output into a structured dictionary and validates required fields.
    Recovers from malformed JSON by returning a safe default structure.
    """
    # Task 2: Parse into a usable object
    try:
        # Strip potential markdown code blocks if the LLM wrapped it
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        parsed_data = json.loads(clean_text)
        
        # Task 4: Validate required fields
        required_fields = ["answer", "source"]
        missing_fields = [field for field in required_fields if field not in parsed_data]
        
        if missing_fields:
            logging.error(f"Validation failed. Missing required fields: {missing_fields}")
            # Recover by injecting missing fields
            for field in missing_fields:
                parsed_data[field] = "N/A - Field missing in response"
            logging.info("Recovered from missing fields by inserting defaults.")

        return parsed_data

    # Task 3: Handle malformed JSON gracefully
    except json.JSONDecodeError as e:
        logging.error(f"Malformed JSON detected. Failed to parse LLM response. Error: {e}")
        logging.warning("Recovering from crash by returning a default fallback structure.")
        
        return {
            "answer": "Error: Unable to process the response from the assistant.",
            "source": "System Fallback"
        }

def simulate_llm_call(prompt: str, mock_response: str):
    """Simulates sending a prompt to an LLM and parsing its response."""
    print(f"\n--- Sending Prompt ---")
    print(f"Prompt: {prompt}")
    print(f"Raw LLM Response: {mock_response}")
    print("--- Parsing Response ---")
    
    result = parse_llm_response(mock_response)
    
    print("Final Usable Object (Dict):")
    print(json.dumps(result, indent=2))
    print("-" * 50)

def main():
    # Task 1: Prompt for a defined JSON structure
    system_prompt = (
        "You are an assistant. You must always return your response in JSON format. "
        "Your JSON MUST contain exactly two keys: 'answer' (your response text) and 'source' (the document name you used)."
    )
    print(f"System Prompt configured as:\n{system_prompt}\n")

    # Scenario 1: Perfect JSON Response
    perfect_response = '{"answer": "Your portfolio is up 5% this year.", "source": "Q2_Report.pdf"}'
    simulate_llm_call("How is my portfolio doing?", perfect_response)

    # Scenario 2: Malformed JSON (Missing quote and bracket)
    malformed_response = '{"answer": "You have $10,000 in checking, "source": "Bank_API"'
    simulate_llm_call("What is my checking balance?", malformed_response)

    # Scenario 3: Valid JSON but missing required fields (No 'source')
    missing_fields_response = '{"answer": "The market is currently volatile due to interest rates."}'
    simulate_llm_call("Why is the market volatile?", missing_fields_response)

if __name__ == "__main__":
    main()
