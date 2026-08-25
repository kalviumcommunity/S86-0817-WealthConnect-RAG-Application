import re
import unicodedata

def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted document text to prepare it for RAG embedding.
    """
    # Task 2: Normalize encoding artifacts (e.g. Unicode NFKC)
    text = unicodedata.normalize('NFKC', raw_text)
    
    # Task 1: Remove boilerplate (headers, footers, page numbers)
    # Remove "Page X of Y" or "- Page X -" patterns
    text = re.sub(r'(?i)\bpage\s+\d+\s+(of\s+\d+)?\b', '', text)
    text = re.sub(r'-\s*Page\s*\d+\s*-', '', text)
    
    # Remove typical boilerplate navigation text
    text = re.sub(r'(?i)back to top|click here for more info', '', text)
    
    # Remove completely isolated headers/footers (e.g. "CONFIDENTIAL" at the bottom)
    text = re.sub(r'(?i)\bCONFIDENTIAL\s+DOCUMENT\b', '', text)

    # Task 2: Normalize whitespace and line breaks
    # Collapse 3 or more newlines into exactly 2 newlines (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse runaway spaces (2 or more) into a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def process_corpus(corpus: list) -> list:
    """
    Task 3: Apply consistently across the corpus.
    """
    return [clean_text(doc) for doc in corpus]


def main():
    # Simulated Raw Corpus with noise, encoding artifacts, and boilerplate
    raw_corpus = [
        (
            "COMPANY Q3 EARNINGS REPORT         \n\n\n\n"
            "Page 1 of 45\n\n"
            "We are pleased to announce \u24B6 15% growth in Q3.   The revenue was strong. \n"
            "CONFIDENTIAL DOCUMENT\n"
            "Back to top\n"
        ),
        (
            "\n\n\n- Page 2 -\n\n"
            "The market  has  been very volatile,  but our RAG system   \n"
            "has stabilized support costs. \n"
            "Page 2 of 45\n\n"
            "Click here for more info"
        )
    ]
    
    cleaned_corpus = process_corpus(raw_corpus)
    
    with open("cleaning_results.txt", "w", encoding="utf-8") as f:
        f.write("--- RAG Document Cleaning Pipeline ---\n\n")
        
        # Task 4: Show before/after
        for i, (raw, cleaned) in enumerate(zip(raw_corpus, cleaned_corpus), 1):
            f.write(f"=== Document {i} ===\n")
            f.write("[BEFORE - Raw Extracted Text]\n")
            f.write(repr(raw) + "\n\n")
            f.write("[AFTER - Cleaned & Ready for Retrieval]\n")
            f.write(repr(cleaned) + "\n")
            f.write("="*40 + "\n\n")
            
    print("Successfully wrote results to cleaning_results.txt")


if __name__ == "__main__":
    main()
