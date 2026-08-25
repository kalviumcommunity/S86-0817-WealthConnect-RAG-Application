def chunk_by_fixed_size(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Strategy 1: Fixed-size chunking with overlap.
    Splits text strictly by character count, ensuring context carries over via overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def chunk_by_paragraph(text: str) -> list:
    """
    Strategy 2: Paragraph chunking.
    Splits text by double newlines, keeping semantic blocks completely intact.
    """
    # Split by one or more blank lines (using \n\n as standard paragraph delimiter)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def compute_stats(chunks: list) -> dict:
    """Task 3: Compute chunk count and average chunk size (in characters)."""
    if not chunks:
        return {"count": 0, "avg_size": 0}
    avg_size = sum(len(c) for c in chunks) / len(chunks)
    return {"count": len(chunks), "avg_size": round(avg_size, 2)}


def write_results_to_file(filename: str, document: str, fixed_chunks: list, para_chunks: list):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("--- Document Chunking Strategies Comparison ---\n\n")
        
        # Output Stats
        f.write("=== Strategy 1: Fixed-Size Character Chunking (Size: 150, Overlap: 30) ===\n")
        stats1 = compute_stats(fixed_chunks)
        f.write(f"Stats - Count: {stats1['count']}, Average Size: {stats1['avg_size']} chars\n\n")
        for i, c in enumerate(fixed_chunks, 1):
            f.write(f"Chunk {i}:\n[{c}]\n\n")

        f.write("\n" + "="*50 + "\n\n")
        
        f.write("=== Strategy 2: Paragraph Chunking ===\n")
        stats2 = compute_stats(para_chunks)
        f.write(f"Stats - Count: {stats2['count']}, Average Size: {stats2['avg_size']} chars\n\n")
        for i, c in enumerate(para_chunks, 1):
            f.write(f"Chunk {i}:\n[{c}]\n\n")
            
        f.write("\n" + "="*50 + "\n\n")
        
        # Task 4: Justify the choice
        f.write("=== Task 4: Justification ===\n")
        f.write("For this financial RAG application (WealthConnect), **Strategy 2 (Paragraph Chunking)** is the chosen strategy.\n")
        f.write("Reasoning: Financial documents (like earnings reports and account summaries) are highly structured by topic per paragraph. ")
        f.write("Using fixed-size chunking blindly slices through sentences and numbers (e.g., cutting '$10,000' in half), ")
        f.write("destroying the factual integrity needed for accurate retrieval. Paragraph chunking respects the natural semantic boundaries, ")
        f.write("ensuring that concepts remain whole and contextually complete when fed to the LLM.\n")


def main():
    # A mock document simulating a financial report with natural paragraph breaks
    sample_document = (
        "WealthConnect Q4 Portfolio Analysis and Earnings Overview.\n\n"
        "In the final quarter of the year, our core aggressive growth fund "
        "yielded a 12% return, heavily driven by strategic investments in the "
        "AI and renewable energy sectors. This offset the minor losses experienced "
        "in the traditional real estate trust portfolio.\n\n"
        "Customer acquisition costs dropped by $15 per head following the rollout "
        "of our new Retrieval-Augmented Generation automated support assistant. "
        "The assistant currently resolves 65% of Tier 1 queries without human intervention.\n\n"
        "Looking ahead to next year, we project a stabilization of interest rates. "
        "Consequently, we recommend clients rebalance to hold at least 30% in high-yield bonds."
    )

    # Task 1 & 2: Compare two strategies
    fixed_chunks = chunk_by_fixed_size(sample_document)
    para_chunks = chunk_by_paragraph(sample_document)
    
    # Write to file and print summary
    write_results_to_file("chunking_results.txt", sample_document, fixed_chunks, para_chunks)
    print("Chunking completed. Full analysis and sample chunks written to 'chunking_results.txt'.")


if __name__ == "__main__":
    main()
