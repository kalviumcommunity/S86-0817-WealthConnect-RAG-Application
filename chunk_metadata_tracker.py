"""
GY3.22 — Chunk Metadata & Source Tracking
WealthConnect RAG Application

Every chunk is stored as { "text": ..., "metadata": { ... } } so any retrieved
chunk can be traced back to its exact source and cited in an answer.
"""

# ---------------------------------------------------------------------------
# 1. Core tagging helper — attaches metadata to every chunk
# ---------------------------------------------------------------------------

def tag_chunks(source: str, chunks: list[str], extra_meta: dict = None) -> list[dict]:
    """
    Pair each chunk with a metadata dict.

    Args:
        source     : Document identifier (filename, URL, or logical name).
        chunks     : List of plain-text chunks produced by any chunking strategy.
        extra_meta : Optional dict of additional fields (e.g. page_number, section).

    Returns:
        List of dicts, each with "text" and "metadata" keys.
    """
    tagged = []
    char_cursor = 0

    for i, chunk_text in enumerate(chunks):
        meta = {
            "source": source,          # which document — needed to cite
            "chunk_index": i,          # position in the document
            "char_start": char_cursor, # where this chunk begins (character offset)
            "char_end": char_cursor + len(chunk_text),
        }

        # Merge any caller-supplied metadata (page, section, url, date, …)
        if extra_meta:
            meta.update(extra_meta)

        tagged.append({"text": chunk_text, "metadata": meta})
        char_cursor += len(chunk_text)   # advance cursor (no-overlap approximation)

    return tagged


# ---------------------------------------------------------------------------
# 2. Source-specific taggers — add richer metadata per document type
# ---------------------------------------------------------------------------

def tag_pdf_chunks(filename: str, chunks: list[str], page_number: int = None) -> list[dict]:
    """
    Tag chunks from a PDF.  Page number is the most useful extra field for PDFs.
    """
    extra = {"doc_type": "pdf"}
    if page_number is not None:
        extra["page_number"] = page_number
    return tag_chunks(source=filename, chunks=chunks, extra_meta=extra)


def tag_markdown_chunks(filename: str, chunks: list[str], section_heading: str = None) -> list[dict]:
    """
    Tag chunks from a Markdown file.  Section heading gives readers a human-readable anchor.
    """
    extra = {"doc_type": "markdown"}
    if section_heading:
        extra["section_heading"] = section_heading
    return tag_chunks(source=filename, chunks=chunks, extra_meta=extra)


def tag_html_chunks(url: str, chunks: list[str], page_title: str = None) -> list[dict]:
    """
    Tag chunks from an HTML page.  URL is the source; page title helps the citation.
    """
    extra = {"doc_type": "html"}
    if page_title:
        extra["page_title"] = page_title
    return tag_chunks(source=url, chunks=chunks, extra_meta=extra)


# ---------------------------------------------------------------------------
# 3. Citation helper — format a retrieved chunk into a human-readable cite
# ---------------------------------------------------------------------------

def format_citation(chunk: dict) -> str:
    """
    Build a citation string from a tagged chunk's metadata.

    Examples:
        "refund-policy.pdf, chunk 3 (chars 420–600)"
        "https://wealthconnect.io/faq, chunk 0 — Page title: WealthConnect FAQ"
        "portfolio-report.md, chunk 1 — Section: Q4 Performance"
    """
    meta = chunk.get("metadata", {})
    source = meta.get("source", "unknown source")
    idx    = meta.get("chunk_index", "?")
    start  = meta.get("char_start", "?")
    end    = meta.get("char_end", "?")

    citation = f"{source}, chunk {idx} (chars {start}–{end})"

    if "page_number" in meta:
        citation += f", page {meta['page_number']}"
    if "section_heading" in meta:
        citation += f" — Section: {meta['section_heading']}"
    if "page_title" in meta:
        citation += f" — Page title: {meta['page_title']}"

    return citation


# ---------------------------------------------------------------------------
# 4. Demo / main
# ---------------------------------------------------------------------------

# --- Sample documents -------------------------------------------------------

PDF_CHUNKS = [
    "WealthConnect Q4 Portfolio Analysis and Earnings Overview.",
    "In the final quarter of the year, our core aggressive growth fund yielded a 12% return, "
    "heavily driven by strategic investments in the AI and renewable energy sectors.",
    "Customer acquisition costs dropped by $15 per head following the rollout of our new "
    "Retrieval-Augmented Generation automated support assistant.",
    "Looking ahead to next year, we project a stabilization of interest rates. "
    "We recommend clients rebalance to hold at least 30% in high-yield bonds.",
]

MARKDOWN_CHUNKS = [
    "## Refund Policy\nAll refund requests must be submitted within 30 days of purchase.",
    "## Contact Support\nReach our support team at support@wealthconnect.io.",
]

HTML_CHUNKS = [
    "WealthConnect helps you grow wealth through smart, AI-powered investment insights.",
    "Our RAG assistant answers your financial questions with cited, up-to-date sources.",
]

# --- Simulated retrieval results --------------------------------------------

QUERY = "What was the Q4 portfolio return?"

# Simulate which chunk would be retrieved for the query (index 1 of the PDF)
RETRIEVED_CHUNK_INDEX = 1


def main():
    # Tag each corpus
    pdf_tagged      = tag_pdf_chunks("q4_earnings_report.pdf", PDF_CHUNKS, page_number=2)
    markdown_tagged = tag_markdown_chunks("policies.md", MARKDOWN_CHUNKS, section_heading="Refund Policy")
    html_tagged     = tag_html_chunks("https://wealthconnect.io/about", HTML_CHUNKS, page_title="About WealthConnect")

    all_chunks = pdf_tagged + markdown_tagged + html_tagged

    results = []

    # ---- Section 1: All tagged chunks ----
    results.append("=" * 60)
    results.append("TAGGED CHUNKS — Full Corpus")
    results.append("=" * 60)
    for chunk in all_chunks:
        results.append(f"\n[TEXT]     {chunk['text'][:80]}{'...' if len(chunk['text']) > 80 else ''}")
        results.append(f"[METADATA] {chunk['metadata']}")

    # ---- Section 2: Simulated retrieval + citation ----
    results.append("\n" + "=" * 60)
    results.append(f"SIMULATED RETRIEVAL  —  Query: '{QUERY}'")
    results.append("=" * 60)

    retrieved = pdf_tagged[RETRIEVED_CHUNK_INDEX]
    citation  = format_citation(retrieved)

    results.append(f"\nRetrieved chunk text:\n  {retrieved['text']}")
    results.append(f"\nCitation:\n  {citation}")
    results.append(f"\nAnswer: According to [{citation}], the Q4 portfolio return was 12%.")

    # ---- Section 3: Tracing any chunk back to its source ----
    results.append("\n" + "=" * 60)
    results.append("SOURCE TRACEABILITY — All citations in corpus")
    results.append("=" * 60)
    for chunk in all_chunks:
        results.append(f"  {format_citation(chunk)}")

    output = "\n".join(results)

    # Write to file
    output_file = "chunk_metadata_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\nFull results written to '{output_file}'.")


if __name__ == "__main__":
    main()
