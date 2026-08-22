"""
ingest.py — Document Ingestion Pipeline for WealthConnect RAG

Orchestrates the full intake-to-index pipeline:
  1. Load all documents from data/ via document_loader (multi-format)
  2. Chunk each document's text with overlap
  3. Attach WealthConnect metadata to every chunk
  4. Return chunk records ready for embedding and vector indexing

Now uses document_loader.py (GY3.19) for real multi-format intake:
  .txt  — plain-text policy extracts
  .md   — Markdown guidelines and tax rules
  .pdf  — product brochures (primary format)
  .html — web-exported compliance pages
"""

from dotenv import load_dotenv
from src.document_loader import load_all_documents, print_intake_report

load_dotenv()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping character-level chunks.

    Overlap ensures context is not lost at chunk boundaries —
    a sentence that straddles a boundary appears in both adjacent chunks,
    so retrieval can surface it from either side.

    Args:
        text       : Plain text to chunk.
        chunk_size : Target chunk length in characters.
        overlap    : Number of characters shared between adjacent chunks.

    Returns:
        List of text chunk strings.
    """
    if not text.strip():
        return []

    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def build_metadata(document: dict, chunk_index: int) -> dict:
    """
    Build the metadata dict for a single chunk.

    These fields are stored alongside the embedding in ChromaDB and used
    for metadata filtering at retrieval time — ensuring only current,
    approved documents are ever surfaced in answers (FR-04).

    Fields align with the WealthConnect Document Metadata Schema defined
    in the README. Version, approval status, and effective date are set
    during admin document upload; defaults here keep the pipeline runnable
    before admin tooling is built.

    Args:
        document    : Loaded document record from document_loader.
        chunk_index : Zero-based index of this chunk within its document.

    Returns:
        Metadata dict for ChromaDB upsert.
    """
    return {
        "document_name"   : document["source"],
        "document_type"   : _infer_doc_type(document["source"]),
        "version"         : "unknown",      # Set during admin upload
        "approval_status" : "approved",     # Only approved docs live in data/
        "effective_date"  : "unknown",      # Set during admin upload
        "expiry_review_date": "unknown",    # Set during admin upload
        "product"         : "unknown",      # Set during admin upload
        "owner"           : "unknown",      # Set during admin upload
        "last_updated"    : "unknown",      # Set during admin upload
        "source_format"   : document["extension"],
        "char_count"      : document["char_count"],
        "chunk_index"     : chunk_index,
    }


def _infer_doc_type(filename: str) -> str:
    """
    Infer a rough document type from the filename for metadata tagging.
    Proper categorisation is set during admin upload; this is a best-effort
    fallback so the pipeline can run before admin tooling exists.
    """
    name = filename.lower()
    if "policy"      in name: return "investment_policy"
    if "tax"         in name: return "tax_rules"
    if "brochure"    in name: return "product_brochure"
    if "eligibility" in name: return "eligibility_guidelines"
    if "risk"        in name: return "risk_document"
    if "compliance"  in name: return "compliance_guidelines"
    return "unknown"


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def run_ingestion(data_dir: str = "data", verbose: bool = True) -> list[dict]:
    """
    Full ingestion pipeline:
      1. Load all supported documents from data/ (multi-format via document_loader)
      2. Print intake report showing loaded/skipped/empty counts
      3. Chunk each document's text
      4. Attach metadata to every chunk
      5. Return chunk records ready for embedding

    Args:
        data_dir : Directory containing approved wealth-management documents.
        verbose  : If True, print per-document and summary output.

    Returns:
        List of chunk dicts:
            { "text": str, "metadata": dict }
    """
    # Step 1 — Multi-format document intake
    loaded, skipped = load_all_documents(data_dir)

    # Step 2 — Intake report
    if verbose:
        print_intake_report(loaded, skipped)

    # Step 3 — Chunk and tag
    all_chunks: list[dict] = []

    for doc in loaded:
        if not doc["text"].strip():
            # Empty after loading — scanned PDF or extraction failure
            if verbose:
                print(f"[ingest] Skipping empty document: {doc['source']}")
            continue

        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text"    : chunk,
                "metadata": build_metadata(doc, chunk_index=i),
            })

    if verbose:
        print(f"[ingest] Total chunks produced: {len(all_chunks)}")

    return all_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = run_ingestion()
    print(f"\n[ingest] Ingestion complete — {len(chunks)} chunk(s) ready for embedding.")

    # Inspect first chunk as a sanity check
    if chunks:
        first = chunks[0]
        print(f"\nFirst chunk preview:")
        print(f"  source  : {first['metadata']['document_name']}")
        print(f"  format  : {first['metadata']['source_format']}")
        print(f"  doc_type: {first['metadata']['document_type']}")
        print(f"  chunk   : {first['metadata']['chunk_index']}")
        print(f"  text    : {repr(first['text'][:120])}")
