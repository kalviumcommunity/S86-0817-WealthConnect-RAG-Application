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

from pathlib import Path

from dotenv import load_dotenv
from src.document_loader import load_all_documents, load_text, print_intake_report
from text_cleaner import clean_text

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
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text.strip():
        return []

    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def ingest_corpus(
    data_dir: str = "data",
    chunk_size: int = 500,
    overlap: int = 50,
    verbose: bool = True,
) -> dict:
    """Run load, clean, chunk, and metadata tagging for every file.

    Unlike ``run_ingestion``, this function includes unsupported files in the
    attempted-file count so the returned reconciliation check can detect any
    document that would otherwise disappear silently.
    """
    data_path = Path(data_dir)
    files = sorted(path for path in data_path.rglob("*") if path.is_file())
    chunks: list[dict] = []
    failures: list[dict] = []
    documents_ingested = 0

    for path in files:
        try:
            raw_text = load_text(path)
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                raise ValueError("document contains no text after cleaning")

            document = {
                "source": path.name,
                "extension": path.suffix.lower(),
                "char_count": len(cleaned_text),
            }
            document_chunks = chunk_text(cleaned_text, chunk_size, overlap)
            for chunk_index, chunk in enumerate(document_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": build_metadata(document, chunk_index),
                })
            documents_ingested += 1
        except Exception as exc:
            failures.append({
                "source": path.name,
                "filepath": str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            })

    summary = {
        "files_found": len(files),
        "documents_ingested": documents_ingested,
        "chunks_produced": len(chunks),
        "failures": failures,
        "reconciled": documents_ingested + len(failures) == len(files),
        "sample_chunk": chunks[0] if chunks else None,
        "chunks": chunks,
    }

    if verbose:
        print(
            f"[ingest] files={summary['files_found']} "
            f"docs={summary['documents_ingested']} "
            f"chunks={summary['chunks_produced']} "
            f"failures={len(failures)}"
        )
        for failure in failures:
            print(f"[ingest] FAILED: {failure['source']} - {failure['reason']}")
        if summary["sample_chunk"]:
            sample = summary["sample_chunk"]
            print(
                f"[ingest] sample: {sample['text'][:80]!r} | "
                f"{sample['metadata']}"
            )

    if not summary["reconciled"]:
        raise RuntimeError("ingestion reconciliation failed: a file was silently dropped")

    return summary


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
    report = ingest_corpus()
    print(
        f"\n[ingest] Ingestion complete — "
        f"{report['chunks_produced']} chunk(s) ready for embedding."
    )

    # Inspect first chunk as a sanity check
    if report["sample_chunk"]:
        first = report["sample_chunk"]
        print(f"\nFirst chunk preview:")
        print(f"  source  : {first['metadata']['document_name']}")
        print(f"  format  : {first['metadata']['source_format']}")
        print(f"  doc_type: {first['metadata']['document_type']}")
        print(f"  chunk   : {first['metadata']['chunk_index']}")
        print(f"  text    : {repr(first['text'][:120])}")
