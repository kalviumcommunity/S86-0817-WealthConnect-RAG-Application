"""
ingest.py — Document ingestion pipeline for WealthConnect RAG

Loads approved wealth-management documents from the data/ directory,
extracts text, chunks it, attaches metadata, and indexes it into the
vector store so Relationship Managers can retrieve grounded answers.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def load_documents(data_dir: str = "data") -> list[dict]:
    """
    Scan data/ for supported document files and return a list of raw document records.
    Each record carries the file path and basic file-level metadata.

    Supported formats (extend as needed): .txt, .pdf, .docx
    """
    supported_extensions = {".txt", ".pdf", ".docx"}
    documents = []

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            documents.append(
                {
                    "filename": filename,
                    "filepath": os.path.join(data_dir, filename),
                    "extension": ext,
                }
            )

    print(f"[ingest] Found {len(documents)} document(s) in '{data_dir}'")
    return documents


def extract_text(document: dict) -> str:
    """
    Extract raw text from a document record.
    Currently handles plain .txt files.
    Extend this function to support PDF and DOCX as needed.
    """
    filepath = document["filepath"]
    ext = document["extension"]

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # Placeholder for future parsers
    print(f"[ingest] Skipping unsupported format: {filepath}")
    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    Overlap ensures context is not lost at chunk boundaries.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_metadata(document: dict, chunk_index: int) -> dict:
    """
    Attach document-level metadata to each chunk.
    These fields are used downstream for metadata filtering —
    ensuring only current, approved documents surface in answers.

    Fields align with the WealthConnect Document Metadata Schema.
    """
    return {
        "document_name": document["filename"],
        "document_type": "unknown",       # Set during admin upload
        "version": "unknown",             # Set during admin upload
        "approval_status": "approved",    # Only approved docs should be in data/
        "effective_date": "unknown",      # Set during admin upload
        "expiry_review_date": "unknown",  # Set during admin upload
        "product": "unknown",             # Set during admin upload
        "owner": "unknown",               # Set during admin upload
        "last_updated": "unknown",        # Set during admin upload
        "chunk_index": chunk_index,
    }


def run_ingestion(data_dir: str = "data") -> list[dict]:
    """
    Full ingestion pipeline:
      1. Load documents from data/
      2. Extract text
      3. Chunk text
      4. Attach metadata
    Returns a list of chunk records ready for embedding.
    """
    documents = load_documents(data_dir)
    all_chunks = []

    for doc in documents:
        text = extract_text(doc)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": build_metadata(doc, chunk_index=i),
                }
            )

    print(f"[ingest] Total chunks produced: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = run_ingestion()
    print(f"[ingest] Ingestion complete. {len(chunks)} chunks ready for embedding.")
