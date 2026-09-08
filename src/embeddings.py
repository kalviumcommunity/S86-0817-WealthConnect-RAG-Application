"""
embeddings.py — Embedding and vector store management for WealthConnect RAG

Converts text chunks into vector embeddings using OpenAI's embedding model
and stores them in ChromaDB for semantic retrieval.
"""

import os
import hashlib
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Client setup — credentials loaded from .env, never hard-coded
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION = "wealthconnect_docs"


def batches(items: list, size: int):
    """Yield consecutive batches and reject sizes that cannot make progress."""
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chunk_record_id(chunk: dict) -> str:
    """Build a stable ID from source, position, and content."""
    metadata = chunk["metadata"]
    identity = "\0".join([
        str(metadata.get("document_name", metadata.get("source", "unknown"))),
        str(metadata.get("chunk_index", 0)),
        chunk["text"],
    ])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def to_vector_record(chunk: dict, embedding: list[float]) -> dict:
    """Combine a chunk and embedding into one index-ready record."""
    return {
        "id": chunk_record_id(chunk),
        "vector": embedding,
        "text": chunk["text"],
        "metadata": dict(chunk["metadata"]),
    }


def get_chroma_collection() -> chromadb.Collection:
    """
    Return (or create) the persistent ChromaDB collection that stores
    all WealthConnect document embeddings.
    """
    client = chromadb.PersistentClient(path="outputs/chroma_db")
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _get_openai_client() -> OpenAI:
    """Create the API client only when live embeddings are requested."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API to convert a list of text strings
    into their corresponding vector representations.
    """
    response = _get_openai_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def index_chunks(chunks: list[dict]) -> None:
    """
    Embed each chunk and upsert it into the ChromaDB collection with metadata.

    Each chunk dict is expected to have:
        {
            "text":     str,
            "metadata": dict   # from ingest.build_metadata()
        }
    """
    index_chunks_verified(chunks)


def index_chunks_verified(
    chunks: list[dict],
    batch_size: int = 100,
    collection=None,
    embedder=embed_texts,
) -> dict:
    """Batch-index chunks and verify count plus a stored-record spot check."""
    if collection is None:
        collection = get_chroma_collection()

    records: list[dict] = []
    failures: list[dict] = []
    batches_processed = 0

    for batch_number, chunk_batch in enumerate(batches(chunks, batch_size), start=1):
        try:
            vectors = embedder([chunk["text"] for chunk in chunk_batch])
            if len(vectors) != len(chunk_batch):
                raise ValueError("embedding count does not match chunk count")

            batch_records = [
                to_vector_record(chunk, vector)
                for chunk, vector in zip(chunk_batch, vectors)
            ]
            collection.upsert(
                ids=[record["id"] for record in batch_records],
                embeddings=[record["vector"] for record in batch_records],
                documents=[record["text"] for record in batch_records],
                metadatas=[record["metadata"] for record in batch_records],
            )
            records.extend(batch_records)
            batches_processed += 1
        except Exception as exc:
            failures.append({
                "batch_number": batch_number,
                "error": f"{type(exc).__name__}: {exc}",
            })

    indexed_count = collection.count()
    expected_count = len(chunks)
    spot_check = None
    if records:
        sample = records[0]
        stored = collection.get(ids=[sample["id"]], include=[
            "documents", "metadatas", "embeddings"
        ])
        if not stored.get("ids"):
            raise RuntimeError(f"spot check failed: missing record {sample['id']}")
        stored_text = stored["documents"][0]
        stored_metadata = stored["metadatas"][0]
        stored_vector = stored["embeddings"][0]
        if stored_text != sample["text"] or stored_metadata != sample["metadata"]:
            raise RuntimeError(f"spot check failed: content mismatch for {sample['id']}")
        if len(stored_vector) != len(sample["vector"]):
            raise RuntimeError(f"spot check failed: vector mismatch for {sample['id']}")
        spot_check = {
            "id": sample["id"],
            "source": stored_metadata.get("document_name", stored_metadata.get("source")),
            "text_preview": stored_text[:120],
            "vector_dimensions": len(stored_vector),
        }

    return {
        "expected_count": expected_count,
        "inserted_count": len(records),
        "indexed_count": indexed_count,
        "failures": failures,
        "batches_processed": batches_processed,
        "count_matches": indexed_count == expected_count,
        "spot_check": spot_check,
    }


def query_collection(
    query_text: str,
    n_results: int = 5,
    approval_status: str = "approved",
) -> list[dict]:
    """
    Embed the user's question and retrieve the top-n most semantically
    similar chunks from the vector store.

    Filters to only return chunks where approval_status == 'approved',
    ensuring outdated or unapproved documents are never surfaced.
    """
    collection = get_chroma_collection()

    query_vector = embed_texts([query_text])[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where={"approval_status": approval_status},
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})

    return hits
