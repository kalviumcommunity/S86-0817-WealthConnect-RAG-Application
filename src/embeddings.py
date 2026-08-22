"""
embeddings.py — Embedding and vector store management for WealthConnect RAG

Converts text chunks into vector embeddings using OpenAI's embedding model
and stores them in ChromaDB for semantic retrieval.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Client setup — credentials loaded from .env, never hard-coded
# ---------------------------------------------------------------------------

_openai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION = "wealthconnect_docs"


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API to convert a list of text strings
    into their corresponding vector representations.
    """
    response = _openai_client.embeddings.create(
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
    collection = get_chroma_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [
        f"{m['document_name']}_chunk_{m['chunk_index']}"
        for m in metadatas
    ]

    print(f"[embeddings] Embedding {len(texts)} chunk(s)...")
    vectors = embed_texts(texts)

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[embeddings] Indexed {len(chunks)} chunk(s) into ChromaDB.")


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
