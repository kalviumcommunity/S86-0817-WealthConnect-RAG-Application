"""
GY3.33 — Metadata Filtering & Hybrid Search
WealthConnect RAG Application

Combines dense vector semantic search (cosine similarity) with
sparse keyword search (BM25 / lexical scoring) using Reciprocal Rank
Fusion (RRF) and metadata filtering.
"""

import math
import re
from collections import Counter
import numpy as np

from similarity_search import VectorCollection, cosine_similarity, embed, CORPUS


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    Lightweight, self-contained BM25 ranking implementation
    for sparse lexical retrieval over document chunks.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()
        self.n_docs: int = 0

    def fit(self, corpus: list[dict]):
        """Index a list of chunk dicts (each containing 'text' and 'metadata')."""
        self.docs = corpus
        self.doc_tokens = [tokenize(doc["text"]) for doc in corpus]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.n_docs = len(corpus)
        self.avg_doc_len = (sum(self.doc_lens) / self.n_docs) if self.n_docs > 0 else 0.0

        self.df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def score(self, query: str) -> list[float]:
        """Compute BM25 score of each document against the query."""
        query_tokens = tokenize(query)
        scores = [0.0] * self.n_docs

        for term in query_tokens:
            if term not in self.df:
                continue
            df_val = self.df[term]
            idf = math.log(1 + (self.n_docs - df_val + 0.5) / (df_val + 0.5))

            for i, (tokens, d_len) in enumerate(zip(self.doc_tokens, self.doc_lens)):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        return scores


class HybridSearchEngine:
    """
    Hybrid Search Engine combining dense vector search and sparse BM25
    with strict metadata filtering and Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, collection: VectorCollection | None = None, corpus: list[dict] | None = None):
        self.collection = collection or VectorCollection()
        self.bm25 = BM25Index()
        self.corpus = corpus or []
        if self.corpus:
            self.bm25.fit(self.corpus)

    def index_corpus(self, chunks: list[dict], embedder=None):
        """Index chunks into both dense vector collection and sparse BM25 index."""
        self.corpus = chunks
        self.bm25.fit(chunks)

        if embedder is None:
            embedder = lambda texts: embed(texts, dry_run=True)

        embeddings = embedder([c["text"] for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            self.collection.add(
                text=chunk["text"],
                embedding=emb,
                metadata=chunk.get("metadata", {}),
            )

    @staticmethod
    def match_metadata(metadata: dict, metadata_filter: dict | None) -> bool:
        """Check whether metadata satisfies all filter conditions."""
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            actual = metadata.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        top_k: int = 5,
        alpha: float = 0.6,
        metadata_filter: dict | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Execute hybrid search using Reciprocal Rank Fusion (RRF).

        Args:
            query           : User text query (used for BM25)
            query_vector    : Dense vector embedding of query (used for vector search)
            top_k           : Number of chunks to return
            alpha           : Dense vs sparse weight (1.0 = pure dense, 0.0 = pure sparse)
            metadata_filter : Dict of metadata fields to require
            rrf_k           : Smoothing constant for RRF (default 60)

        Returns:
            List of ranked results with hybrid scores and attribution.
        """
        # 1. Sparse BM25 candidate scoring
        bm25_scores = self.bm25.score(query)
        sparse_ranked = []
        for i, (doc, sc) in enumerate(zip(self.corpus, bm25_scores)):
            meta = doc.get("metadata", {})
            if self.match_metadata(meta, metadata_filter):
                sparse_ranked.append({
                    "corpus_idx": i,
                    "text": doc["text"],
                    "metadata": meta,
                    "sparse_score": sc,
                })
        sparse_ranked.sort(key=lambda x: x["sparse_score"], reverse=True)

        # 2. Dense vector candidate scoring
        dense_ranked = []
        if query_vector is not None:
            recs = getattr(self.collection, "_records", getattr(self.collection, "records", []))
            raw_dense = self.collection.search(
                vector=query_vector,
                top_k=len(recs) if recs else 100,
                metadata_filter=metadata_filter,
            )
            for item in raw_dense:
                dense_ranked.append({
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_score": item["score"],
                })

        # 3. Reciprocal Rank Fusion (RRF)
        # Key on (source, chunk_index) or text
        sparse_rank_map = {}
        for rank, item in enumerate(sparse_ranked, start=1):
            key = item["text"]
            sparse_rank_map[key] = (rank, item["sparse_score"], item)

        dense_rank_map = {}
        for rank, item in enumerate(dense_ranked, start=1):
            key = item["text"]
            dense_rank_map[key] = (rank, item["dense_score"], item)

        all_keys = set(sparse_rank_map.keys()) | set(dense_rank_map.keys())
        fused = []

        for key in all_keys:
            dense_rank, dense_score, dense_item = dense_rank_map.get(key, (1000, 0.0, None))
            sparse_rank, sparse_score, sparse_item = sparse_rank_map.get(key, (1000, 0.0, None))

            # RRF formula
            dense_rrf = 1.0 / (rrf_k + dense_rank)
            sparse_rrf = 1.0 / (rrf_k + sparse_rank)
            hybrid_score = (alpha * dense_rrf) + ((1.0 - alpha) * sparse_rrf)

            item = dense_item or sparse_item
            fused.append({
                "text": key,
                "metadata": item["metadata"],
                "score": round(hybrid_score, 6),
                "dense_score": round(dense_score, 4),
                "sparse_score": round(sparse_score, 4),
                "dense_rank": dense_rank if dense_rank < 1000 else None,
                "sparse_rank": sparse_rank if sparse_rank < 1000 else None,
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
