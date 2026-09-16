"""
GY3.48 — Caching Layer
WealthConnect RAG Application

Provides thread-safe in-memory caching with TTL (Time To Live)
and LRU (Least Recently Used) eviction for repeated advisory queries.
"""

import time
import threading
from collections import OrderedDict


class QueryCache:
    """
    Thread-safe LRU cache with TTL support for grounded RAG query results.
    """
    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize_key(question: str) -> str:
        """Normalize query text for consistent cache lookup."""
        return " ".join(question.strip().lower().split())

    def get(self, question: str) -> dict | None:
        """Retrieve cached result if present and unexpired."""
        key = self.normalize_key(question)
        with self._lock:
            if key in self._cache:
                timestamp, data = self._cache[key]
                if time.time() - timestamp <= self.ttl:
                    # Move to most-recently-used
                    self._cache.move_to_end(key)
                    self.hits += 1
                    return data
                else:
                    # Expired entry
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, question: str, data: dict):
        """Store query result in cache, evicting oldest if capacity reached."""
        key = self.normalize_key(question)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest entry
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), data)

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def stats(self) -> dict:
        """Return cache performance statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 4),
            }
