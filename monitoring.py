"""
GY3.48 — Logging & Usage Monitoring
WealthConnect RAG Application

Provides structured audit logging and real-time operational metrics:
1. Append-only JSON audit log for regulatory compliance.
2. Latency, token usage, cost estimation, and refusal rate monitoring.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path


LOGS_DIR = Path("logs")
AUDIT_LOG_FILE = LOGS_DIR / "rag_audit.log"


class MetricsTracker:
    """
    Thread-safe tracker for real-time application metrics and telemetry.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_queries = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_refusals = 0
        self.total_errors = 0
        self.latencies: list[float] = []

    def record_query(
        self,
        duration_ms: float,
        status: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ):
        with self._lock:
            self.total_queries += 1
            self.latencies.append(duration_ms)
            if len(self.latencies) > 1000:
                self.latencies.pop(0)

            self.total_tokens_in += tokens_in
            self.total_tokens_out += tokens_out

            if status.startswith("refused"):
                self.total_refusals += 1
            elif status == "error":
                self.total_errors += 1

    def get_metrics(self) -> dict:
        with self._lock:
            uptime = round(time.time() - self.start_time, 2)
            avg_lat = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
            p95_lat = 0.0
            if self.latencies:
                sorted_lats = sorted(self.latencies)
                p95_idx = int(len(sorted_lats) * 0.95)
                p95_lat = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

            refusal_rate = (self.total_refusals / self.total_queries) if self.total_queries > 0 else 0.0

            # Cost estimates ($0.150 / 1M input tokens, $0.600 / 1M output tokens for gpt-4o-mini)
            cost_in = (self.total_tokens_in / 1_000_000) * 0.15
            cost_out = (self.total_tokens_out / 1_000_000) * 0.60
            total_cost = cost_in + cost_out

            return {
                "uptime_seconds": uptime,
                "total_queries": self.total_queries,
                "total_refusals": self.total_refusals,
                "refusal_rate": round(refusal_rate, 4),
                "total_errors": self.total_errors,
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "token_usage": {
                    "input_tokens": self.total_tokens_in,
                    "output_tokens": self.total_tokens_out,
                    "total_tokens": self.total_tokens_in + self.total_tokens_out,
                    "estimated_cost_usd": round(total_cost, 6),
                },
            }


# Global metrics instance
metrics = MetricsTracker()


def log_audit_event(
    question: str,
    status: str,
    sources: list[dict],
    duration_ms: float,
    diagnostics: dict | None = None,
    client_ip: str | None = None,
):
    """
    Write a structured JSON audit record to logs/rag_audit.log.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip or "127.0.0.1",
        "question": question,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "sources_retrieved": len(sources),
        "source_files": [s.get("source") or s.get("document") for s in sources],
        "diagnostics": diagnostics or {},
    }

    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
