"""Quick test script for the streaming endpoint."""
import urllib.request
import json

BASE = "http://localhost:8002"

# Health check
req = urllib.request.Request(BASE + "/health")
with urllib.request.urlopen(req) as r:
    h = json.loads(r.read())
print("=== GET /health ===")
print(json.dumps(h, indent=2))

# Stream test
payload = json.dumps({"question": "What was the Q4 portfolio return?"}).encode()
req = urllib.request.Request(
    BASE + "/query/stream",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("\n=== POST /query/stream ===")
token_count = 0
full_answer = ""

with urllib.request.urlopen(req) as r:
    for raw in r:
        line = raw.decode("utf-8").rstrip()
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])

        if ev["type"] == "citations":
            num_sources = len(ev["sources"])
            print(f"[citations] {num_sources} source(s):")
            for s in ev["sources"]:
                print(f"  {s['label']} {s['document']}  chunk_id={s['chunk_id']}")

        elif ev["type"] == "token":
            token_count += 1
            full_answer += ev["text"]

        elif ev["type"] == "done":
            print(f"[done] received {token_count} token(s)")
            print(f"[answer] {full_answer.strip()}")
            break

        elif ev["type"] == "error":
            print(f"[error] {ev['message']}")
            break

# Refusal test
print("\n=== POST /query/stream (should refuse) ===")
payload2 = json.dumps({"question": "What is the capital of France?"}).encode()
req2 = urllib.request.Request(
    BASE + "/query/stream",
    data=payload2,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req2) as r:
    for raw in r:
        line = raw.decode("utf-8").rstrip()
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev["type"] == "token":
            print(f"[refusal text] {ev['text'][:80]}...")
        elif ev["type"] == "done":
            print("[done]")
            break
