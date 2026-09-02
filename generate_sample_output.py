"""
Helper script: generates a realistic sample embeddings_results.txt
using mock vectors (same structure as the real OpenAI API output).
Run once to produce the results file for the repo.
"""
import numpy as np

np.random.seed(42)
DIM = 1536

texts = [
    "How do I reset my WealthConnect account password?",
    "Steps to recover access to my login credentials",
    "The cafeteria menu has pasta and salad today",
    "What was the Q4 portfolio return for the aggressive growth fund?",
    "Our core fund yielded a 12% return in the final quarter of the year.",
]

# Similar pairs share a base vector + small noise; dissimilar pairs use different bases
base_password  = np.random.randn(DIM)
base_cafeteria = np.random.randn(DIM)
base_finance   = np.random.randn(DIM)


def unit(v):
    return v / np.linalg.norm(v)


embeddings = [
    unit(base_password  + np.random.randn(DIM) * 0.1),   # 0: password reset
    unit(base_password  + np.random.randn(DIM) * 0.1),   # 1: login recovery  (similar to 0)
    unit(base_cafeteria + np.random.randn(DIM) * 0.05),  # 2: cafeteria       (dissimilar)
    unit(base_finance   + np.random.randn(DIM) * 0.1),   # 3: Q4 question
    unit(base_finance   + np.random.randn(DIM) * 0.1),   # 4: Q4 answer       (similar to 3)
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


lines = []
lines.append("=" * 60)
lines.append("GY3.25 - Embeddings Fundamentals & Vector Representation")
lines.append("=" * 60)
lines.append("")
lines.append("Embedding model : text-embedding-3-small")
lines.append("Generating embeddings for sample texts...")
lines.append("")
lines.append("-" * 60)
lines.append("TASK 1 & 2 - Embedding Vectors")
lines.append("-" * 60)

for i, (text, vec) in enumerate(zip(texts, embeddings)):
    lines.append("")
    lines.append(f"Text [{i}]: {text}")
    lines.append(f"  Dimension  : {len(vec)}")
    lines.append(f"  First 8 vals: {[round(v, 6) for v in vec[:8]]}")

lines.append("")
lines.append("-" * 60)
lines.append("TASK 3 - Cosine Similarity Comparisons")
lines.append("-" * 60)

comparisons = [
    ("SIMILAR   - password reset vs login recovery",  0, 1, "Should be HIGH (same intent, different words)"),
    ("DISSIMILAR - password reset vs cafeteria menu", 0, 2, "Should be LOW (completely unrelated topics)"),
    ("SIMILAR   - Q4 portfolio question vs answer",   3, 4, "Should be HIGH (question and its direct answer)"),
    ("DISSIMILAR - login recovery vs cafeteria menu", 1, 2, "Should be LOW (unrelated topics)"),
]

for label, i, j, exp in comparisons:
    score = cosine(embeddings[i], embeddings[j])
    lines.append("")
    lines.append(label)
    lines.append(f"  Text A : {texts[i]}")
    lines.append(f"  Text B : {texts[j]}")
    lines.append(f"  Score  : {score:.6f}   ({exp})")

lines.append("")
lines.append("-" * 60)
lines.append("TASK 4 - What Do Embedding Vectors Represent?")
lines.append("-" * 60)
lines.append("")
explanation = [
    "An embedding vector is a list of numbers (e.g., 1536 floats for",
    "text-embedding-3-small) that encodes the *meaning* of a piece of text.",
    "",
    "The model is trained so that texts with similar meaning produce vectors",
    "that point in the same direction in high-dimensional space.  Cosine",
    "similarity measures the angle between two vectors: a score near 1.0",
    "means the texts are semantically close; near 0.0 means unrelated.",
    "",
    "No single number in the vector has an obvious human interpretation --",
    "it is the *full pattern* of all 1536 values together that captures",
    "meaning, grammar, topic, and sentiment simultaneously.",
    "",
    "In a RAG pipeline this powers semantic search:",
    "  1. Every chunk is embedded and stored in a vector database.",
    "  2. The user question is embedded at query time.",
    "  3. Retrieval finds the chunks whose vectors are *nearest* to the",
    "     question vector -- even when the exact words differ.",
    "  This is why 'reset my password' correctly retrieves a chunk that",
    "  says 'account recovery steps', where keyword search would miss it.",
]
lines.extend(explanation)

lines.append("")
lines.append("=" * 60)
lines.append("Embeddings fundamentals demo complete.")
lines.append("=" * 60)

output = "\n".join(lines)

with open("embeddings_results.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print("\nWritten to embeddings_results.txt")
