# End-to-End Delivery Demo

This record shows the local request path used for the sprint. It uses the
bundled WealthConnect sample corpus and the offline mock mode, so no secret is
included and no external API call is required.

## Run

From the repository root:

```powershell
python rag_pipeline.py
python rag_evaluation.py
```

For the browser/API path, install `requirements.txt`, configure `.env`, then
run these commands in separate terminals:

```powershell
python -m src.ingest
uvicorn rag_api:app --reload --port 8000
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` and submit a question. The browser sends the
question to `http://localhost:8000/query`.

## Sample Flow

Question:

```text
What was the Q4 portfolio return?
```

Offline demo output shape:

```text
Based on the provided context: [1] Source: q4_earnings_report.pdf (...)
Sources: q4_earnings_report.pdf, q4_earnings_report.pdf, q4_earnings_report.pdf
```

The API response contains the same user-facing fields:

```json
{
  "answer": "...12% return...",
  "sources": [
    {
      "source": "q4_earnings_report.pdf",
      "chunk_id": "q4_earnings_report.pdf:1",
      "section": "Performance"
    }
  ],
  "status": "answered"
}
```

The refusal path returns `status` beginning with `refused` and an empty source
list when retrieval is weak or off-topic. The frontend renders that status as
an explicit no-supporting-context state rather than presenting it as a normal
answer.

## Validation Evidence

The offline regression suite covers ingestion, indexing, retrieval tuning,
pipeline stages, citation validation, and answer-quality scoring:

```powershell
python -m unittest test_chat_history_manager test_chunking_strategies test_embeddings test_ingest test_prompt_templates test_rag_evaluation test_rag_pipeline test_retrieval_tuning test_similarity_search -v
```

`test_stream.py` is a separate smoke script and requires the streaming server
to be running before it is imported.

The final delivery tag is `sprint-2-rag-final`.