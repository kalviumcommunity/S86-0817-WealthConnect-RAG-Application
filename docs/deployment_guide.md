# Deployment & Operations Guide
## WealthConnect — Grounded Wealth Advisory RAG Assistant

**Target Milestone**: 3.49 Deployment, Documentation & Delivery  
**Application**: WealthConnect Enterprise RAG Service  

---

## 1. Architecture Overview

WealthConnect is designed to be deployed either locally, in a staging environment, or in a hardened enterprise container environment:

```text
               +----------------------------------------+
               | Relationship Manager Browser (Port 5500)|
               +-------------------+--------------------+
                                   |
                         HTTP / SSE Requests
                                   |
                                   v
             +---------------------+---------------------+
             |     Unified FastAPI Service (Port 8000)   |
             |  /health  |  /query  |  /stream  | /upload|
             +---------------------+---------------------+
                                   |
             +---------------------+---------------------+
             |         Query Cache & Audit Logger        |
             +---------------------+---------------------+
                                   |
             +---------------------+---------------------+
             |     Hybrid Search Engine & Guardrails     |
             +----------+-----------------------+--------+
                        |                       |
                        v                       v
               +-----------------+     +-----------------+
               | Vector Collection|    |  BM25 Lexical   |
               | (Chroma / Memory)|    |      Index      |
               +-----------------+     +-----------------+
```

---

## 2. Local Setup & Execution

### Prerequisites
- Python 3.10 to 3.12
- Git

### Step 1: Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Environment Configuration
Copy the template configuration and fill in optional secrets:
```powershell
Copy-Item .env.example .env
```
*(Note: If no OpenAI API key is supplied, WealthConnect runs automatically in deterministic offline mock mode).*

### Step 4: Ingestion Verification
Validate that all approved policies and brochures are parsed:
```powershell
python -m src.ingest
```

### Step 5: Start the API
```powershell
uvicorn rag_api:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`
- Real-time Metrics: `http://localhost:8000/metrics`

### Step 6: Start the Frontend UI
In a separate terminal:
```powershell
python -m http.server 5500 --directory frontend
```
Open `http://localhost:5500` in your web browser.

---

## 3. Docker Container Deployment

### Build the Image
```bash
docker build -t wealthconnect-rag:latest .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Verify Container Health
```bash
docker ps
curl http://localhost:8000/health
```

---

## 4. Operational Monitoring & Auditing

- **Audit Logs**: All queries, retrieved chunks, and refusal verdicts are recorded in JSON Lines format in `logs/rag_audit.log`.
- **Metrics Telemetry**: Query `/metrics` to monitor:
  - Cache hit ratio
  - P95 latency
  - Refusal rates
  - Token counts and estimated USD costs
