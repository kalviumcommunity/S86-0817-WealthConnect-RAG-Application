# WealthConnect — AI-Powered Wealth Advisory Knowledge Assistant

> **Ask the Question. Find the Approved Source. Advise with Confidence.**

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/kalviumcommunity/S86-0817-WealthConnect-RAG-Application.git
cd S86-0817-WealthConnect-RAG-Application
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Open .env and fill in your API keys — never commit this file
```

### 5. Add approved wealth documents
Place your approved documents (`.txt`, `.pdf`, `.docx`) in the `data/` folder.

### 6. Ingest documents into the vector store
```bash
python -m src.ingest
```

### 7. Run the assistant
```bash
python -m src.app
```

---

## Project Structure

```
S86-0817-WealthConnect-RAG-Application/
├── data/                        # Approved wealth documents (git-ignored — sensitive)
├── src/                         # Application source code
│   ├── __init__.py
│   ├── document_loader.py       # Multi-format intake: .txt .md .pdf .html
│   ├── ingest.py                # Chunking, metadata tagging, full pipeline
│   ├── embeddings.py            # OpenAI embeddings + ChromaDB vector store
│   ├── retrieval.py             # Semantic search + source formatting
│   ├── prompt_builder.py        # System/user role construction, prompt variants
│   ├── prompt_experiments.py    # GY3.13 prompt comparison & demonstration runner
│   ├── model_params.py          # Parameter presets: temperature, max_tokens, etc.
│   ├── parameter_experiments.py # GY3.16 parameter comparison runner
│   └── app.py                   # Main RAG pipeline + LLM answer generation
├── prompts/                     # Prompt templates (editable without touching code)
│   ├── system_prompt_strict.txt   # Full rules, prose output, source citation
│   ├── system_prompt_json.txt     # Rules + structured JSON output format
│   ├── system_prompt_concise.txt  # Minimal, 2–3 sentence cap
│   ├── rag_system_prompt.txt      # Original base system prompt
│   └── fallback_message.txt       # Safe fallback message template
├── outputs/                     # Generated answers, logs, evaluation results (git-ignored)
├── .env                         # Real secrets — NEVER committed
├── .env.example                 # Template showing required keys — committed, no values
├── .gitignore
├── requirements.txt             # Pinned dependency versions
└── README.md
```

## Prompt Design (GY3.13)

WealthConnect uses a deliberate system/user role separation to control every answer.

### System vs User Role

| Role | Purpose |
|------|---------|
| `system` | Sets who the assistant is, the grounding rules, output format, and fallback behaviour. Static per session. |
| `user` | Carries the RM's question + retrieved document context for that specific turn. Changes every query. |

The system message is where grounding is enforced — *"answer only from the context"* and *"use the fallback when unsure"* both live there. Keeping prompt templates in `prompts/` files means the Wealth team can adjust tone and rules without touching Python code.

### Prompt Variants

Three variants are available, each with a different output contract:

| Variant | File | Output |
|---------|------|--------|
| `strict` | `system_prompt_strict.txt` | Prose answer with source line — default for production |
| `concise` | `system_prompt_concise.txt` | 2–3 sentence cap — for quick lookups |
| `json` | `system_prompt_json.txt` | Structured `{ answer, source, confidence }` — for API consumers |

### Running the Prompt Experiments

```bash
python -m src.prompt_experiments
```

Runs four experiments:
1. **Role separation** — prints the raw system and user messages sent to the API
2. **Variant comparison** — same question through all three variants side by side
3. **Ambiguous vs clear** — shows how task + scope + format + fallback in the user prompt produces better output
4. **Fallback behaviour** — verifies all variants return the safe fallback when no context is found

---

## Model Parameters (GY3.16)

WealthConnect uses named parameter presets to control LLM output determinism, length, and cost.

| Parameter | Role |
|-----------|------|
| `temperature` | Randomness (0.0 = deterministic/factual, 2.0 = very creative). RAG uses 0.0. |
| `max_tokens` | Hard cap on output length. Output tokens are billed — caps cost directly. |
| `top_p` | Nucleus sampling — alternative to temperature. Tune one, not both. |
| `stop` | Sequence(s) that end generation early, e.g. `["\n\n"]` to cut at first paragraph. |

### Named Presets

| Preset | temperature | max_tokens | top_p | stop | Use |
|--------|------------|-----------|-------|------|-----|
| `rag_grounded` | 0.0 | 400 | 1.0 | none | **Production default** — grounded prose |
| `rag_strict` | 0.0 | 200 | 1.0 | `\n\n` | One paragraph, cut early |
| `rag_json` | 0.0 | 500 | 1.0 | none | JSON output variant |
| `high_temperature` | 1.0 | 400 | 1.0 | none | Demo only — shows drift |
| `low_top_p` | 1.0 | 400 | 0.1 | none | Tight nucleus sampling demo |

### Running the Parameter Experiments

```bash
python -m src.parameter_experiments
```

Runs five experiments:
1. **Temperature 0.0 vs 1.0** — same prompt twice at each setting; stable vs drifting
2. **max_tokens cap** — 50 / 150 / 400 tokens; shows `finish_reason` flipping to `length`
3. **Stop sequences** — with and without `\n\n` stop string
4. **top_p vs temperature** — two routes to focused output; why not to tune both
5. **Production preset comparison** — all named presets on the same wealth question

---

## Document Loading (GY3.19)

WealthConnect's approved document corpus arrives in multiple formats. `document_loader.py` converts all of them into a single common form — plain text — before chunking and embedding.

### Supported Formats

| Format | Loader | Notes |
|--------|--------|-------|
| `.txt` | `path.read_text()` | Plain-text policy extracts, notes |
| `.md` | `path.read_text()` | Markdown guidelines, tax rules |
| `.pdf` | `pypdf.PdfReader` | Product brochures, investment policies (primary format) |
| `.html` / `.htm` | `BeautifulSoup.get_text()` | Web-exported compliance pages, product pages |

### PDF handling
PDFs are the hard case — product brochures may have multi-column layouts, and some legacy documents may be scanned images with no extractable text. The loader warns on empty-text PDFs so the admin team can re-upload a text-selectable version.

### Failure handling
One corrupt file never stops the rest of the corpus. Every file is loaded inside a `try/except` — failures are logged to a skipped list and reported at the end. The intake report shows loaded count, skipped count, empty documents, and total characters loaded.

### Running document intake

```bash
# Install new dependencies first
pip install -r requirements.txt

# Run intake against data/
python -m src.document_loader

# Run the full ingest pipeline (intake → chunk → metadata)
python -m src.ingest
```

### Sample Documents

Four sample documents are included in `data/` covering all supported formats:

| File | Format | Type |
|------|--------|------|
| `sample_investment_policy.txt` | `.txt` | Investment policy |
| `sample_tax_rules.md` | `.md` | Tax rules |
| `sample_product_brochure.html` | `.html` | Product brochure |
| `sample_eligibility_guidelines.txt` | `.txt` | Eligibility guidelines |

---

## Overview

WealthConnect is an AI-powered knowledge assistant built for a retail bank's wealth division. It enables **Relationship Managers** to ask natural-language questions and receive accurate, source-grounded answers drawn exclusively from the bank's current approved wealth-management documents — investment policies, product brochures, tax rules, eligibility guidelines, and more.

The goal is not to replace Relationship Managers or make investment decisions. WealthConnect exists to help them **find approved information faster, verify sources instantly, and advise customers with greater consistency and confidence**.

---

## Team

| Name | Role |
|------|------|
| G Yashmieen | Team Member |
| Dodla Bhanu Teja Reddy | Team Member |
| Kishore R | Team Member |

---

## The Problem

Relationship Managers regularly need answers to questions like:

- What are the key features of a particular investment product?
- What are the eligibility requirements?
- What are the applicable tax rules?
- What are the investment limits and restrictions?
- Which policy version is current and approved?

Without a centralized tool, they manually search through multiple documents — leading to **inconsistent advice, longer response times, and difficulty verifying information sources**.

---

## Solution

WealthConnect implements a **Retrieval-Augmented Generation (RAG)** pipeline over the bank's approved wealth-management document corpus. Every answer is:

- Grounded in approved documents
- Paired with source references (document name, version, section)
- Filtered to use only the **latest approved version** of each document
- Accompanied by a safe fallback if sufficient information is not found

---

## Key Features

### For Relationship Managers
- Secure login with role-based access
- Natural-language question input
- AI-generated answers grounded in approved documents
- Source references with document name, version, and section
- Conversation history
- 👍 / 👎 answer feedback

### For Wealth Admins
- Document upload and categorization
- Product assignment and version management
- Approval status management
- Document replacement and removal
- Monitoring of unanswered questions and feedback
- Frequently asked question tracking

### AI / RAG Pipeline
- Document parsing and text extraction
- Chunking and cleaning
- Metadata tagging (product, version, approval status, effective date)
- Semantic embedding and vector indexing
- Metadata-filtered retrieval (current + approved versions only)
- Grounded answer generation via LLM
- Source citation and fallback handling

---

## User Workflow

```
Relationship Manager
        │
        ▼
     Login
        │
        ▼
  Ask a Question (natural language)
        │
        ▼
  RAG Pipeline searches approved wealth documents
        │
        ▼
  Filter: Current + Approved versions only
        │
        ▼
  Retrieve relevant chunks
        │
        ▼
  LLM generates grounded answer
        │
        ▼
  Display Answer + Source Reference
        │
        ▼
  Relationship Manager verifies → Customer Interaction
        │
        ▼
     Feedback
```

---

## Document Sources

| Data Source | Purpose | Owner |
|-------------|---------|-------|
| Investment Policies | Investment rules and guidelines | Wealth Team |
| Product Brochures | Product features and details | Product Team |
| Tax Rules | Applicable tax information | Tax/Compliance Team |
| Product Terms | Conditions and restrictions | Wealth Team |
| Eligibility Guidelines | Customer/product eligibility | Wealth Team |
| Risk Documents | Product risk information | Wealth/Compliance |
| Compliance Guidelines | Approved communication guidelines | Compliance Team |

---

## Technical Architecture

```
Relationship Manager
        │
        ▼
    Frontend
        │
        ▼
   Backend API
        │
   ┌────┴────┐
   ▼         ▼
Auth &     AI / RAG
Access     Pipeline
Control        │
               ▼
        Vector / Search DB
               │
               ▼
    Approved Wealth Documents
```

> The final technology stack will be confirmed during technical validation against available bank infrastructure.

---

## Document Metadata Schema

Each uploaded document carries metadata used for filtering and version control:

| Field | Description |
|-------|-------------|
| `document_name` | Name of the document |
| `document_type` | Category (policy, brochure, tax rule, etc.) |
| `version` | Version number |
| `approval_status` | Approved / Pending / Superseded |
| `effective_date` | Date the document became active |
| `expiry_review_date` | Next review or expiry date |
| `product` | Associated product(s) |
| `owner` | Document owner team |
| `last_updated` | Last modification date |

---

## Example Interaction

**Question:**
> "What are the tax implications of this investment product?"

**WealthConnect Response:**
> Based on the current approved tax guidance and product documentation, the applicable tax treatment is described in the relevant approved material.

**Sources:**
> - Tax Rules — Version 3.2 | Approved: XX/XX/XXXX
> - Product Brochure — Version 4.1 | Section: Tax Treatment

**When information is not found:**
> "I couldn't find enough information in the current approved wealth documents to answer this question. Please verify with the appropriate wealth, tax, legal, or compliance team."

WealthConnect will **never** generate unsupported investment, tax, or policy information.

---

## User Stories (V1.0)

| ID | Story |
|----|-------|
| US-01 | As a RM, I want to ask wealth questions in natural language to find information quickly |
| US-02 | As a RM, I want WealthConnect to prioritize the latest approved documents |
| US-03 | As a RM, I want to see the source document and section used for each answer |
| US-04 | As a RM, I want to ask about investment product features and conditions |
| US-05 | As a RM, I want to find relevant approved tax information |
| US-06 | As a Wealth Admin, I want to upload and manage approved documents |
| US-07 | As a Wealth Admin, I want to replace outdated documents |
| US-08 | As a Wealth Admin, I want to view unanswered questions to identify documentation gaps |
| US-09 | As a RM, I want to provide feedback so incorrect responses can be reviewed |

---

## V1.0 Scope

### In Scope
- Secure RM and Admin login
- Natural-language question input
- AI-generated, document-grounded answers
- Current and approved document retrieval
- Product, tax, and policy search
- Source and version references
- Conversation history
- Answer feedback (👍 / 👎)
- Admin: document upload, versioning, approval, categorization
- Admin: question and feedback monitoring
- Fallback handling for unanswered questions

### Out of Scope (V1.0)
- Automatic investment or portfolio decisions
- Personalized financial advice without human review
- Automated tax filing or transaction execution
- Customer account modification
- Legal advice
- Voice assistant
- Predictive investment recommendations
- Full wealth-management system replacement

---

## Security & Access Control

| Role | Permissions |
|------|------------|
| Relationship Manager | Ask questions, view permitted documents, view sources, view conversation history, submit feedback |
| Wealth Administrator | All RM permissions + upload/manage documents, update versions, view all questions, monitor system usage |

Access follows the principle of least privilege. Sensitive internal documents are not exposed to unauthorized roles.

---

## Non-Functional Requirements

| Requirement | Detail |
|-------------|--------|
| Accuracy | Answers must be supported by approved documents only |
| Performance | Response time target to be finalized during technical validation |
| Security | Authentication and role-based access controls |
| Privacy | Data used only for WealthConnect service purposes |
| Maintainability | Admins can update documents without developer involvement |
| Reliability | Graceful handling of missing docs, search failures, and conflicting versions |

---

## Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Outdated documents used | Medium | High | Version tracking and effective dates |
| AI generates unsupported answers | Medium | High | Retrieval-grounded generation only |
| Wrong document retrieved | Medium | High | Metadata filtering |
| Unapproved document used | Medium | High | Approval-status filtering |
| Conflicting policies | Medium | High | Prioritize current approved version |
| Sensitive information exposed | Low/Medium | High | Auth + role-based access |
| Low RM adoption | Medium | Medium | Simple, intuitive UX |

---

## KPIs

| KPI | Measurement | Target |
|-----|------------|--------|
| Search Time Reduction | Avg. time to find information | TBD — baseline at launch |
| Answer Accuracy | Correct answers vs approved docs | TBD — 30 days |
| RM Adoption | Active RMs using WealthConnect | TBD — 30 days |
| Answer Satisfaction | Helpful / total rated responses | TBD — 30 days |
| Response Time | Avg. time to generate answer | TBD — at launch |
| Source Verification | Answers with valid source references | TBD — at launch |
| Unanswered Questions | Questions requiring escalation | TBD reduction — 60 days |

---

## Launch Criteria

WealthConnect is ready for pilot when:

- [ ] Approved wealth documents are uploaded and indexed
- [ ] Document versions are validated
- [ ] Approval status filtering is working
- [ ] Authentication and role-based access are functional
- [ ] Questions return answers grounded in retrieved documents
- [ ] Every answer includes a source reference
- [ ] Current approved documents are prioritized over outdated ones
- [ ] Unsupported questions trigger a safe fallback response
- [ ] Wealth admins can update documents independently
- [ ] Feedback collection is functional
- [ ] Initial accuracy testing is complete
- [ ] Wealth stakeholders approve the pilot

---

## Future Enhancements (V2+)

- CRM and banking system integration
- Customer profile-aware retrieval
- Product comparison features
- Automated policy change detection and notifications
- Multilingual support
- Voice assistant
- Compliance review workflow
- Microsoft Teams / Slack integration
- Advanced analytics and reporting
- Automated escalation to compliance

---

## Stakeholders

| Stakeholder | Role |
|-------------|------|
| Relationship Managers | Primary users |
| Wealth Division | Business owner |
| Wealth / Policy Team | Document owners |
| Compliance Team | Compliance validation |
| Legal Team | Legal and tax content review |
| IT / Engineering Team | Implementation and maintenance |
| Data / AI Team | AI and document-processing pipeline |
| Bank Management | Business approver |

---

*WealthConnect v1.0 — Status: Draft*
