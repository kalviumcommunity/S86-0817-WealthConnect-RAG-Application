# WealthConnect — AI-Powered Wealth Advisory Knowledge Assistant

> **Ask the Question. Find the Approved Source. Advise with Confidence.**

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
