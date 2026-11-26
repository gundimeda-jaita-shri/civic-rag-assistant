# 🏛️ Multilingual Civic Scheme & Document Assistant

A citation-grounded Retrieval-Augmented Generation (RAG) system built to help citizens and caseworkers query government welfare schemes, eligibility rules, benefit schedules, and official circulars across **English, Hindi, Telugu, Tamil, Kannada, Marathi, and Bengali**.

---

## ✨ Key Features

- **Hybrid Semantic + Lexical Retrieval (RRF):** Combines dense multilingual embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via FAISS) with sparse exact-match scoring (`BM25Plus` via `rank_bm25`) fused through **Reciprocal Rank Fusion (RRF)**—preserving exact scheme codes, acreage figures, dates, and monetary thresholds alongside cross-lingual semantic intent.
- **Table-Aware PDF Ingestion & Deduplication:** Extracts page-level text and structured Markdown tables using `PyMuPDF` and `pdfplumber`, chunks with source/page provenance metadata, and prevents duplicate indexing via **SHA-256** file hashing backed by **SQLite**.
- **Strict Grounding & Verbatim Citations:** Uses **Google Gemini** with Pydantic-enforced structured outputs (`GroundedCivicResponse`) requiring verbatim source quotes, page numbers, confidence tiers, and explicit abstention when evidence is missing.
- **Interactive Streamlit Portal & Audit Reports:** Upload PDFs, filter queries by document, inspect raw FAISS + BM25 evidence chunks, view the SQLite Q&A audit trail, and export downloadable Markdown verification reports.
- **Headless CLI & Automated Benchmark Suite:** Run ingestion, queries, and `Recall@K` / `MRR` / grounding accuracy evaluations directly from the terminal.

---

## 🏗️ Project Architecture

```text
civic_rag_assistant/
├── core/                  # Configuration, structured logging, custom exceptions, SQLite registry
├── ingestion/             # SHA-256 hashing, PyMuPDF + pdfplumber PDF parser, provenance chunker
├── retrieval/             # Multilingual MiniLM embeddings, FAISS store, BM25Plus store, RRF hybrid retriever
├── generation/            # Pydantic citation schemas, anti-hallucination prompts, Gemini grounded generator
├── services/              # End-to-end RAG orchestrator & document lifecycle/rebuild manager
├── ui/                    # Institutional slate CSS styles & Markdown verification report exporter
├── eval/                  # Multilingual benchmark dataset, Recall@K/MRR metrics & CLI runner
├── tests/                 # Pytest unit & integration test suite
├── app.py                 # Main Streamlit Civic Registry web application
├── cli.py                 # Headless terminal interface
├── Dockerfile             # Production container definition
└── .github/workflows/ci.yml