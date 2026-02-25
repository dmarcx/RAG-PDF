# AI-Powered Engineering Document Control System

An intelligent document retrieval and Q&A system for engineering specifications, technical reports, and procurement documents — built for the **MANARA Project** by **SLD Engineering**.

---

## Overview

This system enables engineering teams to query large collections of technical PDFs using natural language — in Hebrew or English. It combines semantic search, keyword matching, and AI-based reranking to deliver precise, source-cited answers from complex engineering documents.

---

## Key Capabilities

| Feature | Description |
|---------|-------------|
| Natural Language Q&A | Ask questions in Hebrew or English across multiple documents |
| Engineering Translation | Hebrew queries auto-translated using domain-specific terminology (שנאים→Transformers, משאבות→Pumps, etc.) |
| Hybrid Search | Semantic embeddings + BM25 keyword search combined via Reciprocal Rank Fusion |
| Contextual Retrieval | LLM-generated context enriches each chunk at indexing time |
| Query Expansion | Auto-generates alternative queries with technical unit variations (mio m³, MCM, million cubic meters) |
| Cross-Encoder Reranking | Optional Cohere reranker for precision-optimized result ordering |
| Multi-Document Filtering | Query all documents, one, or any custom selection simultaneously |
| Document Summarization | Full-document AI summary on demand |
| Bilingual UI | Full English / Hebrew interface with RTL support |
| Pre-built Index | ChromaDB vector index included — no re-indexing required on deployment |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language Model | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| Vector Database | ChromaDB (local persistent, tracked via Git LFS) |
| PDF Extraction | pdfplumber (including table-to-text conversion) |
| Keyword Search | BM25Okapi (rank-bm25) |
| Reranker | Cohere rerank-v3.5 (optional) |
| Web Interface | Streamlit |

---

## Setup

### Prerequisites
- Python 3.10+
- Anthropic API key
- Cohere API key *(optional — enables cross-encoder reranking)*
- Git LFS *(required to pull the pre-built vector index)*

### Installation

```bash
git lfs pull          # downloads the pre-built ChromaDB index
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
```

> On cloud deployments, set these as environment variables in your hosting dashboard instead of a `.env` file.

### Running the Application

```bash
streamlit run app.py
```

### Indexing New Documents

Place PDF files in the `pdfs/` directory, then use the **Admin** section in the sidebar (password required), or run directly:

```bash
python rag.py
```

---

## Architecture

### Retrieval Pipeline

```
User Question (Hebrew/English)
        ↓
[1] Engineering Translation → professional English (Claude Haiku)
        ↓
[2] Query Expansion → 3 variants (abbreviated + expanded technical terms)
        ↓
[3] Hybrid Search × 3 queries → top-50 chunks each (BM25 + Semantic, RRF)
        ↓
[4] Merge by (source, page) → keep best RRF score per page
        ↓
[5] Cohere Reranker → top-10 pages from up to 100 candidates
        ↓
[6] Claude → Answer with source + page citations (in original question language)
```

### Indexing Pipeline

```
PDF File
  ↓ pdfplumber → text + tables (structured)
  ↓ Section header extraction (deepest numbered heading)
  ↓ LLM context generation (1 call per page, Contextual Retrieval)
  ↓ Chunking (500 chars, 100 overlap)
  ↓ Each chunk = [LLM context] + [prefix] + [text]
  ↓ ChromaDB — chunk text as vector, full page stored in metadata
```

---

## Web Interface

### Sidebar
- **Language toggle** — switch between English / עברית (RTL layout applied automatically)
- **Loaded Documents** — list of indexed PDFs with per-document delete
- **Admin section** (password-protected) — upload new PDFs or scan the `pdfs/` folder

### Main Area
| Mode | Description |
|------|-------------|
| ❓ Free Question | Multi-turn chat with document filter (all / one / custom selection) |
| 📋 Summarize Document | Full AI summary of a selected document |

---

## Project Structure

```
├── app.py              # Streamlit web interface
├── rag.py              # Core RAG pipeline
├── debug_retrieval.py  # Full pipeline diagnostics
├── debug_page.py       # Per-page chunk inspection
├── requirements.txt
├── SLD LOGO.png        # Company logo
├── .gitattributes      # Git LFS configuration (chroma.sqlite3)
├── pdfs/               # PDF documents (not tracked in git)
└── chroma_db/          # Vector database (tracked via Git LFS)
```
