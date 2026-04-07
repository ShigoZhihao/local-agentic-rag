# Level 4 — Advanced RAG

**Core loop:** `user input → hybrid search → filter → re-rank → LLM + citations → output`

Upgrade retrieval quality with hybrid search, metadata filtering, and
cross-encoder re-ranking. Also extends ingestion to PDF and PPTX with Vision LLM.

**Model:** `gemma4:e4b` | **Framework:** LangChain

## What you learn here

- Hybrid search: BM25 (keyword) + Semantic (vector) combined via alpha parameter
- Metadata filtering: filter by source_type, date range, custom tags
- Cross-encoder re-ranking: `ms-marco-MiniLM-L-6-v2` for precise top-k selection
- ColBERT late-interaction search (optional, on-demand)
- Multimodal ingestion: PDF (PyMuPDF) + PPTX (python-pptx), Vision LLM for visual elements
- All 6 chunking strategies: Recursive, HTML, Python, Semantic, Example, Visual
- BM25 parameter tuning: k1 x b grid search, optimised for MAP@10
- Evaluation metrics: Precision@k, Recall@k, MAP@k, MRR@k

## Setup

```bash
cd levels/level_04_advanced_rag
uv venv && .venv/Scripts/activate
uv pip install -e .
docker compose up -d
ollama pull gemma4:e4b
streamlit run src/app.py
```

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Tool use (ReAct agent) | Level 5 |
| Workflow patterns (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |
| MCP tools | Level 8 |

## Status

> **Placeholder** — implementation coming soon.
