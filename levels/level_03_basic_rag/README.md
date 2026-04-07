# Level 3 — Basic RAG

**Core loop:** `user input → semantic search → LLM + context → output`

Introduce external knowledge. The model can now answer questions about your
documents instead of relying solely on its training data.

**Model:** `gemma4:e2b` | **Framework:** LangChain (first appearance)

## What you learn here

- LangChain LCEL chains (`ChatPromptTemplate | LLM | OutputParser`)
- Document ingestion: load, chunk (RecursiveCharacterTextSplitter), embed (BGE-M3)
- Weaviate vector storage (HNSW, cosine similarity)
- Semantic vector search — pure embedding-based retrieval
- Context injection: append retrieved chunks to the prompt
- Basic citation: show which document the answer came from

## Setup

```bash
cd levels/level_03_basic_rag
uv venv && .venv/Scripts/activate
uv pip install -e .
docker compose up -d   # start Weaviate
ollama pull gemma4:e2b
streamlit run src/app.py
```

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| BM25 keyword search | Level 4 |
| Metadata filtering | Level 4 |
| Cross-encoder re-ranking | Level 4 |
| PDF/PPTX ingestion | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow patterns (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |

## Status

> **Placeholder** — implementation coming soon.
