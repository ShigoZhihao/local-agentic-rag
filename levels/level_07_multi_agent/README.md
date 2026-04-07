# Level 7 — Multi-Agent RAG

**Core loop:** `user input → Facilitator → Synthesizer ⇄ Researcher → Validator → output`

The full production-grade system. A 4-agent loop controlled by LangGraph.
Zero API costs, 100% open-source, fully local.

**Models:** `gemma4:e4b` (planner) / `gemma4:e2b` (executor) — thinking models with reasoning visible in UI

## What you learn here

- LangGraph StateGraph: nodes, conditional edges, checkpointing, Human-in-the-Loop
- 4-agent architecture: Facilitator / Synthesizer / Researcher / Validator
- Prompt enrichment: Facilitator understands intent and asks clarifying questions
- Citation-grounded generation (NotebookLM style): preserve `original_text`, no hallucination
- LLM-as-Judge validation: 4-axis scoring (completeness, accuracy, relevance, faithfulness)
- Feedback loop: Validator → Facilitator → User → re-run (max 3 iterations)
- Thinking model integration: `<think>` tag parsing, thinking/answer separation
- Reflex UI: Chat / Ingest pages with multi-page routing

## Architecture

```
User Query
    │
    ▼
┌──────────────┐  needs clarification  ┌──────────────┐
│  Facilitator │ ◄──────────────────── │  User Input  │
│ (qwen2.5:9b) │ ──────────────────►  │              │
└──────┬───────┘                       └──────────────┘
       │ enriched_prompt
       ▼
┌──────────────┐   can answer directly  ┌──────────────┐
│  Synthesizer │ ──────────────────►    │  Validator   │
│ (qwen2.5:2b) │                        │ (qwen2.5:9b) │
└──────┬───────┘                        └──────┬───────┘
       │ needs retrieval                       │
       ▼                                       │ FAIL (avg < 80)
┌──────────────┐                               │
│  Researcher  │                               ▼
│  (no LLM)   │ ──── citations ──────► Facilitator
└──────────────┘                          (max 3 loops)
```

| Agent | Role | Model |
|-------|------|-------|
| **Facilitator** | Query enrichment, clarification questions | qwen2.5:9b |
| **Synthesizer** | Direct answer or citation-grounded generation | qwen2.5:2b |
| **Researcher** | Hybrid Search → Filter → Re-rank → Citation stitching | No LLM |
| **Validator** | LLM-as-Judge: completeness, accuracy, relevance, faithfulness | qwen2.5:9b |

## Setup

```bash
cd levels/level_05_agentic_rag

# Start Weaviate
docker compose up -d

# Install dependencies
uv venv
.venv/Scripts/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e . --no-deps

# Run Reflex UI
reflex run
```

## Files

```
level_05_agentic_rag/
├── src/
│   ├── config.py               # config.yaml → Pydantic Settings
│   ├── models.py               # Shared Pydantic data models
│   ├── agents/
│   │   ├── state.py            # RAGState (LangGraph shared state)
│   │   ├── facilitator.py
│   │   ├── synthesizer.py
│   │   ├── researcher.py
│   │   ├── validator.py
│   │   └── graph.py            # LangGraph StateGraph definition
│   ├── ingestion/
│   │   ├── loaders.py          # .txt .md .html .py .pdf .pptx
│   │   ├── chunkers.py         # 6 chunking strategies
│   │   ├── embedder.py         # BGE-M3 GPU wrapper
│   │   ├── vision_describer.py # Vision LLM for PDF/PPTX pages
│   │   └── pipeline.py         # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── weaviate_client.py
│   │   ├── hybrid_search.py    # BM25 + Semantic, alpha control
│   │   ├── metadata_filter.py
│   │   └── colbert_search.py   # Optional on-demand ColBERT
│   ├── reranking/
│   │   └── cross_encoder.py    # ms-marco-MiniLM-L-6-v2
│   ├── generation/
│   │   ├── llm_client.py       # Ollama OpenAI-compatible client
│   │   ├── prompts.py          # All prompt templates
│   │   └── generator.py
│   ├── evaluation/
│   │   ├── metrics.py          # Precision, Recall, MAP, MRR
│   │   ├── bm25_tuner.py       # k1 × b grid search
│   │   └── eval_runner.py
├── rxconfig.py                 # Reflex configuration
├── app/
│   ├── app.py                  # Reflex UI (Chat + Ingest pages)
│   └── state.py                # Reactive state management
├── tests/
├── config.yaml
├── docker-compose.yaml
└── pyproject.toml
```
