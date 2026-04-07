# Level 6 — Workflow Patterns

**Core loop:** `user input → retrieve → generate → evaluate → [loop OR end]`

Introduce LangGraph for the first time. A simple Evaluator-Optimizer loop:
generate an answer, evaluate it, and loop back if quality is insufficient.

**Models:** `gemma4:e2b` (generator) / `gemma4:e4b` (evaluator) | **Framework:** LangGraph

## What you learn here

- LangGraph `StateGraph`: nodes, edges, conditional routing
- `TypedDict` shared state across nodes
- `MemorySaver` checkpointing
- Evaluator-Optimizer loop pattern (self-correcting RAG)
- Two-model architecture: small model generates, large model evaluates

## Setup

```bash
cd levels/level_06_workflow_patterns
uv venv && .venv/Scripts/activate
uv pip install -e .
docker compose up -d
ollama pull gemma4:e2b gemma4:e4b
streamlit run src/app.py
```

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Multiple specialised agents | Level 7 |
| Human-in-the-Loop | Level 7 |
| Prompt enrichment (Facilitator) | Level 7 |
| MCP tools | Level 8 |

## Status

> **Placeholder** — implementation coming soon.
