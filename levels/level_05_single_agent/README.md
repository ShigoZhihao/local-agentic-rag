# Level 5 — Single Agent

**Core loop:** `user input → ReAct Agent → [Tool: search] → LLM + citations → output`

The model decides *when* and *whether* to search. Instead of always retrieving,
a LangChain ReAct agent reasons about the query and invokes tools as needed.

**Model:** `gemma4:e4b` | **Framework:** LangChain (`create_react_agent`, `AgentExecutor`)

## What you learn here

- LangChain `create_react_agent` and `AgentExecutor`
- `Tool` abstraction: wrapping retrieval functions as agent tools
- ReAct prompting: Thought → Action → Observation loop
- Agent deciding when/whether to search vs. answer directly
- Tool call visibility in the UI

## Setup

```bash
cd levels/level_05_single_agent
uv venv && .venv/Scripts/activate
uv pip install -e .
docker compose up -d
ollama pull gemma4:e4b
streamlit run src/app.py
```

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Workflow patterns (StateGraph) | Level 6 |
| Multi-agent specialisation | Level 7 |
| MCP tools | Level 8 |

## Status

> **Placeholder** — implementation coming soon.
