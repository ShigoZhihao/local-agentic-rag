# Level 1 — Prompt Only

**Core loop:** `user input → LLM → output`

The simplest possible chatbot. No memory, no prompt engineering, no external data.
Build a working MVP and confirm what it can and cannot do.

**Model:** `gemma4:e4b`

## What you learn here

- Connecting to Ollama via the OpenAI-compatible API
- Loading config from YAML with Pydantic type validation
- Streaming responses token by token
- Parsing `<think>` tags from reasoning models
- The baseline limitations: no memory, no context, no tools

## Setup

```bash
cd levels/level_01_prompt_only
uv venv
.venv/Scripts/activate
uv pip install -e .
```

```bash
ollama pull gemma4:e4b
ollama serve
```

## Run

```bash
python main.py
```

## What to try

1. Ask a simple question and observe the streamed response.
2. Ask a follow-up question — notice the model has no memory of the previous turn.
3. Ask something that requires reasoning — observe the `<think>` section if the model supports it.
4. Type `quit` or press `Ctrl+C` to exit.

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Conversation memory | Level 2 |
| Prompt engineering (CoT, few-shot) | Level 2 |
| External knowledge (documents) | Level 3 |
| Hybrid search, re-ranking | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow patterns (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |

## Files

```
level_01_prompt_only/
├── main.py             # Entry point, user input loop
├── src/
│   ├── config.py       # Pydantic config loader
│   └── llm_client.py   # OpenAI-compatible Ollama wrapper with <think> parsing
├── config.yaml         # Model, base_url, temperature, system_prompt
└── pyproject.toml
```
