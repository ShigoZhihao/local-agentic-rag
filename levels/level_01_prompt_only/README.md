# Level 1 — Prompt Only

**Core loop:** `user input → LLM → output`

The simplest possible agent. No tools, no retrieval, no external memory.
The only levers are the system prompt and model parameters.

## What you learn here

- How an LLM call works at the API level
- How conversation history (message list) creates the illusion of memory
- How much the system prompt alone shapes behaviour

## Setup

```bash
cd levels/level_01_prompt_only
uv venv && source .venv/bin/activate
uv pip install -e .
```

Make sure Ollama is running and the model is available:

```bash
ollama pull qwen3.5:2b
ollama serve
```

## Run

```bash
streamlit run src/app.py
```

## What to try

1. Ask a factual question. Note the model answers from its training data only — no external lookup.
2. Edit the system prompt in the sidebar (e.g. "Answer only in Japanese") and send the next message.
3. Ask something the model doesn't know (recent events, your personal data). It will hallucinate or say it doesn't know. **This is the motivation for Level 3 (RAG).**

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Better prompts (CoT, few-shot) | Level 2 |
| External knowledge (documents) | Level 3 |
| Tool use | Level 5 |
| Agent loop | Level 5 |

## Files

```
level_01_prompt_only/
├── src/
│   ├── config.py       # Pydantic config loader
│   ├── llm_client.py   # Thin OpenAI-compatible wrapper for Ollama
│   └── app.py          # Streamlit chat UI
├── config.yaml         # Model & system prompt settings
└── pyproject.toml      # Dependencies: openai, pydantic, pyyaml, streamlit
```
