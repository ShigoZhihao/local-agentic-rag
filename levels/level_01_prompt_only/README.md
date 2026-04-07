# Level 1 — Prompt Only

**Core loop:** `user input → LLM → output`

The simplest possible agent. No tools, no retrieval, no external memory.
The only levers are the system prompt and model parameters.

**Model:** `gemma4:e2b` (thinking model — reasoning is shown in a collapsible panel)

## What you learn here

- How an LLM call works at the API level (raw openai SDK)
- How conversation history (message list) creates the illusion of memory
- How much the system prompt alone shapes behaviour
- How thinking models expose their reasoning via `<think>` tags
- Claude-inspired chat UI with Siemens brand colour

## Setup

```bash
cd levels/level_01_prompt_only
uv venv
.venv/Scripts/activate
uv pip install -e .
```

Make sure Ollama is running and the model is available:

```bash
ollama pull gemma4:e2b
ollama serve
```

## Run

```bash
reflex run
```

## What to try

1. Ask a factual question. Note the model answers from its training data only — no external lookup.
2. Expand the **"Thinking"** step to see the model's internal reasoning process.
3. Edit the system prompt in the sidebar (e.g. "Answer only in Japanese") and send the next message.
4. Ask something the model doesn't know (recent events, your personal data). It will hallucinate or say it doesn't know. **This is the motivation for Level 3 (RAG).**

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| Better prompts (CoT, few-shot) | Level 2 |
| External knowledge (documents) | Level 3 |
| Hybrid search, re-ranking | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow patterns (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |

## Files

```
level_01_prompt_only/
├── rxconfig.py         # Reflex configuration
├── app/
│   ├── app.py          # Reflex chat UI (pages + components)
│   └── state.py        # Reactive state management
├── src/
│   ├── config.py       # Pydantic config loader
│   ├── llm_client.py   # OpenAI-compatible wrapper with <think> tag parsing
│   ├── ollama_models.py # Ollama model listing & context window
│   └── metrics.py      # CPU/GPU/VRAM/RAM resource metrics
├── config.yaml         # Model & system prompt settings
└── pyproject.toml      # Dependencies: openai, pydantic, pyyaml, reflex
```
