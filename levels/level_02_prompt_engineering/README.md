# Level 2 — Prompt Engineering

**Core loop:** `user input + conversation history → [prompt template] → LLM → output`

Build on Level 1 by applying prompt engineering techniques that dramatically
improve response quality without changing the model or adding external data.

**Model:** `gemma4:e2b` (thinking model — reasoning shown in collapsible panel)

## What you learn here

- Conversation history management with configurable window size
- Chain-of-Thought (CoT) prompting: step-by-step reasoning
- Few-shot prompting: in-context examples steer output format
- Structured output: JSON responses via prompt constraints
- System prompt design patterns (persona, constraints, output spec)
- How thinking models expose reasoning via `<think>` tags

## Setup

```bash
cd levels/level_02_prompt_engineering
uv venv
.venv/Scripts/activate
uv pip install -e .
```

```bash
ollama pull gemma4:e2b
ollama serve
```

## Run

```bash
reflex run
```

## What to try

1. Switch between prompt modes (basic / CoT / few-shot / structured) in the sidebar.
2. Ask a math or logic problem — compare basic vs. CoT mode.
3. Try "structured" mode and see the model respond in JSON format.
4. Have a multi-turn conversation and observe how history affects answers.
5. Expand the **"Thinking"** step to see the model's internal reasoning.

## What this level intentionally lacks

| Missing | Added at |
|---------|----------|
| External knowledge (documents) | Level 3 |
| Hybrid search, re-ranking | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow patterns (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |

## Files

```
level_02_prompt_engineering/
├── rxconfig.py         # Reflex configuration
├── app/
│   ├── app.py          # Reflex chat UI (pages + components)
│   └── state.py        # Reactive state management
├── src/
│   ├── config.py       # Pydantic config loader (extended)
│   ├── llm_client.py   # OpenAI-compatible wrapper with <think> parsing
│   ├── ollama_models.py # Ollama model listing & context window
│   ├── prompts.py      # CoT / few-shot / structured templates
│   └── metrics.py      # CPU/GPU/VRAM/RAM resource metrics
├── config.yaml         # Model, prompt_mode, max_history_turns
└── pyproject.toml
```
