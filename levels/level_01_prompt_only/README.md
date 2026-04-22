# Level 1 — Prompt Only

> *This README was restructured and organized by Claude Opus 4.7.*

**Core loop:** `user input → LLM → output`

The simplest possible chatbot. No memory, no system prompt, no external data. Build a working MVP and see firsthand what a bare LLM call can and cannot do — this baseline is what every later level extends.

**Model:** `gemma4:e4b` (thinking model — emits `<think>...</think>` tags)
**UI:** CLI only (no GUI)

---

## What you learn here

- Connecting to a local Ollama server via the OpenAI-compatible API
- Loading config from YAML into validated Pydantic models
- Streaming tokens from the server and displaying them in real time
- Parsing `<think>` tags from reasoning models using a state machine
- The baseline limitations of a stateless single-turn chatbot: no memory, no context control, no tools

---

## Architecture

```
User input
  ↓
wrap into messages=[{"role": "user", "content": ...}]
  ↓
Ollama LLM (no system prompt, no history)
  ↓
Stream tokens back
  ↓
State machine strips <think>...</think>
  ↓
Display answer on screen
```

Every turn is **stateless** — each request sends only the current user input. The model has no memory of previous turns; ask a follow-up and it will not recall what you just said.

---

## Project layout

```
level_01_prompt_only/
├── README.md            # this file
├── config.yaml          # Ollama connection info only
├── pyproject.toml       # deps: openai, pydantic, pyyaml
├── main.py              # entry point, conversation loop
└── src/
    ├── __init__.py
    ├── config.py        # YAML → Pydantic
    └── llm_client.py    # Ollama call with streaming + <think> parsing
```

---

## All defined classes and functions

### `src/config.py`

| Item | Kind | Description |
|---|---|---|
| `OllamaConfig` | Pydantic Model | `base_url`, `model`, `temperature` (0.0-2.0), `max_tokens` (>0) |
| `Config` | Pydantic Model | Root config. Holds `ollama: OllamaConfig` |
| `get_config()` | function | Loads `config.yaml` and returns a validated `Config` |

### `src/llm_client.py`

| Item | Kind | Description |
|---|---|---|
| `create_client(cfg)` | function | Builds an `OpenAI` client pointed at the Ollama OpenAI-compatible endpoint |
| `stream_response(client, cfg, message)` | function | **Streaming** call — prints tokens as they arrive and returns the final string. Uses a state machine to display `<think>` content in dim style while excluding it from the returned string |

### `main.py`

| Item | Kind | Description |
|---|---|---|
| `main()` | function | Entry point; contains the conversation loop |

---

## While-loop structure (main.py)

A single `while True` loop — the simplest possible shape:

```python
while True:                              # conversation loop
    user_input = input("You: ")
    if Ctrl+C: break
    if empty: continue
    if quit:  break

    messages = [{"role": "user", "content": user_input}]  # rebuilt every turn

    print(f"{model}: ", end="", flush=True)
    stream_response(client, cfg.ollama, messages)         # no return-value use
```

**Key point:** `messages` is re-created inside the loop every iteration. Nothing is carried over between turns — this is what makes Level 1 *stateless*. Level 2 fixes this by moving `messages` outside the loop and appending to it.

---

## What the user can do

| Action | Effect |
|---|---|
| Type a normal input | Sent directly to LLM |
| Empty Enter | Skip to next input |
| `quit` | Exit |
| `Ctrl+C` | Force exit |

That's the whole interface. No system prompt to tune, no history to manage, no mode to switch.

---

## Why this counts as an MVP

MVP = **the minimum that works and expresses the essence**. This level meets all five criteria:

1. **Minimal dependencies** — only openai / pydantic / pyyaml. No GUI, no history store, no prompt templates
2. **Minimal surface expressing the level's essence** — "talk to a local LLM from Python" is the entire point. One user turn → one model turn
3. **Single entry point** — `python main.py` runs the whole thing
4. **Baseline for every later level** — every level above adds one concept on top of this foundation (L2 adds history + system prompt, L3 adds retrieval, etc.). If L1 is wrong, everything above breaks
5. **Everything removable has been removed** — intentionally excluded:
   - System prompt — the whole point of L1 is to feel its absence
   - Conversation history — stateless by design
   - Prompt templates — introduced at Level 2
   - Metrics and logging — not the theme

"Any further reduction would not talk to an LLM anymore" — that's the bar.

---

## Implementation notes (not the main story, but worth recording)

### Ollama cold start (why the first request is slow)

The first request takes 30 seconds to 1 minute. When Ollama loads a model for the first time, the following happens in order:

1. **Hardware inventory** — scan available GPUs and free VRAM
2. **Layer offloading decision** — decide how many layers go to GPU vs CPU
3. **llama runner process spawn** — launch a dedicated inference process for the model
4. **.gguf file read** — load weights from disk (several GB)
5. **VRAM transfer & computation graph build** — dequantize and lay out weights
6. **CUDA kernel first-time compile** — generate GPU kernels (the biggest time sink)
7. **KV cache allocation** — reserve memory for the context window

Subsequent requests hit the resident model and return immediately. By default Ollama unloads the model 5 minutes after the last request. Set `OLLAMA_KEEP_ALIVE=-1` to keep it loaded indefinitely.

---

### `<think>` tag removal strategy (streaming)

Reasoning models (gemma4 family) emit `<think>...</think>` tags containing internal reasoning. Leaving them on screen is fine (dim style is nice for transparency), but `stream_response()` still returns the **cleaned** string so the caller never receives the thinking fragment.

Why a state machine instead of `re.sub`:
- Tokens arrive one at a time; `<think>` or `</think>` can split across chunks (e.g., `<thi` then `nk>`)
- `re.sub` only works on a completed string — using it would require buffering everything silently, which defeats streaming
- The state machine has three states (`init` → `thinking` → `answering`) and handles partial tags by holding unmatched bytes in `buf` until enough context arrives

This pattern is carried into Level 2 unchanged; get comfortable with it here.

---

### Why `messages` is rebuilt every loop

```python
while True:
    ...
    messages = [{"role": "user", "content": user_input}]   # fresh every turn
    stream_response(client, cfg.ollama, messages)
```

This is **intentional**, not laziness. Level 1's definition of "Prompt Only" is exactly that each call carries **no prior context**. Moving `messages = [...]` outside the loop and appending to it would turn Level 1 into Level 2's history-aware version. The placement of this one line is what makes Level 1 what it is.

---

### `if __name__ == "__main__":` guard

The entry-point file wraps `main()` in the standard Python idiom:

```python
if __name__ == "__main__":
    main()
```

This matters because Python's `multiprocessing` (used by some libraries downstream in later levels) re-imports the entry-point file in each worker process. Without this guard, every worker would call `main()` again on import, creating an infinite fork spiral that can freeze the machine. Level 1 doesn't use multiprocessing yet, but the habit is established from day one.

---

### `pyreadline3` is not needed here

Level 2 depends on `pyreadline3` (so that `input()` supports arrow-key history on Windows). Level 1 omits it intentionally — a minimal-dependency stance. If you want history within a single session here, add it manually; it won't break anything.

---

## How to run

### Setup

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

### Launch

```bash
python main.py
```

### Try this

1. Ask any question and watch tokens stream in
2. Ask a follow-up referring to the previous answer — notice the model has no idea what "that" means. This is the absence of history, and it's exactly what Level 2 solves
3. Ask a reasoning-heavy question — `<think>...</think>` appears in dim style, then the answer
4. Type `quit` or press `Ctrl+C` to exit

---

## What this level intentionally lacks

| Missing | Added at |
|---|---|
| Conversation memory | Level 2 |
| Prompt engineering (system prompt, CoT, few-shot) | Level 2 |
| External knowledge (documents) | Level 3 |
| Hybrid search / re-ranking | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |
| Evaluation metrics | Level 10+ |
