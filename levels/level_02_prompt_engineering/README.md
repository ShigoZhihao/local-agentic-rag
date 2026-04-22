# Level 2 — Prompt Engineering

> *This README was restructured and organized by Claude Opus 4.7.*

**Core loop:** `user input → (Clarifier → Rewriter) → [system prompt + history] → LLM → output`

A CLI chatbot that extends Level 1 with a **system prompt**, **conversation history**, and a **query-improvement pipeline**. No external data (no RAG) — the goal is to raise response quality purely through prompt engineering.

**Model:** `gemma4:e4b` (thinking model — emits `<think>...</think>` tags)
**UI:** CLI only (no GUI)

---

## What you learn here

- Defining persona and output rules via system prompt
- Managing multi-turn conversation history and trimming strategies
- The **Clarifier → Rewriter pattern**: ask the user clarification questions before rewriting, so the rewriter works on *user-provided facts* instead of guesses (hallucination mitigation)
- Why a naive Rewriter alone is risky, and how inserting a Clarifier improves it
- Streaming output and `<think>` tag handling (state machine vs `re.sub`)

---

## Architecture

### Main version (`main.py`) — transparent and interactive

```
User input
  ↓
"Improve? (y/n)"
  ├─ n ────────────────────────────────┐
  │                                    ↓
  └─ y                              Main LLM (history + system prompt)
       ↓                                ↓
     Clarifier LLM                   Display answer
       → generates Q1, Q2, Q3
       ↓
     User answers Q1-Q3
       ↓
     Rewriter LLM
       → generates improved prompt
       ↓
     "use / edit / improve ?"
       ├─ use    → send improved prompt to Main LLM
       ├─ edit   → user edits manually → Main LLM
       └─ improve → feed improved prompt back into Clarifier (loop)
```

History keeps only **the user's original input** and **the final answer**; Clarifier and Rewriter outputs are ephemeral.

### Experimental version (`main_copy.py`) — silent auto-improvement

```
User input → Clarifier → user answers → Rewriter (silent) → Main LLM → answer
```

Runs the improvement pipeline automatically on every turn without showing the improved prompt to the user. Simpler UX but **cannot detect Rewriter hallucinations**. Kept as a comparison study — `main.py` is the primary version for learning.

---

## Project layout

```
level_02_prompt_engineering/
├── README.md            # this file
├── config.yaml          # Ollama connection info only
├── pyproject.toml       # deps: openai, pydantic, pyyaml, pyreadline3
├── main.py              # interactive main version
├── main_copy.py         # silent auto-improve experimental version
└── src/
    ├── __init__.py
    ├── config.py        # YAML → Pydantic
    ├── llm_client.py    # Ollama calls (stream / non-stream)
    └── prompts.py       # three system-prompt constants
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
| `response(client, cfg, message)` | function | **Non-streaming** completion. Strips `<think>...</think>` via `re.sub` |
| `stream_response(client, cfg, message)` | function | **Streaming** — prints tokens as they arrive and returns the final string. Uses a state machine to display `<think>` content in dim style while excluding it from the returned string |

### `src/prompts.py`

| Constant | Purpose |
|---|---|
| `MAIN_SYSTEM` | For Main LLM. Persona and hard constraints (e.g., no emoji) |
| `CLARIFIER_SYSTEM` | For Clarifier LLM. Instructs it to output three questions in `Q1:/Q2:/Q3:` format |
| `REWRITER_SYSTEM` | For Rewriter LLM. Combines Q&A pairs into a structured prompt. **Forbids broadening scope or inventing new facts** |

### `main.py` / `main_copy.py`

| Item | Kind | Description |
|---|---|---|
| `max_history_turns` | module constant | 10 (keeps system + last 10 turns = 21 messages) |
| `main()` | function | Entry point; contains the conversation loop |

---

## While-loop structure (main.py)

Two nested `while True` loops:

```python
while True:                              # ① outer: conversation loop
    user_input = input("You: ")
    if quit: break

    if rewrite == "yes":
        while True:                      # ② inner: improvement loop
            questions = stream_response(CLARIFIER, user_input)
            answers = input("You: ")
            improved = stream_response(REWRITER, [user_input, questions, answers])

            choice = input()
            if choice == "use":  break   # send improved → exit inner
            if choice == "edit": break   # send edited → exit inner
            if choice == "improve":      # re-feed improved into clarifier
                user_input = improved
    else:
        # no improvement, send as-is
        ...

    # history trimming (every turn, after both branches)
    if len(messages) > 1 + 2*max_history_turns:
        messages = [messages[0]] + messages[-(2*max_history_turns):]
```

**Why two loops:**
- Outer = one "chat session" with the user
- Inner = one "improvement cycle" for a single query
- Picking `improve` cycles only the inner loop; the outer history doesn't advance until the user sends something

---

## What the user can do

| Action | Effect |
|---|---|
| Type a normal input | Sent directly to Main LLM |
| Answer `yes` / `y` to the improve prompt | Clarifier generates 3 questions → user answers → Rewriter produces improved version |
| `use` | Send improved version to Main LLM as-is |
| `edit` | Hand-edit the improved version before sending (empty Enter keeps it unchanged) |
| `improve` | Feed improved version back into Clarifier for another pass |
| Empty Enter | Skip to next input |
| `quit` | Exit |
| `Ctrl+C` | Force exit |

History stores the **original user utterance**, not the improved version. No matter how many `improve` passes you run, subsequent turns see the user's **actual** words.

---

## Why this counts as an MVP

MVP = **the minimum that works and expresses the essence**. This level meets all five criteria:

1. **Minimal dependencies** — only openai / pydantic / pyyaml / pyreadline3. No GUI, no DB, no RAG
2. **Minimal surface expressing the level's essence** — the Clarifier → Rewriter pipeline demonstrates that changing prompts alone changes output quality
3. **Single entry point** — `python main.py` runs the whole thing
4. **Bridge to higher levels** — Clarifier → Rewriter → Main mirrors Level 7's Facilitator → Synthesizer pattern in miniature. The same structure reappears at L7
5. **Everything removable has been removed** — intentionally excluded:
   - Multiple prompt modes (cot / few-shot / structured) — the Clarifier + Rewriter flow already teaches the lesson
   - Metrics (token / GPU usage) — not the theme of this level
   - Model auto-enumeration (`ollama list`) — config is enough
   - `$EDITOR` integration for editing improved prompts — `input()` suffices

"Any further reduction would break the Prompt Engineering lesson" — that's the bar.

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

A warmup request before the first real query would hide this from the user (not implemented here).

---

### History trimming pitfalls

```python
if len(messages) > 1 + 2*max_history_turns:
    messages = [messages[0]] + messages[-(2*max_history_turns):]
```

**Watch out for:**

1. **Put it inside the `while` loop but outside `if/else`**. Placing it in only one branch lets the other branch grow history forever
2. **Preserve the system prompt as `messages[0]` explicitly**. A plain `messages[-N:]` alone would push system out
3. **Don't forget the `>` guard**. Without it, running `[messages[0]] + messages[-N:]` unconditionally would, early in the conversation, pull system into `messages[-N:]` and **duplicate** it

Threshold `1 + 2*max_history_turns` breakdown:
- `1` = system prompt
- `2*max_history_turns` = user/assistant pairs × turns

---

### `<think>` tag removal strategy

Reasoning models (gemma4 family) emit `<think>...</think>` tags containing internal reasoning. Leaving them in history degrades next-turn quality, so they must be stripped.

| Situation | Method | Reason |
|---|---|---|
| Non-streaming (`response()`) | `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)` | One-shot regex on the finished string is sufficient |
| Streaming (`stream_response()`) | State machine (`init` → `thinking` → `answering`) | Tokens arrive one at a time and the `<think>` tag itself may be split; progressive state tracking is required |

**Why `re.sub` doesn't work for streaming:**

- You *could* buffer all tokens then apply `re.sub`, but "nothing shows on screen until generation finishes" — **the whole point of streaming is lost**
- The state machine can display `[thinking]` in dim color in real time while excluding it from the returned answer — streaming UX and clean history coexist

---

### Variable/function name shadowing

Inside `llm_client.py`'s `response()` function, a local variable is named `response = client.chat...create(...)`, shadowing the outer function name. It runs fine, but you can't recursively reference the `response` function from inside its own body. Rename to `resp` or `completion` once imports get tangled in Level 3+.

---

### `content=None` defense

Ollama / OpenAI APIs occasionally return `response.choices[0].message.content = None` (notably on `max_tokens` overrun). The current `response()` pulls `.content` directly, so `None` would crash `re.sub` with `AttributeError`. Safer:

```python
text = (response.choices[0].message.content or "") if response.choices else ""
```

Low occurrence rate in this MVP, so not hardened yet. Likely to surface once Level 3+ handles retrieved chunks; will fix then.

---

### Why both `main.py` and `main_copy.py` are kept

`main_copy.py` is an experiment with "run the Rewriter silently, every turn". `main.py` (visible improvement process) is the recommended version for teaching. The contrast itself is the note:

| Aspect | `main.py` (main) | `main_copy.py` (experimental) |
|---|---|---|
| Shows improved prompt to user | Yes | No |
| Can catch Rewriter hallucinations | Yes (via use/edit/improve) | No |
| UX | More interactive | Simpler |
| Educational value | High — process is observable | Low — black box |
| Use case | Prototype / teaching | Closer to production feel |

---

## How to run

### Setup

```bash
cd levels/level_02_prompt_engineering
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
# main version (interactive)
python main.py

# experimental version (silent auto-improve)
python main_copy.py
```

---

## What this level intentionally lacks

| Missing | Added at |
|---|---|
| External knowledge (documents) | Level 3 |
| Hybrid search / re-ranking | Level 4 |
| Tool use (ReAct agent) | Level 5 |
| Workflow (LangGraph) | Level 6 |
| Multi-agent loop | Level 7 |
| Evaluation metrics | Level 10+ |
