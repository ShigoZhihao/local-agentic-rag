# Agentic RAG — 12-Level Hands-On Learning Path

**[日本語版はこちら](README_JP.md)**

---

> **About this repository**
>
> This repository is a personal hands-on learning project by the author ([@ShigoZhihao](https://github.com/ShigoZhihao)). The goal is to internalize how modern agentic RAG systems are built by implementing them from scratch in 12 progressive levels — from a bare LLM call to a fully autonomous multi-agent system.
>
> - **All code is written by the author.** Every `main.py`, module, and test is hand-written, typo-for-typo, as part of the learning process. No code generation, no copy-paste shortcuts.
> - **All README documentation is authored by Claude (Anthropic).** The author discusses design, trade-offs, and bugs with Claude; Claude then structures the findings into the README files you see in each level. This division keeps the *engineering* human-owned and the *documentation* consistent across levels.

---

> **Acknowledgements**
>
> This project is inspired by the concepts taught in the
> **[Retrieval Augmented Generation (RAG) Course](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/)**
> by [DeepLearning.AI](https://www.deeplearning.ai/). A huge thank you to the DeepLearning.AI team for making such an outstanding course freely available.

---

## Background and Purpose

Modern AI systems are evolving from **single-turn LLM calls** toward **multi-agent systems** that search, reason, validate, and act autonomously. Understanding this stack top-to-bottom requires building each layer yourself — there is no shortcut.

This repository breaks that journey into **12 self-contained levels**, each a working MVP. Every level:

- Has its own `src/`, `pyproject.toml`, `config.yaml`, and `README.md`
- Runs independently — no cross-level imports
- Adds exactly **one new concept** on top of the previous level
- Runs 100% locally (Ollama + Weaviate), zero API cost

The curriculum progresses from raw OpenAI-compatible SDK calls (L1–L2), through LangChain-based RAG (L3–L5), into LangGraph workflows and multi-agent loops (L6–L7), and finally into advanced agent harness techniques (L8–L12).

### Current progress

| Status | Levels |
|---|---|
| ✅ Implemented | Level 1, Level 2 |
| 🚧 Planned | Level 3 through Level 12 |

---

## Level comparison

### Overview

| # | Level | Folder | Framework | Model | MVP does this |
|---|---|---|---|---|---|
| 1 | Prompt Only | `level_01_prompt_only` | openai SDK | gemma4:e4b | Stateless single-turn chat — user typed → LLM → streamed answer |
| 2 | Prompt Engineering | `level_02_prompt_engineering` | openai SDK | gemma4:e4b | Multi-turn chat + Clarifier→Rewriter pipeline for prompt improvement |
| 3 | Basic RAG | `level_03_basic_rag` | LangChain | gemma4:e2b | Semantic search over local docs via Weaviate |
| 4 | Advanced RAG | `level_04_advanced_rag` | LangChain | gemma4:e4b | Hybrid search (BM25 + semantic) + re-ranking + PDF/PPTX ingest |
| 5 | Single Agent | `level_05_single_agent` | LangChain | gemma4:e4b | ReAct-style agent with tool use |
| 6 | Workflow Patterns | `level_06_workflow_patterns` | LangGraph | e2b+e4b | Evaluator-Optimizer loop explicitly modeled as a state graph |
| 7 | Multi-Agent | `level_07_multi_agent` | LangGraph | e2b+e4b | 4-agent loop (Facilitator / Synthesizer / Researcher / Validator) with HITL |
| 8 | MCP | `level_08_mcp` | LangGraph + MCP | TBD | External tools via Model Context Protocol |
| 9 | Harness | `level_09_harness` | LangGraph | TBD | Context compaction, persistence, session management |
| 10 | Sub-Agents | `level_10_sub_agents` | LangGraph | TBD | Parallel subgraph spawning for decomposable tasks |
| 11 | Skills | `level_11_skills` | LangGraph | TBD | Composable skill units (`SKILL.md` pattern) |
| 12 | Autonomous | `level_12_autonomous` | LangGraph | TBD | Scheduled tasks + CI/CD integration |

### Implemented levels — defined functions

**Level 1** (`level_01_prompt_only`)

| Module | Function / Class | Purpose |
|---|---|---|
| `src/config.py` | `OllamaConfig`, `Config`, `get_config()` | YAML → Pydantic config loading |
| `src/llm_client.py` | `create_client(cfg)` | Builds an OpenAI client pointing at Ollama |
| `src/llm_client.py` | `stream_response(client, cfg, message)` | Streaming response with state-machine-based `<think>` parsing |
| `main.py` | `main()` | Single `while True` loop; stateless per turn |

**Level 2** (`level_02_prompt_engineering`)

| Module | Function / Class | Purpose |
|---|---|---|
| `src/config.py` | `OllamaConfig`, `Config`, `get_config()` | Same as L1 |
| `src/llm_client.py` | `create_client(cfg)` | Same as L1 |
| `src/llm_client.py` | `response(client, cfg, message)` | **New.** Non-streaming completion; strips `<think>` via `re.sub` |
| `src/llm_client.py` | `stream_response(client, cfg, message)` | Same as L1 |
| `src/prompts.py` | `MAIN_SYSTEM` | Main LLM persona + hard constraints |
| `src/prompts.py` | `CLARIFIER_SYSTEM` | Instructs Clarifier to emit `Q1/Q2/Q3` |
| `src/prompts.py` | `REWRITER_SYSTEM` | Instructs Rewriter to combine Q&A pairs without broadening scope |
| `main.py` | `main()` | Outer conversation loop + inner improvement loop |
| `main_copy.py` | `main()` | Experimental silent auto-improve variant |

Levels 3–12: function inventory will be added as each level is implemented.

---

## Target tech stack (by Level 7)

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (gemma4:e2b / gemma4:e4b, thinking models) |
| Embedding | BAAI/bge-m3 (1024-dim, sentence-transformers, GPU) |
| Vector DB | Weaviate 1.28 (Docker, external embedding) |
| Agent Framework | LangGraph (StateGraph + MemorySaver) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Python | 3.12 |
| UI | CLI only (no GUI) |

Levels 1–2 use only `openai`, `pydantic`, `pyyaml` (and `pyreadline3` at L2).

---

## Prerequisites

### 1. Ollama

Install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull gemma4:e2b    # executor — Levels 3 and 6+
ollama pull gemma4:e4b    # planner — Levels 1, 2, 4+
```

- Ollama listens at `http://127.0.0.1:11434` after startup
- OpenAI-compatible API: `http://127.0.0.1:11434/v1`
- Model names are configurable in each level's `config.yaml`

### 2. Docker Desktop (Level 3+)

Needed for Weaviate. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 3. Python 3.12

Python 3.12 is required (3.13 is incompatible with some downstream dependencies used in higher levels).

---

## Getting started

Every level is self-contained. Pick one and run it:

```bash
cd levels/level_01_prompt_only
uv venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux
uv pip install -e .
python main.py
```

For level-specific setup and usage, see each level's own `README.md`.

---

## Project layout

```
local-agentic-rag/
├── README.md                         # this file
├── README_JP.md                      # Japanese version
├── CLAUDE.md                         # coding conventions for Claude Code / Copilot
├── archive/
│   └── plan.md                       # original planning doc (archived)
└── levels/
    ├── level_01_prompt_only/         # ✅ openai SDK — bare LLM chat
    ├── level_02_prompt_engineering/  # ✅ openai SDK — Clarifier→Rewriter
    ├── level_03_basic_rag/           # 🚧 LangChain — semantic search, Weaviate
    ├── level_04_advanced_rag/        # 🚧 LangChain — hybrid search, re-ranking
    ├── level_05_single_agent/        # 🚧 LangChain — ReAct agent, tool use
    ├── level_06_workflow_patterns/   # 🚧 LangGraph — Evaluator-Optimizer
    ├── level_07_multi_agent/         # 🚧 LangGraph — 4 agents, HITL
    ├── level_08_mcp/                 # 🚧 LangGraph + MCP
    ├── level_09_harness/             # 🚧 context compaction, persistence
    ├── level_10_sub_agents/          # 🚧 parallel subgraph spawning
    ├── level_11_skills/              # 🚧 composable skills
    └── level_12_autonomous/          # 🚧 scheduled tasks, CI/CD
```

---

## Authoring notes

- **Code**: author (human). Written from scratch as a learning exercise, reviewed by Claude for bugs and design.
- **READMEs**: Claude (Anthropic model). Structured after discussion with the author on what each level intentionally includes and excludes.
- **Commits**: co-authored, with Claude as co-author where applicable.

---

## License

[MIT License](LICENSE)
