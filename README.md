# Agentic RAG System

**[日本語版はこちら](README_JA.md)**

---

> **Acknowledgements**
>
> This project is inspired by and built upon the concepts taught in the
> **[Retrieval Augmented Generation (RAG) Course](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/)**
> by [DeepLearning.AI](https://www.deeplearning.ai/).
>
> A huge thank you to the DeepLearning.AI team for making such an outstanding course freely available.
> The depth and clarity of the curriculum made it possible to build a production-grade RAG system from scratch.

---

A progressive, 12-level learning path from bare LLM chat to a fully autonomous
agent system. Zero API costs, 100% local (Ollama + Weaviate).

**Models:** Ollama `gemma4:e2b` / `gemma4:e4b` (thinking models)
**UI:** Claude-inspired chat interface with Siemens brand colour (`#009999`)

## Levels

Each level is **self-contained** with its own `src/`, `pyproject.toml`, and `README.md`.

| | Level | Folder | Framework | Model |
|-|-------|--------|-----------|-------|
| | **raw openai SDK** | | | |
| 1 | Prompt Only | `level_01_prompt_only` | openai SDK | e2b |
| 2 | Prompt Engineering | `level_02_prompt_engineering` | openai SDK | e2b |
| | **LangChain** | | | |
| 3 | Basic RAG | `level_03_basic_rag` | LangChain | e2b |
| 4 | Advanced RAG | `level_04_advanced_rag` | LangChain | e4b |
| 5 | Single Agent | `level_05_single_agent` | LangChain | e4b |
| | **LangGraph** | | | |
| 6 | Workflow Patterns | `level_06_workflow_patterns` | LangGraph | e2b+e4b |
| 7 | Multi-Agent | `level_07_multi_agent` | LangGraph | e2b+e4b |
| | **LangGraph (advanced)** | | | |
| 8 | MCP | `level_08_mcp` | LangGraph+MCP | TBD |
| 9 | Harness | `level_09_harness` | LangGraph | TBD |
| 10 | Sub-Agents | `level_10_sub_agents` | LangGraph | TBD |
| 11 | Skills | `level_11_skills` | LangGraph | TBD |
| 12 | Autonomous | `level_12_autonomous` | LangGraph | TBD |

## Architecture (Level 7 — Multi-Agent)

4-agent loop controlled by LangGraph:

```
User Query
    │
    ▼
┌──────────────┐  needs clarification  ┌──────────────┐
│  Facilitator │ ◄──────────────────── │  User Input  │
│(gemma4:e4b)  │ ──────────────────►  │              │
└──────┬───────┘                       └──────────────┘
       │ enriched_prompt
       ▼
┌──────────────┐   can answer directly  ┌──────────────┐
│  Synthesizer │ ──────────────────►    │  Validator   │
│(gemma4:e2b)  │                        │(gemma4:e4b)  │
└──────┬───────┘                        └──────┬───────┘
       │ needs retrieval                       │
       ▼                                       │ FAIL (avg < 80)
┌──────────────┐                               │
│  Researcher  │                               ▼
│  (no LLM)   │ ──── citations ──────► Facilitator
└──────────────┘                          (max 3 loops)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (gemma4:e2b / gemma4:e4b) |
| Embedding | BAAI/bge-m3 (1024-dim, sentence-transformers, GPU) |
| Vector DB | Weaviate 1.28 (Docker, external embedding) |
| Agent Framework | LangGraph (StateGraph + MemorySaver) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| UI | Reflex (Claude-inspired, Siemens teal `#009999`) |
| Python | 3.12 |

## Prerequisites

### 1. Ollama

Install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull gemma4:e2b    # Levels 1-3, 6-7 executor
ollama pull gemma4:e4b    # Levels 4-7 planner
```

- Ollama listens at `http://127.0.0.1:11434` after startup
- OpenAI-compatible API: `http://127.0.0.1:11434/v1`
- Model names are configurable in each level's `config.yaml`

> **VRAM note (8GB GPU):**
> - gemma4:e4b + gemma4:e2b together consume ~70-80% of VRAM
> - BGE-M3 stays resident on GPU (~2.2GB, levels 3+ only)

### 2. Docker Desktop

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 3. Python 3.12

Python 3.12 is required (3.13 is incompatible with ragatouille).

## Setup

> **If `uv pip install -e .` causes 97%+ CPU usage, use the low-load installation steps below.**

### Standard Installation

```bash
cd levels/level_07_multi_agent

# 1. Create virtual environment
uv venv --python 3.12
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
uv pip install -e . --link-mode=copy

# 3. Pre-download BGE-M3 embedding model (~1.5GB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Low-Load Installation — Avoiding 97%+ CPU

`uv pip install -e .` resolves all dependencies at once, causing high CPU usage during PyTorch
download and extraction. Install in stages to reduce load:

```bash
cd levels/level_07_multi_agent

# Step 1: Install PyTorch first (the main culprit, ~2GB)
# For NVIDIA GPU with CUDA 12.1:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --link-mode=copy

# Step 2: Install remaining packages in small batches
uv pip install weaviate-client --link-mode=copy
uv pip install sentence-transformers --link-mode=copy
uv pip install langgraph langchain-text-splitters --link-mode=copy
uv pip install openai pydantic pyyaml --link-mode=copy
uv pip install pymupdf python-pptx beautifulsoup4 --link-mode=copy
uv pip install reflex pandas pytest pytest-mock --link-mode=copy

# Step 3: Register the project itself without re-resolving dependencies (near-zero CPU)
uv pip install -e . --no-deps --link-mode=copy
```

### Start Weaviate and Verify Connection

```bash
cd levels/level_07_multi_agent

# Start Weaviate
docker compose up -d

# Verify connection
python -c "
from src.retrieval.weaviate_client import get_client, ensure_collection
client = get_client()
ensure_collection(client)
print('Weaviate OK')
client.close()
"
```

## Usage

### Launch Reflex UI

```bash
cd levels/level_07_multi_agent
reflex run
```

Opens at `http://localhost:3000`.

### Pages

| Page | Description |
|------|-------------|
| **Chat** | Agentic RAG chat. Agents search, answer, and validate automatically. Displays citations and validation scores. |
| **Ingest** | Upload documents (TXT, MD, HTML, PY, PDF, PPTX). Vision LLM toggle for PDF/PPTX. |
| **BM25 Tuning** | Grid search over k1/b parameters. Input evaluation data in JSONL format. |
| **Evaluation** | Compute Precision@k / Recall@k / MAP@k / MRR@k with per-query drill-down. |

### Supported Document Types

| Extension | Chunking Strategy | Notes |
|-----------|------------------|-------|
| `.txt`, `.md` | RecursiveChunker (512 chars) | Text splitting |
| `.html` | HTMLChunker | Split on p, h1-h4, li tags |
| `.py` | PythonChunker | Split on def / class boundaries |
| `.pdf` | VisualChunker | 1 page = 1 chunk + optional Vision LLM |
| `.pptx` | VisualChunker | 1 slide = 1 chunk + optional Vision LLM |

## Project Structure

```
local-agentic-rag/
├── levels/
│   ├── level_01_prompt_only/           # raw openai SDK — bare LLM chat
│   ├── level_02_prompt_engineering/    # raw openai SDK — CoT, few-shot, structured
│   ├── level_03_basic_rag/            # LangChain — semantic search, Weaviate
│   ├── level_04_advanced_rag/         # LangChain — hybrid search, re-ranking, PDF/PPTX
│   ├── level_05_single_agent/         # LangChain — ReAct agent, tool use
│   ├── level_06_workflow_patterns/    # LangGraph — Evaluator-Optimizer loop
│   ├── level_07_multi_agent/          # LangGraph — 4 agents, Human-in-the-Loop ← main
│   ├── level_08_mcp/                  # LangGraph + MCP tools (placeholder)
│   ├── level_09_harness/              # context compaction, persistence (placeholder)
│   ├── level_10_sub_agents/           # parallel subgraph spawning (placeholder)
│   ├── level_11_skills/               # SKILL.md composable skills (placeholder)
│   └── level_12_autonomous/           # scheduled tasks, CI/CD (placeholder)
├── data/
├── models/
├── logs/
├── plan.md
├── CLAUDE.md
└── README.md
```

## Configuration

All settings are centralized in `config.yaml`. Hardcoding is prohibited.

```python
from src.config import get_config
cfg = get_config()
print(cfg.ollama.planner_model)    # gemma4:e4b
print(cfg.retrieval.hybrid.alpha)  # 0.5
```

Key settings (Level 7):

| Key | Description | Default |
|-----|-------------|---------|
| `ollama.planner_model` | Facilitator/Validator model | gemma4:e4b |
| `ollama.executor_model` | Synthesizer model | gemma4:e2b |
| `retrieval.hybrid.alpha` | BM25/Semantic balance (0=BM25, 1=Semantic) | 0.5 |
| `retrieval.bm25.k1` | BM25 term frequency saturation | 1.2 |
| `retrieval.bm25.b` | BM25 document length normalization | 0.75 |
| `reranking.top_k` | Results returned after re-ranking | 5 |
| `agents.max_loop_count` | Max Validator→Facilitator feedback loops | 3 |
| `agents.validation_threshold` | Validator pass threshold (avg score) | 80 |

## How Embeddings Work

Weaviate does **not** generate vectors itself (`DEFAULT_VECTORIZER_MODULE: none`).

```
Python (src/ingestion/embedder.py)
  └── sentence-transformers (BAAI/bge-m3, GPU-accelerated)
        ↓ 1024-dim vectors
Weaviate  ← storage + HNSW/BM25 search only
```

- **Ingestion**: `Embedder.embed_texts()` → stored in Weaviate
- **Query time**: `Embedder.embed_query()` → passed to Weaviate hybrid search
- Downloaded automatically from HuggingFace on first run (~1.5GB)

## Testing

```bash
cd levels/level_07_multi_agent
pytest tests/ -v
```

- Unit tests: no external dependencies (Weaviate and Ollama are mocked)
- Integration tests: require Docker + downloaded models
- Fixtures: `tests/conftest.py`

## License

[MIT License](LICENSE)
