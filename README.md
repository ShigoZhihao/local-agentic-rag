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

A fully local Agentic RAG (Retrieval-Augmented Generation) system. Zero API costs, 100% open-source.

## Architecture

4-agent loop controlled by LangGraph:

```
User Query
    │
    ▼
┌──────────────┐  needs clarification  ┌──────────────┐
│  Facilitator │ ◄──────────────────── │  User Input  │
│ (qwen2.5:9b) │ ──────────────────►  │              │
└──────┬───────┘                       └──────────────┘
       │ enriched_prompt
       ▼
┌──────────────┐   can answer directly  ┌──────────────┐
│  Synthesizer │ ──────────────────►    │  Validator   │
│ (qwen2.5:2b) │                        │ (qwen2.5:9b) │
└──────┬───────┘                        └──────┬───────┘
       │ needs retrieval                       │
       ▼                                       │ FAIL (avg < 80)
┌──────────────┐                               │
│  Researcher  │                               ▼
│  (no LLM)   │ ──── citations ──────► Facilitator
└──────────────┘                          (max 3 loops)
```

| Agent | Role | Model |
|-------|------|-------|
| **Facilitator** | Query understanding, enrichment, clarification questions | qwen2.5:9b (planner) |
| **Synthesizer** | Direct answer or citation-grounded generation | qwen2.5:2b (executor) |
| **Researcher** | Hybrid Search → Filter → Re-rank → Citation stitching | No LLM |
| **Validator** | LLM-as-Judge on 4 axes: completeness, accuracy, relevance, faithfulness | qwen2.5:9b (planner) |

> **Note:** Current model sizes are for testing. For production use, upgrade planner to `qwen2.5:27b`
> and executor to `qwen2.5:9b`.

### Retrieval Pipeline

```
Hybrid Search (BM25 + Semantic HNSW, alpha=0.5)
    ↓
Metadata Filter (source_type, min_score, etc.)
    ↓
Cross-Encoder Re-rank (ms-marco-MiniLM-L-6-v2)
    ↓
[Optional] ColBERT Late Interaction (top_k=10)
    ↓
Citation Stitching (NotebookLM style: preserve original_text)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (qwen2.5:9b / qwen2.5:2b) |
| Vision LLM | Ollama (qwen3-vl:8b, ingestion only) |
| Embedding | BAAI/bge-m3 (1024-dim, sentence-transformers, GPU) |
| Vector DB | Weaviate 1.28 (Docker, external embedding) |
| Agent Framework | LangGraph (StateGraph + MemorySaver) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| UI | Streamlit (4 pages: Chat / Ingest / Tuning / Evaluation) |
| Python | 3.12 |

## Prerequisites

### 1. Ollama

Install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull qwen2.5:9b    # Facilitator / Validator (planner)
ollama pull qwen2.5:2b    # Synthesizer (executor)
ollama pull qwen3-vl:8b   # Vision — PDF/PPTX ingestion only
```

- Ollama listens at `http://127.0.0.1:11434` after startup
- OpenAI-compatible API: `http://127.0.0.1:11434/v1`
- Model names are configurable in `config.yaml` under `ollama.planner_model` / `ollama.executor_model`

> **VRAM note (8GB GPU):**
> - qwen2.5:9b + qwen2.5:2b together consume ~70-80% of VRAM
> - Vision LLM (qwen3-vl:8b) is only loaded during ingestion — unload with `ollama stop qwen3-vl:8b` when not in use
> - BGE-M3 stays resident on GPU (~2.2GB)

### 2. Docker Desktop

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 3. Python 3.12

Python 3.12 is required (3.13 is incompatible with ragatouille).

## Setup

> **If `uv pip install -e .` causes 97%+ CPU usage, use the low-load installation steps below.**

### Standard Installation

```bash
# 1. Create virtual environment
uv venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
uv pip install -e .

# 3. Pre-download BGE-M3 embedding model (~1.5GB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Low-Load Installation — Avoiding 97%+ CPU

`uv pip install -e .` resolves all dependencies at once, causing high CPU usage during PyTorch
download and extraction. Install in stages to reduce load:

```bash
# Step 1: Install PyTorch first (the main culprit, ~2GB)
# For NVIDIA GPU with CUDA 12.1:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install remaining packages in small batches
uv pip install weaviate-client
uv pip install sentence-transformers
uv pip install langgraph langchain-core langchain-text-splitters
uv pip install openai pydantic pydantic-settings pyyaml
uv pip install pymupdf python-pptx beautifulsoup4
uv pip install streamlit pandas ranx pytest pytest-mock

# Step 3: Register the project itself without re-resolving dependencies (near-zero CPU)
uv pip install -e . --no-deps
```

### Start Weaviate and Verify Connection

```bash
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

### Launch Streamlit UI

```bash
streamlit run src/ui/app.py
```

Opens at `http://localhost:8501`.

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
src/
├── models.py                  # Pydantic data models
├── config.py                  # config.yaml → Settings loader
├── ingestion/
│   ├── loaders.py             # File loaders (all formats)
│   ├── chunkers.py            # 6 chunking strategies + factory
│   ├── embedder.py            # BGE-M3 embedding wrapper
│   ├── vision_describer.py    # Vision LLM image → text
│   └── pipeline.py            # Ingestion pipeline orchestrator
├── retrieval/
│   ├── weaviate_client.py     # Weaviate connection & CRUD
│   ├── hybrid_search.py       # BM25 + Semantic search
│   ├── metadata_filter.py     # Post-retrieval metadata filtering
│   └── colbert_search.py      # ColBERT (optional)
├── reranking/
│   └── cross_encoder.py       # Cross-Encoder re-ranking
├── generation/
│   ├── llm_client.py          # Ollama client (planner/executor/vision)
│   └── prompts.py             # All prompt templates
├── agents/
│   ├── state.py               # RAGState TypedDict
│   ├── graph.py               # LangGraph StateGraph definition
│   ├── facilitator.py         # Query understanding & enrichment
│   ├── synthesizer.py         # Answer generation
│   ├── researcher.py          # Retrieval & citation stitching
│   └── validator.py           # LLM-as-Judge 4-axis scoring
├── evaluation/
│   ├── metrics.py             # Precision/Recall/MAP/MRR
│   └── bm25_tuner.py          # BM25 k1/b grid search
└── ui/
    ├── app.py                 # Streamlit main app
    └── pages/
        ├── chat.py            # Chat page
        ├── ingest.py          # Ingest page
        ├── tuning.py          # BM25 tuning page
        └── evaluation.py      # Evaluation page
```

## Configuration

All settings are centralized in `config.yaml`. Hardcoding is prohibited.

```python
from src.config import get_config
cfg = get_config()
print(cfg.ollama.planner_model)    # qwen2.5:9b
print(cfg.retrieval.hybrid.alpha)  # 0.5
```

Key settings:

| Key | Description | Default |
|-----|-------------|---------|
| `ollama.planner_model` | Facilitator/Validator model | qwen2.5:9b |
| `ollama.executor_model` | Synthesizer model | qwen2.5:2b |
| `vision.model_name` | Vision LLM model | qwen3-vl:8b |
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
pytest tests/ -v
```

- Unit tests: no external dependencies (Weaviate and Ollama are mocked)
- Integration tests: require Docker + downloaded models
- Fixtures: `tests/conftest.py`

## License

[MIT License](LICENSE)
