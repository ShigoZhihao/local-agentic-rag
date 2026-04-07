# Agentic RAG System - Project Rules

This file is read by Claude Code and GitHub Copilot.
All code in this project MUST follow the conventions below.

## Architecture

12レベル段階的構成。Level 7 がメイン実装（LangGraph 4エージェント）。

Level 7: LangGraphで制御する4エージェント構成:
- **Facilitator** (`levels/level_07_multi_agent/src/agents/facilitator.py`): prompt理解・エンリッチ (gemma4:e4b)
- **Synthesizer** (`levels/level_07_multi_agent/src/agents/synthesizer.py`): 自力回答判断・回答生成 (gemma4:e2b)
- **Researcher** (`levels/level_07_multi_agent/src/agents/researcher.py`): Hybrid Search → Filter → Re-rank → 引用ステッチ (LLM不使用)
- **Validator** (`levels/level_07_multi_agent/src/agents/validator.py`): LLM-as-Judge 4軸採点・ループ制御 (gemma4:e4b)

グラフ定義: `levels/level_07_multi_agent/src/agents/graph.py` (LangGraph StateGraph + MemorySaver)
共有状態: `levels/level_07_multi_agent/src/agents/state.py` (RAGState TypedDict)

Key paths (Level 7):
- `levels/level_07_multi_agent/src/models.py` — 全Pydanticデータモデル
- `levels/level_07_multi_agent/src/config.py` — config.yaml → Pydantic Settings
- `levels/level_07_multi_agent/src/generation/llm_client.py` — Ollama接続 (全LLM呼び出しはここ経由)
- `levels/level_07_multi_agent/src/generation/prompts.py` — 全プロンプトテンプレート定数
- `levels/level_07_multi_agent/src/retrieval/weaviate_client.py` — Weaviate接続・スキーマ・CRUD

Models: gemma4:e2b (executor), gemma4:e4b (planner) — thinking models with `<think>` tag support
UI: Reflex with Siemens teal `#009999`, thinking shown in collapsible accordion

## Coding Conventions

### Type Hints (必須)
Every function must have type hints on ALL parameters AND return value:
```python
# Good
def search(query: str, top_k: int = 10) -> list[SearchResult]:
    ...

# Bad - missing type hints
def search(query, top_k=10):
    ...
```

### Docstrings (公開関数は必須)
Use Google-style docstrings for all public functions:
```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using BGE-M3.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each vector is 1024 floats).
    """
```

### Data Models
- ALL data structures must use `src/models.py` Pydantic BaseModel — never plain dicts
- Never create local dataclasses or TypedDicts for domain objects

### Configuration
- ALL config values come from `src/config.py` — never hardcode URLs, model names, thresholds
- Access config via: `from src.config import get_config; cfg = get_config()`

### LLM Calls
- ALL LLM calls go through `src/generation/llm_client.py` — never import openai directly
- Use `call_planner()` for Facilitator/Validator (gemma4:e4b) — returns `LLMResponse(thinking, answer)`
- Use `call_executor()` for Synthesizer (gemma4:e2b) — returns `LLMResponse(thinking, answer)`
- Thinking models: `<think>...</think>` tags are automatically parsed by `strip_thinking()`

### Prompts
- ALL prompt text lives in `src/generation/prompts.py` as string constants
- Never embed prompt strings inside agent logic files

### Error Handling
```python
# Good
import logging
logger = logging.getLogger(__name__)

try:
    result = weaviate_client.search(query)
except Exception as e:
    logger.error("Search failed: %s", e)
    raise

# Bad - silent pass
try:
    result = weaviate_client.search(query)
except Exception:
    pass
```

### Import Order
```python
# 1. Standard library
import logging
from pathlib import Path

# 2. Third-party
import weaviate
from pydantic import BaseModel

# 3. Local
from src.config import get_config
from src.models import Chunk
```

### Max Line Length
100 characters per line.

## Module Interfaces

### Chunkers (`src/ingestion/chunkers.py`)
All chunkers inherit from `BaseChunker` ABC:
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...
```
Factory function: `get_chunker(source_type: SourceType, config: ChunkingConfig) -> BaseChunker`

### Search Modules (`src/retrieval/`)
All search functions expose:
```python
def search(query: str, top_k: int, **kwargs) -> list[SearchResult]:
    ...
```

### Agent Nodes (`src/agents/`)
All agent nodes expose a single function:
```python
def run(state: RAGState) -> RAGState:
    ...
```
Agents do NOT inherit from a base class. They are plain function-based modules.

## Running the App

```bash
cd levels/level_07_multi_agent

# Start Weaviate
docker compose up -d

# Run Reflex UI
reflex run
```

## Testing

```bash
pytest tests/ -v
```

- Unit tests: no external dependencies (mock Weaviate and LM Studio)
- Integration tests: require Docker + downloaded models
- Fixtures in `tests/conftest.py`
