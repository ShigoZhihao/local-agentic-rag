"""
Configuration loader.

Reads config.yaml and exposes a typed Settings object.
All modules must get config values from here — never hardcode.

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.lm_studio.base_url)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------

class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434/v1"
    planner_model: str = "qwen2.5:9b"   # Facilitator / Validator
    executor_model: str = "qwen2.5:2b"  # Synthesizer
    temperature: float = 0.2
    max_tokens: int = 2048


class VisionConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434/v1"
    model_name: str = "qwen3-vl:8b"
    max_tokens: int = 1024
    dpi: int = 200


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 32
    dimension: int = 1024


class WeaviateConfig(BaseModel):
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    collection_name: str = "Documents"


class RecursiveChunkConfig(BaseModel):
    chunk_size: int = 512
    overlap_percent: int = 0
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


class HTMLChunkConfig(BaseModel):
    tags_to_split: list[str] = Field(
        default_factory=lambda: ["p", "h1", "h2", "h3", "h4", "li"]
    )


class PythonChunkConfig(BaseModel):
    split_on: list[str] = Field(default_factory=lambda: ["def ", "class "])
    include_decorator: bool = True


class SemanticChunkConfig(BaseModel):
    cosine_threshold: float = 0.75
    min_chunk_size: int = 100
    max_chunk_size: int = 1000


class ExampleChunkConfig(BaseModel):
    delimiter: str = "---"


class VisualChunkConfig(BaseModel):
    one_chunk_per_page: bool = True


class ChunkingConfig(BaseModel):
    default_strategy: str = "recursive"
    recursive: RecursiveChunkConfig = Field(default_factory=RecursiveChunkConfig)
    html: HTMLChunkConfig = Field(default_factory=HTMLChunkConfig)
    python: PythonChunkConfig = Field(default_factory=PythonChunkConfig)
    semantic: SemanticChunkConfig = Field(default_factory=SemanticChunkConfig)
    example: ExampleChunkConfig = Field(default_factory=ExampleChunkConfig)
    visual: VisualChunkConfig = Field(default_factory=VisualChunkConfig)


class BM25RetrievalConfig(BaseModel):
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 20


class SemanticRetrievalConfig(BaseModel):
    top_k: int = 20
    distance_metric: str = "cosine"


class HybridRetrievalConfig(BaseModel):
    alpha: float = 0.5
    top_k: int = 20


class ColBERTConfig(BaseModel):
    enabled: bool = False
    model_name: str = "colbert-ir/colbertv2.0"
    top_k: int = 10


class RetrievalConfig(BaseModel):
    bm25: BM25RetrievalConfig = Field(default_factory=BM25RetrievalConfig)
    semantic: SemanticRetrievalConfig = Field(default_factory=SemanticRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    colbert: ColBERTConfig = Field(default_factory=ColBERTConfig)


class RerankingConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 16


class AgentRoleConfig(BaseModel):
    model: str = "planner"


class ValidatorConfig(BaseModel):
    model: str = "planner"
    scoring_criteria: list[str] = Field(
        default_factory=lambda: ["completeness", "accuracy", "relevance", "faithfulness"]
    )


class AgentsConfig(BaseModel):
    max_loop_count: int = 3
    validation_threshold: int = 80
    facilitator: AgentRoleConfig = Field(default_factory=AgentRoleConfig)
    synthesizer: AgentRoleConfig = Field(
        default_factory=lambda: AgentRoleConfig(model="executor")
    )
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)


class BM25TuningConfig(BaseModel):
    k1_range: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
    )
    b_range: list[float] = Field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    optimize_metric: str = "map@10"


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["precision@5", "recall@10", "map@10", "mrr@10"]
    )
    bm25_tuning: BM25TuningConfig = Field(default_factory=BM25TuningConfig)


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> Settings:
    """Load and return the Settings object from config.yaml.

    The result is cached so the file is only read once per process.

    Args:
        config_path: Optional path to a custom config file.
                     Defaults to config.yaml in the project root.

    Returns:
        Parsed and validated Settings object.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH

    if not path.exists():
        logger.warning("config.yaml not found at %s — using defaults", path)
        return Settings()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    settings = Settings.model_validate(raw or {})
    logger.debug("Config loaded from %s", path)
    return settings
