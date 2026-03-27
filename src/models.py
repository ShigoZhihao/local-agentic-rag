"""
Pydantic data models shared across the entire RAG system.

All domain objects are defined here. Import from this module
instead of creating local dataclasses or plain dicts.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """File type of the original document."""

    TXT = "txt"
    MD = "md"
    HTML = "html"
    PY = "py"
    PDF = "pdf"
    PPTX = "pptx"


class ChunkStrategy(str, Enum):
    """Which chunking strategy was used to create a Chunk."""

    RECURSIVE = "recursive"
    HTML = "html"
    PYTHON = "python"
    SEMANTIC = "semantic"
    EXAMPLE = "example"
    VISUAL = "visual"  # Vision LLMで生成したスライド/ページ説明


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A raw document before chunking."""

    doc_id: str
    content: str
    source_file: str
    source_type: SourceType
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A document chunk ready for embedding and storage in Weaviate."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    chunk_strategy: ChunkStrategy
    source_file: str
    source_type: SourceType
    page_number: int | None = None    # PDF/PPTXのページ/スライド番号
    image_path: str | None = None     # レンダリングされた画像パス（UI表示用）
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """A single retrieval result with relevance score."""

    chunk: Chunk
    score: float
    search_type: str  # "bm25", "semantic", "hybrid", "colbert"


class Citation(BaseModel):
    """
    A source citation in NotebookLM style.

    Stores the original text as-is rather than re-generating it,
    which helps prevent hallucinations.
    """

    citation_id: int
    source_file: str
    page_number: int | None = None
    original_text: str    # 元のテキストそのまま（生成し直さない）
    relevance_score: float


# ---------------------------------------------------------------------------
# Agent / workflow models
# ---------------------------------------------------------------------------

class ValidationScores(BaseModel):
    """4-axis quality scores from the Validator (LLM-as-Judge)."""

    completeness: int   # 0-100: enriched promptの全要件カバー率
    accuracy: int       # 0-100: 引用元テキストとの整合性
    relevance: int      # 0-100: ユーザ元意図との関連性
    faithfulness: int   # 0-100: 引用外情報を生成していないか
    average: float      # 4点平均（80以上で合格）


class ValidationResult(BaseModel):
    """Output from the Validator agent."""

    scores: ValidationScores
    is_valid: bool              # average >= threshold
    reason: str                 # 判断理由（低スコア項目の具体的説明）
    missing_info: list[str] = Field(default_factory=list)  # 具体的な不足情報


class ConversationTurn(BaseModel):
    """One turn in the chat history shown in the Streamlit UI."""

    role: str                   # "user", "facilitator", "synthesizer", "system"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    validation: ValidationResult | None = None
    loop_count: int = 0


# ---------------------------------------------------------------------------
# Query / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming user query with optional search settings."""

    query: str
    use_colbert: bool = False
    metadata_filters: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Final response returned to the Streamlit UI."""

    answer: str
    citations: list[Citation]
    enriched_prompt: str
    validation: ValidationResult | None = None
    loop_count: int = 0
