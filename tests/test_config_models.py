"""
Unit tests for src/config.py and src/models.py.
No external dependencies required.
"""

from __future__ import annotations

from src.config import get_config
from src.models import (
    Chunk,
    ChunkStrategy,
    Citation,
    ConversationTurn,
    Document,
    QueryRequest,
    QueryResponse,
    SearchResult,
    SourceType,
    ValidationResult,
    ValidationScores,
)


class TestConfig:
    def test_loads_from_yaml(self) -> None:
        cfg = get_config()
        assert cfg.ollama.base_url == "http://127.0.0.1:11434/v1"
        assert cfg.weaviate.port == 8080
        assert cfg.retrieval.bm25.k1 == 1.2
        assert cfg.retrieval.bm25.b == 0.75
        assert cfg.agents.max_loop_count == 3
        assert cfg.agents.validation_threshold == 80

    def test_chunking_defaults(self) -> None:
        cfg = get_config()
        assert cfg.chunking.default_strategy == "recursive"
        assert cfg.chunking.recursive.overlap_percent == 0
        assert cfg.chunking.recursive.chunk_size == 512

    def test_evaluation_metrics(self) -> None:
        cfg = get_config()
        assert "map@10" in cfg.evaluation.metrics
        assert "mrr@10" in cfg.evaluation.metrics


class TestModels:
    def test_document_creation(self, sample_document: Document) -> None:
        assert sample_document.doc_id == "doc-001"
        assert sample_document.source_type == SourceType.TXT

    def test_chunk_creation(self, sample_chunk: Chunk) -> None:
        assert sample_chunk.chunk_id == "chunk-001"
        assert sample_chunk.chunk_strategy == ChunkStrategy.RECURSIVE
        assert len(sample_chunk.embedding) == 1024

    def test_citation_preserves_original_text(self, sample_citation: Citation) -> None:
        """Citation must store original_text without modification."""
        assert sample_citation.original_text == "The system must process requests within 200ms."
        assert sample_citation.citation_id == 1

    def test_validation_scores_average(self) -> None:
        scores = ValidationScores(
            completeness=80,
            accuracy=80,
            relevance=80,
            faithfulness=80,
            average=80.0,
        )
        assert scores.average == 80.0

    def test_passing_validation(self, passing_validation: ValidationResult) -> None:
        assert passing_validation.is_valid is True
        assert passing_validation.scores.average >= 80

    def test_failing_validation(self, failing_validation: ValidationResult) -> None:
        assert failing_validation.is_valid is False
        assert len(failing_validation.missing_info) > 0

    def test_source_type_enum_values(self) -> None:
        assert SourceType.PDF.value == "pdf"
        assert SourceType.PPTX.value == "pptx"
        assert SourceType.PY.value == "py"

    def test_chunk_strategy_enum_values(self) -> None:
        assert ChunkStrategy.VISUAL.value == "visual"
        assert ChunkStrategy.EXAMPLE.value == "example"

    def test_query_request_defaults(self) -> None:
        req = QueryRequest(query="test question")
        assert req.use_colbert is False
        assert req.metadata_filters == {}
