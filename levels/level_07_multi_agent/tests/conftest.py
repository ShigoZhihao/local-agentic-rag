"""
Shared pytest fixtures for the RAG test suite.

Fixtures that require external services (Weaviate, LM Studio) are
marked with the 'integration' pytest mark and skipped in unit test runs.
"""

from __future__ import annotations

import pytest

from src.models import (
    Chunk,
    ChunkStrategy,
    Citation,
    Document,
    SearchResult,
    SourceType,
    ValidationResult,
    ValidationScores,
)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> Document:
    """A minimal Document for use in chunker tests."""
    return Document(
        doc_id="doc-001",
        content="This is the first paragraph.\n\nThis is the second paragraph.",
        source_file="sample.txt",
        source_type=SourceType.TXT,
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    """A minimal Chunk with a placeholder embedding."""
    return Chunk(
        chunk_id="chunk-001",
        doc_id="doc-001",
        content="This is the first paragraph.",
        chunk_index=0,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        source_file="sample.txt",
        source_type=SourceType.TXT,
        embedding=[0.1] * 1024,
    )


@pytest.fixture
def sample_citation() -> Citation:
    """A Citation with original text preserved."""
    return Citation(
        citation_id=1,
        source_file="manual.pdf",
        page_number=3,
        original_text="The system must process requests within 200ms.",
        relevance_score=0.92,
    )


@pytest.fixture
def passing_validation() -> ValidationResult:
    """A ValidationResult that passes (average >= 80)."""
    scores = ValidationScores(
        completeness=85,
        accuracy=90,
        relevance=80,
        faithfulness=88,
        average=85.75,
    )
    return ValidationResult(
        scores=scores,
        is_valid=True,
        reason="All requirements covered with accurate citations.",
        missing_info=[],
    )


@pytest.fixture
def failing_validation() -> ValidationResult:
    """A ValidationResult that fails (average < 80)."""
    scores = ValidationScores(
        completeness=60,
        accuracy=70,
        relevance=75,
        faithfulness=65,
        average=67.5,
    )
    return ValidationResult(
        scores=scores,
        is_valid=False,
        reason="Completeness low: missing information about error handling.",
        missing_info=["Error handling procedures", "Rollback steps"],
    )
