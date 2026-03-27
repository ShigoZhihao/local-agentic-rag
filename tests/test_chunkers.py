"""
Unit tests for src/ingestion/chunkers.py and src/ingestion/loaders.py.

No external services required (Weaviate, LM Studio, GPU).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.ingestion.chunkers import (
    ExampleChunker,
    HTMLChunker,
    PythonChunker,
    RecursiveChunker,
    VisualChunker,
    get_chunker,
)
from src.models import Chunk, ChunkStrategy, Document, SourceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(
    content: str,
    source_type: SourceType = SourceType.TXT,
    metadata: dict | None = None,
) -> Document:
    return Document(
        doc_id="test-doc-001",
        content=content,
        source_file="test.txt",
        source_type=source_type,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_splits_long_text(self) -> None:
        # Create text longer than default chunk_size (512)
        content = "A sentence. " * 60  # ~720 chars
        doc = make_doc(content)
        chunker = RecursiveChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_chunk_ids_are_unique(self) -> None:
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = make_doc(content)
        chunks = RecursiveChunker().chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential(self) -> None:
        content = "Word " * 200
        doc = make_doc(content)
        chunks = RecursiveChunker().chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_strategy_is_recursive(self) -> None:
        doc = make_doc("Some content here.")
        chunks = RecursiveChunker().chunk(doc)
        assert all(c.chunk_strategy == ChunkStrategy.RECURSIVE for c in chunks)

    def test_short_text_single_chunk(self) -> None:
        doc = make_doc("Short text.")
        chunks = RecursiveChunker().chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_empty_content_returns_no_chunks(self) -> None:
        doc = make_doc("   ")
        chunks = RecursiveChunker().chunk(doc)
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# HTMLChunker
# ---------------------------------------------------------------------------

class TestHTMLChunker:
    def test_extracts_paragraph_tags(self) -> None:
        html = "<html><body><p>First para.</p><p>Second para.</p></body></html>"
        doc = make_doc(html, source_type=SourceType.HTML)
        chunks = HTMLChunker().chunk(doc)
        assert len(chunks) == 2
        assert "First para." in chunks[0].content
        assert "Second para." in chunks[1].content

    def test_extracts_heading_tags(self) -> None:
        html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>"
        doc = make_doc(html, source_type=SourceType.HTML)
        chunks = HTMLChunker().chunk(doc)
        contents = [c.content for c in chunks]
        assert any("Title" in c for c in contents)
        assert any("Subtitle" in c for c in contents)

    def test_fallback_on_no_tags(self) -> None:
        html = "<div>Plain text with no target tags.</div>"
        doc = make_doc(html, source_type=SourceType.HTML)
        chunks = HTMLChunker().chunk(doc)
        # Should fall back to plain text extraction (at least 1 chunk)
        assert len(chunks) >= 1

    def test_strategy_is_html(self) -> None:
        html = "<p>Content.</p>"
        doc = make_doc(html, source_type=SourceType.HTML)
        chunks = HTMLChunker().chunk(doc)
        assert all(c.chunk_strategy == ChunkStrategy.HTML for c in chunks)


# ---------------------------------------------------------------------------
# PythonChunker
# ---------------------------------------------------------------------------

class TestPythonChunker:
    def test_splits_on_functions(self) -> None:
        code = textwrap.dedent("""\
            import os

            def foo():
                return 1

            def bar():
                return 2
        """)
        doc = make_doc(code, source_type=SourceType.PY)
        chunks = PythonChunker().chunk(doc)
        # Should have: prologue (imports) + foo + bar = 3 parts
        assert len(chunks) == 3

    def test_splits_on_class(self) -> None:
        code = textwrap.dedent("""\
            class MyClass:
                def method(self):
                    pass

            def standalone():
                pass
        """)
        doc = make_doc(code, source_type=SourceType.PY)
        chunks = PythonChunker().chunk(doc)
        assert len(chunks) == 2

    def test_no_functions_single_chunk(self) -> None:
        code = "x = 1\ny = 2\nz = x + y\n"
        doc = make_doc(code, source_type=SourceType.PY)
        chunks = PythonChunker().chunk(doc)
        assert len(chunks) == 1

    def test_strategy_is_python(self) -> None:
        code = "def f():\n    pass\n"
        doc = make_doc(code, source_type=SourceType.PY)
        chunks = PythonChunker().chunk(doc)
        assert all(c.chunk_strategy == ChunkStrategy.PYTHON for c in chunks)

    def test_decorated_function(self) -> None:
        code = textwrap.dedent("""\
            @property
            def value(self):
                return self._value
        """)
        doc = make_doc(code, source_type=SourceType.PY)
        chunks = PythonChunker().chunk(doc)
        assert len(chunks) >= 1
        assert "@property" in chunks[0].content


# ---------------------------------------------------------------------------
# ExampleChunker
# ---------------------------------------------------------------------------

class TestExampleChunker:
    def test_splits_on_delimiter(self) -> None:
        content = "Example 1\n---\nExample 2\n---\nExample 3"
        doc = make_doc(content)
        chunks = ExampleChunker().chunk(doc)
        assert len(chunks) == 3
        assert "Example 1" in chunks[0].content
        assert "Example 2" in chunks[1].content
        assert "Example 3" in chunks[2].content

    def test_no_delimiter_single_chunk(self) -> None:
        content = "One example without any delimiter."
        doc = make_doc(content)
        chunks = ExampleChunker().chunk(doc)
        assert len(chunks) == 1

    def test_strategy_is_example(self) -> None:
        content = "A\n---\nB"
        doc = make_doc(content)
        chunks = ExampleChunker().chunk(doc)
        assert all(c.chunk_strategy == ChunkStrategy.EXAMPLE for c in chunks)


# ---------------------------------------------------------------------------
# VisualChunker
# ---------------------------------------------------------------------------

class TestVisualChunker:
    def test_one_chunk_per_page(self) -> None:
        metadata = {
            "pages": [
                {"page_number": 1, "text": "Page one content.", "image_path": None},
                {"page_number": 2, "text": "Page two content.", "image_path": None},
                {"page_number": 3, "text": "Page three content.", "image_path": None},
            ]
        }
        doc = make_doc("", source_type=SourceType.PDF, metadata=metadata)
        chunks = VisualChunker().chunk(doc)
        assert len(chunks) == 3

    def test_page_number_set_correctly(self) -> None:
        metadata = {
            "pages": [
                {"page_number": 5, "text": "Content.", "image_path": None},
            ]
        }
        doc = make_doc("", source_type=SourceType.PDF, metadata=metadata)
        chunks = VisualChunker().chunk(doc)
        assert chunks[0].page_number == 5

    def test_image_path_preserved(self) -> None:
        metadata = {
            "pages": [
                {"page_number": 1, "text": "Content.", "image_path": "/tmp/slide.png"},
            ]
        }
        doc = make_doc("", source_type=SourceType.PDF, metadata=metadata)
        chunks = VisualChunker().chunk(doc)
        assert chunks[0].image_path == "/tmp/slide.png"

    def test_empty_pages_skipped(self) -> None:
        metadata = {
            "pages": [
                {"page_number": 1, "text": "Content.", "image_path": None},
                {"page_number": 2, "text": "   ", "image_path": None},  # blank
                {"page_number": 3, "text": "More.", "image_path": None},
            ]
        }
        doc = make_doc("", source_type=SourceType.PDF, metadata=metadata)
        chunks = VisualChunker().chunk(doc)
        assert len(chunks) == 2

    def test_fallback_on_missing_metadata(self) -> None:
        doc = make_doc("Full content fallback.", source_type=SourceType.PDF)
        chunks = VisualChunker().chunk(doc)
        assert len(chunks) == 1
        assert "Full content fallback." in chunks[0].content

    def test_strategy_is_visual(self) -> None:
        metadata = {
            "pages": [
                {"page_number": 1, "text": "Content.", "image_path": None},
            ]
        }
        doc = make_doc("", source_type=SourceType.PDF, metadata=metadata)
        chunks = VisualChunker().chunk(doc)
        assert all(c.chunk_strategy == ChunkStrategy.VISUAL for c in chunks)


# ---------------------------------------------------------------------------
# get_chunker factory
# ---------------------------------------------------------------------------

class TestGetChunkerFactory:
    def test_txt_returns_recursive(self) -> None:
        assert isinstance(get_chunker(SourceType.TXT), RecursiveChunker)

    def test_md_returns_recursive(self) -> None:
        assert isinstance(get_chunker(SourceType.MD), RecursiveChunker)

    def test_html_returns_html(self) -> None:
        assert isinstance(get_chunker(SourceType.HTML), HTMLChunker)

    def test_py_returns_python(self) -> None:
        assert isinstance(get_chunker(SourceType.PY), PythonChunker)

    def test_pdf_returns_visual(self) -> None:
        assert isinstance(get_chunker(SourceType.PDF), VisualChunker)

    def test_pptx_returns_visual(self) -> None:
        assert isinstance(get_chunker(SourceType.PPTX), VisualChunker)

    def test_strategy_override(self) -> None:
        # Override PDF to use recursive chunker
        chunker = get_chunker(SourceType.PDF, strategy=ChunkStrategy.RECURSIVE)
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            get_chunker(SourceType.TXT, strategy="nonexistent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chunk model properties
# ---------------------------------------------------------------------------

class TestChunkProperties:
    def test_source_file_preserved(self) -> None:
        doc = Document(
            doc_id="d1",
            content="Content.",
            source_file="my_doc.txt",
            source_type=SourceType.TXT,
        )
        chunks = RecursiveChunker().chunk(doc)
        assert all(c.source_file == "my_doc.txt" for c in chunks)

    def test_doc_id_preserved(self) -> None:
        doc = Document(
            doc_id="unique-123",
            content="Content.",
            source_file="f.txt",
            source_type=SourceType.TXT,
        )
        chunks = RecursiveChunker().chunk(doc)
        assert all(c.doc_id == "unique-123" for c in chunks)

    def test_embedding_none_before_embed(self) -> None:
        doc = make_doc("Content.")
        chunks = RecursiveChunker().chunk(doc)
        assert all(c.embedding is None for c in chunks)
