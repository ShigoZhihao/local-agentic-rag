"""
Chunking strategies for all document types.

Each chunker inherits BaseChunker and implements chunk(doc) -> list[Chunk].
Use get_chunker() to obtain the right chunker for a given source type.

Strategy selection:
  TXT, MD       → RecursiveChunker
  HTML          → HTMLChunker
  PY            → PythonChunker
  PDF, PPTX     → VisualChunker (1 page/slide = 1 chunk)
  example files → ExampleChunker  (delimiter-based)
  semantic      → SemanticChunker (cosine-threshold)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from src.config import get_config
from src.models import Chunk, ChunkStrategy, Document, SourceType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Abstract base for all chunkers.

    Subclasses must implement chunk().
    """

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split the document into chunks.

        Args:
            doc: Source document to split.

        Returns:
            Ordered list of Chunk objects (chunk_index is set automatically).
        """

    # ------------------------------------------------------------------
    # Helpers shared by all chunkers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        """Generate a deterministic chunk_id from doc_id + index."""
        raw = f"{doc_id}-{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _build_chunk(
        self,
        doc: Document,
        content: str,
        index: int,
        strategy: ChunkStrategy,
        *,
        page_number: int | None = None,
        image_path: str | None = None,
        extra_metadata: dict | None = None,
    ) -> Chunk:
        """Create a Chunk from document metadata + content slice."""
        meta = dict(doc.metadata)
        if extra_metadata:
            meta.update(extra_metadata)
        return Chunk(
            chunk_id=self._make_chunk_id(doc.doc_id, index),
            doc_id=doc.doc_id,
            content=content.strip(),
            chunk_index=index,
            chunk_strategy=strategy,
            source_file=doc.source_file,
            source_type=doc.source_type,
            page_number=page_number,
            image_path=image_path,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# RecursiveChunker — TXT, MD (and fallback)
# ---------------------------------------------------------------------------

class RecursiveChunker(BaseChunker):
    """Split text recursively by paragraph/sentence/word boundaries.

    Uses LangChain's RecursiveCharacterTextSplitter under the hood.
    """

    def __init__(self) -> None:
        cfg = get_config().chunking.recursive
        self._chunk_size = cfg.chunk_size
        self._overlap = int(cfg.chunk_size * cfg.overlap_percent / 100)
        self._separators = cfg.separators

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split the document into fixed-size chunks with optional overlap."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._overlap,
            separators=self._separators,
        )
        texts = splitter.split_text(doc.content)
        chunks = [
            self._build_chunk(doc, text, i, ChunkStrategy.RECURSIVE)
            for i, text in enumerate(texts)
            if text.strip()
        ]
        logger.debug("RecursiveChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# HTMLChunker — HTML
# ---------------------------------------------------------------------------

class HTMLChunker(BaseChunker):
    """Split HTML by semantic tag boundaries (p, h1-h4, li, etc.)."""

    def __init__(self) -> None:
        cfg = get_config().chunking.html
        self._tags = cfg.tags_to_split

    def chunk(self, doc: Document) -> list[Chunk]:
        """Extract text from each HTML tag as a separate chunk."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(doc.content, "html.parser")
        texts: list[str] = []

        for tag in soup.find_all(self._tags):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)

        # Fallback: if no tags matched, treat as plain text
        if not texts:
            logger.debug("HTMLChunker: no target tags found in %s, falling back", doc.source_file)
            plain = BeautifulSoup(doc.content, "html.parser").get_text(separator="\n", strip=True)
            texts = [plain] if plain else []

        chunks = [
            self._build_chunk(doc, text, i, ChunkStrategy.HTML)
            for i, text in enumerate(texts)
            if text.strip()
        ]
        logger.debug("HTMLChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# PythonChunker — .py files
# ---------------------------------------------------------------------------

class PythonChunker(BaseChunker):
    """Split Python source at function/class boundaries.

    Each top-level function or class (including its decorators) becomes
    a separate chunk. Code before the first definition goes into its own
    chunk (module-level code / imports).
    """

    # Regex to find the start of a top-level def/class (not indented)
    _DEF_PATTERN = re.compile(r"^((?:@\S+\s*\n)*(?:async\s+)?(?:def|class)\s+\w+)", re.MULTILINE)

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split at top-level def/class boundaries."""
        text = doc.content
        boundaries = [m.start() for m in self._DEF_PATTERN.finditer(text)]

        if not boundaries:
            # No functions/classes — return whole file as one chunk
            return [self._build_chunk(doc, text, 0, ChunkStrategy.PYTHON)]

        parts: list[str] = []

        # Code before first function/class
        prologue = text[: boundaries[0]].strip()
        if prologue:
            parts.append(prologue)

        # Each function/class block
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            parts.append(text[start:end].strip())

        chunks = [
            self._build_chunk(doc, part, i, ChunkStrategy.PYTHON)
            for i, part in enumerate(parts)
            if part.strip()
        ]
        logger.debug("PythonChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# SemanticChunker — cosine-similarity boundary detection
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """Merge sentences into chunks until cosine similarity drops below threshold.

    This uses the embedder at chunking time, so it is slower than the others.
    Only recommended for documents where semantic coherence matters most.
    """

    def __init__(self) -> None:
        cfg = get_config().chunking.semantic
        self._threshold = cfg.cosine_threshold
        self._min_size = cfg.min_chunk_size
        self._max_size = cfg.max_chunk_size
        self._embedder: Any = None  # lazy load

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from src.ingestion.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def chunk(self, doc: Document) -> list[Chunk]:
        """Group sentences into semantically coherent chunks."""
        import numpy as np

        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s+", doc.content) if s.strip()]
        if not sentences:
            return []

        embedder = self._get_embedder()
        embeddings = embedder.embed_texts(sentences)

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        current_emb = embeddings[0]

        for i in range(1, len(sentences)):
            sent_emb = embeddings[i]
            # Cosine similarity between current group centroid and next sentence
            norm_curr = current_emb / (np.linalg.norm(current_emb) + 1e-10)
            norm_sent = sent_emb / (np.linalg.norm(sent_emb) + 1e-10)
            sim = float(np.dot(norm_curr, norm_sent))

            current_text = " ".join(current)
            if sim >= self._threshold and len(current_text) < self._max_size:
                current.append(sentences[i])
                # Update centroid (running average)
                n = len(current)
                current_emb = current_emb * (n - 1) / n + sent_emb / n
            else:
                if len(current_text) >= self._min_size:
                    groups.append(current)
                    current = [sentences[i]]
                    current_emb = sent_emb
                else:
                    # Too short — absorb into current group
                    current.append(sentences[i])

        if current:
            groups.append(current)

        chunks = [
            self._build_chunk(doc, " ".join(group), i, ChunkStrategy.SEMANTIC)
            for i, group in enumerate(groups)
        ]
        logger.debug("SemanticChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# ExampleChunker — delimiter-based (script/example collections)
# ---------------------------------------------------------------------------

class ExampleChunker(BaseChunker):
    """Split an example-collection file by a fixed delimiter.

    The delimiter defaults to "---" (three dashes on its own line) but is
    configurable via config.yaml (chunking.example.delimiter).
    """

    def __init__(self) -> None:
        cfg = get_config().chunking.example
        self._delimiter = cfg.delimiter

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split the document at each occurrence of the delimiter."""
        parts = re.split(
            rf"^\s*{re.escape(self._delimiter)}\s*$",
            doc.content,
            flags=re.MULTILINE,
        )
        chunks = [
            self._build_chunk(doc, part, i, ChunkStrategy.EXAMPLE)
            for i, part in enumerate(parts)
            if part.strip()
        ]
        logger.debug("ExampleChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# VisualChunker — PDF/PPTX (1 page/slide = 1 chunk)
# ---------------------------------------------------------------------------

class VisualChunker(BaseChunker):
    """Create one chunk per page (PDF) or slide (PPTX).

    Each chunk's content is the extracted text for that page/slide.
    The image_path field is populated if the document metadata contains
    rendered image paths (set by load_pdf/load_pptx with rendered_dir).

    Vision LLM descriptions are appended to the content by the ingestion
    pipeline (vision_describer.py) after chunking.
    """

    def chunk(self, doc: Document) -> list[Chunk]:
        """One chunk per page or slide."""
        is_pdf = doc.source_type == SourceType.PDF
        key = "pages" if is_pdf else "slides"
        page_key = "page_number" if is_pdf else "slide_number"

        pages: list[dict] = doc.metadata.get(key, [])

        if not pages:
            # Fallback: treat entire content as a single chunk
            logger.warning(
                "VisualChunker: no %s metadata in %s, using full content",
                key,
                doc.source_file,
            )
            return [self._build_chunk(doc, doc.content, 0, ChunkStrategy.VISUAL)]

        chunks = [
            self._build_chunk(
                doc,
                page["text"],
                i,
                ChunkStrategy.VISUAL,
                page_number=page[page_key],
                image_path=page.get("image_path"),
            )
            for i, page in enumerate(pages)
            if page["text"].strip()
        ]
        logger.debug("VisualChunker: %s → %d chunks", doc.source_file, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[ChunkStrategy, type[BaseChunker]] = {
    ChunkStrategy.RECURSIVE: RecursiveChunker,
    ChunkStrategy.HTML: HTMLChunker,
    ChunkStrategy.PYTHON: PythonChunker,
    ChunkStrategy.SEMANTIC: SemanticChunker,
    ChunkStrategy.EXAMPLE: ExampleChunker,
    ChunkStrategy.VISUAL: VisualChunker,
}

_SOURCE_TYPE_DEFAULT: dict[SourceType, ChunkStrategy] = {
    SourceType.TXT: ChunkStrategy.RECURSIVE,
    SourceType.MD: ChunkStrategy.RECURSIVE,
    SourceType.HTML: ChunkStrategy.HTML,
    SourceType.PY: ChunkStrategy.PYTHON,
    SourceType.PDF: ChunkStrategy.VISUAL,
    SourceType.PPTX: ChunkStrategy.VISUAL,
}


def get_chunker(
    source_type: SourceType,
    *,
    strategy: ChunkStrategy | None = None,
) -> BaseChunker:
    """Return the appropriate chunker for a source type.

    Args:
        source_type: The type of the source document.
        strategy: Override the default strategy for this source type.

    Returns:
        An instantiated BaseChunker subclass.

    Raises:
        ValueError: If the requested strategy is unknown.
    """
    resolved = strategy or _SOURCE_TYPE_DEFAULT.get(source_type, ChunkStrategy.RECURSIVE)
    chunker_cls = _STRATEGY_MAP.get(resolved)
    if chunker_cls is None:
        raise ValueError(f"Unknown chunk strategy: {resolved!r}")
    return chunker_cls()
