"""
Document ingestion pipeline orchestrator.

Takes raw document files, chunks them, embeds the chunks,
and upserts them into Weaviate. Handles all supported file types.

Usage:
    pipeline = IngestionPipeline()
    stats = pipeline.ingest_file("path/to/document.pdf")
    stats = pipeline.ingest_directory("data/raw/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import get_config
from src.ingestion.chunkers import get_chunker
from src.ingestion.embedder import Embedder
from src.ingestion.loaders import get_source_type, load_document
from src.models import Chunk, ChunkStrategy, SourceType

logger = logging.getLogger(__name__)

# File extensions supported by the pipeline
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".py", ".pdf", ".pptx"}


@dataclass
class IngestionStats:
    """Statistics returned after ingesting a file or directory."""

    total_files: int = 0
    success_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (file, error_message)


class IngestionPipeline:
    """Full ingestion pipeline: load → chunk → embed → upsert to Weaviate.

    Args:
        use_vision: If True, call the Vision LLM for PDF/PPTX images.
                    Requires a rendered_dir and Vision LLM loaded in LM Studio.
        rendered_dir: Directory to save rendered page/slide images.
                      Required when use_vision=True.
        progress_callback: Optional callable(file_name, chunks_done, total_chunks)
                           for UI progress reporting.
    """

    def __init__(
        self,
        *,
        use_vision: bool = False,
        rendered_dir: str | Path | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self._use_vision = use_vision
        self._rendered_dir = Path(rendered_dir) if rendered_dir else None
        self._progress_callback = progress_callback
        self._embedder = Embedder()
        # Weaviate client is imported lazily to avoid import-time connection
        self._weaviate_client = None

    def _get_weaviate_client(self):  # type: ignore[return]
        """Lazy-load and cache the Weaviate client."""
        if self._weaviate_client is None:
            from src.retrieval.weaviate_client import get_client
            self._weaviate_client = get_client()
        return self._weaviate_client

    def _embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks and return them with .embedding set."""
        texts = [c.content for c in chunks]
        vectors = self._embedder.embed_texts(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
        return chunks

    def _apply_vision(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich PDF/PPTX chunks with Vision LLM descriptions."""
        from src.ingestion.vision_describer import enrich_chunk_with_vision

        enriched: list[Chunk] = []
        for chunk in chunks:
            if chunk.source_type in (SourceType.PDF, SourceType.PPTX) and chunk.image_path:
                new_content = enrich_chunk_with_vision(chunk.content, chunk.image_path)
                # Build new chunk with updated content (Pydantic models are immutable by default)
                enriched.append(chunk.model_copy(update={"content": new_content}))
            else:
                enriched.append(chunk)
        return enriched

    def ingest_file(
        self,
        path: str | Path,
        *,
        strategy: ChunkStrategy | None = None,
    ) -> IngestionStats:
        """Ingest a single file into Weaviate.

        Args:
            path: Path to the document file.
            strategy: Override the default chunking strategy for this file.

        Returns:
            IngestionStats with chunk count and any errors.
        """
        p = Path(path)
        stats = IngestionStats(total_files=1)

        try:
            source_type = get_source_type(p)
            logger.info("Ingesting: %s (%s)", p.name, source_type.value)

            # 1. Load
            rendered_dir = self._rendered_dir if source_type in (SourceType.PDF, SourceType.PPTX) else None
            doc = load_document(p, rendered_dir=rendered_dir)

            # 2. Chunk
            chunker = get_chunker(source_type, strategy=strategy)
            chunks = chunker.chunk(doc)
            logger.info("  → %d chunks", len(chunks))

            if not chunks:
                logger.warning("No chunks produced for %s", p.name)
                stats.success_files += 1
                return stats

            # 3. Vision enrichment (PDF/PPTX only, if enabled)
            if self._use_vision:
                chunks = self._apply_vision(chunks)

            # 4. Embed
            chunks = self._embed_chunks(chunks)

            # 5. Progress callback
            if self._progress_callback:
                self._progress_callback(p.name, 0, len(chunks))

            # 6. Upsert to Weaviate
            from src.retrieval.weaviate_client import upsert_chunks
            client = self._get_weaviate_client()
            upsert_chunks(client, chunks)

            stats.success_files += 1
            stats.total_chunks += len(chunks)
            logger.info("  → upserted %d chunks for %s", len(chunks), p.name)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", p.name, exc, exc_info=True)
            stats.failed_files += 1
            stats.errors.append((p.name, str(exc)))

        return stats

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        strategy_overrides: dict[SourceType, ChunkStrategy] | None = None,
    ) -> IngestionStats:
        """Ingest all supported files in a directory.

        Args:
            directory: Path to the directory containing documents.
            recursive: If True, also scan subdirectories.
            strategy_overrides: Map from SourceType to ChunkStrategy to override
                                 the default strategy for specific file types.

        Returns:
            Aggregated IngestionStats across all files.
        """
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")

        pattern = "**/*" if recursive else "*"
        files = [
            f for f in d.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info("Found %d supported files in %s", len(files), d)
        total = IngestionStats(total_files=len(files))

        for f in files:
            try:
                source_type = get_source_type(f)
                strategy = (strategy_overrides or {}).get(source_type)
            except ValueError:
                continue  # Unsupported extension (shouldn't happen given filter above)

            stats = self.ingest_file(f, strategy=strategy)
            total.success_files += stats.success_files
            total.failed_files += stats.failed_files
            total.total_chunks += stats.total_chunks
            total.errors.extend(stats.errors)

        logger.info(
            "Ingestion complete: %d/%d files, %d chunks, %d errors",
            total.success_files,
            total.total_files,
            total.total_chunks,
            total.failed_files,
        )
        return total

    def close(self) -> None:
        """Close the Weaviate client connection."""
        if self._weaviate_client is not None:
            self._weaviate_client.close()
            self._weaviate_client = None
