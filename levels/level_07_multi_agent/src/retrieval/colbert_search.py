"""
ColBERT on-demand search via RAGatouille.

ColBERT is disabled by default (config.yaml: retrieval.colbert.enabled=false).
When enabled, it provides late-interaction token-level matching that catches
precise terminology that dense vectors miss.

Results are limited to top_k=10 (fixed) and immediately converted to Python
lists to avoid holding the full ColBERT index in RAM.

Note: ragatouille is optional. If not installed, this module logs a warning
and returns an empty list rather than crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_config
from src.models import SearchResult

logger = logging.getLogger(__name__)

_INDEX_DIR = Path("data/colbert_index")


class ColBERTSearcher:
    """On-demand ColBERT search using RAGatouille.

    The ColBERT index is built once from the current Weaviate collection
    content and stored on disk. It is loaded lazily on first search call.
    """

    # Fixed top_k per plan specification (RAM efficiency)
    TOP_K = 10

    def __init__(self) -> None:
        self._model = None  # Lazy load

    def _is_enabled(self) -> bool:
        return get_config().retrieval.colbert.enabled

    def _load_model(self) -> None:
        """Load the RAGatouille model (lazy)."""
        if self._model is not None:
            return
        try:
            from ragatouille import RAGPretrainedModel  # type: ignore[import]
        except ImportError:
            logger.warning("ragatouille not installed — ColBERT search disabled")
            return

        cfg = get_config().retrieval.colbert
        index_path = _INDEX_DIR / "index"
        if index_path.exists():
            logger.info("Loading ColBERT index from %s", index_path)
            self._model = RAGPretrainedModel.from_index(str(index_path))
        else:
            logger.info("Loading ColBERT model %s (no index yet)", cfg.model_name)
            self._model = RAGPretrainedModel.from_pretrained(cfg.model_name)

    def build_index(self, texts: list[str], doc_ids: list[str]) -> None:
        """Build a ColBERT index from the provided texts.

        This is called once during or after ingestion when ColBERT is enabled.
        Should not be called at query time.

        Args:
            texts: List of chunk content strings.
            doc_ids: Corresponding chunk IDs (used as document identifiers).
        """
        if not self._is_enabled():
            logger.debug("ColBERT disabled, skipping index build")
            return

        self._load_model()
        if self._model is None:
            return

        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Building ColBERT index for %d documents", len(texts))
        self._model.index(
            collection=texts,
            document_ids=doc_ids,
            index_name="index",
            overwrite_index=True,
        )
        logger.info("ColBERT index built")

    def search(self, query: str) -> list[SearchResult]:
        """Search using ColBERT late-interaction scoring.

        Returns at most TOP_K=10 results. If ColBERT is disabled or
        ragatouille is not installed, returns an empty list.

        Args:
            query: Search query string.

        Returns:
            List of SearchResult objects (search_type="colbert").
            The chunk objects have minimal data (content + chunk_id only).
            Caller should enrich from Weaviate if needed.
        """
        from src.models import Chunk, ChunkStrategy, SourceType

        if not self._is_enabled():
            logger.debug("ColBERT disabled, returning empty results")
            return []

        self._load_model()
        if self._model is None:
            return []

        index_path = _INDEX_DIR / "index"
        if not index_path.exists():
            logger.warning("ColBERT index not found at %s — run build_index first", index_path)
            return []

        logger.debug("ColBERT search: query=%r, top_k=%d", query, self.TOP_K)
        raw_results = self._model.search(query, k=self.TOP_K)

        # Convert immediately to Python list (don't hold full index in RAM)
        results: list[SearchResult] = []
        for item in raw_results:
            # RAGatouille returns: {"content": ..., "document_id": ..., "score": ...}
            chunk = Chunk(
                chunk_id=item.get("document_id", ""),
                doc_id="",
                content=item.get("content", ""),
                chunk_index=0,
                chunk_strategy=ChunkStrategy.RECURSIVE,
                source_file="",
                source_type=SourceType.TXT,
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(item.get("score", 0.0)),
                    search_type="colbert",
                )
            )

        logger.debug("ColBERT returned %d results", len(results))
        return results
