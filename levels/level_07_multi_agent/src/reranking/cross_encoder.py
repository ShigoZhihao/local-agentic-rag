"""
Cross-Encoder re-ranker using ms-marco-MiniLM-L-6-v2.

Runs on CPU (22M parameters, fast enough for top_k=20 candidates).
Takes the initial retrieval results and re-scores each (query, passage) pair
using a cross-attention model for more precise relevance ranking.

Usage:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("user query", initial_results)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_config
from src.models import SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank search results using a cross-encoder model.

    The model is loaded lazily on first use and cached on the instance.
    Always runs on CPU regardless of GPU availability (model is small enough).

    Attributes:
        model_name: HuggingFace model identifier.
        top_k: Maximum number of results to return after re-ranking.
    """

    def __init__(self) -> None:
        cfg = get_config().reranking
        self.model_name: str = cfg.model_name
        self.top_k: int = cfg.top_k
        self.batch_size: int = cfg.batch_size
        self._model = None  # Lazy load

    def _load_model(self) -> None:
        """Load the cross-encoder model (lazy, cached)."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model: %s", self.model_name)
        self._model = CrossEncoder(self.model_name, device="cpu")
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-score and re-rank search results using cross-attention.

        Args:
            query: The original search query.
            results: Candidate results from initial retrieval (BM25 + semantic).
            top_k: Number of results to return. Defaults to config value.

        Returns:
            Re-ranked list of SearchResult objects (new scores from cross-encoder).
            Length is min(top_k, len(results)).
        """
        if not results:
            return []

        k = top_k if top_k is not None else self.top_k
        self._load_model()

        # Build (query, passage) pairs for scoring
        pairs = [(query, r.chunk.content) for r in results]

        logger.debug(
            "CrossEncoder: reranking %d results (batch_size=%d)",
            len(pairs),
            self.batch_size,
        )
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach new scores and sort descending
        scored = [
            SearchResult(chunk=r.chunk, score=float(s), search_type="reranked")
            for r, s in zip(results, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "CrossEncoder top score: %.4f, bottom of top-%d: %.4f",
            scored[0].score if scored else 0.0,
            k,
            scored[min(k - 1, len(scored) - 1)].score if scored else 0.0,
        )

        return scored[:k]
