"""
Post-retrieval metadata filtering.

After Weaviate returns search results, this module applies additional
Python-side filtering that is either too complex for Weaviate's where
clause or needs to run after re-ranking.

Usage:
    filter = MetadataFilter(source_types=["pdf"], min_score=0.3)
    filtered = filter.apply(results)
"""

from __future__ import annotations

import logging

from src.models import SearchResult, SourceType

logger = logging.getLogger(__name__)


class MetadataFilter:
    """Filter search results by metadata criteria.

    All criteria are optional. If not specified, that criterion is ignored.

    Args:
        source_types: Keep only results with these source types.
        min_score: Discard results below this relevance score.
        max_results: Return at most this many results.
        source_files: Keep only results from these specific files.
    """

    def __init__(
        self,
        *,
        source_types: list[str] | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        source_files: list[str] | None = None,
    ) -> None:
        self._source_types = set(source_types) if source_types else None
        self._min_score = min_score
        self._max_results = max_results
        self._source_files = set(source_files) if source_files else None

    def apply(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply all configured filters to a list of search results.

        Args:
            results: Ordered list of SearchResult objects from Weaviate.

        Returns:
            Filtered (and possibly truncated) list. Order is preserved.
        """
        filtered = results

        if self._source_types:
            filtered = [
                r for r in filtered
                if r.chunk.source_type.value in self._source_types
            ]

        if self._source_files:
            filtered = [
                r for r in filtered
                if r.chunk.source_file in self._source_files
            ]

        if self._min_score is not None:
            filtered = [r for r in filtered if r.score >= self._min_score]

        if self._max_results is not None:
            filtered = filtered[: self._max_results]

        logger.debug(
            "MetadataFilter: %d → %d results", len(results), len(filtered)
        )
        return filtered


def apply_filters(
    results: list[SearchResult],
    metadata_filters: dict,
) -> list[SearchResult]:
    """Convenience function: filter results using a dict of criteria.

    Supported keys:
      - "source_types": list[str] — e.g. ["pdf", "pptx"]
      - "source_files": list[str] — e.g. ["manual.pdf"]
      - "min_score": float
      - "max_results": int

    Args:
        results: Search results to filter.
        metadata_filters: Dict of filter criteria.

    Returns:
        Filtered results list.
    """
    f = MetadataFilter(
        source_types=metadata_filters.get("source_types"),
        source_files=metadata_filters.get("source_files"),
        min_score=metadata_filters.get("min_score"),
        max_results=metadata_filters.get("max_results"),
    )
    return f.apply(results)
