"""
Hybrid search: BM25 + Semantic (dense vector) via Weaviate.

The alpha parameter controls the balance between BM25 and semantic search:
  alpha=0.0  → pure BM25
  alpha=1.0  → pure semantic
  alpha=0.5  → equal weight (default)

Usage:
    searcher = HybridSearcher()
    results = searcher.search("how to configure logging", top_k=20)
"""

from __future__ import annotations

import logging

import weaviate
import weaviate.classes as wvc

from src.config import get_config
from src.ingestion.embedder import Embedder
from src.models import Chunk, SearchResult
from src.retrieval.weaviate_client import COLLECTION_NAME, _row_to_chunk

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Perform hybrid BM25 + semantic search against Weaviate.

    Args:
        client: An open WeaviateClient instance. If None, a new client is
                created using get_client().
        embedder: Embedder instance for query vectorization. If None,
                  a new Embedder is created (loads model on first use).
    """

    def __init__(
        self,
        client: weaviate.WeaviateClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if client is None:
            from src.retrieval.weaviate_client import get_client
            client = get_client()
        self._client = client
        self._embedder = embedder or Embedder()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        alpha: float | None = None,
        metadata_filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run a hybrid search and return ranked results.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return (defaults to config value).
            alpha: BM25/semantic balance (0.0=BM25, 1.0=semantic).
                   Defaults to config value.
            metadata_filters: Optional dict of field→value pairs for pre-
                               filtering. Applied as Weaviate where filters.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        cfg = get_config().retrieval
        k = top_k or cfg.hybrid.top_k
        a = alpha if alpha is not None else cfg.hybrid.alpha

        # Embed query for vector component
        query_vector = self._embedder.embed_query(query)

        # Build optional metadata filters
        weaviate_filter = _build_filter(metadata_filters) if metadata_filters else None

        logger.debug("HybridSearch: query=%r, top_k=%d, alpha=%.2f", query, k, a)

        collection = self._client.collections.get(COLLECTION_NAME)

        kwargs: dict = {
            "query": query,
            "vector": query_vector,
            "alpha": a,
            "limit": k,
            "return_metadata": wvc.query.MetadataQuery(score=True, distance=True),
        }
        if weaviate_filter:
            kwargs["filters"] = weaviate_filter

        response = collection.query.hybrid(**kwargs)

        results: list[SearchResult] = []
        for obj in response.objects:
            try:
                chunk = _row_to_chunk(obj)
                score = obj.metadata.score or 0.0
                results.append(SearchResult(chunk=chunk, score=score, search_type="hybrid"))
            except Exception as exc:
                logger.warning("Skipping malformed search result: %s", exc)

        logger.debug("HybridSearch returned %d results", len(results))
        return results

    def bm25_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filters: dict | None = None,
    ) -> list[SearchResult]:
        """Pure BM25 keyword search.

        Args:
            query: Keyword search query.
            top_k: Maximum results.
            metadata_filters: Optional field→value pre-filters.

        Returns:
            List of SearchResult objects sorted by BM25 score.
        """
        cfg = get_config().retrieval
        k = top_k or cfg.bm25.top_k
        weaviate_filter = _build_filter(metadata_filters) if metadata_filters else None

        collection = self._client.collections.get(COLLECTION_NAME)

        kwargs: dict = {
            "query": query,
            "limit": k,
            "return_metadata": wvc.query.MetadataQuery(score=True),
        }
        if weaviate_filter:
            kwargs["filters"] = weaviate_filter

        response = collection.query.bm25(**kwargs)

        results: list[SearchResult] = []
        for obj in response.objects:
            try:
                chunk = _row_to_chunk(obj)
                score = obj.metadata.score or 0.0
                results.append(SearchResult(chunk=chunk, score=score, search_type="bm25"))
            except Exception as exc:
                logger.warning("Skipping malformed BM25 result: %s", exc)

        return results

    def semantic_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filters: dict | None = None,
    ) -> list[SearchResult]:
        """Pure semantic (dense vector) search.

        Args:
            query: Natural language query.
            top_k: Maximum results.
            metadata_filters: Optional field→value pre-filters.

        Returns:
            List of SearchResult objects sorted by cosine similarity.
        """
        cfg = get_config().retrieval
        k = top_k or cfg.semantic.top_k
        query_vector = self._embedder.embed_query(query)
        weaviate_filter = _build_filter(metadata_filters) if metadata_filters else None

        collection = self._client.collections.get(COLLECTION_NAME)

        kwargs: dict = {
            "near_vector": query_vector,
            "limit": k,
            "return_metadata": wvc.query.MetadataQuery(distance=True),
        }
        if weaviate_filter:
            kwargs["filters"] = weaviate_filter

        response = collection.query.near_vector(**kwargs)

        results: list[SearchResult] = []
        for obj in response.objects:
            try:
                chunk = _row_to_chunk(obj)
                distance = obj.metadata.distance or 1.0
                score = 1.0 - distance  # Convert distance to similarity
                results.append(SearchResult(chunk=chunk, score=score, search_type="semantic"))
            except Exception as exc:
                logger.warning("Skipping malformed semantic result: %s", exc)

        return results


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter(metadata_filters: dict) -> wvc.query.Filter:
    """Convert a simple dict of field→value pairs to a Weaviate Filter.

    Supports:
      - Exact match: {"source_type": "pdf"}
      - List of values (OR): {"source_type": ["pdf", "pptx"]}

    Multiple keys are ANDed together.

    Args:
        metadata_filters: Dict of property_name → value or list of values.

    Returns:
        A Weaviate Filter object.
    """
    filters: list[wvc.query.Filter] = []

    for field, value in metadata_filters.items():
        if isinstance(value, list):
            # OR across list values
            sub = [wvc.query.Filter.by_property(field).equal(v) for v in value]
            filters.append(wvc.query.Filter.any_of(sub))
        else:
            filters.append(wvc.query.Filter.by_property(field).equal(value))

    if len(filters) == 1:
        return filters[0]
    return wvc.query.Filter.all_of(filters)
