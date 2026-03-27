"""
Retrieval evaluation metrics: Precision@k, Recall@k, MAP@k, MRR@k.

These metrics require a ground-truth relevance set (list of relevant chunk IDs
or source files) and the ordered list of retrieved SearchResult objects.

Usage:
    from src.evaluation.metrics import evaluate_retrieval
    scores = evaluate_retrieval(results, relevant_ids={"chunk-001", "chunk-002"}, k=10)
    print(scores)  # {"precision@10": 0.2, "recall@10": 1.0, "map@10": 0.6, "mrr@10": 0.5}
"""

from __future__ import annotations

import logging

from src.models import SearchResult

logger = logging.getLogger(__name__)


def precision_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k.

    Args:
        results: Ordered list of retrieved results.
        relevant_ids: Set of chunk_ids that are considered relevant.
        k: Cut-off rank.

    Returns:
        Precision@k in [0.0, 1.0].
    """
    if k == 0 or not results:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.chunk.chunk_id in relevant_ids)
    return hits / k


def recall_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k.

    Args:
        results: Ordered list of retrieved results.
        relevant_ids: Set of chunk_ids that are considered relevant.
        k: Cut-off rank.

    Returns:
        Recall@k in [0.0, 1.0]. Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids or not results:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.chunk.chunk_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Compute Average Precision@k for a single query.

    Args:
        results: Ordered list of retrieved results.
        relevant_ids: Set of relevant chunk_ids.
        k: Cut-off rank.

    Returns:
        AP@k in [0.0, 1.0].
    """
    if not relevant_ids or not results:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, result in enumerate(results[:k], start=1):
        if result.chunk.chunk_id in relevant_ids:
            hits += 1
            sum_precision += hits / i

    if hits == 0:
        return 0.0
    return sum_precision / min(len(relevant_ids), k)


def mean_average_precision(
    queries_results: list[list[SearchResult]],
    queries_relevant: list[set[str]],
    k: int,
) -> float:
    """Compute MAP@k across multiple queries.

    Args:
        queries_results: One ordered result list per query.
        queries_relevant: One relevant_ids set per query.
        k: Cut-off rank.

    Returns:
        MAP@k in [0.0, 1.0].
    """
    if not queries_results:
        return 0.0
    aps = [
        average_precision_at_k(results, relevant, k)
        for results, relevant in zip(queries_results, queries_relevant)
    ]
    return sum(aps) / len(aps)


def reciprocal_rank_at_k(
    results: list[SearchResult], relevant_ids: set[str], k: int
) -> float:
    """Compute Reciprocal Rank@k for a single query.

    Args:
        results: Ordered list of retrieved results.
        relevant_ids: Set of relevant chunk_ids.
        k: Cut-off rank.

    Returns:
        RR@k in [0.0, 1.0]. Returns 0 if no relevant result in top-k.
    """
    for i, result in enumerate(results[:k], start=1):
        if result.chunk.chunk_id in relevant_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    queries_results: list[list[SearchResult]],
    queries_relevant: list[set[str]],
    k: int,
) -> float:
    """Compute MRR@k across multiple queries.

    Args:
        queries_results: One ordered result list per query.
        queries_relevant: One relevant_ids set per query.
        k: Cut-off rank.

    Returns:
        MRR@k in [0.0, 1.0].
    """
    if not queries_results:
        return 0.0
    rrs = [
        reciprocal_rank_at_k(results, relevant, k)
        for results, relevant in zip(queries_results, queries_relevant)
    ]
    return sum(rrs) / len(rrs)


def evaluate_retrieval(
    results: list[SearchResult],
    relevant_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a single query retrieval against a ground-truth set.

    Computes Precision@k, Recall@k, AP@k, and RR@k.

    Args:
        results: Ordered search results for one query.
        relevant_ids: Set of chunk_ids that are considered correct.
        k: Cut-off rank for all metrics.

    Returns:
        Dict with keys: "precision@{k}", "recall@{k}", "map@{k}", "mrr@{k}".
    """
    return {
        f"precision@{k}": precision_at_k(results, relevant_ids, k),
        f"recall@{k}": recall_at_k(results, relevant_ids, k),
        f"map@{k}": average_precision_at_k(results, relevant_ids, k),
        f"mrr@{k}": reciprocal_rank_at_k(results, relevant_ids, k),
    }


def evaluate_retrieval_batch(
    queries_results: list[list[SearchResult]],
    queries_relevant: list[set[str]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate retrieval across a batch of queries.

    Aggregates per-query scores into MAP and MRR.

    Args:
        queries_results: One result list per query.
        queries_relevant: One relevant_ids set per query.
        k: Cut-off rank.

    Returns:
        Dict with keys: "precision@{k}", "recall@{k}", "map@{k}", "mrr@{k}".
        Precision and Recall are macro-averaged across queries.
    """
    if not queries_results:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"map@{k}": 0.0, f"mrr@{k}": 0.0}

    precisions = [
        precision_at_k(r, rel, k)
        for r, rel in zip(queries_results, queries_relevant)
    ]
    recalls = [
        recall_at_k(r, rel, k)
        for r, rel in zip(queries_results, queries_relevant)
    ]

    return {
        f"precision@{k}": sum(precisions) / len(precisions),
        f"recall@{k}": sum(recalls) / len(recalls),
        f"map@{k}": mean_average_precision(queries_results, queries_relevant, k),
        f"mrr@{k}": mean_reciprocal_rank(queries_results, queries_relevant, k),
    }
