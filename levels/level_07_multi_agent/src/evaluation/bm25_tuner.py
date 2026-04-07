"""
BM25 parameter tuner: grid search over k1 and b values.

Because Weaviate sets BM25 k1/b at collection creation time, tuning requires
deleting and recreating the collection for each (k1, b) combination, re-indexing
all documents, running evaluation queries, and recording the MAP@k score.

Usage:
    tuner = BM25Tuner(eval_queries, eval_relevant_ids)
    best = tuner.run()
    print(best)  # {"k1": 1.2, "b": 0.75, "map@10": 0.72}
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from src.config import get_config
from src.evaluation.metrics import evaluate_retrieval_batch
from src.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result of a single (k1, b) evaluation."""

    k1: float
    b: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def optimize_metric(self) -> float:
        """Return the primary optimization metric score."""
        cfg = get_config().evaluation.bm25_tuning
        return self.scores.get(cfg.optimize_metric, 0.0)


@dataclass
class BM25TuningReport:
    """Full grid-search results."""

    results: list[TuningResult] = field(default_factory=list)
    best: TuningResult | None = None

    def as_records(self) -> list[dict]:
        """Convert to list of dicts suitable for display in a DataFrame."""
        rows = []
        for r in self.results:
            row = {"k1": r.k1, "b": r.b}
            row.update(r.scores)
            rows.append(row)
        return rows


class BM25Tuner:
    """Grid-search BM25 k1/b parameters by evaluating MAP@k.

    For each (k1, b) pair:
    1. Recreate the Weaviate collection with the new BM25 parameters.
    2. Re-index all documents from an existing corpus.
    3. Run evaluation queries via HybridSearcher (alpha=0 for pure BM25).
    4. Compute metrics and record.

    Args:
        eval_queries: List of natural language query strings.
        eval_relevant_ids: List of sets of relevant chunk_ids (one per query).
        corpus_chunks: Pre-embedded chunks to re-index for each parameter combo.
                       If None, uses whatever is already indexed.
        k: Metric cut-off rank (default from config metrics).
        progress_callback: Optional callable(iteration, total, k1, b) for UI updates.
    """

    def __init__(
        self,
        eval_queries: list[str],
        eval_relevant_ids: list[set[str]],
        corpus_chunks: list | None = None,
        *,
        k: int = 10,
        progress_callback=None,
    ) -> None:
        self._queries = eval_queries
        self._relevant_ids = eval_relevant_ids
        self._corpus_chunks = corpus_chunks
        self._k = k
        self._progress_callback = progress_callback

    def run(self) -> BM25TuningReport:
        """Execute the BM25 grid search and return a report.

        Returns:
            BM25TuningReport with all (k1, b) scores and the best configuration.
        """
        cfg_eval = get_config().evaluation.bm25_tuning
        k1_values: list[float] = cfg_eval.k1_range
        b_values: list[float] = cfg_eval.b_range
        optimize_metric: str = cfg_eval.optimize_metric

        combos = list(itertools.product(k1_values, b_values))
        total = len(combos)
        logger.info("BM25 tuner: %d combinations to test (k1=%s, b=%s)", total, k1_values, b_values)

        report = BM25TuningReport()

        from src.retrieval.weaviate_client import ensure_collection, get_client, upsert_chunks
        from src.retrieval.hybrid_search import HybridSearcher
        from src.ingestion.embedder import Embedder

        client = get_client()
        embedder = Embedder()

        for iteration, (k1, b) in enumerate(combos, start=1):
            logger.info("Tuning %d/%d: k1=%.2f, b=%.2f", iteration, total, k1, b)

            try:
                # 1. Recreate collection with new BM25 params
                ensure_collection(client, bm25_k1=k1, bm25_b=b, recreate=True)

                # 2. Re-index corpus if provided
                if self._corpus_chunks:
                    upsert_chunks(client, self._corpus_chunks)
                    logger.info("  Re-indexed %d chunks", len(self._corpus_chunks))

                # 3. Run evaluation queries (pure BM25: alpha=0)
                searcher = HybridSearcher(client=client, embedder=embedder)
                queries_results: list[list[SearchResult]] = []
                for query in self._queries:
                    results = searcher.bm25_search(query, top_k=self._k)
                    queries_results.append(results)

                # 4. Compute metrics
                scores = evaluate_retrieval_batch(
                    queries_results, self._eval_relevant_ids, k=self._k
                )

                result = TuningResult(k1=k1, b=b, scores=scores)
                report.results.append(result)

                logger.info(
                    "  k1=%.2f b=%.2f → %s=%.4f",
                    k1, b, optimize_metric, result.optimize_metric,
                )

                if self._progress_callback:
                    self._progress_callback(iteration, total, k1, b)

            except Exception as exc:
                logger.error("Tuning failed for k1=%.2f b=%.2f: %s", k1, b, exc)
                report.results.append(
                    TuningResult(k1=k1, b=b, scores={optimize_metric: 0.0})
                )

        client.close()

        # Find best
        if report.results:
            report.best = max(report.results, key=lambda r: r.optimize_metric)
            logger.info(
                "BM25 tuning done. Best: k1=%.2f, b=%.2f, %s=%.4f",
                report.best.k1,
                report.best.b,
                optimize_metric,
                report.best.optimize_metric,
            )

        return report

    @property
    def _eval_relevant_ids(self) -> list[set[str]]:
        return self._relevant_ids
