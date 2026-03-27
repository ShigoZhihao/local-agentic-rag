"""
Evaluation page — Retrieval metric evaluation.

Features:
- Input evaluation queries and relevant chunk IDs (JSONL)
- Run retrieval with current BM25/hybrid settings
- Display Precision@k, Recall@k, MAP@k, MRR@k per query and aggregated
- Per-query result drill-down
"""

from __future__ import annotations

import json
import logging

import streamlit as st

logger = logging.getLogger(__name__)

_EXAMPLE_JSONL = """\
{"query": "How to output error logs", "relevant_ids": ["chunk-001", "chunk-002"]}
{"query": "How to load a config file", "relevant_ids": ["chunk-010", "chunk-011"]}"""


def _parse_eval_data(text: str) -> tuple[list[str], list[set[str]]]:
    """Parse JSONL eval data lines."""
    queries: list[str] = []
    relevant: list[set[str]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            queries.append(obj["query"])
            relevant.append(set(obj.get("relevant_ids", [])))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping line: %s", exc)
    return queries, relevant


def render() -> None:
    """Render the Evaluation page."""
    st.title("📊 Retrieval Accuracy Evaluation")
    st.caption("Computes Precision@k / Recall@k / MAP@k / MRR@k.")

    # Settings
    with st.sidebar:
        st.subheader("Evaluation Settings")
        k = st.slider("@k (cutoff)", min_value=1, max_value=20, value=10)
        search_mode = st.selectbox(
            "Search mode",
            ["hybrid", "bm25", "semantic"],
            index=0,
        )
        alpha = st.slider(
            "alpha (hybrid only)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0=BM25, 1=Semantic",
        )

    # Evaluation data input
    st.subheader("Evaluation Data (JSONL)")
    eval_text = st.text_area(
        "Each line: {\"query\": \"...\", \"relevant_ids\": [\"...\", ...]}",
        value=_EXAMPLE_JSONL,
        height=200,
    )

    queries, relevant_ids = _parse_eval_data(eval_text)
    if queries:
        st.success(f"{len(queries)} query/queries loaded.")
    else:
        st.warning("No valid queries found.")
        return

    if st.button("Run evaluation", type="primary"):
        _run_evaluation(queries, relevant_ids, k=k, search_mode=search_mode, alpha=alpha)


def _run_evaluation(
    queries: list[str],
    relevant_ids: list[set[str]],
    *,
    k: int,
    search_mode: str,
    alpha: float,
) -> None:
    """Run retrieval for all queries and compute metrics."""
    from src.evaluation.metrics import evaluate_retrieval, evaluate_retrieval_batch
    from src.retrieval.hybrid_search import HybridSearcher
    from src.ingestion.embedder import Embedder
    import pandas as pd

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        embedder = Embedder()
        searcher = HybridSearcher(embedder=embedder)
    except Exception as exc:
        st.error(f"Search engine initialization error: {exc}")
        return

    queries_results = []
    per_query_scores = []

    for i, (query, rel) in enumerate(zip(queries, relevant_ids)):
        status_text.info(f"Query {i+1}/{len(queries)}: {query[:60]}")
        progress_bar.progress(i / len(queries))

        try:
            if search_mode == "hybrid":
                results = searcher.search(query, top_k=k, alpha=alpha)
            elif search_mode == "bm25":
                results = searcher.bm25_search(query, top_k=k)
            else:
                results = searcher.semantic_search(query, top_k=k)
        except Exception as exc:
            logger.error("Search failed for %r: %s", query, exc)
            results = []

        queries_results.append(results)
        scores = evaluate_retrieval(results, rel, k=k)
        per_query_scores.append({"query": query, **scores})

    progress_bar.progress(1.0)
    status_text.success("Evaluation complete!")

    # Aggregated scores
    agg = evaluate_retrieval_batch(queries_results, relevant_ids, k=k)

    st.subheader("Aggregated Scores")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Precision@{k}", f"{agg[f'precision@{k}']:.4f}")
    col2.metric(f"Recall@{k}", f"{agg[f'recall@{k}']:.4f}")
    col3.metric(f"MAP@{k}", f"{agg[f'map@{k}']:.4f}")
    col4.metric(f"MRR@{k}", f"{agg[f'mrr@{k}']:.4f}")

    st.divider()

    # Per-query breakdown
    st.subheader("Per-Query Scores")
    df = pd.DataFrame(per_query_scores)
    st.dataframe(df, use_container_width=True)

    # Query drill-down
    st.subheader("Search Result Details")
    selected_idx = st.selectbox(
        "Select a query",
        range(len(queries)),
        format_func=lambda i: queries[i][:60],
    )

    if selected_idx is not None:
        selected_results = queries_results[selected_idx]
        selected_rel = relevant_ids[selected_idx]
        if selected_results:
            rows = []
            for rank, r in enumerate(selected_results, start=1):
                rows.append({
                    "rank": rank,
                    "hit": "✅" if r.chunk.chunk_id in selected_rel else "❌",
                    "score": f"{r.score:.4f}",
                    "source": r.chunk.source_file,
                    "chunk_id": r.chunk.chunk_id,
                    "preview": r.chunk.content[:100],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No results")
