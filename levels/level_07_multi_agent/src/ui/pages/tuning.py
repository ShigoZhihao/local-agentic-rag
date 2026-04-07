"""
BM25 Tuning page — Grid search over k1 and b parameters.

Features:
- Input evaluation queries and relevant chunk IDs via text area (JSONL format)
- Run grid search with live progress
- Display results as a heatmap and sortable table
- Apply best parameters to the live collection
"""

from __future__ import annotations

import json
import logging

import streamlit as st

logger = logging.getLogger(__name__)

_EXAMPLE_JSONL = """\
{"query": "How to configure logging", "relevant_ids": ["chunk-001", "chunk-002"]}
{"query": "How to handle authentication errors", "relevant_ids": ["chunk-010"]}"""


def _parse_eval_data(text: str) -> tuple[list[str], list[set[str]]]:
    """Parse JSONL evaluation data.

    Each line: {"query": str, "relevant_ids": list[str]}

    Returns:
        (queries, relevant_ids_per_query)
    """
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
            logger.warning("Skipping invalid JSONL line: %s — %s", line[:60], exc)
    return queries, relevant


def render() -> None:
    """Render the BM25 Tuning page."""
    st.title("🎛️ BM25 Parameter Tuning")
    st.caption("Search for optimal k1 and b values using grid search. Evaluation queries and their ground-truth chunk IDs are required.")

    # Evaluation data input
    st.subheader("1. Evaluation Data (JSONL)")
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

    # Config display
    with st.expander("📋 Grid Settings (config.yaml)"):
        from src.config import get_config
        cfg = get_config().evaluation.bm25_tuning
        st.json({
            "k1_range": cfg.k1_range,
            "b_range": cfg.b_range,
            "optimize_metric": cfg.optimize_metric,
            "total_combinations": len(cfg.k1_range) * len(cfg.b_range),
        })

    st.divider()

    if not queries:
        st.info("Please enter evaluation data before starting the tuning.")
        return

    # Run tuning
    if st.button("Start tuning", type="primary"):
        _run_tuning(queries, relevant_ids)

    # Display previous results if stored
    if "tuning_report" in st.session_state:
        _display_results(st.session_state.tuning_report)


def _run_tuning(queries: list[str], relevant_ids: list[set[str]]) -> None:
    """Execute BM25 grid search and store results in session state."""
    from src.evaluation.bm25_tuner import BM25Tuner
    from src.config import get_config
    cfg = get_config().evaluation.bm25_tuning
    total_combos = len(cfg.k1_range) * len(cfg.b_range)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(iteration: int, total: int, k1: float, b: float) -> None:
        progress_bar.progress(iteration / total)
        status_text.info(f"Testing {iteration}/{total}: k1={k1:.2f}, b={b:.2f}")

    tuner = BM25Tuner(
        queries,
        relevant_ids,
        k=10,
        progress_callback=on_progress,
    )

    with st.spinner(f"Testing {total_combos} parameter combinations..."):
        try:
            report = tuner.run()
            st.session_state.tuning_report = report
            status_text.success("Tuning complete!")
            progress_bar.progress(1.0)
            _display_results(report)
        except Exception as exc:
            logger.error("Tuning failed: %s", exc, exc_info=True)
            st.error(f"Tuning error: {exc}")


def _display_results(report) -> None:
    """Display tuning results as table and heatmap."""
    import pandas as pd

    if not report.results:
        st.warning("No results available.")
        return

    st.subheader("2. Results")

    # Best result callout
    if report.best:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best k1", f"{report.best.k1:.2f}")
        with col2:
            st.metric("Best b", f"{report.best.b:.2f}")
        with col3:
            from src.config import get_config
            metric = get_config().evaluation.bm25_tuning.optimize_metric
            st.metric(f"Best {metric}", f"{report.best.optimize_metric:.4f}")

    # Results table
    records = report.as_records()
    df = pd.DataFrame(records)
    st.dataframe(df.sort_values("map@10", ascending=False), use_container_width=True)

    # Apply best button
    if report.best and st.button("Apply best parameters and recreate collection"):
        _apply_best_params(report.best.k1, report.best.b)


def _apply_best_params(k1: float, b: float) -> None:
    """Recreate Weaviate collection with best-found BM25 params."""
    try:
        from src.retrieval.weaviate_client import get_client, ensure_collection
        client = get_client()
        with st.spinner("Recreating collection..."):
            ensure_collection(client, bm25_k1=k1, bm25_b=b, recreate=True)
        client.close()
        st.success(
            f"✅ Collection recreated (k1={k1:.2f}, b={b:.2f}). "
            "Please re-ingest your documents."
        )
    except Exception as exc:
        st.error(f"Parameter application error: {exc}")
