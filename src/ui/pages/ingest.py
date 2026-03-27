"""
Ingest page — Upload and index documents into Weaviate.

Features:
- File upload (drag & drop) for all supported formats
- Vision LLM toggle for PDF/PPTX visual content
- Per-file progress display
- Re-index / delete document buttons
- Collection stats display
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ["txt", "md", "html", "htm", "py", "pdf", "pptx"]


def _get_collection_stats() -> dict:
    """Fetch document count from Weaviate."""
    try:
        from src.retrieval.weaviate_client import get_client, COLLECTION_NAME
        client = get_client()
        collection = client.collections.get(COLLECTION_NAME)
        count = collection.aggregate.over_all(total_count=True).total_count
        client.close()
        return {"count": count or 0, "error": None}
    except Exception as exc:
        return {"count": 0, "error": str(exc)}


def render() -> None:
    """Render the Ingest page."""
    st.title("📥 Document Ingestion")

    # Sidebar settings
    with st.sidebar:
        st.subheader("Ingestion Settings")
        use_vision = st.checkbox(
            "Use Vision LLM (PDF/PPTX)",
            value=False,
            help="Converts slide images into text descriptions using the Vision LLM. Requires a Vision model in LM Studio.",
        )
        st.caption("⚠️ Vision mode slows down ingestion")

    # Collection stats
    with st.expander("📊 Weaviate Collection Info", expanded=True):
        stats = _get_collection_stats()
        if stats["error"]:
            st.error(f"Weaviate connection error: {stats['error']}")
            st.info("Start Docker Desktop and run `docker compose up -d`.")
        else:
            st.metric("Indexed chunk count", stats["count"])

    st.divider()

    # File upload
    uploaded_files = st.file_uploader(
        "Drag & drop files here (multiple files supported)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Supported files: .txt, .md, .html, .py, .pdf, .pptx")
        return

    st.write(f"{len(uploaded_files)} file(s) selected.")

    if st.button("Start ingestion", type="primary"):
        _run_ingestion(uploaded_files, use_vision=use_vision)


def _run_ingestion(uploaded_files, *, use_vision: bool) -> None:
    """Ingest all uploaded files and display progress."""
    from src.ingestion.pipeline import IngestionPipeline

    # Temporary directory for rendered images
    with tempfile.TemporaryDirectory() as tmp_dir:
        rendered_dir = Path(tmp_dir) / "rendered"

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        total = len(uploaded_files)
        all_stats = []

        pipeline = IngestionPipeline(
            use_vision=use_vision,
            rendered_dir=rendered_dir if use_vision else None,
        )

        try:
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.info(f"Processing: {uploaded_file.name} ({i+1}/{total})")
                progress_bar.progress((i) / total)

                # Save to temp file
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False, dir=tmp_dir
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                # Rename to original filename for correct source_type detection
                named_path = Path(tmp_dir) / uploaded_file.name
                Path(tmp_path).rename(named_path)

                stats = pipeline.ingest_file(named_path)
                all_stats.append((uploaded_file.name, stats))

                progress_bar.progress((i + 1) / total)

        finally:
            pipeline.close()

        status_text.success("Ingestion complete!")
        progress_bar.progress(1.0)

        # Display results
        with results_container:
            st.subheader("Results")
            for name, stats in all_stats:
                if stats.failed_files > 0:
                    st.error(f"❌ {name}: Error — {stats.errors[0][1] if stats.errors else 'Unknown'}")
                else:
                    st.success(f"✅ {name}: {stats.total_chunks} chunk(s) ingested successfully")

        # Refresh stats
        updated = _get_collection_stats()
        if not updated["error"]:
            st.metric("Total chunk count (updated)", updated["count"])
