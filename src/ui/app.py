"""
Streamlit main application entry point.

Run with:
    streamlit run src/ui/app.py

Pages:
    1. Chat — Agentic RAG chat with citation display
    2. Ingest — Upload and index documents
    3. Tuning — BM25 parameter grid search
    4. Evaluation — Retrieval metric evaluation
"""

import logging

import streamlit as st

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "💬 Chat": "chat",
    "📥 Ingest": "ingest",
    "🎛️ BM25 Tuning": "tuning",
    "📊 Evaluation": "evaluation",
}

with st.sidebar:
    st.title("Agentic RAG")
    st.caption("Local LLM · Weaviate · LangGraph")
    st.divider()
    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

page_key = PAGES[selected]

if page_key == "chat":
    from src.ui.pages.chat import render
    render()
elif page_key == "ingest":
    from src.ui.pages.ingest import render
    render()
elif page_key == "tuning":
    from src.ui.pages.tuning import render
    render()
elif page_key == "evaluation":
    from src.ui.pages.evaluation import render
    render()
