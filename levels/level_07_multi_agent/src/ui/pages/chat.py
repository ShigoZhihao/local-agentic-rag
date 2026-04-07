"""
Chat page — Agentic RAG chat interface.

Features:
- Multi-turn conversation with the 4-agent LangGraph pipeline
- Real-time streaming of agent events (facilitator → synthesizer → researcher → validator)
- Citation display with source/page info
- Validator score display
- Human-in-the-Loop: Facilitator clarification questions
- Metadata filter sidebar controls
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

from src.config import get_config
from src.generation.ollama_models import get_context_window, list_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "citations", "scores"}
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "waiting_for_user" not in st.session_state:
        st.session_state.waiting_for_user = False
    if "pending_state" not in st.session_state:
        st.session_state.pending_state = None


def _get_graph():
    if st.session_state.graph is None:
        from src.agents.graph import build_graph
        st.session_state.graph = build_graph(debug=False)
    return st.session_state.graph


# ---------------------------------------------------------------------------
# Citation display
# ---------------------------------------------------------------------------

def _render_citations(citations: list) -> None:
    if not citations:
        return
    with st.expander(f"📎 Citations ({len(citations)})", expanded=False):
        for cit in citations:
            page_info = f" — p.{cit.page_number}" if cit.page_number else ""
            st.markdown(f"**[{cit.citation_id}] {cit.source_file}{page_info}**")
            st.caption(cit.original_text[:400] + ("..." if len(cit.original_text) > 400 else ""))
            st.divider()


def _render_validation_score(scores: dict) -> None:
    if not scores:
        return
    avg = scores.get("average", 0)
    color = "green" if avg >= 80 else "orange" if avg >= 60 else "red"
    st.markdown(
        f"<span style='color:{color}'>■ Validation score: **{avg:.1f}/100**</span> "
        f"(Completeness:{scores.get('completeness',0)} / "
        f"Accuracy:{scores.get('accuracy',0)} / "
        f"Relevance:{scores.get('relevance',0)} / "
        f"Faithfulness:{scores.get('faithfulness',0)})",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Chat page."""
    _init_session()

    st.title("💬 Agentic RAG Chat")

    # Sidebar: model settings + filters
    cfg = get_config()
    ollama_base = cfg.ollama.base_url.replace("/v1", "")

    with st.sidebar:
        st.subheader("Model Settings")
        available_models = list_models(ollama_base)

        default_planner_idx = (
            available_models.index(cfg.ollama.planner_model)
            if cfg.ollama.planner_model in available_models
            else 0
        )
        planner_model = st.selectbox(
            "Planner Model (Facilitator/Validator)",
            available_models,
            index=default_planner_idx,
        ) if available_models else cfg.ollama.planner_model

        default_executor_idx = (
            available_models.index(cfg.ollama.executor_model)
            if cfg.ollama.executor_model in available_models
            else 0
        )
        executor_model = st.selectbox(
            "Executor Model (Synthesizer)",
            available_models,
            index=default_executor_idx,
        ) if available_models else cfg.ollama.executor_model

        temperature = st.slider(
            "Temperature", 0.0, 2.0, cfg.ollama.temperature, 0.1,
        )

        ctx_window = get_context_window(planner_model, ollama_base)
        max_tokens = st.number_input(
            "Max tokens",
            min_value=1,
            max_value=ctx_window,
            value=min(128000, ctx_window),
        )

        # Save to session state for agents to use
        st.session_state.planner_model = planner_model
        st.session_state.executor_model = executor_model
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens

        st.divider()
        st.subheader("Search Filters")
        source_filter = st.multiselect(
            "File type",
            ["txt", "md", "html", "py", "pdf", "pptx"],
            default=[],
        )
        use_colbert = st.checkbox("Use ColBERT search", value=False)
        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.waiting_for_user = False
            st.session_state.pending_state = None
            st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("thinking"):
                for entry in msg["thinking"]:
                    with st.expander(
                        f"🧠 {entry['agent']} thinking", expanded=False,
                    ):
                        st.markdown(
                            f'<div class="thinking-block">'
                            f'{entry["thinking"]}</div>',
                            unsafe_allow_html=True,
                        )
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(msg["citations"])
            if msg.get("scores"):
                _render_validation_score(msg["scores"])

    # Input area
    if st.session_state.waiting_for_user:
        prompt = st.chat_input("Enter your response to the Facilitator...")
    else:
        prompt = st.chat_input("Enter your question...")

    if not prompt:
        return

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    graph = _get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    metadata_filters = {"source_types": source_filter} if source_filter else {}

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()

        final_answer = ""
        final_citations = []
        final_scores = {}
        final_thinking: list[dict] = []

        try:
            if st.session_state.waiting_for_user:
                # Resume graph after user provided clarification
                from src.agents.facilitator import enrich_with_user_response
                current_state = st.session_state.pending_state or {}
                new_state = enrich_with_user_response(current_state, prompt)
                graph.update_state(config, new_state)
                st.session_state.waiting_for_user = False
                st.session_state.pending_state = None
                stream_input = None  # Resume from current position
            else:
                from src.agents.graph import make_initial_state
                initial = make_initial_state(prompt)
                # Inject metadata filters and colbert setting into state via user_query
                # (they are accessed from QueryRequest in a full implementation;
                #  for simplicity we pass them as extra state keys here)
                initial["metadata_filters"] = metadata_filters  # type: ignore[typeddict-unknown-key]
                initial["use_colbert"] = use_colbert  # type: ignore[typeddict-unknown-key]
                stream_input = initial

            # Stream graph events
            for event in graph.stream(stream_input, config, stream_mode="values"):
                # Track agent progress
                if "facilitator" in str(event):
                    status_placeholder.info("🧠 Facilitator: Analyzing query...")
                if "synthesizer" in str(event):
                    status_placeholder.info("✍️ Synthesizer: Generating answer...")
                if "researcher" in str(event):
                    status_placeholder.info("🔍 Researcher: Searching documents...")
                if "validator" in str(event):
                    status_placeholder.info("✅ Validator: Evaluating answer...")

                # Extract current state values
                state = event if isinstance(event, dict) else {}

                # Check for Human-in-the-Loop pause
                if state.get("needs_user_input"):
                    feedback = state.get("feedback_to_user", "")
                    st.session_state.waiting_for_user = True
                    st.session_state.pending_state = state
                    answer_placeholder.warning(f"**From Facilitator:**\n\n{feedback}")
                    status_placeholder.empty()
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"**From Facilitator:**\n\n{feedback}"}
                    )
                    return

                # Capture answer
                if state.get("answer"):
                    final_answer = state["answer"]
                    answer_placeholder.markdown(final_answer)

                # Capture citations
                if state.get("citations"):
                    final_citations = state["citations"]

                # Capture agent thinking
                if state.get("agent_thinking"):
                    final_thinking = state["agent_thinking"]

                # Capture validation scores
                if state.get("validation") and state["validation"]:
                    v = state["validation"]
                    final_scores = {
                        "completeness": v.scores.completeness,
                        "accuracy": v.scores.accuracy,
                        "relevance": v.scores.relevance,
                        "faithfulness": v.scores.faithfulness,
                        "average": v.scores.average,
                    }

            status_placeholder.empty()

            # Final display
            if final_answer:
                # Show agent thinking in expandable panels
                if final_thinking:
                    for entry in final_thinking:
                        with st.expander(
                            f"🧠 {entry['agent']} thinking", expanded=True,
                        ):
                            st.markdown(
                                f'<div class="thinking-block">'
                                f'{entry["thinking"]}</div>',
                                unsafe_allow_html=True,
                            )
                answer_placeholder.markdown(final_answer)
                if final_citations:
                    _render_citations(final_citations)
                if final_scores:
                    _render_validation_score(final_scores)
                st.caption(
                    f"planner: {planner_model} | "
                    f"executor: {executor_model}"
                )
            else:
                answer_placeholder.warning("Could not generate an answer.")

        except Exception as exc:
            logger.error("Graph execution error: %s", exc, exc_info=True)
            status_placeholder.empty()
            answer_placeholder.error(f"An error occurred: {exc}")
            final_answer = f"Error: {exc}"

    # Save to message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "citations": final_citations,
        "scores": final_scores,
        "thinking": final_thinking,
    })
