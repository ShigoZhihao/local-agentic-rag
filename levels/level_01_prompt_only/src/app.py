"""Level 1 — Prompt Only.

What this level demonstrates:
    The most basic agent loop: user input → LLM → output.
    Tokens are streamed in real time.
    Resource usage (CPU, GPU, VRAM, RAM) is measured as delta during generation.
"""

import time

import streamlit as st

from config import get_config
from llm_client import StreamingChat, create_client
from metrics import format_metrics, measure_delta, take_snapshot

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Level 1 — Prompt Only", layout="wide")
st.title("Level 1 — Prompt Only")
st.caption("Core loop: user input → LLM → output. No tools, no retrieval.")

# ── Load config once per session ──────────────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = get_config()

cfg = st.session_state.cfg

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value=cfg.agent.system_prompt,
        height=160,
        help="Edit and press Ctrl+Enter to apply. Takes effect on the next message.",
    )
    st.divider()
    st.markdown(f"**Model:** `{cfg.ollama.model}`")
    st.markdown(f"**Temperature:** `{cfg.ollama.temperature}`")
    st.markdown(f"**Max tokens:** `{cfg.ollama.max_tokens}`")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "metrics"}

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("metrics"):
            st.caption(msg["metrics"])

# ── Handle new input ──────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask anything..."):
    # Timer starts the moment the user submits
    t_start = time.perf_counter()

    # Baseline snapshot primes CPU/GPU counters for delta measurement
    baseline = take_snapshot()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    full_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        client = create_client(cfg.ollama)
        stream = StreamingChat(client, cfg.ollama, full_messages)

        # st.write_stream() renders each token as it arrives, returns full reply
        reply = st.write_stream(stream)

        elapsed = time.perf_counter() - t_start
        metrics = measure_delta(
            baseline=baseline,
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=stream.completion_tokens,
            elapsed_sec=elapsed,
        )
        metrics_str = format_metrics(metrics)
        st.caption(metrics_str)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "metrics": metrics_str,
    })
