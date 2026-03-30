"""Level 1 — Prompt Only.

What this level demonstrates:
    The most basic agent loop: user input → LLM → output.
    No tools, no retrieval, no external memory.
    The only levers are the system prompt and model settings.
"""

import streamlit as st

from src.config import get_config
from src.llm_client import chat, create_client

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Level 1 — Prompt Only", layout="wide")
st.title("Level 1 — Prompt Only")
st.caption("Core loop: user input → LLM → output. No tools, no retrieval.")

# ── Load config once per session ──────────────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = get_config()

cfg = st.session_state.cfg

# ── Sidebar: editable system prompt & model info ──────────────────────────────
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

# ── Session state: conversation history ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new user input ─────────────────────────────────────────────────────
if user_input := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build full message list: system prompt + history
    full_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            client = create_client(cfg.ollama)
            reply = chat(client, cfg.ollama, full_messages)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
