"""Shared CSS constants — Claude-inspired UI with Siemens brand color."""

SIEMENS_TEAL = "#009999"
SIEMENS_TEAL_HOVER = "#007a7a"
SIEMENS_TEAL_LIGHT = "#f0fafa"

CUSTOM_CSS = f"""
<style>
    /* ── Accent colour (buttons, links, highlights) ─────────────────────── */
    .stButton > button {{
        background-color: {SIEMENS_TEAL};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1rem;
    }}
    .stButton > button:hover {{
        background-color: {SIEMENS_TEAL_HOVER};
        color: white;
    }}

    /* ── Thinking block (collapsed by default, scrollable) ──────────────── */
    .thinking-block {{
        border-left: 3px solid {SIEMENS_TEAL};
        padding: 8px 12px;
        margin: 8px 0;
        background-color: {SIEMENS_TEAL_LIGHT};
        max-height: 200px;
        overflow-y: auto;
        font-size: 0.85em;
        color: #555;
        border-radius: 0 6px 6px 0;
    }}

    /* ── Chat messages ──────────────────────────────────────────────────── */
    .stChatMessage {{
        padding: 12px 16px;
    }}

    /* ── Sidebar (dark background) ──────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: #1a1a2e;
        color: #e0e0e0;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stSelectbox label {{
        color: #e0e0e0 !important;
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {SIEMENS_TEAL} !important;
    }}

    /* ── Expander header colour ──────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        color: {SIEMENS_TEAL} !important;
        font-size: 0.85em;
    }}

    /* ── Metrics caption ────────────────────────────────────────────────── */
    .stCaption {{
        font-size: 0.75em;
        color: #888;
    }}

    /* ── Page title accent ──────────────────────────────────────────────── */
    h1 {{
        color: white !important;
    }}
</style>
"""
