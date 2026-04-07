"""Level 7 — Multi-Agent RAG: Reflex UI entry point.

Two-page application:
  - / (Chat): 4-agent RAG pipeline with citations and validation
  - /ingest: Document upload and indexing into Weaviate

Theme uses Siemens teal (#009999) in dark mode.
"""

import reflex as rx

from .state import ChatState, IngestState

# ── Style Constants ─────────────────────────────────────────────────────

SIEMENS_TEAL = "#009999"
METRICS_GRAY = "rgb(127, 127, 127)"


# ── Shared Components ───────────────────────────────────────────────────


def navbar() -> rx.Component:
    """Render the top navigation bar with page links."""
    return rx.hstack(
        rx.heading("Level 7 — Multi-Agent RAG", size="4"),
        rx.spacer(),
        rx.link("Chat", href="/", padding_x="1em"),
        rx.link("Ingest", href="/ingest", padding_x="1em"),
        width="100%",
        padding="1em",
        background=rx.color("gray", 2),
        border_bottom=f"1px solid {rx.color('gray', 5)}",
        align="center",
    )


# ── Chat Page ───────────────────────────────────────────────────────────


def chat_sidebar() -> rx.Component:
    """Render the chat settings sidebar with dual model selectors."""
    return rx.box(
        rx.vstack(
            rx.heading("Settings", size="4"),
            rx.separator(),
            rx.vstack(
                rx.text("Planner Model", weight="bold", size="2"),
                rx.text(
                    "Facilitator / Validator",
                    size="1", color="gray",
                ),
                rx.select(
                    ChatState.available_models,
                    value=ChatState.planner_model,
                    on_change=ChatState.set_planner_model,
                    width="100%",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text("Executor Model", weight="bold", size="2"),
                rx.text("Synthesizer", size="1", color="gray"),
                rx.select(
                    ChatState.available_models,
                    value=ChatState.executor_model,
                    on_change=ChatState.set_executor_model,
                    width="100%",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text(
                    "Temperature: ", ChatState.temperature,
                    weight="bold", size="2",
                ),
                rx.slider(
                    default_value=[0.7],
                    min=0, max=2, step=0.1,
                    on_value_commit=ChatState.set_temperature_value,
                    width="100%",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text(
                    "Max Tokens: ", ChatState.max_tokens,
                    weight="bold", size="2",
                ),
                rx.slider(
                    default_value=[128000],
                    min=1024, max=131072, step=1024,
                    on_value_commit=ChatState.set_max_tokens_value,
                    width="100%",
                ),
                spacing="1",
            ),
            rx.hstack(
                rx.switch(
                    checked=ChatState.use_colbert,
                    on_change=ChatState.set_use_colbert,
                ),
                rx.text("ColBERT search", size="2"),
                spacing="2",
            ),
            spacing="4",
            padding="1.5em",
            width="100%",
        ),
        width="280px",
        min_width="280px",
        height="100%",
        background=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 5)}",
        overflow_y="auto",
    )


def chat_message(msg: dict[str, str]) -> rx.Component:
    """Render a single chat message with optional thinking and metrics.

    Args:
        msg: Message dict with keys: role, content, thinking, metrics.
    """
    return rx.cond(
        msg["role"] == "user",
        rx.box(
            rx.box(
                rx.text(msg["content"], color="white"),
                background=SIEMENS_TEAL,
                padding="0.75em 1em",
                border_radius="16px 16px 4px 16px",
                max_width="70%",
            ),
            display="flex",
            justify_content="flex-end",
            width="100%",
            padding_y="0.25em",
        ),
        rx.box(
            rx.box(
                rx.markdown(msg["content"]),
                rx.cond(
                    msg["metrics"] != "",
                    rx.box(
                        rx.separator(margin_y="0.5em"),
                        rx.text(
                            msg["metrics"],
                            color=METRICS_GRAY,
                            size="1",
                        ),
                    ),
                ),
                background=rx.color("gray", 3),
                padding="1em",
                border_radius="16px 16px 16px 4px",
                max_width="85%",
            ),
            display="flex",
            justify_content="flex-start",
            width="100%",
            padding_y="0.25em",
        ),
    )


def agent_step(step: dict[str, str]) -> rx.Component:
    """Render an agent thinking step as a collapsible accordion.

    Args:
        step: Dict with agent name and thinking text.
    """
    return rx.accordion.root(
        rx.accordion.item(
            header=rx.text(
                step["agent"], " Thinking", size="2",
            ),
            content=rx.box(
                rx.text(
                    step["thinking"],
                    white_space="pre-wrap",
                    size="2",
                    color=rx.color("gray", 11),
                ),
                max_height="200px",
                overflow_y="auto",
            ),
            value="s",
        ),
        type="single",
        collapsible=True,
        variant="ghost",
        margin_bottom="0.25em",
    )


def streaming_chat() -> rx.Component:
    """Render the currently streaming chat response with agent steps."""
    return rx.box(
        rx.box(
            # Agent thinking steps
            rx.foreach(ChatState.agent_steps, agent_step),
            # Current answer or spinner
            rx.cond(
                ChatState.current_answer != "",
                rx.markdown(ChatState.current_answer),
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Processing...", size="2", color="gray"),
                    spacing="2",
                ),
            ),
            background=rx.color("gray", 3),
            padding="1em",
            border_radius="16px 16px 16px 4px",
            max_width="85%",
        ),
        display="flex",
        justify_content="flex-start",
        width="100%",
        padding_y="0.25em",
    )


def chat_input() -> rx.Component:
    """Render the chat message input form."""
    return rx.form(
        rx.hstack(
            rx.input(
                name="message",
                placeholder="Type a message...",
                size="3",
                flex="1",
                auto_focus=True,
                disabled=ChatState.is_generating,
            ),
            rx.button(
                rx.cond(
                    ChatState.is_generating,
                    rx.spinner(size="1"),
                    rx.text("Send"),
                ),
                type="submit",
                size="3",
                disabled=ChatState.is_generating,
            ),
            width="100%",
            spacing="2",
        ),
        on_submit=ChatState.handle_submit,
        reset_on_submit=True,
        width="100%",
        padding="1em",
        border_top=f"1px solid {rx.color('gray', 5)}",
    )


def chat_page() -> rx.Component:
    """Chat page: sidebar + message area + input."""
    return rx.flex(
        navbar(),
        rx.flex(
            chat_sidebar(),
            rx.flex(
                rx.box(
                    rx.vstack(
                        rx.foreach(ChatState.messages, chat_message),
                        rx.cond(
                            ChatState.is_generating,
                            streaming_chat(),
                        ),
                        spacing="1",
                        padding="1em",
                        width="100%",
                    ),
                    flex="1",
                    overflow_y="auto",
                ),
                chat_input(),
                direction="column",
                flex="1",
            ),
            direction="row",
            flex="1",
        ),
        direction="column",
        width="100vw",
        height="100vh",
    )


# ── Ingest Page ─────────────────────────────────────────────────────────


def ingest_page() -> rx.Component:
    """Document ingestion page with file upload and results display."""
    return rx.flex(
        navbar(),
        rx.container(
            rx.vstack(
                rx.heading("Document Ingestion", size="5"),
                rx.text(
                    "Upload files to index into Weaviate. "
                    "Supported: .txt, .md, .html, .py, .pdf, .pptx",
                    color="gray",
                ),
                rx.separator(),
                # Upload area
                rx.upload(
                    rx.vstack(
                        rx.button(
                            "Select Files",
                            color=SIEMENS_TEAL,
                            variant="outline",
                        ),
                        rx.text(
                            "or drag and drop files here",
                            size="2",
                            color="gray",
                        ),
                        align="center",
                        spacing="2",
                    ),
                    id="ingest_upload",
                    multiple=True,
                    accept={
                        "text/plain": [".txt", ".md", ".py"],
                        "text/html": [".html", ".htm"],
                        "application/pdf": [".pdf"],
                        "application/vnd.openxmlformats-officedocument"
                        ".presentationml.presentation": [".pptx"],
                    },
                    border=f"2px dashed {rx.color('gray', 6)}",
                    padding="2em",
                    width="100%",
                    border_radius="8px",
                ),
                rx.button(
                    rx.cond(
                        IngestState.is_processing,
                        rx.hstack(
                            rx.spinner(size="1"),
                            rx.text("Processing..."),
                        ),
                        rx.text("Upload & Ingest"),
                    ),
                    on_click=IngestState.handle_upload(
                        rx.upload_files(upload_id="ingest_upload"),
                    ),
                    disabled=IngestState.is_processing,
                    width="100%",
                    size="3",
                ),
                # Results
                rx.cond(
                    IngestState.results.length() > 0,
                    rx.box(
                        rx.heading("Results", size="4"),
                        rx.foreach(
                            IngestState.results,
                            lambda r: rx.text(f"• {r}", size="2"),
                        ),
                        rx.cond(
                            IngestState.collection_count > 0,
                            rx.text(
                                f"Total indexed chunks: ",
                                rx.text(
                                    IngestState.collection_count,
                                    weight="bold",
                                ),
                                size="2",
                                margin_top="0.5em",
                            ),
                        ),
                        padding="1em",
                        background=rx.color("gray", 3),
                        border_radius="8px",
                        width="100%",
                    ),
                ),
                spacing="4",
                padding="2em",
                max_width="600px",
                width="100%",
            ),
            padding_top="2em",
        ),
        direction="column",
        width="100vw",
        height="100vh",
    )


# ── App Definition ──────────────────────────────────────────────────────

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="teal",
    ),
)
app.add_page(
    chat_page,
    route="/",
    title="Level 7 — Chat",
    on_load=ChatState.on_load,
)
app.add_page(
    ingest_page,
    route="/ingest",
    title="Level 7 — Ingest",
)
