"""Level 1 — Prompt Only: Reflex chat UI entry point.

Defines the main page layout with a settings sidebar and streaming
chat interface. Theme uses Siemens teal (#009999) in dark mode.
"""

import reflex as rx

from .state import ChatState

# ── Style Constants ─────────────────────────────────────────────────────

SIEMENS_TEAL = "#009999"
METRICS_GRAY = "rgb(127, 127, 127)"


# ── Sidebar ─────────────────────────────────────────────────────────────


def sidebar() -> rx.Component:
    """Render the settings sidebar with model and parameter controls."""
    return rx.box(
        rx.vstack(
            rx.heading("Settings", size="4"),
            rx.separator(),
            rx.vstack(
                rx.text("Model", weight="bold", size="2"),
                rx.select(
                    ChatState.available_models,
                    value=ChatState.model,
                    on_change=ChatState.set_model,
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
            spacing="4",
            padding="1.5em",
            width="100%",
        ),
        width="280px",
        min_width="280px",
        height="100vh",
        background=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 5)}",
        overflow_y="auto",
    )


# ── Message Components ──────────────────────────────────────────────────


def message_bubble(msg: dict[str, str]) -> rx.Component:
    """Render a single chat message with optional thinking and metrics.

    Args:
        msg: Message dict with keys: role, content, thinking, metrics.
    """
    return rx.cond(
        msg["role"] == "user",
        _user_bubble(msg),
        _assistant_bubble(msg),
    )


def _user_bubble(msg: dict[str, str]) -> rx.Component:
    """Render a user message bubble aligned to the right."""
    return rx.box(
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
    )


def _assistant_bubble(msg: dict[str, str]) -> rx.Component:
    """Render an assistant message with thinking, content, and metrics."""
    return rx.box(
        rx.box(
            # Thinking (collapsible accordion)
            rx.cond(
                msg["thinking"] != "",
                rx.box(
                    rx.accordion.root(
                        rx.accordion.item(
                            header=rx.text("Thinking", size="2"),
                            content=rx.box(
                                rx.text(
                                    msg["thinking"],
                                    white_space="pre-wrap",
                                    size="2",
                                    color=rx.color("gray", 11),
                                ),
                                max_height="300px",
                                overflow_y="auto",
                            ),
                            value="t",
                        ),
                        type="single",
                        collapsible=True,
                        variant="ghost",
                    ),
                    margin_bottom="0.5em",
                ),
            ),
            # Answer content (markdown)
            rx.markdown(msg["content"]),
            # Metrics footer (gray)
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
    )


def streaming_response() -> rx.Component:
    """Render the currently streaming LLM response with live thinking."""
    return rx.box(
        rx.box(
            # Live thinking (accordion open by default)
            rx.cond(
                ChatState.current_thinking != "",
                rx.box(
                    rx.accordion.root(
                        rx.accordion.item(
                            header=rx.text("Thinking...", size="2"),
                            content=rx.box(
                                rx.text(
                                    ChatState.current_thinking,
                                    white_space="pre-wrap",
                                    size="2",
                                    color=rx.color("gray", 11),
                                ),
                                max_height="300px",
                                overflow_y="auto",
                            ),
                            value="thinking",
                        ),
                        type="single",
                        collapsible=True,
                        variant="ghost",
                        default_value=["thinking"],
                    ),
                    margin_bottom="0.5em",
                ),
            ),
            # Live answer or spinner
            rx.cond(
                ChatState.current_answer != "",
                rx.markdown(ChatState.current_answer),
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Generating...", size="2", color="gray"),
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


# ── Chat Input ──────────────────────────────────────────────────────────


def chat_input() -> rx.Component:
    """Render the message input form at the bottom of the chat area."""
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


# ── Page Layout ─────────────────────────────────────────────────────────


def index() -> rx.Component:
    """Main page: settings sidebar + chat area with streaming."""
    return rx.flex(
        sidebar(),
        rx.flex(
            # Scrollable message area
            rx.box(
                rx.vstack(
                    rx.foreach(ChatState.messages, message_bubble),
                    rx.cond(ChatState.is_generating, streaming_response()),
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
            height="100vh",
        ),
        direction="row",
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
    index,
    title="Level 1 — Prompt Only",
    on_load=ChatState.on_load,
)
