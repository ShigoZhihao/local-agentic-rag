"""Application state for Level 1 — Prompt Only.

Manages chat history, LLM streaming, model settings, and system metrics.
State is server-side per-user, synchronized via WebSocket.
"""

from __future__ import annotations

import logging
import time

import reflex as rx

logger = logging.getLogger(__name__)


class ChatState(rx.State):
    """Reactive state for the chat application.

    Handles model selection, parameter tuning, message history,
    and streaming LLM responses with thinking display.
    """

    # ── Settings ───────────────────────────────────────────────────────
    available_models: list[str] = []
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 128000

    # ── Chat ───────────────────────────────────────────────────────────
    messages: list[dict[str, str]] = []
    current_thinking: str = ""
    current_answer: str = ""
    is_generating: bool = False

    # ── Private (backend-only, not sent to frontend) ──────────────────
    _history: list[dict[str, str]] = []

    @rx.event
    def on_load(self) -> None:
        """Load configuration and populate available models on page load."""
        from src.config import get_config
        from src.ollama_models import list_models

        cfg = get_config()
        ollama_base = cfg.ollama.base_url.replace("/v1", "")
        models = list_models(ollama_base) or [cfg.ollama.model]

        self.available_models = models
        self.model = (
            cfg.ollama.model if cfg.ollama.model in models else models[0]
        )
        self.temperature = cfg.ollama.temperature

    @rx.event
    def set_temperature_value(self, value: list[float]) -> None:
        """Update temperature from slider value."""
        self.temperature = round(value[0], 1)

    @rx.event
    def set_max_tokens_value(self, value: list[float]) -> None:
        """Update max tokens from slider value."""
        self.max_tokens = int(value[0])

    @rx.event
    async def handle_submit(self, form_data: dict) -> None:
        """Process user message: stream LLM response with thinking.

        Args:
            form_data: Form submission data containing the message text.
        """
        message = form_data.get("message", "").strip()
        if not message or self.is_generating:
            return

        # Add user message
        self.messages.append({
            "role": "user",
            "content": message,
            "thinking": "",
            "metrics": "",
        })
        self.is_generating = True
        self.current_thinking = ""
        self.current_answer = ""
        yield

        from src.config import OllamaConfig, get_config
        from src.llm_client import StreamingChat, create_client
        from src.metrics import format_metrics, measure_delta, take_snapshot

        cfg = get_config()
        runtime_cfg = OllamaConfig(
            base_url=cfg.ollama.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        client = create_client(runtime_cfg)

        # Build message list with conversation history
        full_messages = [
            {"role": "system", "content": cfg.agent.system_prompt},
        ]
        full_messages.extend(self._history)
        full_messages.append({"role": "user", "content": message})

        t_start = time.perf_counter()
        baseline = take_snapshot()

        metrics_str = ""
        try:
            stream = StreamingChat(client, runtime_cfg, full_messages)

            for state, token in stream:
                if state == "thinking":
                    self.current_thinking += token
                else:
                    self.current_answer += token
                yield

            elapsed = time.perf_counter() - t_start
            metrics = measure_delta(
                baseline=baseline,
                prompt_tokens=stream.prompt_tokens,
                completion_tokens=stream.completion_tokens,
                elapsed_sec=elapsed,
            )
            metrics_str = format_metrics(metrics) + f" | model: {self.model}"

        except Exception as e:
            logger.error("LLM streaming error: %s", e, exc_info=True)
            self.current_answer = f"Error: {e}"

        # Finalize assistant message
        self.messages.append({
            "role": "assistant",
            "content": self.current_answer,
            "thinking": self.current_thinking,
            "metrics": metrics_str,
        })

        # Update conversation history
        self._history.append({"role": "user", "content": message})
        self._history.append({
            "role": "assistant",
            "content": self.current_answer,
        })

        self.is_generating = False
        self.current_thinking = ""
        self.current_answer = ""
        yield
