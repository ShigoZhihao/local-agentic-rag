"""Application state for Level 2 — Prompt Engineering.

Extends Level 1 with prompt mode selection (basic, CoT, few-shot, structured)
and conversation history management with configurable window size.
"""

from __future__ import annotations

import logging
import time

import reflex as rx

logger = logging.getLogger(__name__)


class ChatState(rx.State):
    """Reactive state for the prompt engineering chat application.

    Adds prompt mode selection and history trimming on top of
    the basic chat streaming from Level 1.
    """

    # ── Settings ───────────────────────────────────────────────────────
    available_models: list[str] = []
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 128000
    prompt_modes: list[str] = []
    prompt_mode: str = "basic"

    # ── Chat ───────────────────────────────────────────────────────────
    messages: list[dict[str, str]] = []
    current_thinking: str = ""
    current_answer: str = ""
    is_generating: bool = False

    # ── Private (backend-only) ─────────────────────────────────────────
    _history: list[dict[str, str]] = []

    @rx.event
    def on_load(self) -> None:
        """Load configuration, models, and prompt modes on page load."""
        from src.config import get_config
        from src.ollama_models import list_models
        from src.prompts import PROMPT_MODES

        cfg = get_config()
        ollama_base = cfg.ollama.base_url.replace("/v1", "")
        models = list_models(ollama_base) or [cfg.ollama.model]

        self.available_models = models
        self.model = (
            cfg.ollama.model if cfg.ollama.model in models else models[0]
        )
        self.temperature = cfg.ollama.temperature

        self.prompt_modes = list(PROMPT_MODES.keys())
        self.prompt_mode = (
            cfg.agent.prompt_mode
            if cfg.agent.prompt_mode in self.prompt_modes
            else self.prompt_modes[0]
        )

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
        """Process user message with selected prompt mode and stream response.

        Args:
            form_data: Form submission data containing the message text.
        """
        message = form_data.get("message", "").strip()
        if not message or self.is_generating:
            return

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
        from src.prompts import PROMPT_MODES

        cfg = get_config()
        runtime_cfg = OllamaConfig(
            base_url=cfg.ollama.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        client = create_client(runtime_cfg)

        # Get system prompt for selected mode
        system_prompt = PROMPT_MODES.get(
            self.prompt_mode, PROMPT_MODES["basic"],
        )

        # Build messages with trimmed history
        max_turns = cfg.agent.max_history_turns
        trimmed = self._history[-(max_turns * 2):]
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(trimmed)
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
            metrics_str = (
                format_metrics(metrics)
                + f" | model: {self.model} | mode: {self.prompt_mode}"
            )

        except Exception as e:
            logger.error("LLM streaming error: %s", e, exc_info=True)
            self.current_answer = f"Error: {e}"

        self.messages.append({
            "role": "assistant",
            "content": self.current_answer,
            "thinking": self.current_thinking,
            "metrics": metrics_str,
        })

        self._history.append({"role": "user", "content": message})
        self._history.append({
            "role": "assistant",
            "content": self.current_answer,
        })

        self.is_generating = False
        self.current_thinking = ""
        self.current_answer = ""
        yield
