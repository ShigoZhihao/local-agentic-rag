"""
Ollama LLM client.

All LLM calls in this project go through this module.
Never import openai directly from other modules.

Two roles:
- planner: gemma4:e4b   (Facilitator, Validator — reasoning tasks)
- executor: gemma4:e2b  (Synthesizer — generation tasks)

Ollama exposes an OpenAI-compatible API at http://127.0.0.1:11434/v1,
so we use the openai SDK with a custom base_url.
All models are called via the `model` parameter in each API request.

Thinking support:
    gemma4 models emit <think>...</think> before the answer.
    strip_thinking() separates thinking from the final answer.

Note on JSON mode: Ollama supports structured output via response_format
with models that have a template supporting it. If a model ignores it,
the JSON parsing in each agent falls back gracefully.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import get_config

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLMResponse:
    """Separates thinking from the final answer."""
    thinking: str
    answer: str


def strip_thinking(text: str) -> LLMResponse:
    """Parse ``<think>...</think>`` from model output.

    Args:
        text: Raw model response.

    Returns:
        LLMResponse with thinking and answer separated.
    """
    match = _THINK_RE.search(text)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return LLMResponse(thinking=thinking, answer=answer)
    return LLMResponse(thinking="", answer=text.strip())


def _get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at Ollama."""
    cfg = get_config()
    return OpenAI(
        base_url=cfg.ollama.base_url,
        api_key="ollama",  # Ollama does not require a real key
    )


def call_planner(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Call the planner model (gemma4:e4b) for reasoning tasks.

    Used by: Facilitator, Validator.

    Args:
        messages: List of chat messages in OpenAI format.
        model: Override the planner model name from config.
        temperature: Override the default temperature from config.
        max_tokens: Override the default max_tokens from config.
        json_mode: If True, request JSON output format.

    Returns:
        LLMResponse with thinking and answer separated.
    """
    cfg = get_config().ollama
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": model or cfg.planner_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    logger.debug("call_planner: model=%s, messages=%d", cfg.planner_model, len(messages))
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return strip_thinking(content)


def call_executor(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Call the executor model (gemma4:e2b) for generation tasks.

    Used by: Synthesizer.

    Args:
        messages: List of chat messages in OpenAI format.
        model: Override the executor model name from config.
        temperature: Override the default temperature from config.
        max_tokens: Override the default max_tokens from config.
        json_mode: If True, request JSON output format.

    Returns:
        LLMResponse with thinking and answer separated.
    """
    cfg = get_config().ollama
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": model or cfg.executor_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    logger.debug("call_executor: model=%s, messages=%d", cfg.executor_model, len(messages))
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return strip_thinking(content)


def call_vision(
    image_path: str,
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> str:
    """Call the vision model to describe an image.

    Used during ingestion of PDF/PPTX files.
    Only loaded when needed — do not call during query time.

    Args:
        image_path: Absolute path to the image file (PNG/JPG).
        prompt: Instruction for the vision model.
        max_tokens: Override the default max_tokens from vision config.

    Returns:
        The model's description of the image.
    """
    import base64
    from pathlib import Path

    cfg = get_config().vision
    client = _get_client()

    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.debug("call_vision: model=%s, image=%s", cfg.model_name, image_path)
    response = client.chat.completions.create(
        model=cfg.model_name,
        messages=messages,
        max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()
