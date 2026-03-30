"""Thin OpenAI-compatible client for Ollama.

Core loop at this level:
    user input → LLM → output

No tools, no retrieval, no memory beyond what the caller passes in.
"""

import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from config import OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    reply: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float


def create_client(cfg: OllamaConfig) -> OpenAI:
    """Create an OpenAI client pointed at a local Ollama instance.

    Args:
        cfg: Ollama connection settings.

    Returns:
        Configured OpenAI client.
    """
    return OpenAI(base_url=cfg.base_url, api_key="ollama")


def chat(
    client: OpenAI,
    cfg: OllamaConfig,
    messages: list[dict[str, str]],
) -> ChatResult:
    """Send messages to the model and return the reply with usage metrics.

    Args:
        client: OpenAI client instance.
        cfg: Ollama model settings.
        messages: Full conversation history including system prompt.

    Returns:
        ChatResult with reply text, token counts, and elapsed time.
    """
    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        elapsed = time.perf_counter() - t0
        usage = response.usage
        return ChatResult(
            reply=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            elapsed_sec=elapsed,
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise
